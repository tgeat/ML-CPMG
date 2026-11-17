"""Direct-CSR assembly plus energy-lumped non-Galerkin coarse hierarchy.

Only the PRECONDITIONER's coarse operators are changed.  The fine matrix is
the original harmonic face graph, assembled without COO expansion.  Public
state.A, state.preconditioner and state.setup_meta match the v2 interface.
All construction is per-build and timed; there is no cross-run cache.
"""
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_variable] = '1'

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits
from darcy_core import edges, action
from path_multigrid_v2 import Multigrid as _Base, _aggregate, _csr_bytes, _project


def _fine_csr(k, edge):
    """Fill sorted seven-point CSR directly from unchanged oriented faces."""
    n = k.shape[0]
    N = k.size
    tail, head, weights = edge
    degree = np.bincount(tail, minlength=N) + np.bincount(head, minlength=N) + 1
    nonzeros = int(np.sum(degree))
    index_type = np.int32 if max(N, nonzeros) < np.iinfo(np.int32).max else np.int64
    indptr = np.empty(N + 1, dtype=index_type)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    indices = np.empty(nonzeros, dtype=index_type)
    data = np.empty(nonzeros, dtype=np.float64)
    cursor = indptr[:-1].copy()
    diagonal = (np.bincount(tail, weights=weights, minlength=N)
                + np.bincount(head, weights=weights, minlength=N))
    faces_per_axis = (n - 1) * n * n

    def insert(rows, columns, values):
        positions = cursor[rows]
        indices[positions] = columns
        data[positions] = values
        # Every row appears at most once in any oriented axis batch.
        cursor[rows] += 1

    # Sorted columns in every row: -n^2, -n, -1, 0, 1, n, n^2.
    for axis in range(3):
        section = slice(axis * faces_per_axis, (axis + 1) * faces_per_axis)
        insert(head[section], tail[section], -weights[section])
    cells = np.arange(N, dtype=index_type)
    insert(cells, cells, diagonal)
    for axis in (2, 1, 0):
        section = slice(axis * faces_per_axis, (axis + 1) * faces_per_axis)
        insert(tail[section], head[section], -weights[section])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    matrix.has_sorted_indices = True
    return matrix


def _coarse_energy_lump(matrix):
    """Remove positive offdiagonals while adding PSD edge energies.

    For each symmetric positive entry c=Aij>0, removing Aij and Aji and
    adding c to both diagonals adds c*(e_i-e_j)*(e_i-e_j)^T.  Thus the
    known constant nullspace and quotient positivity survive.  This does
    NOT claim exact Galerkin energy after filtering, or identical PCG rates.
    """
    matrix = matrix.tocsr()
    row_drift = np.asarray(matrix.sum(axis=1)).ravel()
    row_ids = np.repeat(np.arange(matrix.shape[0], dtype=matrix.indices.dtype),
                        np.diff(matrix.indptr))
    positive = (matrix.data > 0.0) & (matrix.indices != row_ids)
    removed = int(np.count_nonzero(positive))
    if removed:
        diagonal_add = np.bincount(row_ids[positive], weights=matrix.data[positive],
                                   minlength=matrix.shape[0])
        diagonal = matrix.diagonal() + diagonal_add - row_drift
        matrix.data[positive] = 0.0
        matrix.setdiag(diagonal)
    else:
        matrix.setdiag(matrix.diagonal() - row_drift)
    matrix.eliminate_zeros()
    return matrix, removed


class Multigrid(_Base):
    """v2-compatible reusable state with a filtered coarse preconditioner."""
    def __init__(self, k, deadline=None, smoothing_steps=1, coarse_size=96,
                 lump_positive=True):
        started = time.perf_counter()
        self.k = np.asarray(k, dtype=np.float64)
        if self.k.ndim != 3 or len(set(self.k.shape)) != 1:
            raise ValueError('expected cubic permeability')
        if np.any(self.k <= 0) or not np.all(np.isfinite(self.k)):
            raise ValueError('permeability must be finite and positive')
        self.smoothing_steps = int(smoothing_steps)
        if self.smoothing_steps < 1:
            raise ValueError('smoothing_steps must be positive')
        self.edge = edges(self.k)
        self.A = _fine_csr(self.k, self.edge)
        assembled = time.perf_counter()
        self.levels = []
        self.last_pressure = None
        self.last_meta = None
        self.solve_calls = 0
        xyz = np.indices(self.k.shape).reshape(3, -1).T
        phase = (self.k.ravel() > np.min(self.k)).astype(np.int64)
        matrix = self.A
        level_sizes, level_nnz, interpolation_nnz = [], [], []
        galerkin_nnz, lumped_positive_nnz = [], []
        with threadpool_limits(limits=1):
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError('deadline during hierarchy construction')
                matrix = matrix.tocsr()
                diagonal = matrix.diagonal()
                if matrix.shape[0] > 1 and np.any(diagonal <= 0):
                    raise ValueError('nonpositive multigrid diagonal')
                if matrix.shape[0] == 1:
                    inverse_diagonal, omega = np.zeros(1), 0.0
                else:
                    inverse_diagonal = 1.0 / diagonal
                    if lump_positive or len(self.levels) == 0:
                        # The fine graph, and filtered coarse graphs, have
                        # nonpositive offdiagonals and zero row sum: bound=2.
                        omega = 2.0 / 3.0
                    else:
                        bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()
                                             * inverse_diagonal))
                        omega = min(2.0 / 3.0, 4.0 / (3.0 * bound))
                level = {'A': matrix, 'di': omega * inverse_diagonal}
                self.levels.append(level)
                level_sizes.append(int(matrix.shape[0]))
                level_nnz.append(int(matrix.nnz))
                if matrix.shape[0] <= coarse_size:
                    level['factor'] = (splu(matrix[1:, 1:].tocsc())
                                       if matrix.shape[0] > 1 else None)
                    break
                inverse, coarse_xyz, coarse_phase = _aggregate(xyz, phase)
                nc = len(coarse_phase)
                if nc >= matrix.shape[0] or len(self.levels) >= 8:
                    level['factor'] = splu(matrix[1:, 1:].tocsc())
                    break
                rows = matrix.shape[0]
                Z = sparse.csr_matrix((np.ones(rows), inverse,
                                       np.arange(rows + 1, dtype=np.int64)),
                                      shape=(rows, nc))
                AZ = (matrix @ Z).tocsr()
                AZ.data *= np.repeat(level['di'], np.diff(AZ.indptr))
                P = (Z - AZ).tocsr()
                P.eliminate_zeros()
                row_sum = np.asarray(P.sum(axis=1)).ravel()
                P.data *= np.repeat(1.0 / row_sum, np.diff(P.indptr))
                R = P.T.tocsr()
                level['P'], level['R'] = P, R
                interpolation_nnz.append(int(P.nnz))
                coarse = (R @ (matrix @ P)).tocsr()
                coarse = ((coarse + coarse.T) * 0.5).tocsr()
                galerkin_nnz.append(int(coarse.nnz))
                if lump_positive:
                    coarse, removed = _coarse_energy_lump(coarse)
                else:
                    drift = np.asarray(coarse.sum(axis=1)).ravel()
                    coarse.setdiag(coarse.diagonal() - drift)
                    coarse.eliminate_zeros()
                    removed = 0
                lumped_positive_nnz.append(removed)
                matrix, xyz, phase = coarse, coarse_xyz, coarse_phase

        storage = sum(_csr_bytes(level['A']) + level['di'].nbytes
                      + (_csr_bytes(level['P']) + _csr_bytes(level['R'])
                         if 'P' in level else 0)
                      for level in self.levels)
        self.preconditioner = LinearOperator(self.A.shape,
            matvec=self.apply_preconditioner, dtype=np.float64)
        self.sparse_operator = LinearOperator(self.A.shape,
            matvec=lambda x: _project(self.A @ _project(x)), dtype=np.float64)
        self.graph_operator = LinearOperator(self.A.shape,
            matvec=lambda x: _project(action(_project(x), self.edge)), dtype=np.float64)
        self.setup_meta = {
            'assembly_seconds': assembled - started,
            'hierarchy_seconds': 0.0,
            'setup_seconds': 0.0,
            'level_sizes': level_sizes,
            'level_nnz': level_nnz,
            'interpolation_nnz': interpolation_nnz,
            'galerkin_nnz_before_lump': galerkin_nnz,
            'positive_offdiagonals_lumped': lumped_positive_nnz,
            'lump_positive': bool(lump_positive),
            'operator_complexity': float(sum(level_nnz) / max(1, level_nnz[0])),
            'hierarchy_array_bytes': int(storage),
            'edge_array_bytes': int(sum(array.nbytes for array in self.edge)),
            'array_bytes_note': 'persistent CSR/diagonal/edge arrays; excludes LU internals, Python, temporary setup allocations',
            'smoothing_steps_each_side': self.smoothing_steps,
            'fine_operator_unchanged': True,
            'preconditioner_mechanism': 'phase-separated SA; positive-offdiagonal PSD energy lump on coarse operators only',
        }
        finished = time.perf_counter()
        self.setup_meta['setup_seconds'] = finished - started
        self.setup_meta['hierarchy_seconds'] = finished - assembled

    def solve(self, *args, **kwargs):
        p, meta = super().solve(*args, **kwargs)
        meta['mechanism'] = 'direct seven-point fine CSR; phase-separated SA; coarse positive-offdiagonal PSD energy lump; symmetric V-cycle; graph-checked PCG'
        self.last_meta = dict(meta)
        return p, meta


def build(k, deadline=None, smoothing_steps=1, coarse_size=96,
          lump_positive=True):
    return Multigrid(k, deadline=deadline, smoothing_steps=smoothing_steps,
                     coarse_size=coarse_size, lump_positive=lump_positive)


def solve(k, b, deadline=None):
    started = time.perf_counter()
    try:
        hierarchy = build(k, deadline=deadline)
    except TimeoutError:
        elapsed = time.perf_counter() - started
        return np.zeros(np.asarray(k).size, dtype=np.float64), {
            'iterations': 0, 'setup_seconds': elapsed, 'solve_seconds': 0.0,
            'total_seconds': elapsed, 'status': 'deadline_in_setup',
            'mechanism': 'direct CSR plus phase-separated energy-lumped multigrid'}
    p, meta = hierarchy.solve(b, deadline=deadline)
    meta['total_seconds'] = time.perf_counter() - started
    return p, meta
