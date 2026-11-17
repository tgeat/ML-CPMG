"""Configurable fixed symmetric multigrid for the common extension kernel.

S2 source delta from the immutable extension_multigrid.py: coarse_size's
accepted domain additionally contains 512.  No numerical kernel, interface,
original CONFIGS entry or DEFAULT_CONFIG changes.  S2's four proposed presets
are supplied externally so the original preset registry also stays unchanged.

No numerical experiment runs on import.  This file only adds new code and
uses unchanged helpers from path_multigrid_v2/v3.  The fine harmonic graph,
source, gauge and common DD/true-check policy are outside this experiment.

Mechanisms
----------
* Direct seven-point fine CSR and integer aggregate encoding retain cheap
  setup.  High-coefficient FACE-CONNECTED COMPONENT identities survive every
  coarse level.  This matters when an odd coarse grid would otherwise merge
  several disconnected inclusions in one binary-phase aggregate (e.g. n=96).
* Independent fine/coarse symmetric Jacobi sweep counts recover the stronger
  v1 smoothing without restoring its slower aggregate/assembly construction.
* Optional weak-positive-offdiagonal lumping changes coarse preconditioners
  only: deleting Aij=Aji=c>0 and adding c to both diagonals adds the PSD energy
  c*(x_i-x_j)^2.  A relative cutoff keeps stronger positive Galerkin couplings.
* Optional TWO fixed residual-correction V-cycles apply 2B-BAB.  This is linear
  and symmetric.  In exact arithmetic our damped symmetric Galerkin/PSD-lumped
  hierarchy has 0<B<=A^dagger on the quotient; consequently 2B-BAB is SPD too.
  No residual-dependent choice, variable sweep count or nonlinear MG is used.

Cost / limits
-------------
Fine assembly and connected-component labels cost O(N+E).  A V-cycle costs
O(sum(nnz(A_l)+nnz(P_l))) with sweep constants below; double-cycle repeats that
work and adds one fine CSR product.  Sparse Galerkin construction can still
dominate setup or memory.  Preserving labels can increase very small coarse
dimensions; coarsest LU is bounded by the selected 64/96/192/512 threshold unless
aggregation stops reducing.  The array-byte metadata excludes LU/Python and
temporary setup storage, so RSS monitoring remains authoritative.

Expected failure modes: weak lumping may still stiffen useful coarse modes;
more sweeps or double cycles can reduce outer iterations yet lose time; a
larger exact coarse solve can add apply cost; preserving many conducting
components can prevent useful coarsening.  No speed/accuracy success is
claimed without the common root-run pilot and independent certification.

CONFIGS is the complete proposed pilot set (at most four configurations).
The default first restores strong smoothing with exact Galerkin energies;
the other three test a cheaper deep smoother, weak deep filtering, and fewer
DD outer iterations through a fixed symmetric repeated cycle respectively.
"""
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_variable] = '1'

import time
import numpy as np
from scipy import sparse
from scipy.ndimage import label
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits
from darcy_core import edges, action
from path_multigrid_v2 import Multigrid as _SolveBase, _csr_bytes, _project
from path_multigrid_v3 import _fine_csr


PARAMETER_DOMAIN = {
    'fine_sweeps': (1, 2),
    'coarse_sweeps': (1, 2),
    'coarse_size': (64, 96, 192, 512),
    'lump_max_relative': 'None (exact Galerkin) or a float in [0,1]',
    'lump_start_level': (1, 2, 3),
    'cycle_repeats': (1, 2),
    'preserve_components': (False, True),
}

CONFIGS = {
    'galerkin_2_2': {
        'fine_sweeps': 2, 'coarse_sweeps': 2, 'coarse_size': 192,
        'lump_max_relative': None, 'lump_start_level': 2,
        'cycle_repeats': 1, 'preserve_components': True,
    },
    'galerkin_2_1': {
        'fine_sweeps': 2, 'coarse_sweeps': 1, 'coarse_size': 192,
        'lump_max_relative': None, 'lump_start_level': 2,
        'cycle_repeats': 1, 'preserve_components': True,
    },
    'weak_lump_2_1': {
        'fine_sweeps': 2, 'coarse_sweeps': 1, 'coarse_size': 192,
        'lump_max_relative': 0.02, 'lump_start_level': 2,
        'cycle_repeats': 1, 'preserve_components': True,
    },
    'two_cycle_1_1': {
        'fine_sweeps': 1, 'coarse_sweeps': 1, 'coarse_size': 192,
        'lump_max_relative': None, 'lump_start_level': 2,
        'cycle_repeats': 2, 'preserve_components': True,
    },
}
DEFAULT_CONFIG = 'galerkin_2_2'


def _config(config):
    if config is None:
        name = DEFAULT_CONFIG
        result = dict(CONFIGS[name])
    elif isinstance(config, str):
        if config not in CONFIGS:
            raise ValueError('unknown multigrid configuration: ' + config)
        name = config
        result = dict(CONFIGS[name])
    else:
        name = 'custom'
        supplied = dict(config)
        unknown = set(supplied) - set(PARAMETER_DOMAIN)
        if unknown:
            raise ValueError('unknown multigrid parameter(s): ' + ', '.join(sorted(unknown)))
        result = dict(CONFIGS[DEFAULT_CONFIG])
        result.update(supplied)
    for key in ('fine_sweeps', 'coarse_sweeps', 'coarse_size',
                'lump_start_level', 'cycle_repeats'):
        if result[key] not in PARAMETER_DOMAIN[key]:
            raise ValueError('unsupported ' + key)
        result[key] = int(result[key])
    if not isinstance(result['preserve_components'], (bool, np.bool_)):
        raise ValueError('preserve_components must be boolean')
    result['preserve_components'] = bool(result['preserve_components'])
    cutoff = result['lump_max_relative']
    if cutoff is not None:
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or not (0.0 <= cutoff <= 1.0):
            raise ValueError('lump_max_relative must be None or lie in [0,1]')
        result['lump_max_relative'] = cutoff
    return name, result


def _aggregate_components(xyz, component, radix):
    """Keep each high component distinct even inside a shared coarse block."""
    blocks = xyz // 2
    side = int(np.max(blocks)) + 1
    code = (((blocks[:, 0] * side + blocks[:, 1]) * side
             + blocks[:, 2]) * radix + component)
    # The target geometries have radix<=9.  Avoid a huge mostly empty radix
    # table for arbitrary externally supplied geometries with many components.
    table_size = int(np.max(code)) + 1
    if table_size <= max(1024, 16 * len(code)):
        count = np.bincount(code, minlength=table_size)
        occupied = np.flatnonzero(count)
        mapping = np.empty(table_size, dtype=np.int64)
        mapping[occupied] = np.arange(len(occupied), dtype=np.int64)
        inverse = mapping[code]
    else:
        occupied, inverse = np.unique(code, return_inverse=True)
    geometric = occupied // radix
    coarse_xyz = np.column_stack((geometric // (side * side),
                                  (geometric // side) % side,
                                  geometric % side))
    return inverse, coarse_xyz, occupied % radix


def _coarse_filter(matrix, relative_cutoff):
    """Return a symmetric coarse operator, directed removed count and count+.

    Only positive offdiagonals c <= cutoff*sqrt(Aii*Ajj) are lumped.
    Each removed symmetric pair adds PSD edge energy; negative couplings and
    stronger positive couplings are retained.  None leaves exact Galerkin
    energies apart from the same constant-row-drift repair used in v1-v3.
    """
    matrix = matrix.tocsr()
    drift = np.asarray(matrix.sum(axis=1)).ravel()
    diagonal = matrix.diagonal()
    if relative_cutoff is None:
        matrix.setdiag(diagonal - drift)
        matrix.eliminate_zeros()
        return matrix, 0, 0
    rows = np.repeat(np.arange(matrix.shape[0], dtype=matrix.indices.dtype),
                     np.diff(matrix.indptr))
    positive = (matrix.data > 0.0) & (matrix.indices != rows)
    positive_count = int(np.count_nonzero(positive))
    locations = np.flatnonzero(positive)
    if len(locations):
        pair_scale = np.sqrt(diagonal[rows[locations]]
                             * diagonal[matrix.indices[locations]])
        locations = locations[matrix.data[locations] <= relative_cutoff * pair_scale]
    if len(locations):
        addition = np.bincount(rows[locations], weights=matrix.data[locations],
                                minlength=matrix.shape[0])
        diagonal = diagonal + addition
        matrix.data[locations] = 0.0
    matrix.setdiag(diagonal - drift)
    matrix.eliminate_zeros()
    return matrix, int(len(locations)), positive_count


class Multigrid(_SolveBase):
    def __init__(self, k, config=None, deadline=None):
        started = time.perf_counter()
        self.config_name, self.config = _config(config)
        self.k = np.asarray(k, dtype=np.float64)
        if self.k.ndim != 3 or len(set(self.k.shape)) != 1:
            raise ValueError('expected cubic permeability')
        if np.any(self.k <= 0) or not np.all(np.isfinite(self.k)):
            raise ValueError('permeability must be finite and positive')
        self.edge = edges(self.k)
        self.A = _fine_csr(self.k, self.edge)
        assembled = time.perf_counter()
        high = self.k > np.min(self.k)
        if self.config['preserve_components']:
            component_grid, number = label(high)
            component = component_grid.ravel().astype(np.int64, copy=False)
        else:
            component = high.ravel().astype(np.int64)
            number = int(np.any(high))
        radix = int(number) + 1
        labelled = time.perf_counter()
        self.cycle_repeats = self.config['cycle_repeats']
        self.levels = []
        self.solve_calls = 0
        self.last_pressure = None
        self.last_meta = None
        xyz = np.indices(self.k.shape).reshape(3, -1).T
        matrix = self.A
        sizes, nnzs, pnnzs = [], [], []
        galerkin_nnzs, lumped, positive_counts = [], [], []
        sweeps_per_level, tag_counts = [], []
        with threadpool_limits(limits=1):
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError('deadline during extension hierarchy construction')
                matrix = matrix.tocsr()
                diagonal = matrix.diagonal()
                if matrix.shape[0] > 1 and np.any(diagonal <= 0.0):
                    raise ValueError('nonpositive multigrid diagonal')
                if matrix.shape[0] == 1:
                    damped_inverse = np.zeros(1)
                else:
                    inverse_diagonal = 1.0 / diagonal
                    if not self.levels:
                        omega = 2.0 / 3.0
                    else:
                        bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()
                                             * inverse_diagonal))
                        omega = min(2.0 / 3.0, 4.0 / (3.0 * bound))
                    damped_inverse = omega * inverse_diagonal
                sweeps = (self.config['fine_sweeps'] if not self.levels
                          else self.config['coarse_sweeps'])
                level = {'A': matrix, 'di': damped_inverse, 'sweeps': sweeps}
                self.levels.append(level)
                sizes.append(int(matrix.shape[0]))
                nnzs.append(int(matrix.nnz))
                sweeps_per_level.append(sweeps)
                tag_counts.append(int(np.count_nonzero(np.bincount(component))))
                if matrix.shape[0] <= self.config['coarse_size']:
                    level['factor'] = (splu(matrix[1:, 1:].tocsc())
                                       if matrix.shape[0] > 1 else None)
                    break
                inverse, coarse_xyz, coarse_component = _aggregate_components(xyz, component, radix)
                nc = len(coarse_component)
                if nc >= matrix.shape[0] or len(self.levels) >= 10:
                    level['factor'] = splu(matrix[1:, 1:].tocsc())
                    break
                rows = matrix.shape[0]
                Z = sparse.csr_matrix((np.ones(rows), inverse,
                                       np.arange(rows + 1, dtype=np.int64)),
                                      shape=(rows, nc))
                AZ = (matrix @ Z).tocsr()
                AZ.data *= np.repeat(damped_inverse, np.diff(AZ.indptr))
                P = (Z - AZ).tocsr()
                P.eliminate_zeros()
                row_sum = np.asarray(P.sum(axis=1)).ravel()
                P.data *= np.repeat(1.0 / row_sum, np.diff(P.indptr))
                R = P.T.tocsr()
                level['P'], level['R'] = P, R
                pnnzs.append(int(P.nnz))
                coarse = (R @ (matrix @ P)).tocsr()
                coarse = ((coarse + coarse.T) * 0.5).tocsr()
                galerkin_nnzs.append(int(coarse.nnz))
                # After appending level 0, len(levels)==1 denotes coarse level1.
                cutoff = (self.config['lump_max_relative']
                          if len(self.levels) >= self.config['lump_start_level']
                          else None)
                coarse, removed, positives = _coarse_filter(coarse, cutoff)
                lumped.append(removed)
                positive_counts.append(positives)
                matrix, xyz, component = coarse, coarse_xyz, coarse_component

        self.preconditioner = LinearOperator(self.A.shape,
            matvec=self.apply_preconditioner, dtype=np.float64)
        self.sparse_operator = LinearOperator(self.A.shape,
            matvec=lambda x: _project(self.A @ _project(x)), dtype=np.float64)
        self.graph_operator = LinearOperator(self.A.shape,
            matvec=lambda x: _project(action(_project(x), self.edge)), dtype=np.float64)
        storage = sum(_csr_bytes(level['A']) + level['di'].nbytes
                      + (_csr_bytes(level['P']) + _csr_bytes(level['R'])
                         if 'P' in level else 0)
                      for level in self.levels)
        self.setup_meta = {
            'config_name': self.config_name, 'config': dict(self.config),
            'assembly_seconds': assembled - started,
            'component_label_seconds': labelled - assembled,
            'hierarchy_seconds': 0.0, 'setup_seconds': 0.0,
            'level_sizes': sizes, 'level_nnz': nnzs,
            'interpolation_nnz': pnnzs,
            'galerkin_nnz_before_lump': galerkin_nnzs,
            'positive_offdiagonals_lumped': lumped,
            'positive_offdiagonals_before_filter': positive_counts,
            'smoothing_steps_per_level_each_side': sweeps_per_level,
            'cycle_repeats': self.cycle_repeats,
            'high_component_count': int(number),
            'component_tag_count_per_level': tag_counts,
            'operator_complexity': float(sum(nnzs) / max(1, nnzs[0])),
            'hierarchy_array_bytes': int(storage),
            'edge_array_bytes': int(sum(array.nbytes for array in self.edge)),
            'array_bytes_note': 'persistent CSR/diagonal/edge arrays only; RSS includes LU, temporaries and Python',
            'fine_operator_unchanged': True,
            'preconditioner_is_fixed_linear_symmetric': True,
            'mechanism': 'component-preserving SA; configurable symmetric sweeps; optional weak positive coarse energy lump; fixed repeated V-cycle',
        }
        finished = time.perf_counter()
        self.setup_meta['setup_seconds'] = finished - started
        self.setup_meta['hierarchy_seconds'] = finished - labelled

    def _cycle(self, index, rhs):
        level = self.levels[index]
        rhs = _project(rhs)
        if 'factor' in level:
            x = np.zeros_like(rhs)
            if level['factor'] is not None:
                x[1:] = level['factor'].solve(rhs[1:])
            return _project(x)
        matrix, diagonal = level['A'], level['di']
        x = diagonal * rhs
        for _ in range(level['sweeps'] - 1):
            x += diagonal * (rhs - matrix @ x)
        residual = _project(rhs - matrix @ x)
        x += level['P'] @ self._cycle(index + 1, level['R'] @ residual)
        for _ in range(level['sweeps']):
            x += diagonal * (rhs - matrix @ x)
        return _project(x)

    def apply_preconditioner(self, rhs):
        rhs = np.asarray(rhs, dtype=np.float64).ravel()
        x = self._cycle(0, rhs)
        if self.cycle_repeats == 2:
            # _cycle projects its input and output; avoid duplicate fine-grid
            # reductions/copies on the usual single-cycle application.
            residual = rhs - self.A @ x
            x += self._cycle(0, residual)
            return _project(x)
        return x

    def solve(self, *args, **kwargs):
        # Compatibility only.  Formal extension comparisons should use the
        # common DD kernel/flux/certificate policy for every method.
        p, meta = super().solve(*args, **kwargs)
        meta['mechanism'] = self.setup_meta['mechanism']
        self.last_meta = dict(meta)
        return p, meta


def build(k, config=None, deadline=None):
    """Return state.A / preconditioner / setup_meta for the common DD solver."""
    return Multigrid(k, config=config, deadline=deadline)


def solve(k, b, deadline=None, config=None):
    """Compatibility wrapper; formal studies should use the common DD kernel."""
    started = time.perf_counter()
    try:
        state = build(k, config=config, deadline=deadline)
    except TimeoutError:
        elapsed = time.perf_counter() - started
        return np.zeros(np.asarray(k).size, dtype=np.float64), {
            'iterations': 0, 'setup_seconds': elapsed, 'solve_seconds': 0.0,
            'total_seconds': elapsed, 'status': 'deadline_in_setup'}
    p, meta = state.solve(b, deadline=deadline)
    meta['total_seconds'] = time.perf_counter() - started
    return p, meta
