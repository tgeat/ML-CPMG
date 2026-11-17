"""Lower-construction-cost variant of the independent SA multigrid candidate.

There are no experiments, caches, or hidden setup costs on import.  build(k)
creates reusable state and records all construction time; solve(k,b) includes
that time.  The pressure operator, gauge and final graph checks match v1.
"""
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_variable] = '1'

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu
from threadpoolctl import threadpool_limits
from darcy_core import assemble, edges, action, metrics


def _project(x):
    return x - np.mean(x)


def _csr_bytes(a):
    return a.data.nbytes + a.indices.nbytes + a.indptr.nbytes


class _Stopped(Exception):
    def __init__(self, x, reason):
        self.x = x.copy()
        self.reason = reason


def _aggregate(xyz, phase):
    """Linear-time deterministic phase-separated geometric aggregation."""
    blocks = xyz // 2
    side = int(np.max(blocks)) + 1
    code = (((blocks[:, 0] * side + blocks[:, 1]) * side
             + blocks[:, 2]) * 2 + phase)
    counts = np.bincount(code, minlength=2 * side ** 3)
    occupied = np.flatnonzero(counts)
    mapping = np.empty(len(counts), dtype=np.int64)
    mapping[occupied] = np.arange(len(occupied), dtype=np.int64)
    inverse = mapping[code]
    geometric = occupied // 2
    coarse_xyz = np.column_stack((geometric // (side * side),
                                  (geometric // side) % side,
                                  geometric % side))
    return inverse, coarse_xyz, occupied % 2


class Multigrid:
    def __init__(self, k, deadline=None, smoothing_steps=1, coarse_size=96):
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
        self.A = assemble(self.k)
        assembled = time.perf_counter()
        self.levels = []
        self.last_pressure = None
        self.last_meta = None
        self.solve_calls = 0
        xyz = np.indices(self.k.shape).reshape(3, -1).T
        phase = (self.k.ravel() > np.min(self.k)).astype(np.int64)
        matrix = self.A
        level_sizes, level_nnz, interpolation_nnz = [], [], []
        with threadpool_limits(limits=1):
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError('deadline during hierarchy construction')
                matrix = matrix.tocsr()
                diagonal = matrix.diagonal()
                if matrix.shape[0] > 1 and np.any(diagonal <= 0):
                    raise ValueError('nonpositive multigrid diagonal')
                if matrix.shape[0] == 1:
                    inverse_diagonal = np.zeros(1)
                    omega = 0.0
                else:
                    inverse_diagonal = 1.0 / diagonal
                    bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()
                                         * inverse_diagonal))
                    omega = min(2.0 / 3.0, 4.0 / (3.0 * bound))
                level = {'A': matrix, 'di': omega * inverse_diagonal}
                self.levels.append(level)
                level_sizes.append(int(matrix.shape[0]))
                level_nnz.append(int(matrix.nnz))
                if matrix.shape[0] <= coarse_size:
                    if matrix.shape[0] > 1:
                        level['factor'] = splu(matrix[1:, 1:].tocsc())
                    else:
                        level['factor'] = None
                    break
                inverse, coarse_xyz, coarse_phase = _aggregate(xyz, phase)
                nc = len(coarse_phase)
                if nc >= matrix.shape[0] or len(self.levels) >= 8:
                    level['factor'] = splu(matrix[1:, 1:].tocsc())
                    break
                rows = matrix.shape[0]
                # Exactly one nonzero per row; construct CSR directly.
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
                drift = np.asarray(coarse.sum(axis=1)).ravel()
                coarse.setdiag(coarse.diagonal() - drift)
                coarse.eliminate_zeros()
                matrix, xyz, phase = coarse, coarse_xyz, coarse_phase
        ready = time.perf_counter()
        storage = sum(_csr_bytes(level['A']) + level['di'].nbytes
                      + (_csr_bytes(level['P']) + _csr_bytes(level['R'])
                         if 'P' in level else 0)
                      for level in self.levels)
        edge_storage = sum(array.nbytes for array in self.edge)
        self.setup_meta = {
            'assembly_seconds': assembled - started,
            'hierarchy_seconds': ready - assembled,
            'setup_seconds': ready - started,
            'level_sizes': level_sizes,
            'level_nnz': level_nnz,
            'interpolation_nnz': interpolation_nnz,
            'operator_complexity': float(sum(level_nnz) / max(1, level_nnz[0])),
            'hierarchy_array_bytes': int(storage),
            'edge_array_bytes': int(edge_storage),
            'array_bytes_note': 'persistent CSR/diagonal/edge arrays; excludes LU internals, Python, temporary setup allocations',
            'smoothing_steps_each_side': self.smoothing_steps,
        }
        self.preconditioner = LinearOperator(self.A.shape,
            matvec=self.apply_preconditioner, dtype=np.float64)
        self.sparse_operator = LinearOperator(self.A.shape,
            matvec=lambda x: _project(self.A @ _project(x)), dtype=np.float64)
        self.graph_operator = LinearOperator(self.A.shape,
            matvec=lambda x: _project(action(_project(x), self.edge)), dtype=np.float64)
        # Include array accounting and operator construction as real setup.
        construction_finished = time.perf_counter()
        self.setup_meta['setup_seconds'] = construction_finished - started
        self.setup_meta['hierarchy_seconds'] = construction_finished - assembled

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
        for _ in range(self.smoothing_steps - 1):
            x += diagonal * (rhs - matrix @ x)
        residual = _project(rhs - matrix @ x)
        x += level['P'] @ self._cycle(index + 1, level['R'] @ residual)
        for _ in range(self.smoothing_steps):
            x += diagonal * (rhs - matrix @ x)
        return _project(x)

    def apply_preconditioner(self, rhs):
        """Apply the same fixed, symmetric V-cycle to any residual."""
        return self._cycle(0, np.asarray(rhs, dtype=np.float64).ravel())

    def solve(self, b, deadline=None, target_tol=1e-9, x0=None,
              precisiontail=True, maxiter=1200):
        """Solve with optional explicit restart pressure and strict graph gates.

        The main PCG uses CSR matvecs.  If their recursive residual is less
        accurate than the graph residual, the correction tail uses graph
        matvecs plus explicit residual replacement.  No reference is used.
        x0 is explicit: repeated calls with x0=None always start from zero.
        A deadline result can be resumed by passing last_pressure as x0.
        """
        began = time.perf_counter()
        b = np.asarray(b, dtype=np.float64).ravel()
        if len(b) != self.A.shape[0] or not np.all(np.isfinite(b)):
            raise ValueError('expected finite source for every cell')
        if abs(float(np.sum(b))) > 1e-12 * max(1.0, float(np.sum(abs(b)))):
            raise ValueError('Neumann source must have zero integral')
        if not (0 < target_tol < 1):
            raise ValueError('target_tol must lie between zero and one')
        b = _project(b)
        p = np.zeros_like(b) if x0 is None else _project(np.asarray(x0, dtype=np.float64).ravel().copy())
        if p.shape != b.shape or not np.all(np.isfinite(p)):
            raise ValueError('x0 must contain one finite pressure per cell')
        iterations, passes, tail_iterations = 0, 0, 0
        reason, info = 'krylov_finished', 0
        values = metrics(p, b, self.edge)
        accepted = values['rho'] <= target_tol and values['eta'] <= target_tol
        main_iteration_limit = int(maxiter)
        with threadpool_limits(limits=1):
            for pass_index in range(4 if precisiontail else 1):
                if accepted:
                    break
                if deadline is not None and time.perf_counter() >= deadline - 0.002:
                    reason = 'deadline'
                    break
                passes += 1
                graph_tail = pass_index > 0
                operator = self.graph_operator if graph_tail else self.sparse_operator
                residual = _project(b - action(p, self.edge))
                local_iterations = 0

                def callback(correction):
                    nonlocal iterations, tail_iterations, local_iterations
                    iterations += 1
                    local_iterations += 1
                    if graph_tail:
                        tail_iterations += 1
                    if deadline is not None and time.perf_counter() >= deadline - 0.002:
                        raise _Stopped(correction, 'deadline')
                    if local_iterations == 1 or local_iterations % 8 == 0:
                        check = metrics(_project(p + correction), b, self.edge)
                        if check['rho'] <= target_tol * 0.1 and check['eta'] <= target_tol * 0.1:
                            raise _Stopped(correction, 'true_residual_converged')

                try:
                    correction, info = cg(operator, residual,
                        M=self.preconditioner, rtol=min(1e-12, target_tol * 0.01),
                        atol=0.0, maxiter=main_iteration_limit, callback=callback)
                except _Stopped as stopped:
                    correction = stopped.x
                    reason = stopped.reason
                p = _project(p + correction)
                values = metrics(p, b, self.edge)
                accepted = values['rho'] <= target_tol and values['eta'] <= target_tol
                if reason == 'deadline':
                    break
        p = np.asarray(_project(p), dtype=np.float64).ravel()
        values = metrics(p, b, self.edge)
        accepted = values['rho'] <= target_tol and values['eta'] <= target_tol
        self.last_pressure = p.copy()
        ended = time.perf_counter()
        before_deadline = deadline is None or ended < deadline
        meta = {
            **self.setup_meta,
            'solve_seconds': ended - began,
            'total_seconds': self.setup_meta['setup_seconds'] + ended - began,
            'iterations': iterations,
            'precision_tail_iterations': tail_iterations,
            'residual_replacement_passes': passes,
            'status': ('completed' if accepted and before_deadline
                       else ('deadline' if reason == 'deadline' or not before_deadline
                             else 'not_converged')),
            'conservation_compliant': bool(accepted),
            'completed_before_deadline': bool(before_deadline),
            'checkpoint_completed_perf_counter': ended if accepted and before_deadline else None,
            'stop_reason': reason,
            'info': int(info),
            'target_tol': float(target_tol),
            'setup_reused': self.solve_calls > 0,
            'setup_accounting': 'total_seconds includes the full original construction time on every call; solve_seconds is this call only',
            'mechanism': 'linear-time phase/geometric aggregation; CSR-scaled smoothed prolongation; symmetric Galerkin V-cycle; CSR PCG with strict graph residual tail',
            **values,
        }
        self.last_meta = dict(meta)
        self.solve_calls += 1
        return p, meta


def build(k, deadline=None, smoothing_steps=1, coarse_size=96):
    """Build reusable state; construction time is available in setup_meta."""
    return Multigrid(k, deadline=deadline, smoothing_steps=smoothing_steps,
                     coarse_size=coarse_size)


def solve(k, b, deadline=None):
    """Compatibility wrapper: always builds once and starts from zero."""
    began = time.perf_counter()
    try:
        hierarchy = build(k, deadline=deadline)
    except TimeoutError:
        elapsed = time.perf_counter() - began
        return np.zeros(np.asarray(k).size, dtype=np.float64), {
            'iterations': 0, 'setup_seconds': elapsed, 'solve_seconds': 0.0,
            'total_seconds': elapsed, 'status': 'deadline_in_setup',
            'mechanism': 'phase/geometric smoothed-aggregation hierarchy'}
    p, meta = hierarchy.solve(b, deadline=deadline)
    meta['total_seconds'] = time.perf_counter() - began
    return p, meta
