"""Four frozen solver configurations for well-grounded SPE10 pressure systems.

The unknown is drawdown x=p-p_BH. The exact serialized face system is
div(T grad_graph(x)) + d*x = injection, with nonnegative producer diagonal d.
Its constant mode is physically grounded. Coarse row sums are therefore kept,
full SPD coarse systems are solved, and no zero-mean projection is applied.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

from importlib import metadata as package_metadata
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.ndimage import label
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits

import graph_precision as gp
from extension_multigrid_v2 import _aggregate_components
from spe10_methods import _check_inputs, assemble_fast, assemble_ordinary


CANDIDATE = 'component_lump_1_1_c512'
STRONG = 'strong_nu1_c512'
METHODS = (CANDIDATE, STRONG, 'sa_theta0_c10', 'rs_theta025_c512')


def _guard(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError('dynamic pressure deadline')


def _lump_positive(matrix):
    """Add PSD pair energies while preserving physical Galerkin row sums."""
    matrix = matrix.tocsr()
    rows = np.repeat(np.arange(matrix.shape[0], dtype=matrix.indices.dtype),
                     np.diff(matrix.indptr))
    positive = (matrix.data > 0) & (matrix.indices != rows)
    # Frozen cutoff is one times the pair diagonal scale. Retain that explicit
    # rule even though exact SPD 2x2 principal minors already imply the bound.
    diagonal = matrix.diagonal()
    locations = np.flatnonzero(positive)
    positive[locations] &= (matrix.data[locations] <=
                            np.sqrt(diagonal[rows[locations]]*
                                    diagonal[matrix.indices[locations]]))
    count = int(np.count_nonzero(positive))
    if count:
        addition = np.bincount(rows[positive], weights=matrix.data[positive],
                              minlength=matrix.shape[0])
        diagonal = matrix.diagonal() + addition
        matrix.data[positive] = 0
        matrix.setdiag(diagonal)
        matrix.eliminate_zeros()
    return matrix, count


def _geometric(A, k_tag, component_preservation, deadline):
    xyz = np.indices(k_tag.shape).reshape(3, -1).T
    high = k_tag > np.min(k_tag)
    if component_preservation:
        components, number = label(high)
        tags = components.ravel().astype(np.int64, copy=False)
    else:
        tags = high.ravel().astype(np.int64)
        number = int(np.any(high))
    matrix = A
    levels, removed = [], []
    limit = 10 if component_preservation else 8
    while True:
        _guard(deadline)
        matrix = matrix.tocsr()
        diagonal = matrix.diagonal()
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
            raise ValueError('nonpositive geometric diagonal')
        inverse = 1/diagonal
        bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()*inverse))
        omega = min(2/3, 4/(3*bound))
        level = {'A': matrix, 'di': omega*inverse}
        levels.append(level)
        if matrix.shape[0] <= 512:
            level['factor'] = splu(matrix.tocsc())
            break
        if component_preservation:
            aggregate, next_xyz, next_tags = _aggregate_components(xyz, tags, number+1)
        else:
            unique, aggregate = np.unique(np.column_stack((xyz//2, tags)), axis=0,
                                          return_inverse=True)
            next_xyz, next_tags = unique[:, :3], unique[:, 3]
        ncoarse = len(next_tags)
        if ncoarse >= matrix.shape[0] or len(levels) >= limit:
            if matrix.shape[0] > 2048:
                raise MemoryError('geometric coarse direct-size guard')
            level['factor'] = splu(matrix.tocsc())
            break
        if component_preservation:
            Z = sparse.csr_matrix((np.ones(matrix.shape[0]), aggregate,
                                  np.arange(matrix.shape[0]+1, dtype=np.int64)),
                                 shape=(matrix.shape[0], ncoarse))
            AZ = (matrix@Z).tocsr()
            AZ.data *= np.repeat(level['di'], np.diff(AZ.indptr))
            P = (Z-AZ).tocsr()
            P.eliminate_zeros()
            P.data *= np.repeat(1/np.asarray(P.sum(axis=1)).ravel(), np.diff(P.indptr))
        else:
            Z = sparse.csr_matrix((np.ones(matrix.shape[0]),
                                  (np.arange(matrix.shape[0]), aggregate)),
                                 shape=(matrix.shape[0], ncoarse))
            P = (Z-(matrix@Z).multiply(level['di'][:, None])).tocsr()
            P = P.multiply((1/np.asarray(P.sum(axis=1)).ravel())[:, None]).tocsr()
        if not np.isfinite(P.data).all():
            raise ValueError('invalid grounded prolongation')
        R = P.T.tocsr()
        level.update(P=P, R=R)
        coarse = R@(matrix@P) if component_preservation else (R@matrix)@P
        matrix = ((coarse+coarse.T)*.5).tocsr()
        if component_preservation:
            matrix, count = _lump_positive(matrix)
            removed.append(count)
        else:
            # SPD well terms are real coarse row sums, never roundoff to erase.
            matrix.eliminate_zeros()
        xyz, tags = next_xyz, next_tags

    def cycle(index, rhs):
        level = levels[index]
        if 'factor' in level:
            return level['factor'].solve(rhs)
        x = level['di']*rhs
        residual = rhs-level['A']@x
        x += level['P']@cycle(index+1, level['R']@residual)
        x += level['di']*(rhs-level['A']@x)
        return x

    sizes = [v['A'].shape[0] for v in levels]
    nnz = [v['A'].nnz for v in levels]
    return LinearOperator(A.shape, matvec=lambda r: cycle(0, r), dtype=np.float64), {
        'level_sizes': sizes, 'level_nnz': nnz,
        'operator_complexity': float(sum(nnz)/nnz[0]),
        'positive_offdiagonals_lumped': removed,
        'component_preservation': component_preservation,
        'high_component_count': int(number),
        'phase_rule': 'Kx > min(Kx)', 'coarse_size': 512,
        'pre_sweeps': 1, 'post_sweeps': 1,
    }


def _algebraic(A, method, deadline):
    vendor = str(Path(__file__).resolve().parent/'.paper_vendor')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import pyamg
    from paper_amg_v7 import EXPECTED_PYAMG_VERSION, PRESETS
    version = package_metadata.version('pyamg')
    if version != EXPECTED_PYAMG_VERSION or pyamg.__version__ != version:
        raise RuntimeError('dynamic baseline PyAMG version differs from frozen release')
    cfg = dict(PRESETS[method])
    H = sparse.csr_array(A, copy=False)
    smoother = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': cfg['nu']})
    parameters = dict(presmoother=smoother, postsmoother=smoother,
                      max_levels=cfg['max_levels'], max_coarse=cfg['max_coarse'],
                      coarse_solver='cholesky', keep=False)
    random_state = np.random.get_state()
    try:
        np.random.seed(cfg['setup_random_seed'])
        _guard(deadline)
        if cfg['family'] == 'sa':
            hierarchy = pyamg.smoothed_aggregation_solver(
                H, B=np.ones((H.shape[0], 1)), symmetry='hermitian',
                strength=('symmetric', {'theta': cfg['strength_theta']}), aggregate='standard',
                smooth=('jacobi', {'omega': 4/3}),
                improve_candidates=(('block_gauss_seidel', {'iterations': 4, 'sweep': 'symmetric'}), None),
                **parameters)
        else:
            hierarchy = pyamg.ruge_stuben_solver(
                H, strength=('classical', {'theta': cfg['strength_theta']}),
                CF=('RS', {'second_pass': False}), interpolation='classical', **parameters)
    finally:
        np.random.set_state(random_state)
    last = hierarchy.levels[-1].A
    if last.shape[0] > 2048:
        raise MemoryError('algebraic coarse dense-size guard')
    hierarchy.coarse_solver(last, np.zeros(last.shape[0]))
    _guard(deadline)
    return hierarchy.aspreconditioner(cycle='V'), {
        'library_version': version, 'config': cfg,
        'level_sizes': [v.A.shape[0] for v in hierarchy.levels],
        'level_nnz': [v.A.nnz for v in hierarchy.levels],
    }


def build(method, k_tag, edge, well_diagonal, deadline=None):
    """Fresh SPD hierarchy, preserving the four frozen configuration choices."""
    begin = time.perf_counter()
    if method not in METHODS:
        raise ValueError('unsupported dynamic pressure method '+str(method))
    k_tag = np.asarray(k_tag, dtype=np.float64)
    edge = tuple(np.asarray(v) for v in edge)
    _check_inputs(k_tag, edge)
    d = np.asarray(well_diagonal, dtype=np.float64).ravel()
    if d.size != k_tag.size or not np.isfinite(d).all() or np.any(d < 0) or not np.any(d > 0):
        raise ValueError('finite nonnegative well diagonal with at least one producer required')
    _guard(deadline)
    assembly = assemble_ordinary if method == STRONG else assemble_fast
    with threadpool_limits(1):
        A = assembly(k_tag.shape, edge)
        A.setdiag(A.diagonal()+d)
        assembled = time.perf_counter()
        if method in (CANDIDATE, STRONG):
            M, info = _geometric(A, k_tag, method == CANDIDATE, deadline)
        else:
            M, info = _algebraic(A, method, deadline)
    _guard(deadline)
    end = time.perf_counter()
    info.update(method=method, assembly=assembly.__name__, assembly_seconds=assembled-begin,
                hierarchy_seconds=end-assembled, setup_seconds=end-begin,
                coordinate_system='physical drawdown p-p_BH; unprojected SPD well system',
                adaptation='Galerkin row sums preserved; full SPD coarse solve; no mean projection',
                mathematical_fine_operator='serialized graph plus nonnegative producer diagonal')
    return {'A': A, 'M': M, 'metadata': info}


def flux_and_residual(hi, lo, edge, well_diagonal, rhs, chunk=65536):
    """Residual of actual returned binary64 face and producer fluxes."""
    i, j, weights = edge
    q = np.empty(len(weights))
    for begin in range(0, q.size, chunk):
        end = min(begin+chunk, q.size)
        q[begin:end] = gp.graph_flux(hi, lo, (i[begin:end], j[begin:end], weights[begin:end]))
    d = np.asarray(well_diagonal, dtype=np.float64).ravel()
    wells = np.flatnonzero(d)
    produced = np.zeros_like(rhs)
    ph, pl = gp._two_product(d[wells], hi[wells])
    produced[wells] = ph+(pl+d[wells]*lo[wells])
    residual = rhs-(np.bincount(i, weights=q, minlength=rhs.size)
                    -np.bincount(j, weights=q, minlength=rhs.size)+produced)
    return q, produced, residual


def solve(method, k_tag, edge, d, rhs, initial=None, target=1e-9, deadline=None):
    """Fresh setup plus warm-start reliable PCG with compensated pressure updates.

    Returned well_rates are positive extraction, expressed in the same integrated
    flow units as RHS. Only the first iterate passing both true residual tests is
    accepted. Timings cover setup and solve, excluding parent file I/O.
    """
    begin = time.perf_counter()
    rhs = np.asarray(rhs, dtype=np.float64).ravel()
    d = np.asarray(d, dtype=np.float64).ravel()
    if (rhs.size != np.size(k_tag) or not np.isfinite(rhs).all()
            or not np.any(rhs) or not 0 < target <= 1e-6):
        raise ValueError('invalid RHS or relative target')
    state = build(method, k_tag, edge, d, deadline)
    setup_end = time.perf_counter()
    A, M = state['A'], state['M']
    if initial is None:
        hi, lo = np.zeros_like(rhs), np.zeros_like(rhs)
    elif isinstance(initial, tuple):
        hi, lo = (np.asarray(v, dtype=np.float64).ravel().copy() for v in initial)
    else:
        hi = np.asarray(initial, dtype=np.float64).ravel().copy()
        lo = np.zeros_like(rhs)
    if hi.shape != rhs.shape or lo.shape != rhs.shape or not np.isfinite(hi).all() or not np.isfinite(lo).all():
        raise ValueError('invalid warm-start pressure')
    rhs_norm, rhs_max = float(np.linalg.norm(rhs)), float(np.max(np.abs(rhs)))
    iterations, checks = 0, 0
    trace = []

    def observe(high, low):
        nonlocal checks
        _guard(deadline)
        q, production, residual = flux_and_residual(high, low, edge, d, rhs)
        rho = float(np.linalg.norm(residual)/rhs_norm)
        eta = float(np.max(np.abs(residual))/rhs_max)
        if not np.isfinite(rho+eta):
            raise FloatingPointError('nonfinite physical pressure residual')
        checks += 1
        trace.append(dict(iterations=iterations, seconds=time.perf_counter()-setup_end,
                          rho=rho, eta=eta))
        return q, production, residual, rho, eta

    def result(high, low, q, production, rho, eta, outer):
        now = time.perf_counter()
        meta = dict(method=method, status='completed', iterations=iterations,
                    true_checks=checks, outer_corrections=outer, rho=rho, eta=eta,
                    setup_seconds=setup_end-begin, solve_seconds=now-setup_end,
                    total_seconds=now-begin, warm_start=initial is not None,
                    input_total=float(np.sum(rhs)), output_total=float(np.sum(production)),
                    min_drawdown=float(np.min(high+low)), max_drawdown=float(np.max(high+low)),
                    setup_metadata=state['metadata'], trace=trace,
                    pressure_representation='double_double', flow_representation='binary64')
        return dict(hi=high, lo=low, q=q, well_rates=production, meta=meta)

    with threadpool_limits(1):
        q, production, residual, rho, eta = observe(hi, lo)
        if max(rho, eta) <= target:
            return result(hi, lo, q, production, rho, eta, 0)
        for outer in range(1, 21):
            _guard(deadline)
            base_hi, base_lo = hi, lo
            delta = np.zeros_like(rhs)
            r = residual.copy()
            z = M@r
            direction = z.copy()
            rz = float(np.dot(r, z))
            initial_norm = float(np.linalg.norm(r))
            rtol = 1e-6 if outer == 1 else max(5e-15, min(1e-3, .1*target/max(rho, eta)))
            last_observed = -1
            for step in range(1, 5001):
                _guard(deadline)
                Ad = A@direction
                denominator = float(np.dot(direction, Ad))
                if not np.isfinite(denominator+rz) or denominator <= 0 or rz <= 0:
                    break
                alpha = rz/denominator
                delta += alpha*direction
                r -= alpha*Ad
                iterations += 1
                estimate = max(float(np.linalg.norm(r))/rhs_norm,
                               float(np.max(np.abs(r)))/rhs_max)
                converged = np.linalg.norm(r) <= rtol*initial_norm
                if step % 50 == 0 or estimate <= .5*target or converged:
                    hi, lo = gp.pair_update(base_hi, base_lo, delta)
                    q, production, residual, rho, eta = observe(hi, lo)
                    last_observed = step
                    if max(rho, eta) <= target:
                        return result(hi, lo, q, production, rho, eta, outer)
                if converged:
                    break
                z = M@r
                next_rz = float(np.dot(r, z))
                if not np.isfinite(next_rz) or next_rz <= 0:
                    break
                direction = z+(next_rz/rz)*direction
                rz = next_rz
            if last_observed != step:
                hi, lo = gp.pair_update(base_hi, base_lo, delta)
                q, production, residual, rho, eta = observe(hi, lo)
                if max(rho, eta) <= target:
                    return result(hi, lo, q, production, rho, eta, outer)
    raise RuntimeError('dynamic pressure did not satisfy physical balance after 20 corrections: '
                       +str(trace[-3:]))
