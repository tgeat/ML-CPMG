"""Rectangular-grid adapters for the five frozen Darcy solver configurations.

Inputs are Cartesian shape (nz, ny, nx) in C order and positive TPFA edges
grouped by increasing array axis. Kx supplies *only* the scalar high-phase
tag in the two geometric MG methods; weights contain all three diagonal K
entries and the cell dimensions. Boundary faces have prescribed zero flux.
No geometry assumption enters the reused algebraic cycles or coarse solves.
"""
import sys
import time
from importlib import metadata
from math import prod
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import label
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits

from extension_multigrid_v2 import Multigrid, _aggregate_components, _coarse_filter
from extension_strong_mg import _hierarchy as legacy_hierarchy
from paper_ablation_v1 import CONFIGS as ABLATION
from path_multigrid_v2 import _project


CANDIDATE = 'component_lump_1_1_c512'
STRONG = 'strong_nu1_c512'
JACOBI = 'jacobi_original'
METHODS = (CANDIDATE, STRONG, JACOBI, 'sa_theta0_c10', 'rs_theta025_c512')


def _sections(shape):
    offset = 0
    for length in shape:
        count = (prod(shape) // length) * (length - 1)
        yield slice(offset, offset + count)
        offset += count


def _check_inputs(k_tag, edge):
    if k_tag.ndim != 3 or k_tag.size < 2 or not np.all(np.isfinite(k_tag)) or np.any(k_tag <= 0):
        raise ValueError('expected three nonempty Kx axes and finite positive permeability')
    tail, head, trans = edge
    if (tail.ndim != 1 or tail.shape != head.shape or tail.shape != trans.shape
            or tail.dtype.kind not in 'iu' or head.dtype.kind not in 'iu'
            or trans.dtype != np.float64 or np.any(trans <= 0) or not np.isfinite(trans).all()):
        raise ValueError('expected integer directed endpoints and positive float64 transmissibilities')
    ids = np.arange(k_tag.size, dtype=tail.dtype).reshape(k_tag.shape)
    sections = list(_sections(k_tag.shape))
    if sections[-1].stop != tail.size:
        raise ValueError('face count differs from full interior Cartesian grid')
    for axis, section in enumerate(sections):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        if not (np.array_equal(tail[section], ids[tuple(lo)].ravel()) and
                np.array_equal(head[section], ids[tuple(hi)].ravel())):
            raise ValueError('face ordering must be axis 0 then 1 then 2, C order in each')


def assemble_fast(shape, edge):
    """Rectangular equivalent of path_multigrid_v3._fine_csr (sorted direct CSR)."""
    size = prod(shape)
    tail, head, trans = edge
    degree = np.bincount(tail, minlength=size) + np.bincount(head, minlength=size) + 1
    nnz = int(degree.sum())
    index_type = np.int32 if nnz < np.iinfo(np.int32).max else np.int64
    indptr = np.empty(size + 1, dtype=index_type)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    indices = np.empty(nnz, dtype=index_type)
    data = np.empty(nnz, dtype=np.float64)
    cursor = indptr[:-1].copy()
    diagonal = np.bincount(tail, weights=trans, minlength=size) + np.bincount(head, weights=trans, minlength=size)

    def insert(rows, cols, values):
        positions = cursor[rows]
        indices[positions] = cols
        data[positions] = values
        cursor[rows] += 1

    sections = list(_sections(shape))
    for section in sections:
        insert(head[section], tail[section], -trans[section])
    ids = np.arange(size, dtype=index_type)
    insert(ids, ids, diagonal)
    for section in reversed(sections):
        insert(tail[section], head[section], -trans[section])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(size, size))
    matrix.has_sorted_indices = True
    return matrix


def assemble_ordinary(shape, edge):
    """darcy_core.assemble COO summation with externally supplied physical faces."""
    tail, head, t = edge
    rows = np.concatenate((tail, head, tail, head))
    cols = np.concatenate((tail, head, head, tail))
    data = np.concatenate((t, t, -t, -t))
    return sparse.coo_matrix((data, (rows, cols)), shape=(prod(shape),) * 2).tocsr()


class _PreparedCycle(Multigrid):
    """Use the original per-application cycle with a preassembled physical A."""

    def __init__(self, A, k_tag, check_time):
        cfg = ABLATION[CANDIDATE]
        self.A = A
        self.cycle_repeats = cfg['cycle_repeats']
        self.levels = []
        high = k_tag > np.min(k_tag)
        components, count = label(high)
        tags = components.ravel().astype(np.int64, copy=False)
        xyz = np.indices(k_tag.shape).reshape(3, -1).T
        matrix = A
        removed = []
        while True:
            check_time()
            diagonal = matrix.diagonal()
            if matrix.shape[0] > 1 and np.any(diagonal <= 0):
                raise ValueError('nonpositive multigrid diagonal')
            if matrix.shape[0] == 1:
                damped = np.zeros(1)
            else:
                inverse = 1.0 / diagonal
                omega = 2.0 / 3.0
                if self.levels:
                    bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel() * inverse))
                    omega = min(omega, 4.0 / (3.0 * bound))
                damped = omega * inverse
            sweeps = cfg['coarse_sweeps'] if self.levels else cfg['fine_sweeps']
            level = {'A': matrix, 'di': damped, 'sweeps': sweeps}
            self.levels.append(level)
            size = matrix.shape[0]
            if size <= cfg['coarse_size']:
                level['factor'] = splu(matrix[1:, 1:].tocsc()) if size > 1 else None
                break
            cells, coarse_xyz, coarse_tags = _aggregate_components(xyz, tags, count + 1)
            ncoarse = len(coarse_tags)
            if ncoarse >= size or len(self.levels) >= 10:
                level['factor'] = splu(matrix[1:, 1:].tocsc())
                break
            Z = sparse.csr_matrix((np.ones(size), cells, np.arange(size + 1, dtype=np.int64)),
                                  shape=(size, ncoarse))
            AZ = (matrix @ Z).tocsr()
            AZ.data *= np.repeat(damped, np.diff(AZ.indptr))
            P = (Z - AZ).tocsr()
            P.eliminate_zeros()
            P.data *= np.repeat(1.0 / np.asarray(P.sum(axis=1)).ravel(), np.diff(P.indptr))
            R = P.T.tocsr()
            level['P'], level['R'] = P, R
            coarse = (R @ (matrix @ P)).tocsr()
            coarse = ((coarse + coarse.T) * 0.5).tocsr()
            cutoff = cfg['lump_max_relative'] if len(self.levels) >= cfg['lump_start_level'] else None
            matrix, nremoved, _ = _coarse_filter(coarse, cutoff)
            removed.append(nremoved)
            xyz, tags = coarse_xyz, coarse_tags
        self.info = {
            'level_sizes': [x['A'].shape[0] for x in self.levels],
            'level_nnz': [x['A'].nnz for x in self.levels],
            'positive_offdiagonals_lumped': removed,
            'high_component_count': int(count),
            'high_fraction': float(high.mean()),
            'phase_rule': 'Kx > min(Kx)',
            'config': dict(cfg),
        }


def _amg(A, name, check_time):
    vendor = str(Path(__file__).resolve().parent / '.paper_vendor')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from paper_amg_v7 import PRESETS, EXPECTED_PYAMG_VERSION, _quotient_coarse_solver
    import pyamg

    version = metadata.version('pyamg')
    if version != EXPECTED_PYAMG_VERSION or pyamg.__version__ != version:
        raise RuntimeError('expected the frozen PyAMG release ' + EXPECTED_PYAMG_VERSION)
    cfg = dict(PRESETS[name])
    H = sparse.csr_array(A, dtype=np.float64, copy=False)
    H.sum_duplicates()
    H.sort_indices()
    smoother = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': cfg['nu']})
    common = dict(presmoother=smoother, postsmoother=smoother, max_levels=cfg['max_levels'],
                  max_coarse=cfg['max_coarse'], coarse_solver='pinv', keep=False)
    rng = np.random.get_state()
    try:
        np.random.seed(cfg['setup_random_seed'])
        check_time()
        if cfg['family'] == 'sa':
            hierarchy = pyamg.smoothed_aggregation_solver(
                H, B=np.ones((H.shape[0], 1)), symmetry='hermitian',
                strength=('symmetric', {'theta': cfg['strength_theta']}),
                aggregate='standard', smooth=('jacobi', {'omega': 4.0 / 3.0}),
                improve_candidates=(('block_gauss_seidel', {'iterations': 4, 'sweep': 'symmetric'}), None),
                **common)
        else:
            hierarchy = pyamg.ruge_stuben_solver(
                H, strength=('classical', {'theta': cfg['strength_theta']}),
                CF=('RS', {'second_pass': False}), interpolation='classical', **common)
    finally:
        np.random.set_state(rng)
    check_time()
    solve_coarse, info = _quotient_coarse_solver(hierarchy, cfg['family'], check_time)
    hierarchy.coarse_solver = pyamg.coarse_grid_solver(solve_coarse)
    last = hierarchy.levels[-1].A
    hierarchy.coarse_solver(last, np.zeros(last.shape[0]))
    raw = hierarchy.aspreconditioner(cycle=cfg['cycle'])

    def apply(rhs):
        rhs = _project(rhs)
        if rhs.size != H.shape[0] or not np.all(np.isfinite(rhs)):
            raise ValueError('invalid physical residual')
        check_time()
        return _project(np.asarray(raw @ rhs).ravel())

    return LinearOperator(H.shape, matvec=apply, dtype=np.float64), {
        'library_version': version, 'config': cfg, 'level_sizes': [v.A.shape[0] for v in hierarchy.levels],
        'level_nnz': [v.A.nnz for v in hierarchy.levels], **info,
    }


def build(method, k_tag, edge, deadline=None):
    """Build the chosen preconditioner and fine CSR afresh for one cold run."""
    start = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}')
    k_tag = np.asarray(k_tag, dtype=np.float64)
    edge = tuple(map(np.asarray, edge))
    _check_inputs(k_tag, edge)

    def check_time():
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('SPE10 solver deadline')

    check_time()
    with threadpool_limits(1):
        assembly = assemble_ordinary if method in (STRONG, JACOBI) else assemble_fast
        A = assembly(k_tag.shape, edge)
        assembled = time.perf_counter()
        if method == CANDIDATE:
            state = _PreparedCycle(A, k_tag, check_time)
            M = LinearOperator(A.shape, matvec=state.apply_preconditioner, dtype=np.float64)
            info = state.info
        elif method == STRONG:
            M, info = legacy_hierarchy(A, k_tag, nu=1, coarse_size=512)
            info['phase_rule'] = 'Kx > min(Kx)'
        elif method == JACOBI:
            diagonal = A.diagonal()
            M = LinearOperator(A.shape, matvec=lambda rhs: _project(_project(rhs) / diagonal),
                               dtype=np.float64)
            info = {}
        else:
            M, info = _amg(A, method, check_time)
    check_time()
    end = time.perf_counter()
    info.update(name=method, assembly=assembly.__name__, assembly_seconds=assembled-start,
                hierarchy_seconds=end-assembled, setup_seconds=end-start)
    return {'A': A, 'M': M, 'krylov': 'cg', 'metadata': info}
