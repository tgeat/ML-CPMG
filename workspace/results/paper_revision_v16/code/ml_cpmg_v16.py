"""Multi-Level Persistence Component-Separating Aggregation Multigrid (ML-CPMG).

This module generalizes component-preserving smoothed aggregation multigrid
(CP-MG) from binary permeability media (``k > min(k)``) to arbitrary continuous
multi-level heterogeneous and anisotropic Darcy media while preserving exact
binary-medium equivalence and strict SPD / quotient-SPD admissibility for PCG.

Mathematical mechanisms
-----------------------
1. Geometry-normalized face conductivity:
   Given oriented TPFA edges ``e = (u, v)`` with physical transmissibility ``t_e``
   and scalar cell permeabilities ``k_u, k_v``, let ``h_e = 2*k_u*k_v/(k_u+k_v)``.
   Normalizing ``t_e`` by the per-axis median geometric factor ``g_a = median(t/h)``
   yields geometry-free effective face conductivities ``c_e = t_e / g_a``.
   On rectangular grids (e.g. SPE10 ``20 x 10 x 2`` ft with ``100 : 4 : 1`` face
   aspect ratios), homogeneous rock recovers isotropic ``c_e = K_0``, whereas thin
   anisotropic barriers (e.g. ``Kz << Kx``) satisfy ``min(k_u, k_v) / c_e >> 1``
   and are severed cleanly.

2. Superlevel persistence watershed with pedestal separation:
   Active cells and geometry-normalized edges above background ``ell_bg`` are
   swept across descending log-conductivity superlevels in ``O(N + E)`` vectorized
   array passes:
   - Saddle separation: two components that have each achieved log-prominence
     ``>= log(theta_pers)`` and volume ``>= min_cells`` are mature conductive
     peaks and never merge across a lower saddle.
   - Pedestal separation: an intermediate-tier lobe supporting either multiple
     mature cores or volume ``> pedestal_ratio * size(core)`` at contrast
     ``>= theta_pers`` below the core peak refuses upward absorption into the
     nested ultra-high core, preserving nested ``10^6``-in-``10^3`` structures.
   - Onion-shell suppression: monotone smooth flanks around a single peak attach
     directly into that peak's basin without concentric multi-bin rings.
   - Binary fast-path: two-phase fields (``|{k}| <= 2``) without anisotropic
     barriers reduce identically to ``k > min(k)`` face-connected components.

3. Hybrid component-geometric + heavy-edge algebraic hierarchy:
   Coarse levels aggregate ``(xyz // 2, tag)`` via 1D ``int64`` key packing until
   geometric coarsening slows (``n_next > stall_ratio * n_curr``), then switch
   cleanly to vectorized symmetric-strength heavy-edge algebraic aggregation
   respecting disconnected flow components down to ``n_curr <= coarse_size``.
   Coarse direct factorization enforces a hard ``n_curr <= max_direct <= 2048``
   memory guard before every ``splu`` call.

4. Strict SPD / Quotient-SPD admissibility:
   Prolongation ``P = (I - omega * D_tilde^-1 * A_f) Z`` uses nonnegative weights
   with ``A_f @ 1 = 0`` and ``Z @ 1 = 1``, guaranteeing ``P @ 1 = 1`` on both
   zero-mean Neumann quotients and well-grounded SPD systems.  Symmetric Galerkin
   coarse operators ``R @ A @ P`` with ``R = P.T``, PSD positive-offdiagonal
   energy lumping, rank-safe damped Jacobi smoothing, and exact symmetric coarse
   solves ensure a fixed linear SPD preconditioner ``M`` for PCG.

No numerical work, I/O, or RNG mutation executes on import.
"""
from __future__ import annotations

import os
for _env_name in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
):
    os.environ[_env_name] = '1'

from math import prod
import time
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import label
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits

from darcy_core import action, edges
from extension_multigrid_v2 import _coarse_filter
from path_multigrid_v2 import Multigrid as _SolveBase, _csr_bytes, _project
from path_multigrid_v3 import _fine_csr


PHASE_MODES = (
    'ml_cpmg',
    'cliff_only',
    'naive_multibin',
    'single_median',
    'legacy_k_gt_min',
)

PARAMETER_DOMAIN = {
    'phase_mode': PHASE_MODES,
    'fine_sweeps': (1, 2),
    'coarse_sweeps': (1, 2),
    'coarse_size': (64, 96, 192, 512),
    'max_direct': (512, 1024, 2048),
    'lump_max_relative': 'None (exact Galerkin) or a float in [0, 1]',
    'lump_start_level': (1, 2, 3),
    'cycle_repeats': (1, 2),
    'theta_pers': 'float > 1.0 (persistence saddle prominence ratio)',
    'tau_barrier': 'float > 1.0 (geometry-normalized barrier ratio)',
    'tau_cliff': 'float > 1.0 (local cliff ratio for cliff_only mode)',
    'bg_alpha': 'float in (0.0, 0.8) (log-range background fraction)',
    'min_peak_cells': 'positive int (base minimum basin core volume)',
    'pedestal_ratio': 'float >= 1.0 (nested lobe-to-core pedestal volume ratio)',
    'max_basins': 'None or positive int (prominence-ranked basin cap)',
    'prolong_omega': 'float in (0, 0.5) (rank-preserving interpolation damping)',
    'stall_ratio': 'float in (0.15, 0.99] (geometric-to-algebraic switch ratio)',
    'prolong_filter': 'float in [0.0, 0.5] (coarse prolongation strength cutoff)',
    'prolong_filter_start_level': (0, 1, 2, 3),
    'max_alg_group': (4, 6, 8, 12, 16),
}

_BASE_CONTROLS = {
    'fine_sweeps': 1,
    'coarse_sweeps': 1,
    'coarse_size': 512,
    'max_direct': 2048,
    'lump_max_relative': 1.0,
    'lump_start_level': 1,
    'cycle_repeats': 1,
    'theta_pers': 16.0,
    'tau_barrier': 8.0,
    'tau_cliff': 1.0e30,
    'bg_alpha': 0.22,
    'min_peak_cells': 6,
    'pedestal_ratio': 2.0,
    'max_basins': 256,
    'prolong_omega': 0.45,
    'stall_ratio': 0.33,
    'prolong_filter': 0.04,
    'prolong_filter_start_level': 1,
    'max_alg_group': 8,
}

CONFIGS = {
    'ml_cpmg': dict(
        _BASE_CONTROLS,
        phase_mode='ml_cpmg',
    ),
    'cliff_only': dict(
        _BASE_CONTROLS,
        phase_mode='cliff_only',
        tau_cliff=24.0,
    ),
    'naive_multibin': dict(
        _BASE_CONTROLS,
        phase_mode='naive_multibin',
    ),
    'single_median': dict(
        _BASE_CONTROLS,
        phase_mode='single_median',
    ),
    'legacy_k_gt_min': dict(
        _BASE_CONTROLS,
        phase_mode='legacy_k_gt_min',
        prolong_filter=0.0,
        stall_ratio=0.95,
    ),
}

DEFAULT_CONFIG = 'ml_cpmg'


def _resolve_config(config):
    """Validate and resolve a named preset or parameter override dictionary."""
    if config is None:
        name = DEFAULT_CONFIG
        cfg = dict(CONFIGS[name])
    elif isinstance(config, str):
        if config not in CONFIGS:
            raise ValueError(f'unknown multigrid configuration: {config!r}')
        name = config
        cfg = dict(CONFIGS[name])
    elif isinstance(config, dict):
        supplied = dict(config)
        unknown = set(supplied) - set(PARAMETER_DOMAIN)
        if unknown:
            raise ValueError('unknown multigrid parameter(s): ' + ', '.join(sorted(unknown)))
        if not supplied:
            name = DEFAULT_CONFIG
            cfg = dict(CONFIGS[name])
        else:
            base_mode = supplied.get('phase_mode', DEFAULT_CONFIG)
            base_preset = CONFIGS.get(base_mode, CONFIGS[DEFAULT_CONFIG])
            cfg = dict(base_preset)
            cfg.update(supplied)
            matched = [k for k, v in CONFIGS.items() if v == cfg]
            name = matched[0] if len(matched) == 1 else 'custom'
    else:
        raise TypeError('config must be None, a preset name, or a parameter dict')

    if cfg['phase_mode'] not in PHASE_MODES:
        raise ValueError(f"unsupported phase_mode: {cfg['phase_mode']!r}")
    for key in (
        'fine_sweeps',
        'coarse_sweeps',
        'coarse_size',
        'max_direct',
        'lump_start_level',
        'cycle_repeats',
        'prolong_filter_start_level',
        'max_alg_group',
    ):
        val = cfg[key]
        if isinstance(val, (bool, np.bool_)) or val not in PARAMETER_DOMAIN[key]:
            raise ValueError(f'unsupported {key}: {val!r}')
        cfg[key] = int(val)
    if cfg['max_direct'] < cfg['coarse_size'] or cfg['max_direct'] > 2048:
        raise ValueError('max_direct must satisfy coarse_size <= max_direct <= 2048')

    cutoff = cfg['lump_max_relative']
    if cutoff is not None:
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or not (0.0 <= cutoff <= 1.0):
            raise ValueError('lump_max_relative must be None or lie in [0, 1]')
        cfg['lump_max_relative'] = cutoff

    for key, lo in (('theta_pers', 1.0), ('tau_barrier', 1.0), ('tau_cliff', 1.0)):
        val = float(cfg[key])
        if not np.isfinite(val) or val <= lo:
            raise ValueError(f'{key} must be finite and > {lo}')
        cfg[key] = val

    bg_alpha = float(cfg['bg_alpha'])
    if not np.isfinite(bg_alpha) or not (0.0 < bg_alpha < 0.8):
        raise ValueError('bg_alpha must lie in (0.0, 0.8)')
    cfg['bg_alpha'] = bg_alpha

    ped = float(cfg['pedestal_ratio'])
    if not np.isfinite(ped) or ped < 1.0:
        raise ValueError('pedestal_ratio must be >= 1.0')
    cfg['pedestal_ratio'] = ped

    pw = float(cfg['prolong_omega'])
    if not np.isfinite(pw) or not (0.0 < pw < 0.5):
        raise ValueError('prolong_omega must lie in (0, 0.5)')
    cfg['prolong_omega'] = pw

    sr = float(cfg['stall_ratio'])
    if not np.isfinite(sr) or not (0.15 < sr <= 0.99):
        raise ValueError('stall_ratio must lie in (0.15, 0.99]')
    cfg['stall_ratio'] = sr

    pf = float(cfg['prolong_filter'])
    if not np.isfinite(pf) or not (0.0 <= pf <= 0.5):
        raise ValueError('prolong_filter must lie in [0.0, 0.5]')
    cfg['prolong_filter'] = pf

    mpc = cfg['min_peak_cells']
    if isinstance(mpc, (bool, np.bool_)) or not isinstance(mpc, (int, np.integer)) or int(mpc) < 1:
        raise ValueError('min_peak_cells must be a positive integer')
    cfg['min_peak_cells'] = int(mpc)

    mb = cfg['max_basins']
    if mb is not None:
        if isinstance(mb, (bool, np.bool_)) or not isinstance(mb, (int, np.integer)) or int(mb) < 1:
            raise ValueError('max_basins must be None or a positive integer')
        cfg['max_basins'] = int(mb)

    return name, cfg


def _check_deadline(deadline, message='multigrid deadline exceeded'):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError(message)


def _sections(shape):
    offset = 0
    total = prod(shape)
    for length in shape:
        count = (total // length) * (length - 1)
        yield slice(offset, offset + count)
        offset += count


def _is_standard_cartesian_edges(shape, edge):
    """Return True if edge matches axis-0, axis-1, axis-2 C-order Cartesian faces."""
    tail, head, trans = edge
    expected = sum((prod(shape) // length) * (length - 1) for length in shape)
    if tail.size != expected:
        return False
    ids = np.arange(prod(shape), dtype=tail.dtype).reshape(shape)
    for axis, section in enumerate(_sections(shape)):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        if not (
            np.array_equal(tail[section], ids[tuple(lo)].ravel())
            and np.array_equal(head[section], ids[tuple(hi)].ravel())
        ):
            return False
    return True


def _check_rectangular_inputs(k_tag, edge):
    """Validate 3D rectangular Cartesian grid and axis-ordered positive TPFA faces."""
    if k_tag.ndim != 3 or k_tag.size < 2 or not np.all(np.isfinite(k_tag)) or np.any(k_tag <= 0):
        raise ValueError('expected three nonempty Kx axes and finite positive permeability')
    tail, head, trans = edge
    if (
        tail.ndim != 1
        or tail.shape != head.shape
        or tail.shape != trans.shape
        or tail.dtype.kind not in 'iu'
        or head.dtype.kind not in 'iu'
        or trans.dtype != np.float64
        or np.any(trans <= 0)
        or not np.isfinite(trans).all()
    ):
        raise ValueError('expected integer directed endpoints and positive float64 transmissibilities')
    if not _is_standard_cartesian_edges(k_tag.shape, edge):
        raise ValueError('face ordering must be axis 0 then 1 then 2, C order in each')


def assemble_cartesian_fast(shape, edge):
    """Sorted direct 7-point CSR assembly for arbitrary 3D rectangular shape."""
    size = prod(shape)
    tail, head, trans = edge
    degree = np.bincount(tail, minlength=size) + np.bincount(head, minlength=size) + 1
    nnz = int(degree.sum())
    index_type = np.int32 if max(size, nnz) < np.iinfo(np.int32).max else np.int64
    indptr = np.empty(size + 1, dtype=index_type)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    indices = np.empty(nnz, dtype=index_type)
    data = np.empty(nnz, dtype=np.float64)
    cursor = indptr[:-1].copy()
    diagonal = (
        np.bincount(tail, weights=trans, minlength=size)
        + np.bincount(head, weights=trans, minlength=size)
    )

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


def _assemble_general_edges(size, edge, well_diagonal=None):
    """Assemble symmetric Laplacian (plus optional nonnegative grounding d) from edges."""
    tail, head, trans = edge
    if well_diagonal is None:
        rows = np.concatenate((tail, head, tail, head))
        cols = np.concatenate((tail, head, head, tail))
        vals = np.concatenate((trans, trans, -trans, -trans))
    else:
        rows = np.concatenate((tail, head, tail, head, np.arange(size, dtype=tail.dtype)))
        cols = np.concatenate((tail, head, head, tail, np.arange(size, dtype=tail.dtype)))
        vals = np.concatenate((trans, trans, -trans, -trans, well_diagonal))
    matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _lump_positive_grounded(matrix, relative_cutoff):
    """Lump weak positive off-diagonals into the diagonal while preserving row sums."""
    matrix = matrix.tocsr()
    if relative_cutoff is None:
        matrix.eliminate_zeros()
        return matrix, 0, 0
    rows = np.repeat(
        np.arange(matrix.shape[0], dtype=matrix.indices.dtype),
        np.diff(matrix.indptr),
    )
    positive = (matrix.data > 0.0) & (matrix.indices != rows)
    positive_count = int(np.count_nonzero(positive))
    locations = np.flatnonzero(positive)
    diagonal = matrix.diagonal()
    if len(locations):
        pair_scale = np.sqrt(diagonal[rows[locations]] * diagonal[matrix.indices[locations]])
        locations = locations[matrix.data[locations] <= relative_cutoff * pair_scale]
    if len(locations):
        addition = np.bincount(
            rows[locations], weights=matrix.data[locations], minlength=matrix.shape[0]
        )
        matrix.data[locations] = 0.0
        matrix.setdiag(diagonal + addition)
    matrix.eliminate_zeros()
    return matrix, int(len(locations)), positive_count


def is_binary_field(k_flat):
    """Fast O(N) exact check for uniform or two-value permeability fields."""
    if k_flat.size == 0:
        return True
    kmin = float(np.min(k_flat))
    kmax = float(np.max(k_flat))
    if kmax <= kmin:
        return True
    sample = k_flat[:min(512, k_flat.size)]
    if not np.all((sample == kmin) | (sample == kmax)):
        return False
    return bool(np.all((k_flat == kmin) | (k_flat == kmax)))


def normalized_conductivities(k_flat, xyz, edge, cartesian_shape=None):
    """Return scalar cell heights and axis-geometry-normalized face conductivities."""
    tail, head, trans = edge
    if trans.size == 0:
        return k_flat, np.empty(0, dtype=np.float64)
    ku = k_flat[tail]
    kv = k_flat[head]
    harm = (2.0 * ku * kv) / (ku + kv)
    tiny = np.finfo(np.float64).tiny
    c_edge = np.empty_like(trans, dtype=np.float64)

    if cartesian_shape is not None:
        for section in _sections(cartesian_shape):
            cnt = section.stop - section.start
            if cnt > 0:
                step = max(1, cnt // 16384)
                r_sub = trans[section][::step] / np.maximum(harm[section][::step], tiny)
                g_ax = float(np.median(r_sub))
                c_edge[section] = trans[section] / max(g_ax, tiny)
    else:
        ratio = trans / np.maximum(harm, tiny)
        if xyz is not None:
            dxyz = np.abs(xyz[tail] - xyz[head])
            l1 = dxyz.sum(axis=1)
            assigned = np.zeros(len(trans), dtype=bool)
            for ax in range(3):
                mask = (l1 == 1) & (dxyz[:, ax] == 1)
                if np.any(mask):
                    r_sub = ratio[mask]
                    step = max(1, r_sub.size // 16384)
                    g_ax = float(np.median(r_sub[::step]))
                    c_edge[mask] = trans[mask] / max(g_ax, tiny)
                    assigned |= mask
            if not np.all(assigned):
                rem = ~assigned
                step = max(1, ratio.size // 16384)
                g_all = float(np.median(ratio[::step]))
                c_edge[rem] = trans[rem] / max(g_all, tiny)
        else:
            step = max(1, ratio.size // 16384)
            g_all = float(np.median(ratio[::step]))
            c_edge[:] = trans / max(g_all, tiny)

    return k_flat, c_edge


def _graph_mask_components(n, tail, head, node_mask, edge_mask=None):
    """Connected components of active nodes restricted to active undirected edges."""
    ids = np.flatnonzero(node_mask)
    m = len(ids)
    if m == 0:
        return np.zeros(n, dtype=np.int64), 0
    valid = node_mask[tail] & node_mask[head]
    if edge_mask is not None:
        valid &= edge_mask
    inv_map = np.full(n, -1, dtype=np.int32)
    inv_map[ids] = np.arange(m, dtype=np.int32)
    u = inv_map[tail[valid]]
    v = inv_map[head[valid]]
    adj = sparse.coo_matrix(
        (np.ones(2 * len(u), dtype=np.uint8), (np.concatenate((u, v)), np.concatenate((v, u)))),
        shape=(m, m),
    ).tocsr()
    num, comp = connected_components(adj, directed=False)
    tags = np.zeros(n, dtype=np.int64)
    tags[ids] = comp.astype(np.int64) + 1
    return tags, int(num)


def _sv_union_immature(c_parent, c_size, c_peak, c_pedestal, iu, iv, n_comp):
    """Vectorized Shiloach-Vishkin union over immature component roots."""
    is_touched = np.zeros(n_comp, dtype=bool)
    is_touched[iu] = True
    is_touched[iv] = True
    touched = np.flatnonzero(is_touched)

    u, v = iu, iv
    while u.size > 0:
        ru = c_parent[u]
        rv = c_parent[v]
        diff = ru != rv
        if not np.any(diff):
            break
        u, v = u[diff], v[diff]
        ru, rv = ru[diff], rv[diff]
        np.minimum.at(c_parent, np.maximum(ru, rv), np.minimum(ru, rv))
        c_parent[touched] = c_parent[c_parent[touched]]

    r = c_parent[touched]
    while True:
        rr = c_parent[r]
        if np.array_equal(r, rr):
            break
        r = rr
    c_parent[touched] = r

    moved_mask = touched != r
    if np.any(moved_mask):
        moved = touched[moved_mask]
        dest = r[moved_mask]
        np.add.at(c_size, dest, c_size[moved])
        np.maximum.at(c_peak, dest, c_peak[moved])
        np.logical_or.at(c_pedestal, dest, c_pedestal[moved])


def _partition_with_background_zero(n, tail, head, keep_edge):
    """Connected components after severing barriers when scalar k is near-uniform."""
    u, v = tail[keep_edge], head[keep_edge]
    adj = sparse.coo_matrix(
        (np.ones(2 * len(u), dtype=np.uint8), (np.concatenate((u, v)), np.concatenate((v, u)))),
        shape=(n, n),
    ).tocsr()
    num, comp = connected_components(adj, directed=False)
    if num <= 1:
        return np.zeros(n, dtype=np.int64), 0
    counts = np.bincount(comp, minlength=num)
    bg_c = int(np.argmax(counts))
    order = [bg_c] + [i for i in range(num) if i != bg_c]
    remap = np.empty(num, dtype=np.int64)
    remap[order] = np.arange(num, dtype=np.int64)
    return remap[comp], int(num - 1)


def _extract_components_core(k_flat, xyz, edge, cartesian_shape, cfg):
    """Dispatch component extraction according to cfg['phase_mode']."""
    n = k_flat.size
    mode = cfg['phase_mode']
    tail, head, _ = edge

    if mode == 'legacy_k_gt_min':
        high = k_flat > float(np.min(k_flat))
        if not np.any(high):
            return np.zeros(n, dtype=np.int64), 0, 'Kx > min(Kx)'
        if cartesian_shape is not None:
            comp_grid, num = label(high.reshape(cartesian_shape))
            return comp_grid.ravel().astype(np.int64, copy=False), int(num), 'Kx > min(Kx)'
        tags, num = _graph_mask_components(n, tail, head, high)
        return tags, num, 'Kx > min(Kx)'

    if mode == 'single_median':
        ell = np.log(k_flat)
        med = float(np.median(ell))
        high = ell > med
        if not np.any(high):
            return np.zeros(n, dtype=np.int64), 0, 'log(Kx) > median(log(Kx))'
        if cartesian_shape is not None:
            comp_grid, num = label(high.reshape(cartesian_shape))
            return (
                comp_grid.ravel().astype(np.int64, copy=False),
                int(num),
                'log(Kx) > median(log(Kx))',
            )
        tags, num = _graph_mask_components(n, tail, head, high)
        return tags, num, 'log(Kx) > median(log(Kx))'

    if mode == 'naive_multibin':
        ell = np.log(k_flat)
        emin, emax = float(np.min(ell)), float(np.max(ell))
        if emax <= emin + 1e-12:
            return np.zeros(n, dtype=np.int64), 0, 'naive_4bin_log_quantization'
        bins = np.clip(np.floor(4.0 * (ell - emin) / (emax - emin)).astype(np.int32), 0, 3)
        active = bins > 0
        if not np.any(active):
            return np.zeros(n, dtype=np.int64), 0, 'naive_4bin_log_quantization'
        same_bin = active[tail] & active[head] & (bins[tail] == bins[head])
        tags, num = _graph_mask_components(n, tail, head, active, edge_mask=same_bin)
        return tags, num, 'naive_4bin_log_quantization'

    # Modes 'ml_cpmg' and 'cliff_only' use geometry-normalized edge conductivities
    s_node, c_edge = normalized_conductivities(k_flat, xyz, edge, cartesian_shape)
    tau_bar = cfg['tau_barrier']
    su, sv = (s_node[tail], s_node[head]) if tail.size else (np.empty(0), np.empty(0))
    is_bar = (np.minimum(su, sv) > tau_bar * c_edge) if tail.size else np.zeros(0, dtype=bool)
    has_bar = bool(np.any(is_bar))

    if is_binary_field(k_flat):
        high = k_flat > float(np.min(k_flat))
        if not np.any(high) and not has_bar:
            return np.zeros(n, dtype=np.int64), 0, 'binary_exact_fastpath'
        if np.any(high) and not has_bar and cartesian_shape is not None:
            comp_grid, num = label(high.reshape(cartesian_shape))
            return (
                comp_grid.ravel().astype(np.int64, copy=False),
                int(num),
                'binary_exact_fastpath',
            )
        if np.any(high):
            tags, num = _graph_mask_components(n, tail, head, high, edge_mask=~is_bar)
            return tags, num, 'binary_barrier_fastpath'
        tags, num = _partition_with_background_zero(n, tail, head, ~is_bar)
        return tags, num, 'uniform_anisotropic_barrier_partition'

    tau_clf = cfg['tau_cliff']
    if tau_clf < 1.0e20 and tail.size:
        keep_edge = ~(is_bar | (np.maximum(su, sv) > tau_clf * c_edge))
    else:
        keep_edge = ~is_bar

    smin, smax = float(np.min(s_node)), float(np.max(s_node))
    if smax <= 4.0 * smin:
        if np.all(keep_edge):
            return np.zeros(n, dtype=np.int64), 0, 'near_homogeneous_single_basin'
        tags, num = _partition_with_background_zero(n, tail, head, keep_edge)
        return tags, num, 'near_homogeneous_barrier_partition'

    ell = np.log(s_node)
    step_q = max(1, n // 32768)
    ell_lo = float(np.quantile(ell[::step_q], 0.05))
    ell_hi = float(np.max(ell))
    if ell_hi <= ell_lo + 1e-9:
        ell_lo = float(np.min(ell))
    ell_bg = float(ell_lo + cfg['bg_alpha'] * (ell_hi - ell_lo))
    exp_bg = float(np.exp(ell_bg))

    active_mask = ell > ell_bg
    active_ids = np.flatnonzero(active_mask)
    m = len(active_ids)
    if m == 0:
        return np.zeros(n, dtype=np.int64), 0, f'{mode}_empty_foreground'

    eff_min_cells = max(4, min(cfg['min_peak_cells'], max(4, n // 8192)))
    max_basins = cfg['max_basins']

    node_to_act = np.full(n, -1, dtype=np.int32)
    node_to_act[active_ids] = np.arange(m, dtype=np.int32)

    valid_e = keep_edge & active_mask[tail] & active_mask[head] & (c_edge > exp_bg)
    eu = node_to_act[tail[valid_e]]
    ev = node_to_act[head[valid_e]]

    if mode == 'cliff_only':
        adj = sparse.coo_matrix(
            (np.ones(2 * len(eu), dtype=np.uint8), (np.concatenate((eu, ev)), np.concatenate((ev, eu)))),
            shape=(m, m),
        ).tocsr()
        num, comp = connected_components(adj, directed=False)
        sizes = np.bincount(comp, minlength=num)
        kept = np.flatnonzero(sizes >= eff_min_cells)
        if max_basins is not None and len(kept) > max_basins:
            top = np.argsort(sizes[kept])[::-1][:max_basins]
            kept = np.sort(kept[top])
        remap = np.zeros(num, dtype=np.int64)
        remap[kept] = np.arange(1, len(kept) + 1, dtype=np.int64)
        tags = np.zeros(n, dtype=np.int64)
        tags[active_ids] = remap[comp]
        return tags, int(len(kept)), 'cliff_only_superlevel'

    # Full superlevel persistence watershed ('ml_cpmg')
    ec = np.log(c_edge[valid_e])
    ell_act = ell[active_ids]
    log_theta = float(np.log(cfg['theta_pers']))
    pedestal_ratio = cfg['pedestal_ratio']
    delta = max(0.35, 0.25 * log_theta)
    n_steps = int(np.clip(np.ceil((ell_hi - ell_bg) / delta), 4, 20))
    taus = np.linspace(ell_hi - 0.5 * delta, ell_bg, n_steps)

    neg_taus = -taus
    node_bucket = np.clip(np.searchsorted(neg_taus, -ell_act, side='right'), 0, n_steps - 1)
    edge_c_bucket = np.clip(np.searchsorted(neg_taus, -ec, side='right'), 0, n_steps - 1)
    edge_bucket = np.maximum(np.maximum(node_bucket[eu], node_bucket[ev]), edge_c_bucket)

    nb_counts = np.bincount(node_bucket, minlength=n_steps)
    node_splits = np.empty(n_steps + 1, dtype=np.int64)
    node_splits[0] = 0
    np.cumsum(nb_counts, out=node_splits[1:])
    node_order = np.argsort(node_bucket, kind='stable')

    eb_counts = np.bincount(edge_bucket, minlength=n_steps)
    edge_splits = np.empty(n_steps + 1, dtype=np.int64)
    edge_splits[0] = 0
    np.cumsum(eb_counts, out=edge_splits[1:])
    edge_order = np.argsort(edge_bucket, kind='stable')

    comp = np.full(m, -1, dtype=np.int32)
    max_comps = m + 1
    c_parent = np.arange(max_comps, dtype=np.int32)
    c_peak = np.zeros(max_comps, dtype=np.float64)
    c_size = np.zeros(max_comps, dtype=np.int32)
    c_pedestal = np.zeros(max_comps, dtype=bool)
    scratch_min = np.empty(max_comps, dtype=np.int32)
    scratch_max = np.empty(max_comps, dtype=np.int32)
    n_comp = 0

    def find_roots(ids):
        r = c_parent[ids]
        while True:
            rr = c_parent[r]
            if np.array_equal(r, rr):
                return r
            c_parent[ids] = rr
            r = rr

    for b_idx in range(n_steps):
        tau = taus[b_idx]
        new_nds = node_order[node_splits[b_idx]:node_splits[b_idx + 1]]
        e_idx = edge_order[edge_splits[b_idx]:edge_splits[b_idx + 1]]

        if new_nds.size > 0:
            c_ids = np.arange(n_comp, n_comp + new_nds.size, dtype=np.int32)
            comp[new_nds] = c_ids
            c_parent[c_ids] = c_ids
            c_peak[c_ids] = ell_act[new_nds]
            c_size[c_ids] = 1
            n_comp += new_nds.size

        if e_idx.size == 0:
            continue

        cu = find_roots(comp[eu[e_idx]])
        cv = find_roots(comp[ev[e_idx]])
        ew = ec[e_idx]

        diff = cu != cv
        if not np.any(diff):
            continue
        cu, cv, ew = cu[diff], cv[diff], ew[diff]

        mat_u = (c_peak[cu] - tau >= log_theta) & (c_size[cu] >= eff_min_cells)
        mat_v = (c_peak[cv] - tau >= log_theta) & (c_size[cv] >= eff_min_cells)

        imm_mask = (~mat_u) & (~mat_v)
        if np.any(imm_mask):
            _sv_union_immature(c_parent, c_size, c_peak, c_pedestal, cu[imm_mask], cv[imm_mask], n_comp)
            cu = find_roots(cu)
            cv = find_roots(cv)
            keep2 = cu != cv
            if not np.any(keep2):
                continue
            cu, cv, ew = cu[keep2], cv[keep2], ew[keep2]
            mat_u = (c_peak[cu] - tau >= log_theta) & (c_size[cu] >= eff_min_cells)
            mat_v = (c_peak[cv] - tau >= log_theta) & (c_size[cv] >= eff_min_cells)

        one_mat = mat_u ^ mat_v
        if np.any(one_mat):
            imm_c = np.where(mat_u[one_mat], cv[one_mat], cu[one_mat])
            mat_c = np.where(mat_u[one_mat], cu[one_mat], cv[one_mat])
            w_c = ew[one_mat]

            scratch_min[imm_c] = max_comps
            scratch_max[imm_c] = -1
            np.minimum.at(scratch_min, imm_c, mat_c)
            np.maximum.at(scratch_max, imm_c, mat_c)
            touches_multi = (scratch_min[imm_c] < scratch_max[imm_c]) & (
                c_size[imm_c] >= 2 * eff_min_cells
            )

            newly_ped = (
                (
                    (c_size[imm_c] > pedestal_ratio * np.maximum(c_size[mat_c], eff_min_cells))
                    | touches_multi
                )
                & (c_peak[mat_c] - c_peak[imm_c] >= log_theta)
            )
            if np.any(newly_ped):
                c_pedestal[imm_c[newly_ped]] = True

            blocked = c_pedestal[imm_c] & (c_peak[mat_c] - c_peak[imm_c] >= log_theta)
            valid_attach = ~blocked
            if np.any(valid_attach):
                imm_c = imm_c[valid_attach]
                mat_c = mat_c[valid_attach]
                w_c = w_c[valid_attach]
                order = np.lexsort((w_c, imm_c))
                imm_s = imm_c[order]
                mat_s = mat_c[order]
                last_mask = np.empty(len(imm_s), dtype=bool)
                last_mask[:-1] = imm_s[:-1] != imm_s[1:]
                last_mask[-1] = True
                chosen_imm = imm_s[last_mask]
                chosen_mat = mat_s[last_mask]
                c_parent[chosen_imm] = chosen_mat
                np.add.at(c_size, chosen_mat, c_size[chosen_imm])
                np.maximum.at(c_peak, chosen_mat, c_peak[chosen_imm])

    if n_comp == 0:
        return np.zeros(n, dtype=np.int64), 0, 'ml_persistence_watershed'

    final_roots = find_roots(comp)
    r_sizes_all = np.bincount(final_roots, minlength=n_comp)
    occ_roots = np.flatnonzero(r_sizes_all)
    r_sizes = r_sizes_all[occ_roots]
    r_peaks = c_peak[occ_roots]
    min_final = max(eff_min_cells, n // 30000)
    prom = r_peaks - ell_bg
    keep_root = (r_sizes >= min_final) & (prom >= 0.5 * log_theta)
    kept_idx = np.flatnonzero(keep_root)
    if max_basins is not None and len(kept_idx) > max_basins:
        scores = prom[kept_idx] * np.log1p(r_sizes[kept_idx])
        top = np.argsort(scores)[::-1][:max_basins]
        kept_idx = np.sort(kept_idx[top])

    root_to_tag = np.zeros(n_comp, dtype=np.int64)
    root_to_tag[occ_roots[kept_idx]] = np.arange(1, len(kept_idx) + 1, dtype=np.int64)
    tags = np.zeros(n, dtype=np.int64)
    tags[active_ids] = root_to_tag[final_roots]
    return tags, int(len(kept_idx)), 'ml_persistence_watershed'


def extract_multilevel_components(k, edge=None, xyz=None, config=None):
    """Extract multi-level conductive basin labels for 3D grids or active graphs.

    Returns ``(labels, high_component_count, info)`` where ``labels`` matches
    ``k.shape`` when ``k`` is 3D (or 1D ``(len(k),)`` on active graphs), ``0``
    denotes low/background phase, and ``1..high_component_count`` denote
    separated conductive structures.
    """
    t0 = time.perf_counter()
    cfg_name, cfg = _resolve_config(config)
    k_arr = np.asarray(k, dtype=np.float64)
    if k_arr.size == 0 or not np.all(np.isfinite(k_arr)) or np.any(k_arr <= 0):
        raise ValueError('k must be nonempty, finite, and strictly positive')

    if k_arr.ndim == 3:
        cartesian_shape = k_arr.shape
        if edge is None:
            if len(set(cartesian_shape)) != 1:
                raise ValueError('edge is required for non-cubic 3D permeability arrays')
            edge_tup = edges(k_arr)
            std_cart = True
        else:
            edge_tup = (
                np.asarray(edge[0]),
                np.asarray(edge[1]),
                np.asarray(edge[2], dtype=np.float64),
            )
            std_cart = _is_standard_cartesian_edges(cartesian_shape, edge_tup)
        if xyz is None:
            xyz_arr = None if std_cart else np.indices(cartesian_shape).reshape(3, -1).T
        else:
            xyz_arr = np.asarray(xyz)
        tags_flat, count, rule = _extract_components_core(
            k_arr.ravel(), xyz_arr, edge_tup, cartesian_shape if std_cart else None, cfg
        )
        labels_out = tags_flat.reshape(cartesian_shape)
    elif k_arr.ndim == 1:
        if edge is None:
            raise ValueError('1D active-cell permeability requires an explicit edge tuple')
        edge_tup = (
            np.asarray(edge[0]),
            np.asarray(edge[1]),
            np.asarray(edge[2], dtype=np.float64),
        )
        xyz_arr = None if xyz is None else np.asarray(xyz)
        tags_flat, count, rule = _extract_components_core(k_arr, xyz_arr, edge_tup, None, cfg)
        labels_out = tags_flat
    else:
        raise ValueError('k must be a 3D Cartesian array or 1D active-cell vector')

    elapsed = time.perf_counter() - t0
    info = {
        'config_name': cfg_name,
        'phase_mode': cfg['phase_mode'],
        'phase_rule': rule,
        'high_component_count': int(count),
        'high_fraction': float(np.count_nonzero(tags_flat > 0) / max(1, tags_flat.size)),
        'component_label_seconds': elapsed,
    }
    return labels_out, int(count), info


def _aggregate_components_1d(xyz, tags):
    """Component-preserving 2x2x2 geometric aggregation via 1D int64 key packing."""
    n = len(tags)
    bxyz = xyz // 2
    s1 = int(np.max(bxyz[:, 1])) + 1
    s2 = int(np.max(bxyz[:, 2])) + 1
    bcode = (
        bxyz[:, 0].astype(np.int64) * s1 + bxyz[:, 1].astype(np.int64)
    ) * s2 + bxyz[:, 2].astype(np.int64)
    radix = max(1, int(np.max(tags)) + 1)
    max_code = (int(np.max(bcode)) + 1) * radix
    if max_code < np.iinfo(np.int64).max // 2:
        code = bcode * radix + tags
        if max_code <= max(1024, 16 * n):
            cnt = np.bincount(code, minlength=max_code)
            occ = np.flatnonzero(cnt)
            mp = np.empty(max_code, dtype=np.int64)
            mp[occ] = np.arange(len(occ), dtype=np.int64)
            inv = mp[code]
        else:
            occ, inv = np.unique(code, return_inverse=True)
            inv = inv.astype(np.int64, copy=False)
        geom = occ // radix
        c_xyz = np.column_stack((geom // (s1 * s2), (geom // s2) % s1, geom % s2))
        return inv, c_xyz, occ % radix
    keys = np.column_stack((bxyz, tags))
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.astype(np.int64, copy=False), uniq[:, :3], uniq[:, 3]


def _pair_consecutive_within_flow(flow_tags, n):
    """Fallback deterministic pairing within identical flow_tags when edges vanish."""
    if flow_tags is None:
        inv = np.arange(n, dtype=np.int64) // 2
        return inv, int(inv[-1]) + 1, None
    order = np.argsort(flow_tags, kind='stable')
    ft = flow_tags[order]
    same_prev = np.zeros(n, dtype=bool)
    same_prev[1:] = ft[1:] == ft[:-1]
    # Compute position within each contiguous flow_tag run
    starts = np.flatnonzero(~same_prev)
    pos = np.arange(n, dtype=np.int64) - np.repeat(starts, np.diff(np.append(starts, n)))
    new_grp = (~same_prev) | ((pos % 2) == 0)
    grp_id = np.cumsum(new_grp) - 1
    inv = np.empty(n, dtype=np.int64)
    inv[order] = grp_id
    nc = int(grp_id[-1]) + 1
    c_ft = np.empty(nc, dtype=flow_tags.dtype)
    c_ft[grp_id] = ft
    return inv, nc, c_ft


def _coarsen_algebraic(matrix, diag, flow_tags=None, max_group=8):
    """Vectorized symmetric-strength heavy-edge forest aggregation."""
    n = matrix.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(matrix.indptr))
    cols = matrix.indices
    vals = matrix.data
    neg = (cols != rows) & (vals < 0.0)
    if flow_tags is not None:
        neg &= flow_tags[rows] == flow_tags[cols]
    r = rows[neg]
    c = cols[neg]
    w = -vals[neg]
    if len(r) == 0:
        return _pair_consecutive_within_flow(flow_tags, n)

    score = w / np.sqrt(diag[r] * diag[c])
    max_s = np.full(n, -1.0, dtype=np.float64)
    np.maximum.at(max_s, r, score)
    ties = score >= max_s[r] * (1.0 - 1e-15)
    rt = r[ties]
    ct = c[ties]
    best_nbr = np.arange(n, dtype=np.int32)
    best_c = np.full(n, n + 1, dtype=np.int32)
    np.minimum.at(best_c, rt, ct)
    has_nb = best_c <= n
    best_nbr[has_nb] = best_c[has_nb]

    idx = np.arange(n, dtype=np.int32)
    valid_nb = best_nbr != idx
    u_nb, v_nb = idx[valid_nb], best_nbr[valid_nb]
    adj_nb = sparse.coo_matrix(
        (
            np.ones(2 * len(u_nb), dtype=np.uint8),
            (np.concatenate((u_nb, v_nb)), np.concatenate((v_nb, u_nb))),
        ),
        shape=(n, n),
    ).tocsr()
    _, parent = connected_components(adj_nb, directed=False)

    order_p = np.argsort(parent, kind='stable')
    ps = parent[order_p]
    change = np.ones(n, dtype=bool)
    change[1:] = ps[1:] != ps[:-1]
    starts = np.flatnonzero(change)
    pos_in_grp = np.arange(n, dtype=np.int64) - np.repeat(starts, np.diff(np.append(starts, n)))
    split = change | ((pos_in_grp % max_group) == 0)
    agg_sorted = np.cumsum(split) - 1
    inv = np.empty(n, dtype=np.int64)
    inv[order_p] = agg_sorted
    nc = int(agg_sorted[-1]) + 1

    c_ft = None
    if flow_tags is not None:
        c_ft = np.empty(nc, dtype=flow_tags.dtype)
        c_ft[inv] = flow_tags

    # Guarantee steady reduction if isolated singletons dominate a pathological graph
    if nc > int(0.85 * n):
        agg_counts = np.bincount(inv, minlength=nc)
        singletons = np.flatnonzero(agg_counts == 1)
        min_possible = 1 if c_ft is None else int(np.unique(c_ft).size)
        if len(singletons) >= 2 and nc > min_possible:
            s_ft = None if c_ft is None else c_ft[singletons]
            s_inv, s_nc, _ = _pair_consecutive_within_flow(s_ft, len(singletons))
            non_sing = np.flatnonzero(agg_counts > 1)
            remap_agg = np.empty(nc, dtype=np.int64)
            remap_agg[non_sing] = np.arange(len(non_sing), dtype=np.int64)
            remap_agg[singletons] = len(non_sing) + s_inv
            inv = remap_agg[inv]
            nc = len(non_sing) + s_nc
            if flow_tags is not None:
                c_ft = np.empty(nc, dtype=flow_tags.dtype)
                c_ft[inv] = flow_tags

    return inv, nc, c_ft


def _build_prolongation(matrix, diag, omega, di, inv, nc, prolong_filter):
    """Construct nonnegative row-normalized smoothed prolongation P with P @ 1 = 1."""
    n_curr = matrix.shape[0]
    idx_dtype = np.int32 if max(n_curr, nc) < np.iinfo(np.int32).max else np.int64
    indptr_z = np.arange(n_curr + 1, dtype=idx_dtype)
    inv_idx = inv.astype(idx_dtype, copy=False)

    if prolong_filter > 0.0:
        rows_m = np.repeat(np.arange(n_curr, dtype=idx_dtype), np.diff(matrix.indptr))
        keep_off = (matrix.indices != rows_m) & ((-matrix.data) >= prolong_filter * diag[rows_m])
        w_data = np.where(keep_off, -matrix.data, 0.0)
        Wf = sparse.csr_matrix(
            (w_data, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
        )
        Wf.eliminate_zeros()
        df = np.asarray(Wf.sum(axis=1)).ravel()
        alpha = omega / np.maximum(diag, df)
        Z_diag = sparse.csr_matrix((1.0 - alpha * df, inv_idx, indptr_z), shape=(n_curr, nc))
        Z_ones = sparse.csr_matrix(
            (np.ones(n_curr, dtype=np.float64), inv_idx, indptr_z), shape=(n_curr, nc)
        )
        WZ = (Wf @ Z_ones).tocsr()
        WZ.data *= np.repeat(alpha, np.diff(WZ.indptr))
        P = (Z_diag + WZ).tocsr()
    else:
        rows_m = np.repeat(np.arange(n_curr, dtype=idx_dtype), np.diff(matrix.indptr))
        neg_off = (matrix.indices != rows_m) & (matrix.data < 0.0)
        # Check if matrix has positive off-diagonals or grounding row sums;
        # using nonnegative off-diagonal smoother guarantees P >= 0 and P @ 1 = 1!
        has_pos_or_ground = bool(
            np.any((matrix.indices != rows_m) & (matrix.data > 0.0))
            or np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel())) > 1e-11 * float(np.max(diag))
        )
        if not has_pos_or_ground:
            Z = sparse.csr_matrix(
                (np.ones(n_curr, dtype=np.float64), inv_idx, indptr_z), shape=(n_curr, nc)
            )
            AZ = (matrix @ Z).tocsr()
            AZ.data *= np.repeat(di, np.diff(AZ.indptr))
            P = (Z - AZ).tocsr()
        else:
            w_data = np.where(neg_off, -matrix.data, 0.0)
            Wf = sparse.csr_matrix(
                (w_data, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
            )
            Wf.eliminate_zeros()
            df = np.asarray(Wf.sum(axis=1)).ravel()
            alpha = omega / np.maximum(diag, df)
            Z_diag = sparse.csr_matrix((1.0 - alpha * df, inv_idx, indptr_z), shape=(n_curr, nc))
            Z_ones = sparse.csr_matrix(
                (np.ones(n_curr, dtype=np.float64), inv_idx, indptr_z), shape=(n_curr, nc)
            )
            WZ = (Wf @ Z_ones).tocsr()
            WZ.data *= np.repeat(alpha, np.diff(WZ.indptr))
            P = (Z_diag + WZ).tocsr()

    P.eliminate_zeros()
    rsum = np.asarray(P.sum(axis=1)).ravel()
    if not np.all(np.isfinite(rsum)) or np.any(rsum <= 1e-14):
        raise ValueError('invalid prolongation row sum before normalization')
    P.data *= np.repeat(1.0 / rsum, np.diff(P.indptr))
    return P


class _CholeskySolver:
    """Symmetric dense Cholesky coarse-grid solver matching splu.solve(rhs)."""

    def __init__(self, dense_spd):
        self._cf = cho_factor(dense_spd, lower=True, check_finite=False)

    def solve(self, rhs):
        return cho_solve(self._cf, rhs, check_finite=False)


def _factor_coarsest(matrix, grounded, max_direct, verify_spd=False, prefer_splu=False):
    """Enforce direct-solver size guard <= 2048 and factorize coarsest matrix."""
    n_curr = matrix.shape[0]
    hard_limit = min(int(max_direct), 2048)
    if n_curr > hard_limit:
        raise MemoryError(
            f'coarsest matrix size {n_curr} exceeds direct-solver guard {hard_limit}'
        )
    if grounded:
        if n_curr <= 512 and not prefer_splu:
            dense = ((matrix + matrix.T) * 0.5).toarray()
            try:
                return _CholeskySolver(dense)
            except np.linalg.LinAlgError as exc:
                if verify_spd:
                    raise ValueError('coarsest grounded matrix failed Cholesky SPD check') from exc
        return splu(matrix.tocsc())
    if n_curr <= 1:
        return None
    if n_curr <= 512 and not prefer_splu:
        sub = matrix[1:, 1:]
        dense = ((sub + sub.T) * 0.5).toarray()
        try:
            return _CholeskySolver(dense)
        except np.linalg.LinAlgError:
            pass
    return splu(matrix[1:, 1:].tocsc())


def _construct_hierarchy(
    A,
    xyz,
    tags,
    cfg,
    *,
    grounded=False,
    flow_tags=None,
    cartesian_fine=True,
    rank_safe_damping=False,
    verify_coarse_spd=False,
    binary_exact=False,
    deadline=None,
):
    """Build geometric-then-algebraic symmetric Galerkin hierarchy."""
    levels = []
    sizes, nnzs, pnnzs = [], [], []
    galerkin_nnzs, lumped_counts, positive_counts = [], [], []
    sweeps_per_level, tag_counts, modes_used = [], [], []

    matrix = A.tocsr()
    curr_xyz = xyz
    curr_tags = tags
    curr_ft = flow_tags
    alg_mode = curr_xyz is None
    coarse_size = cfg['coarse_size']
    max_direct = cfg['max_direct']
    stall_ratio = cfg['stall_ratio']
    pf = cfg['prolong_filter']
    pf_start = cfg['prolong_filter_start_level']
    max_group = cfg['max_alg_group']

    while True:
        _check_deadline(deadline, 'deadline during multigrid hierarchy construction')
        matrix = matrix.tocsr()
        diag = matrix.diagonal()
        n_curr = matrix.shape[0]
        if (grounded or n_curr > 1) and (not np.all(np.isfinite(diag)) or np.any(diag <= 0.0)):
            raise ValueError('nonpositive or nonfinite multigrid diagonal')

        if n_curr == 1 and not grounded:
            omega = 0.0
            di = np.zeros(1, dtype=np.float64)
            bound = 0.0
        else:
            inv_d = 1.0 / diag
            bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel() * inv_d))
            if not levels and cartesian_fine and not grounded and not rank_safe_damping:
                omega = 2.0 / 3.0
            elif rank_safe_damping:
                omega = min(2.0 / 3.0, 0.9 / bound)
            else:
                omega = min(2.0 / 3.0, 4.0 / (3.0 * bound))
            di = omega * inv_d

        sweeps = cfg['fine_sweeps'] if not levels else cfg['coarse_sweeps']
        level = {'A': matrix, 'di': di, 'sweeps': sweeps, 'omega': omega, 'bound': bound}
        levels.append(level)
        sizes.append(int(n_curr))
        nnzs.append(int(matrix.nnz))
        sweeps_per_level.append(int(sweeps))
        if curr_tags is not None and len(curr_tags) == n_curr:
            tag_counts.append(int(np.unique(curr_tags).size))
        else:
            tag_counts.append(1 if curr_ft is None else int(np.unique(curr_ft).size))

        if n_curr <= coarse_size:
            level['factor'] = _factor_coarsest(
                matrix, grounded, max_direct, verify_spd=verify_coarse_spd, prefer_splu=binary_exact
            )
            break

        step_mode = 'geometric'
        if not alg_mode and curr_xyz is not None and np.max(curr_xyz) > 0:
            inv, next_xyz, next_tags = _aggregate_components_1d(curr_xyz, curr_tags)
            nc = int(np.max(inv)) + 1
            if nc >= n_curr or (nc > coarse_size and nc > int(stall_ratio * n_curr)):
                alg_mode = True
                curr_xyz = None
                curr_tags = None
            else:
                if curr_ft is not None:
                    next_ft = np.empty(nc, dtype=curr_ft.dtype)
                    next_ft[inv] = curr_ft
                    curr_ft = next_ft
                curr_xyz, curr_tags = next_xyz, next_tags
        else:
            alg_mode = True
            curr_xyz = None
            curr_tags = None

        if alg_mode:
            step_mode = 'algebraic'
            inv, nc, curr_ft = _coarsen_algebraic(
                matrix, diag, flow_tags=curr_ft, max_group=max_group
            )
            if nc >= n_curr or len(levels) >= 20:
                level['factor'] = _factor_coarsest(
                    matrix, grounded, max_direct, verify_spd=verify_coarse_spd
                )
                break

        modes_used.append(step_mode)
        cur_lvl_idx = len(levels) - 1
        eff_pf = pf if cur_lvl_idx >= pf_start else 0.0
        # A separate damping gives strict row diagonal dominance of J in P=JZ.
        # It does not alter the relaxation damping used by the V-cycle.
        po = cfg['prolong_omega']
        P = _build_prolongation(matrix, diag, po, po / diag, inv, nc, eff_pf)
        R = P.T.tocsr()
        level['P'], level['R'] = P, R
        pnnzs.append(int(P.nnz))

        coarse = (R @ (matrix @ P)).tocsr()
        coarse = ((coarse + coarse.T) * 0.5).tocsr()
        galerkin_nnzs.append(int(coarse.nnz))

        cutoff = cfg['lump_max_relative'] if len(levels) >= cfg['lump_start_level'] else None
        if grounded:
            matrix, removed, positives = _lump_positive_grounded(coarse, cutoff)
        else:
            matrix, removed, positives = _coarse_filter(coarse, cutoff)
        lumped_counts.append(int(removed))
        positive_counts.append(int(positives))

    meta = {
        'level_sizes': sizes,
        'level_nnz': nnzs,
        'interpolation_nnz': pnnzs,
        'galerkin_nnz_before_lump': galerkin_nnzs,
        'positive_offdiagonals_lumped': lumped_counts,
        'positive_offdiagonals_before_filter': positive_counts,
        'smoothing_steps_per_level_each_side': sweeps_per_level,
        'component_tag_count_per_level': tag_counts,
        'coarsening_modes': modes_used,
        'operator_complexity': float(sum(nnzs) / max(1, nnzs[0])),
    }
    return levels, meta


def _build_anisotropic_line_smoother(shape, edge, diag, omega=2.0 / 3.0, forced_axis=None):
    """Build exact symmetric 1D line-Jacobi LDL^T smoother if one axis dominates >= 4x."""
    if len(shape) != 3 or min(shape) < 2:
        return None
    trans = edge[2]
    medians = []
    sections = list(_sections(shape))
    for sec in sections:
        cnt = sec.stop - sec.start
        if cnt <= 0:
            return None
        step = max(1, cnt // 8192)
        medians.append(float(np.median(trans[sec][::step])))
    dom_ax = int(np.argmax(medians)) if forced_axis is None else int(forced_axis)
    if dom_ax not in (0, 1, 2):
        raise ValueError('forced line axis must be 0, 1 or 2')
    other_max = max(medians[i] for i in range(3) if i != dom_ax)
    if forced_axis is None and medians[dom_ax] < 4.0 * max(other_max, np.finfo(np.float64).tiny):
        return None

    n_line = shape[dom_ax]
    other_shape = tuple(shape[i] for i in range(3) if i != dom_ax)
    n_cols = prod(other_shape)
    if n_cols < 2:
        return None

    t_sec = trans[sections[dom_ax]]
    if dom_ax == 0:
        tz = t_sec.reshape(n_line - 1, n_cols)
        d2 = diag.reshape(n_line, n_cols).copy()
    else:
        lo_shape = list(shape)
        lo_shape[dom_ax] = n_line - 1
        tz = np.moveaxis(t_sec.reshape(lo_shape), dom_ax, 0).reshape(n_line - 1, n_cols)
        d2 = np.moveaxis(diag.reshape(shape), dom_ax, 0).reshape(n_line, n_cols).copy()

    l_sub = np.empty((n_line - 1, n_cols), dtype=np.float64)
    for kk in range(n_line - 1):
        if np.any(d2[kk] <= 0.0):
            return None
        w_col = -tz[kk]
        mult = w_col / d2[kk]
        l_sub[kk] = mult
        d2[kk + 1] -= mult * w_col
    if np.any(d2[-1] <= 0.0) or not np.all(np.isfinite(d2)):
        return None
    inv_d2 = 1.0 / d2

    if dom_ax == 0:
        def apply_line(rhs_vec):
            y = rhs_vec.reshape(n_line, n_cols).copy()
            for kk in range(n_line - 1):
                y[kk + 1] -= l_sub[kk] * y[kk]
            y *= inv_d2
            for kk in range(n_line - 2, -1, -1):
                y[kk] -= l_sub[kk] * y[kk + 1]
            y *= omega
            return y.ravel()
    else:
        perm_shape = (n_line, *other_shape)

        def apply_line(rhs_vec):
            y = np.moveaxis(rhs_vec.reshape(shape), dom_ax, 0).reshape(n_line, n_cols).copy()
            for kk in range(n_line - 1):
                y[kk + 1] -= l_sub[kk] * y[kk]
            y *= inv_d2
            for kk in range(n_line - 2, -1, -1):
                y[kk] -= l_sub[kk] * y[kk + 1]
            y *= omega
            return np.moveaxis(y.reshape(perm_shape), 0, dom_ax).ravel()

    return apply_line


def _make_cycle_callable(levels, *, grounded=False, cycle_repeats=1, fine_A=None):
    """Return fixed symmetric V-cycle matvec callable for Neumann or grounded SPD."""
    if grounded:
        def _vcycle_spd(index, rhs):
            lev = levels[index]
            if 'factor' in lev:
                return lev['factor'].solve(rhs)
            mat, di, sw = lev['A'], lev['di'], lev['sweeps']
            line_fn = lev.get('line_smoother')
            if line_fn is not None:
                x = line_fn(rhs)
                for _ in range(sw - 1):
                    x += line_fn(rhs - mat @ x)
            else:
                x = di * rhs
                for _ in range(sw - 1):
                    x += di * (rhs - mat @ x)
            res = rhs - mat @ x
            x += lev['P'] @ _vcycle_spd(index + 1, lev['R'] @ res)
            if line_fn is not None:
                for _ in range(sw):
                    x += line_fn(rhs - mat @ x)
            else:
                for _ in range(sw):
                    x += di * (rhs - mat @ x)
            return x

        def apply_spd(rhs):
            rhs = np.asarray(rhs, dtype=np.float64).ravel()
            x = _vcycle_spd(0, rhs)
            if cycle_repeats == 2:
                x += _vcycle_spd(0, rhs - fine_A @ x)
            return x

        return apply_spd, _vcycle_spd

    def _vcycle_neumann(index, rhs):
        lev = levels[index]
        rhs = _project(rhs)
        if 'factor' in lev:
            x = np.zeros_like(rhs)
            if lev['factor'] is not None:
                x[1:] = lev['factor'].solve(rhs[1:])
            return _project(x)
        mat, di, sw = lev['A'], lev['di'], lev['sweeps']
        line_fn = lev.get('line_smoother')
        if line_fn is not None:
            x = line_fn(rhs)
            for _ in range(sw - 1):
                x += line_fn(rhs - mat @ x)
        else:
            x = di * rhs
            for _ in range(sw - 1):
                x += di * (rhs - mat @ x)
        res = _project(rhs - mat @ x)
        x += lev['P'] @ _vcycle_neumann(index + 1, lev['R'] @ res)
        if line_fn is not None:
            for _ in range(sw):
                x += line_fn(rhs - mat @ x)
        else:
            for _ in range(sw):
                x += di * (rhs - mat @ x)
        return _project(x)

    def apply_neumann(rhs):
        rhs = np.asarray(rhs, dtype=np.float64).ravel()
        x = _vcycle_neumann(0, rhs)
        if cycle_repeats == 2:
            x += _vcycle_neumann(0, rhs - fine_A @ x)
            return _project(x)
        return x

    return apply_neumann, _vcycle_neumann


class Multigrid(_SolveBase):
    """Cubic Cartesian zero-mean Neumann ML-CPMG state compatible with extension_multigrid_v2."""

    krylov = 'cg'
    coordinate_system = 'physical zero-mean pressure quotient'

    def __init__(self, k, config=None, deadline=None, edge=None):
        started = time.perf_counter()
        _check_deadline(deadline, 'deadline before ML-CPMG setup')
        self.config_name, self.config = _resolve_config(config)
        self.k = np.asarray(k, dtype=np.float64)
        if self.k.ndim != 3 or len(set(self.k.shape)) != 1:
            raise ValueError('expected cubic permeability')
        if np.any(self.k <= 0) or not np.all(np.isfinite(self.k)):
            raise ValueError('permeability must be finite and positive')

        with threadpool_limits(limits=1):
            if edge is None:
                self.edge = edges(self.k)
                std_cart = True
                self.A = _fine_csr(self.k, self.edge)
            else:
                tail, head, trans = (
                    np.asarray(edge[0]),
                    np.asarray(edge[1]),
                    np.asarray(edge[2], dtype=np.float64),
                )
                size = self.k.size
                if (
                    tail.ndim != 1
                    or head.shape != tail.shape
                    or trans.shape != tail.shape
                    or tail.dtype.kind not in 'iu'
                    or head.dtype.kind not in 'iu'
                    or np.any(tail < 0)
                    or np.any(head < 0)
                    or np.any(tail >= size)
                    or np.any(head >= size)
                    or np.any(trans <= 0)
                    or not np.all(np.isfinite(trans))
                ):
                    raise ValueError('invalid serialized edge graph')
                self.edge = (tail, head, trans)
                std_cart = _is_standard_cartesian_edges(self.k.shape, self.edge)
                self.A = (
                    _fine_csr(self.k, self.edge)
                    if std_cart
                    else _assemble_general_edges(size, self.edge)
                )
            assembled = time.perf_counter()
            _check_deadline(deadline, 'deadline after fine assembly')

            xyz = np.indices(self.k.shape).reshape(3, -1).T
            tags, high_count, phase_rule = _extract_components_core(
                self.k.ravel(),
                None if std_cart else xyz,
                self.edge,
                self.k.shape if std_cart else None,
                self.config,
            )
            self.component_labels = tags
            self.high_component_count = int(high_count)
            labelled = time.perf_counter()

            self.cycle_repeats = self.config['cycle_repeats']
            self.solve_calls = 0
            self.last_pressure = None
            self.last_meta = None

            self.levels, hier_meta = _construct_hierarchy(
                self.A,
                xyz,
                tags,
                self.config,
                grounded=False,
                cartesian_fine=std_cart,
                binary_exact=(phase_rule == 'binary_exact_fastpath' and high_count <= 256),
                deadline=deadline,
            )
            if std_cart and edge is not None and self.config['phase_mode'] != 'legacy_k_gt_min':
                line_fn = _build_anisotropic_line_smoother(
                    self.k.shape, self.edge, self.A.diagonal(), omega=self.levels[0]['omega']
                )
                if line_fn is not None:
                    self.levels[0]['line_smoother'] = line_fn

        self._apply_fn, self._vcycle_fn = _make_cycle_callable(
            self.levels, grounded=False, cycle_repeats=self.cycle_repeats, fine_A=self.A
        )
        self.preconditioner = LinearOperator(
            self.A.shape, matvec=self.apply_preconditioner, dtype=np.float64
        )
        self.sparse_operator = LinearOperator(
            self.A.shape, matvec=lambda x: _project(self.A @ _project(x)), dtype=np.float64
        )
        self.graph_operator = LinearOperator(
            self.A.shape,
            matvec=lambda x: _project(action(_project(x), self.edge)),
            dtype=np.float64,
        )

        storage = sum(
            _csr_bytes(lev['A'])
            + lev['di'].nbytes
            + (_csr_bytes(lev['P']) + _csr_bytes(lev['R']) if 'P' in lev else 0)
            for lev in self.levels
        )
        finished = time.perf_counter()
        self.setup_meta = {
            'config_name': self.config_name,
            'config': dict(self.config),
            'phase_mode': self.config['phase_mode'],
            'phase_rule': phase_rule,
            'assembly_seconds': assembled - started,
            'component_label_seconds': labelled - assembled,
            'hierarchy_seconds': finished - labelled,
            'setup_seconds': finished - started,
            **hier_meta,
            'cycle_repeats': self.cycle_repeats,
            'high_component_count': int(high_count),
            'high_fraction': float(np.count_nonzero(tags > 0) / max(1, tags.size)),
            'hierarchy_array_bytes': int(storage),
            'edge_array_bytes': int(sum(arr.nbytes for arr in self.edge)),
            'array_bytes_note': 'persistent CSR/diagonal/edge arrays only; RSS includes LU, temporaries and Python',
            'fine_operator_unchanged': True,
            'preconditioner_is_fixed_linear_symmetric': True,
            'coordinate_system': self.coordinate_system,
            'mechanism': (
                f"ML-CPMG ({self.config['phase_mode']}: {phase_rule}); "
                'hybrid component-geometric + heavy-edge algebraic SA; '
                'symmetric Galerkin + PSD positive-coupling lump; fixed symmetric V-cycle'
            ),
        }

    def _cycle(self, index, rhs):
        return self._vcycle_fn(index, rhs)

    def apply_preconditioner(self, rhs):
        return self._apply_fn(rhs)

    def solve(self, *args, **kwargs):
        p, meta = super().solve(*args, **kwargs)
        meta['mechanism'] = self.setup_meta['mechanism']
        self.last_meta = dict(meta)
        return p, meta


def build(k, config=None, deadline=None, edge=None):
    """Build cubic Cartesian Neumann ML-CPMG state."""
    return Multigrid(k, config=config, deadline=deadline, edge=edge)


def build_from_context(ctx, **parameters):
    """Context builder compatible with extension_worker_v2."""
    return Multigrid(
        ctx['k'],
        config=parameters if parameters else None,
        deadline=ctx.get('deadline'),
        edge=ctx.get('edge'),
    )


def make_precision_setup(config=None):
    """Return a setup(ctx) callback compatible with extension_precision_v2.run."""
    def setup(ctx):
        state = Multigrid(
            ctx['k'],
            config=config,
            deadline=ctx.get('deadline'),
            edge=ctx.get('edge'),
        )
        return {
            'A': state.A,
            'M': state.preconditioner,
            'krylov': 'cg',
            'metadata': state.setup_meta,
        }

    return setup


def solve(k, b, config=None, deadline=None, **kwargs):
    """Compatibility wrapper building ML-CPMG once and solving from zero."""
    if isinstance(config, (int, float)) and not isinstance(config, bool) and deadline is None:
        deadline = float(config)
        config = None
    started = time.perf_counter()
    try:
        state = build(k, config=config, deadline=deadline)
    except TimeoutError:
        elapsed = time.perf_counter() - started
        return np.zeros(np.asarray(k).size, dtype=np.float64), {
            'iterations': 0,
            'setup_seconds': elapsed,
            'solve_seconds': 0.0,
            'total_seconds': elapsed,
            'status': 'deadline_in_setup',
        }
    p, meta = state.solve(b, deadline=deadline, **kwargs)
    meta['total_seconds'] = time.perf_counter() - started
    return p, meta


def build_rectangular(k_tag, edge, d=None, config=None, deadline=None, line_axis=None):
    """Build ML-CPMG on rectangular anisotropic grids (SPE10 static & dynamic).

    - ``d is None``: pure Neumann zero-mean pressure quotient (SPE10 static).
    - ``d is not None``: well-grounded SPD system ``L + diag(d)`` without
      zero-mean projection (SPE10 dynamic IMPES drawdown).
    """
    start = time.perf_counter()
    _check_deadline(deadline, 'SPE10 rectangular solver deadline')
    cfg_name, cfg = _resolve_config(config)
    k_tag = np.asarray(k_tag, dtype=np.float64)
    edge = (
        np.asarray(edge[0]),
        np.asarray(edge[1]),
        np.asarray(edge[2], dtype=np.float64),
    )
    _check_rectangular_inputs(k_tag, edge)

    grounded = d is not None
    if grounded:
        d_arr = np.asarray(d, dtype=np.float64)
        if (
            d_arr.shape != (k_tag.size,)
            or not np.all(np.isfinite(d_arr))
            or np.any(d_arr < 0.0)
            or not np.any(d_arr > 0.0)
        ):
            raise ValueError('d must be finite, nonnegative, shape (N,), with positive grounding')

    with threadpool_limits(1):
        _check_deadline(deadline, 'SPE10 rectangular solver deadline')
        A = assemble_cartesian_fast(k_tag.shape, edge)
        if grounded:
            A.setdiag(A.diagonal() + d_arr)
        assembled = time.perf_counter()

        _check_deadline(deadline, 'SPE10 rectangular solver deadline')
        tags, high_count, phase_rule = _extract_components_core(
            k_tag.ravel(), None, edge, k_tag.shape, cfg
        )
        labelled = time.perf_counter()

        xyz = np.indices(k_tag.shape).reshape(3, -1).T
        levels, hier_meta = _construct_hierarchy(
            A,
            xyz,
            tags,
            cfg,
            grounded=grounded,
            cartesian_fine=(not grounded),
            deadline=deadline,
        )
        line_used = False
        if True:  # identical geometry-triggered line smoothing for every label rule
            line_fn = _build_anisotropic_line_smoother(
                k_tag.shape, edge, A.diagonal(), omega=levels[0]['omega'], forced_axis=line_axis
            )
            if line_fn is not None:
                levels[0]['line_smoother'] = line_fn
                line_used = True
        hier_meta['anisotropic_line_smoother'] = line_used
        hier_meta['forced_line_axis'] = line_axis
        apply_fn, _ = _make_cycle_callable(
            levels, grounded=grounded, cycle_repeats=cfg['cycle_repeats'], fine_A=A
        )
        M = LinearOperator(A.shape, matvec=apply_fn, dtype=np.float64)

    end = time.perf_counter()
    info = {
        'name': f'ml_cpmg_{cfg_name}',
        'config_name': cfg_name,
        'config': dict(cfg),
        'phase_mode': cfg['phase_mode'],
        'phase_rule': phase_rule,
        'grounded': bool(grounded),
        'assembly': 'assemble_cartesian_fast',
        'assembly_seconds': assembled - start,
        'component_label_seconds': labelled - assembled,
        'hierarchy_seconds': end - labelled,
        'setup_seconds': end - start,
        'high_component_count': int(high_count),
        'high_fraction': float(np.count_nonzero(tags > 0) / max(1, tags.size)),
        **hier_meta,
    }
    return {'A': A, 'M': M, 'krylov': 'cg', 'metadata': info}


def build_active_graph(
    graph,
    k,
    d=None,
    config=None,
    coarse_size=512,
    max_direct=512,
    deadline=None,
):
    """Build ML-CPMG on an unstructured / corner-point ActiveGraph (e.g. Brugge).

    Returns ``(A, M, description)`` matching ``results/brugge_validation_v2``.
    """
    started = time.perf_counter()
    _check_deadline(deadline, 'active-graph solver deadline')
    for attr in ('xyz', 'tail', 'head', 'trans', 'shape'):
        if not hasattr(graph, attr):
            raise TypeError(f'graph is missing required ActiveGraph attribute {attr!r}')

    if coarse_size < 1 or max_direct < coarse_size or max_direct > 2048:
        raise ValueError('require 1 <= coarse_size <= max_direct <= 2048')

    cfg_name, cfg = _resolve_config(config)
    cfg = dict(cfg)
    if coarse_size in PARAMETER_DOMAIN['coarse_size']:
        cfg['coarse_size'] = int(coarse_size)
    else:
        cfg['coarse_size'] = int(coarse_size)
    cfg['max_direct'] = int(max_direct)

    xyz = np.asarray(graph.xyz, dtype=np.int64)
    n = len(xyz)
    k_arr = np.asarray(k, dtype=np.float64)
    if k_arr.shape != (n,) or not np.all(np.isfinite(k_arr)) or np.any(k_arr <= 0.0):
        raise ValueError('one finite positive permeability per active cell required')

    tail = np.asarray(graph.tail, dtype=np.int64)
    head = np.asarray(graph.head, dtype=np.int64)
    trans = np.asarray(graph.trans, dtype=np.float64)
    if (
        tail.ndim != 1
        or head.shape != tail.shape
        or trans.shape != tail.shape
        or np.any(tail < 0)
        or np.any(head < 0)
        or np.any(tail >= n)
        or np.any(head >= n)
        or np.any(tail == head)
        or not np.all(np.isfinite(trans))
        or np.any(trans <= 0.0)
    ):
        raise ValueError('invalid active-graph edge arrays')

    if hasattr(graph, 'flow_components'):
        flow_count, flow_comp = graph.flow_components()
    else:
        adj = sparse.coo_matrix(
            (np.ones(2 * len(tail), dtype=np.uint8), (np.concatenate((tail, head)), np.concatenate((head, tail)))),
            shape=(n, n),
        ).tocsr()
        flow_count, flow_comp = connected_components(adj, directed=False)
    flow_count = int(flow_count)
    flow_comp = np.asarray(flow_comp, dtype=np.int64)

    grounded = d is not None
    if grounded:
        d_arr = np.asarray(d, dtype=np.float64)
        if d_arr.shape != (n,) or not np.all(np.isfinite(d_arr)) or np.any(d_arr < 0.0):
            raise ValueError('well diagonal must be finite, nonnegative, and one per active cell')
        grounding = np.bincount(flow_comp, weights=d_arr, minlength=flow_count)
        if np.any(grounding <= 0.0) or not np.all(np.isfinite(grounding)):
            raise ValueError('every flow component must be pressure grounded')
    else:
        d_arr = None
        grounding = None
        if flow_count != 1:
            raise ValueError('ungrounded pure-Neumann active graph must be connected')

    with threadpool_limits(1):
        A = _assemble_general_edges(n, (tail, head, trans), well_diagonal=d_arr)
        assembled = time.perf_counter()

        # Deduplicate parallel faces via upper triangle of assembled Laplacian couplings
        rows_a = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
        upper = (A.indices > rows_a) & (A.data < 0.0)
        eff_edge = (rows_a[upper], A.indices[upper].astype(np.int64), -A.data[upper])

        high_tags, high_count, phase_rule = _extract_components_core(
            k_arr, xyz, eff_edge, None, cfg
        )
        if flow_count > 1:
            packed = flow_comp * (int(high_count) + 1) + high_tags
            _, comb_tags = np.unique(packed, return_inverse=True)
            comb_tags = comb_tags.astype(np.int64, copy=False)
        else:
            comb_tags = high_tags
        labelled = time.perf_counter()

        levels, hier_meta = _construct_hierarchy(
            A,
            xyz.copy(),
            comb_tags,
            cfg,
            grounded=grounded,
            flow_tags=flow_comp if flow_count > 1 else None,
            cartesian_fine=False,
            rank_safe_damping=True,
            verify_coarse_spd=grounded,
            deadline=deadline,
        )
        apply_fn, _ = _make_cycle_callable(
            levels, grounded=grounded, cycle_repeats=cfg['cycle_repeats'], fine_A=A
        )
        M = LinearOperator(A.shape, matvec=apply_fn, dtype=np.float64)

    finished = time.perf_counter()
    description = {
        'method': 'ml_cpmg_active_graph',
        'config_name': cfg_name,
        'config': dict(cfg),
        'phase_mode': cfg['phase_mode'],
        'phase_rule': phase_rule,
        'grounded': bool(grounded),
        'flow_components': flow_count,
        'grounding_by_component': grounding.tolist() if grounding is not None else None,
        'high_component_count': int(high_count),
        'high_fraction': float(np.count_nonzero(high_tags > 0) / max(1, n)),
        'initial_combined_tags': int(np.unique(comb_tags).size),
        'assembly_seconds': assembled - started,
        'component_label_seconds': labelled - assembled,
        'hierarchy_seconds': finished - labelled,
        'setup_seconds': finished - started,
        'levels': [
            {
                'size': int(s),
                'nnz': int(nz),
                'tags': int(tc),
            }
            for s, nz, tc in zip(
                hier_meta['level_sizes'],
                hier_meta['level_nnz'],
                hier_meta['component_tag_count_per_level'],
            )
        ],
        **hier_meta,
    }
    return A, M, description
