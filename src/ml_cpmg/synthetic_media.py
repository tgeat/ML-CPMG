"""Deterministic grid-independent 3D continuous multi-level permeability media.

This module defines four continuous heterogeneous permeability families on the
unit cube ``[0, 1]^3`` sampled at cell centres ``(i + 0.5) / n``:

1. ``multilevel_nested_lobes``:
   Continuous smooth log-background + undulating intermediate fan lobes
   (``~contrast**0.45``) containing closely spaced but separated ultra-high
   cores/streaks (``~contrast**1.0``, separated by ~1.8-2.2 grid cells at
   ``n=64``), plus independent mid-high lobe bodies (``~contrast**0.72``).
2. ``multilevel_braided_channels``:
   Groups of closely spaced sinuous continuous channel ribbons across three
   distinct permeability tiers (``~contrast**0.38``, ``~contrast**0.68``,
   ``~contrast**1.0``) separated by thin (1.5-2 cell at ``n=64``) continuous
   low-permeability floodplain barriers (``k ~ 1``).
3. ``multilevel_layered_barriers``:
   Continuous Tarbert/Upper-Ness-inspired undulating multi-tier shoreface sand
   layers (``~contrast**0.4``, ``~contrast**0.7``, ``~contrast**1.0``)
   separated by thin continuous low-permeability shale barriers (``k ~ 1``),
   with continuous fine-scale heterogeneity throughout both sands and shales.
4. ``multilevel_anisotropic_streaks``:
   Continuous multi-scale log-Gaussian-spectrum background with embedded
   multi-tier close elongated triaxial ellipsoidal streak clusters spanning
   three orders of magnitude (``~contrast**0.38``, ``~contrast**0.69``,
   ``~contrast**1.0``).

Every family is a fixed analytical function of physical coordinates ``(x, y, z)``
and ``seed`` via ``numpy.random.Generator(PCG64(...))``.  Changing ``n`` never
alters the random draws and never normalizes by sampled array extrema.  Because
background and internal heterogeneity are strictly continuous everywhere,
``k > min(k)`` marks ``(n**3 - 1) / n**3 > 0.999`` of cells as high phase,
collapsing existing ``CP-MG`` (``extension_multigrid_v2`` / ``paper_ablation_v1``)
to ``high_component_count == 1``.

No experiment, file I/O, or global RNG mutation occurs on import.
"""

from __future__ import annotations

import os
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_env_var] = "1"

import numpy as np
from scipy.ndimage import binary_dilation, label
from threadpoolctl import threadpool_limits


SCHEMA = "multilevel_continuous_media_v1"
FAMILIES = (
    "multilevel_nested_lobes",
    "multilevel_braided_channels",
    "multilevel_layered_barriers",
    "multilevel_anisotropic_streaks",
)
DEFAULT_CONTRAST = 1.0e6
DEFAULT_SEED = 20261102
TUNING_SEEDS = (20261101,)
HELDOUT_SEEDS = (20261102, 20261103)
CONTRASTS = (1.0e4, 1.0e6, 1.0e8)
SUGGESTED_GRIDS = (32, 48, 64, 96)

FAMILY_TIERS = {
    "multilevel_nested_lobes": {1: 0.45, 2: 0.72, 3: 1.00},
    "multilevel_braided_channels": {1: 0.38, 2: 0.68, 3: 1.00},
    "multilevel_layered_barriers": {1: 0.40, 2: 0.70, 3: 1.00},
    "multilevel_anisotropic_streaks": {1: 0.38, 2: 0.69, 3: 1.00},
}

_FAMILY_SALTS = {
    "multilevel_nested_lobes": 101,
    "multilevel_braided_channels": 202,
    "multilevel_layered_barriers": 303,
    "multilevel_anisotropic_streaks": 404,
}

# Every mode has an odd sum (m0 + m1 + m2).  Consequently all cosines equal +1
# at x = shift and -1 at x = shift + (0.5, 0.5, 0.5) modulo periodicity, so
# 0.5 * (1 + sum_m w_m cos(...)) spans [0, 1] analytically without grid-dependent
# min/max rescaling.
_BG_ODD_MODES = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 1), (2, 1, 0), (0, 2, 1),
    (1, 0, 2), (1, -1, 1), (2, 2, 1),
)

_MULTISCALE_ODD_MODES = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 1), (1, -1, 1), (1, 1, -1),
    (2, 1, 0), (0, 2, 1), (1, 0, 2), (2, -1, 0), (0, 2, -1), (-1, 0, 2),
    (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, -2, 1),
    (3, 0, 0), (0, 3, 0), (0, 0, 3),
    (3, 1, 1), (1, 3, -1), (1, -1, 3), (3, 2, 0), (0, 3, 2),
)


def _validate_arguments(n, family, contrast, seed, return_meta=False):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    n = int(n)
    if n < 8:
        raise ValueError("n must be at least 8; formal tests should use n >= 32")
    if family not in FAMILIES:
        raise ValueError("unknown continuous multi-level medium family: " + str(family))
    if isinstance(contrast, (bool, np.bool_)):
        raise TypeError("contrast must be a real number > 1.0")
    contrast = float(contrast)
    if not np.isfinite(contrast) or contrast <= 1.0:
        raise ValueError("contrast must be finite and strictly greater than 1.0")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not isinstance(return_meta, (bool, np.bool_)):
        raise TypeError("return_meta must be a boolean")
    return n, contrast, seed, bool(return_meta)


def _rng_pair(seed: int, family: str) -> tuple[np.random.Generator, np.random.Generator]:
    """Return independent deterministic (geometry_rng, texture_rng) generators."""
    seq = np.random.SeedSequence([int(seed), _FAMILY_SALTS[family], 20261102])
    g_seq, t_seq = seq.spawn(2)
    return np.random.Generator(np.random.PCG64(g_seq)), np.random.Generator(np.random.PCG64(t_seq))


def _coordinates(n: int):
    c = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    return c[:, None, None], c[None, :, None], c[None, None, :]


def _smooth_fourier_unit(coords, rng: np.random.Generator, modes=_BG_ODD_MODES, decay: float = 1.0):
    """Grid-independent smooth field in [0, 1] evaluated via exact 2D+1D trig contraction."""
    shift = rng.uniform(0.0, 1.0, size=3)
    raw_weights = rng.uniform(0.5, 1.0, size=len(modes))
    modes_arr = np.asarray(modes, dtype=np.float64)
    freq_norm = np.linalg.norm(modes_arr, axis=1)
    weights = raw_weights / (freq_norm ** decay)
    weights /= np.sum(weights)

    x1 = coords[0][:, 0, 0] - shift[0]
    y1 = coords[1][0, :, 0] - shift[1]
    z1 = coords[2][0, 0, :] - shift[2]

    ax = 2.0 * np.pi * modes_arr[:, 0:1] * x1[None, :]
    ay = 2.0 * np.pi * modes_arr[:, 1:2] * y1[None, :]
    az = 2.0 * np.pi * modes_arr[:, 2:3] * z1[None, :]

    axy = ax[:, :, None] + ay[:, None, :]
    c_xy = np.cos(axy)
    s_xy = np.sin(axy)
    wc_z = weights[:, None] * np.cos(az)
    ws_z = weights[:, None] * np.sin(az)

    field = np.einsum("mij,mk->ijk", c_xy, wc_z) - np.einsum("mij,mk->ijk", s_xy, ws_z)
    return np.clip(0.5 * (1.0 + field), 0.0, 1.0)


def _tube_profile(coords, x0: float, x1: float, y_fn, z_fn, ry: float, rz: float,
                  rx_cap: float = 0.028, power: float = 1.9):
    """Smooth super-Gaussian ribbon/tube profile in [0, 1] along x in [x0, x1]."""
    x, y, z = coords
    yc = y_fn(x)
    zc = z_fn(x)
    dx = np.maximum(0.0, x0 - x) + np.maximum(0.0, x - x1)
    d2_xy = ((y - yc) / ry) ** 2 + (dx / rx_cap) ** 2
    d2_xz = ((z - zc) / rz) ** 2
    return np.exp(-((d2_xy + d2_xz + 1.0e-12) ** power))


def _ellipsoid_profile(coords, center: np.ndarray, radii: np.ndarray,
                       rot: np.ndarray, power: float = 1.95):
    """Smooth rotated triaxial super-ellipsoid profile in [0, 1]."""
    dx = coords[0] - center[0]
    dy = coords[1] - center[1]
    dz = coords[2] - center[2]
    d2 = np.zeros((dx.shape[0], dy.shape[1], dz.shape[2]), dtype=np.float64)
    for axis in range(3):
        inv_r = 1.0 / radii[axis]
        xy_part = (rot[0, axis] * dx + rot[1, axis] * dy) * inv_r
        z_part = (rot[2, axis] * dz) * inv_r
        d2 += (xy_part + z_part) ** 2
    return np.exp(-((d2 + 1.0e-12) ** power))


def _tilt_rotation(ay: float, az: float, ax: float = 0.0) -> np.ndarray:
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    cx, sx = np.cos(ax), np.sin(ax)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return rz @ ry @ rx


def _finalize_meta(n: int, family: str, contrast: float, seed: int,
                   peak_id_grid: np.ndarray, basin_id_grid: np.ndarray,
                   peak_specs: list[dict], close_pairs: list[dict], k: np.ndarray | None = None):
    # Ensure every reference peak core mask is a single contiguous 6-connected component
    # (pruning any 1-cell staircase crumbs at pointed ellipsoid/tube tips on coarse grids).
    for spec in peak_specs:
        pid = int(spec["peak_id"])
        raw_mask = peak_id_grid == pid
        lbl, num_lbl = label(raw_mask)
        if num_lbl > 1:
            counts = np.bincount(lbl.ravel())
            counts[0] = 0
            keep_lbl = int(np.argmax(counts))
            peak_id_grid[raw_mask & (lbl != keep_lbl)] = 0

    tiers_map = dict(FAMILY_TIERS[family])
    tier_lookup = np.zeros(len(peak_specs) + 1, dtype=np.int8)
    for spec in peak_specs:
        tier_lookup[spec["peak_id"]] = spec["tier"]
    tier_grid = tier_lookup[peak_id_grid]

    core_masks = {}
    peaks_info = []
    core_counts = np.bincount(peak_id_grid.ravel(), minlength=len(peak_specs) + 1)
    basin_counts = np.bincount(basin_id_grid.ravel(), minlength=len(peak_specs) + 1)

    for spec in peak_specs:
        pid = int(spec["peak_id"])
        mask = peak_id_grid == pid
        core_masks[pid] = mask
        peaks_info.append({
            "peak_id": pid,
            "name": spec["name"],
            "tier": int(spec["tier"]),
            "alpha": float(spec["alpha"]),
            "nominal_permeability": float(contrast ** spec["alpha"]),
            "role": spec["role"],
            "parent_peak_id": spec["parent_peak_id"],
            "core_cells": int(core_counts[pid]),
            "basin_cells": int(basin_counts[pid]),
        })

    meta = {
        "schema": SCHEMA,
        "family": family,
        "n": int(n),
        "contrast": float(contrast),
        "seed": int(seed),
        "tiers": tiers_map,
        "num_peaks": len(peak_specs),
        "peak_id_grid": peak_id_grid,
        "tier_grid": tier_grid,
        "basin_id_grid": basin_id_grid,
        "core_masks": core_masks,
        "peaks": peaks_info,
        "close_pairs": close_pairs,
    }
    if k is not None:
        k_min = float(np.min(k))
        k_max = float(np.max(k))
        meta.update({
            "sample_minimum": k_min,
            "sample_maximum": k_max,
            "sampled_contrast": k_max / k_min,
            "sampled_high_fraction": float(np.mean(k > k_min)),
        })
    return meta


def _build_nested_lobes(n: int, contrast: float, seed: int, need_k: bool, need_meta: bool):
    rng_geom, rng_tex = _rng_pair(seed, "multilevel_nested_lobes")
    coords = _coordinates(n)
    bg = 0.026 * _smooth_fourier_unit(coords, rng_tex, _BG_ODD_MODES, decay=0.8) if need_k else None
    het = _smooth_fourier_unit(coords, rng_tex, _MULTISCALE_ODD_MODES[:16], decay=1.0) if need_k else None
    mod = (0.96 + 0.055 * het) if need_k else None

    ph = rng_geom.uniform(-np.pi, np.pi, size=10)
    jit = rng_geom.uniform(-0.003, 0.003, size=6)
    sgn = float(rng_geom.choice([-1.0, 1.0]))

    beta = 36.0
    acc = np.zeros((n, n, n), dtype=np.float64) if need_k else None

    def z_warp(xv):
        return sgn * 0.31 * (xv - 0.5) + 0.011 * np.sin(2.0 * np.pi * xv + ph[0])

    lobe_z0s = (0.345, 0.500, 0.655)
    lobe_y0s = (0.488 + jit[0], 0.512 + jit[1], 0.492 + jit[2])
    mid_z0s = (0.175, 0.243, 0.757, 0.825)
    c_dz = 0.066  # ~4.22 cells center-to-center; core rz=0.0185 -> ~2.0 cell saddle at n=64

    peak_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    basin_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    peak_specs = []
    close_pairs = []

    # Peak IDs:
    # 1..3   : intermediate host lobes (tier 1, alpha=0.45)
    # 4..7   : independent mid-high bodies (tier 2, alpha=0.72)
    # 8..13  : nested ultra-high cores inside host lobes (tier 3, alpha=1.00)
    for l_idx, (z0, y0) in enumerate(zip(lobe_z0s, lobe_y0s)):
        lobe_pid = l_idx + 1
        yc_l = lambda xv, y0=y0, l_idx=l_idx: y0 + 0.016 * np.sin(2.0 * np.pi * xv + ph[1 + l_idx])
        zc_l = lambda xv, z0=z0: z0 - z_warp(xv)
        phi_l = _tube_profile(coords, 0.02, 0.98, yc_l, zc_l, ry=0.175, rz=0.060, rx_cap=0.028, power=1.9)
        if need_k:
            acc += np.expm1(beta * (0.45 * mod * phi_l))
        if need_meta:
            peak_id_grid[phi_l >= 0.44] = lobe_pid
            basin_id_grid[phi_l >= 0.10] = lobe_pid
            peak_specs.append({
                "peak_id": lobe_pid,
                "name": f"host_lobe_{lobe_pid}",
                "tier": 1,
                "alpha": 0.45,
                "role": "host_lobe",
                "parent_peak_id": None,
            })

    if need_meta:
        close_pairs.append({"pair": (1, 2), "relation": "adjacent_host_lobes", "tiers": (1, 1)})
        close_pairs.append({"pair": (2, 3), "relation": "adjacent_host_lobes", "tiers": (1, 1)})

    best_mid = np.full((n, n, n), 0.44, dtype=np.float64) if need_meta else None
    best_mid_basin = np.full((n, n, n), 0.10, dtype=np.float64) if need_meta else None
    for m_idx, z0 in enumerate(mid_z0s):
        mid_pid = 4 + m_idx
        yc_m = lambda xv, m_idx=m_idx: (
            (0.500 + jit[3 + (m_idx % 2)]) + 0.018 * np.cos(2.0 * np.pi * xv + ph[6 + m_idx])
        )
        zc_m = lambda xv, z0=z0: z0 - z_warp(xv)
        phi_m = _tube_profile(coords, 0.03, 0.97, yc_m, zc_m, ry=0.130, rz=0.0185, rx_cap=0.028, power=1.9)
        if need_k:
            acc += np.expm1(beta * (0.72 * mod * phi_m))
        if need_meta:
            m_core = phi_m > best_mid
            peak_id_grid[m_core] = mid_pid
            best_mid[m_core] = phi_m[m_core]
            m_bas = phi_m > best_mid_basin
            basin_id_grid[m_bas] = mid_pid
            best_mid_basin[m_bas] = phi_m[m_bas]
            peak_specs.append({
                "peak_id": mid_pid,
                "name": f"mid_body_{m_idx + 1}",
                "tier": 2,
                "alpha": 0.72,
                "role": "mid_body",
                "parent_peak_id": None,
            })

    if need_meta:
        close_pairs.append({"pair": (4, 5), "relation": "adjacent_mid_bodies", "tiers": (2, 2)})
        close_pairs.append({"pair": (5, 1), "relation": "mid_body_vs_host_lobe", "tiers": (2, 1)})
        close_pairs.append({"pair": (3, 6), "relation": "mid_body_vs_host_lobe", "tiers": (1, 2)})
        close_pairs.append({"pair": (6, 7), "relation": "adjacent_mid_bodies", "tiers": (2, 2)})

    # Nested ultra-high cores overwrite interior core/skirt regions of their host lobe
    # while carving a clean skirt buffer out of the host lobe core mask so host_lobe
    # core_masks strictly represent the intermediate lobe plateau.
    for l_idx, (z0, y0) in enumerate(zip(lobe_z0s, lobe_y0s)):
        lobe_pid = l_idx + 1
        yc_l = lambda xv, y0=y0, l_idx=l_idx: y0 + 0.016 * np.sin(2.0 * np.pi * xv + ph[1 + l_idx])
        zc_l = lambda xv, z0=z0: z0 - z_warp(xv)
        core_pids_in_lobe = []
        for sub_idx, c_sign in enumerate((-0.5, +0.5)):
            core_pid = 8 + 2 * l_idx + sub_idx
            core_pids_in_lobe.append(core_pid)
            zc_c = lambda xv, zc_l=zc_l, c_sign=c_sign: zc_l(xv) + c_sign * c_dz
            yc_c = lambda xv, yc_l=yc_l, c_sign=c_sign, l_idx=l_idx: (
                yc_l(xv) + c_sign * 0.022 * np.cos(2.0 * np.pi * xv + ph[4 + l_idx])
            )
            phi_c = _tube_profile(coords, 0.06, 0.94, yc_c, zc_c, ry=0.105, rz=0.0185, rx_cap=0.025, power=1.95)
            if need_k:
                acc += np.expm1(beta * (1.00 * mod * phi_c))
            if need_meta:
                core_mask = phi_c >= 0.44
                skirt_ring = ((phi_c >= 0.05) | binary_dilation(core_mask)) & (peak_id_grid == lobe_pid)
                peak_id_grid[skirt_ring] = 0
                peak_id_grid[core_mask] = core_pid
                basin_id_grid[phi_c >= 0.16] = core_pid
                peak_specs.append({
                    "peak_id": core_pid,
                    "name": f"nested_core_{l_idx + 1}{'a' if sub_idx == 0 else 'b'}",
                    "tier": 3,
                    "alpha": 1.00,
                    "role": "nested_core",
                    "parent_peak_id": lobe_pid,
                })
                close_pairs.append({
                    "pair": (core_pid, lobe_pid),
                    "relation": "nested_core_vs_host_lobe",
                    "tiers": (3, 1),
                })
        if need_meta:
            # Prune any tiny isolated 1-cell rim crumbs left by dilation carving on host_lobe
            lobe_mask = peak_id_grid == lobe_pid
            lbl, num_lbl = label(lobe_mask)
            if num_lbl > 1:
                counts = np.bincount(lbl.ravel())
                counts[0] = 0
                keep_lbl = int(np.argmax(counts))
                peak_id_grid[lobe_mask & (lbl != keep_lbl)] = 0
            close_pairs.append({
                "pair": (core_pids_in_lobe[0], core_pids_in_lobe[1]),
                "relation": "nested_cores_in_same_lobe",
                "tiers": (3, 3),
            })

    k = None
    if need_k:
        s_struct = np.log1p(acc) / beta
        s = bg + (1.0 - bg) * s_struct
        k = np.ascontiguousarray(np.exp(np.log(contrast) * s), dtype=np.float64)

    meta = None
    if need_meta:
        meta = _finalize_meta(n, "multilevel_nested_lobes", contrast, seed,
                              peak_id_grid, basin_id_grid, peak_specs, close_pairs, k=k)
    return k, meta


def _build_braided_channels(n: int, contrast: float, seed: int, need_k: bool, need_meta: bool):
    rng_geom, rng_tex = _rng_pair(seed, "multilevel_braided_channels")
    coords = _coordinates(n)
    bg = 0.026 * _smooth_fourier_unit(coords, rng_tex, _BG_ODD_MODES, decay=0.8) if need_k else None
    het = _smooth_fourier_unit(coords, rng_tex, _MULTISCALE_ODD_MODES[:16], decay=1.0) if need_k else None
    mod = (0.96 + 0.055 * het) if need_k else None

    ph = rng_geom.uniform(-np.pi, np.pi, size=14)
    jit = rng_geom.uniform(-0.003, 0.003, size=6)
    sign_base = float(rng_geom.choice([-1.0, 1.0]))

    beta = 36.0
    acc = np.zeros((n, n, n), dtype=np.float64) if need_k else None

    sep = 0.068  # ~4.35 cells c-to-c; ry=0.019 -> ~1.7-cell k~1 barrier at n=64
    tier_lookup_by_alpha = {0.38: 1, 0.68: 2, 1.00: 3}
    belts = [
        (0.310, (0.38, 0.68, 1.00, 0.38), +sign_base, ph[0], ph[1], jit[0]),
        (0.386, (0.68, 1.00, 1.00, 0.68), -sign_base, ph[2], ph[3], jit[1]),
        (0.462, (1.00, 1.00, 0.68, 0.38), +sign_base, ph[4], ph[5], jit[2]),
        (0.538, (0.38, 0.68, 1.00, 1.00), -sign_base, ph[6], ph[7], jit[3]),
        (0.614, (0.68, 1.00, 1.00, 0.68), +sign_base, ph[8], ph[9], jit[4]),
        (0.690, (0.38, 1.00, 0.68, 0.38), -sign_base, ph[10], ph[11], jit[5]),
    ]

    peak_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    basin_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    best_core = np.full((n, n, n), 0.44, dtype=np.float64) if need_meta else None
    best_basin = np.full((n, n, n), 0.10, dtype=np.float64) if need_meta else None
    peak_specs = []
    close_pairs = []

    cid = 0
    for b_idx, (z0, alphas, sgn, p1, p2, j0) in enumerate(belts, start=1):
        m_cnt = len(alphas)
        belt_pids = []
        for m_idx, alpha in enumerate(alphas):
            cid += 1
            belt_pids.append((cid, tier_lookup_by_alpha[alpha]))
            offset_y = (m_idx - 0.5 * (m_cnt - 1)) * sep
            phi = _tube_profile(
                coords, 0.02, 0.98,
                lambda x, sgn=sgn, p1=p1, j0=j0, offset_y=offset_y: (
                    0.500 + j0 + sgn * 0.305 * (x - 0.5)
                    + 0.015 * np.sin(2.0 * np.pi * 1.2 * x + p1) + offset_y
                ),
                lambda x, z0=z0, p2=p2: z0 + 0.007 * np.cos(2.0 * np.pi * 1.2 * x + p2),
                ry=0.019, rz=0.021, rx_cap=0.025, power=1.95,
            )
            if need_k:
                acc += np.expm1(beta * (alpha * mod * phi))
            if need_meta:
                m_core = phi > best_core
                peak_id_grid[m_core] = cid
                best_core[m_core] = phi[m_core]
                m_bas = phi > best_basin
                basin_id_grid[m_bas] = cid
                best_basin[m_bas] = phi[m_bas]
                peak_specs.append({
                    "peak_id": cid,
                    "name": f"belt{b_idx}_ch{m_idx + 1}",
                    "tier": tier_lookup_by_alpha[alpha],
                    "alpha": alpha,
                    "role": "braided_channel",
                    "parent_peak_id": None,
                })
        if need_meta:
            for idx_pair in range(len(belt_pids) - 1):
                pa, ta = belt_pids[idx_pair]
                pb, tb = belt_pids[idx_pair + 1]
                close_pairs.append({
                    "pair": (pa, pb),
                    "relation": "adjacent_braided_channels",
                    "tiers": (ta, tb),
                })

    k = None
    if need_k:
        s_struct = np.log1p(acc) / beta
        s = bg + (1.0 - bg) * s_struct
        k = np.ascontiguousarray(np.exp(np.log(contrast) * s), dtype=np.float64)

    meta = None
    if need_meta:
        meta = _finalize_meta(n, "multilevel_braided_channels", contrast, seed,
                              peak_id_grid, basin_id_grid, peak_specs, close_pairs, k=k)
    return k, meta


def _build_layered_barriers(n: int, contrast: float, seed: int, need_k: bool, need_meta: bool):
    rng_geom, rng_tex = _rng_pair(seed, "multilevel_layered_barriers")
    coords = _coordinates(n)
    x, y, z = coords
    bg = 0.025 * _smooth_fourier_unit(coords, rng_tex, _BG_ODD_MODES, decay=0.8) if need_k else None
    het = _smooth_fourier_unit(coords, rng_tex, _MULTISCALE_ODD_MODES, decay=0.95) if need_k else None
    layer_mod = (0.92 + 0.095 * het) if need_k else None

    ph = rng_geom.uniform(-np.pi, np.pi, size=6)
    sign_x = float(rng_geom.choice([-1.0, 1.0]))
    undul = (
        sign_x * 0.31 * (x - 0.5)
        + 0.035 * (y - 0.5)
        + 0.020 * np.sin(2.0 * np.pi * x + ph[0]) * np.cos(2.0 * np.pi * y + ph[1])
        + 0.011 * np.sin(4.0 * np.pi * x + ph[2])
        + 0.011 * np.cos(2.0 * np.pi * y + ph[3])
    )
    zeta = z + undul

    alphas = (0.40, 0.70, 1.00, 0.40, 1.00, 0.70, 1.00, 0.40, 0.70, 1.00, 0.40)
    tier_lookup_by_alpha = {0.40: 1, 0.70: 2, 1.00: 3}
    z_centers = np.linspace(0.16, 0.84, len(alphas))

    beta = 36.0
    acc = np.zeros((n, n, n), dtype=np.float64) if need_k else None
    peak_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    basin_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    best_core = np.full((n, n, n), 0.44, dtype=np.float64) if need_meta else None
    best_basin = np.full((n, n, n), 0.10, dtype=np.float64) if need_meta else None
    peak_specs = []
    close_pairs = []

    rz_half = 0.020
    for idx, (zc, alpha) in enumerate(zip(z_centers, alphas), start=1):
        d2 = ((zeta - zc) / rz_half) ** 2
        phi = np.exp(-((d2 + 1.0e-12) ** 1.9))
        if need_k:
            acc += np.expm1(beta * (alpha * layer_mod * phi))
        if need_meta:
            m_core = phi > best_core
            peak_id_grid[m_core] = idx
            best_core[m_core] = phi[m_core]
            m_bas = phi > best_basin
            basin_id_grid[m_bas] = idx
            best_basin[m_bas] = phi[m_bas]
            peak_specs.append({
                "peak_id": idx,
                "name": f"sand_layer_{idx:02d}",
                "tier": tier_lookup_by_alpha[alpha],
                "alpha": float(alpha),
                "role": "sand_layer",
                "parent_peak_id": None,
            })
            if idx > 1:
                prev_alpha = alphas[idx - 2]
                close_pairs.append({
                    "pair": (idx - 1, idx),
                    "relation": "adjacent_sand_layers_across_shale",
                    "tiers": (tier_lookup_by_alpha[prev_alpha], tier_lookup_by_alpha[alpha]),
                })

    k = None
    if need_k:
        s_struct = np.log1p(acc) / beta
        s = bg + (1.0 - bg) * s_struct
        k = np.ascontiguousarray(np.exp(np.log(contrast) * s), dtype=np.float64)

    meta = None
    if need_meta:
        meta = _finalize_meta(n, "multilevel_layered_barriers", contrast, seed,
                              peak_id_grid, basin_id_grid, peak_specs, close_pairs, k=k)
    return k, meta


def _build_anisotropic_streaks(n: int, contrast: float, seed: int, need_k: bool, need_meta: bool):
    rng_geom, rng_tex = _rng_pair(seed, "multilevel_anisotropic_streaks")
    coords = _coordinates(n)
    if need_k:
        bg_ms = _smooth_fourier_unit(coords, rng_tex, _MULTISCALE_ODD_MODES, decay=0.75)
        bg = 0.026 * bg_ms + 0.055 * (bg_ms ** 3)
        het = _smooth_fourier_unit(coords, rng_tex, _MULTISCALE_ODD_MODES[:16], decay=1.0)
        mod = 0.95 + 0.065 * het
    else:
        bg = None
        mod = None

    jit = rng_geom.uniform(-0.003, 0.003, size=(3, 3))
    sgn = float(rng_geom.choice([-1.0, 1.0]))
    sep = 0.068
    tier_lookup_by_alpha = {0.38: 1, 0.69: 2, 1.00: 3}

    clusters = [
        (np.array([0.50, 0.350, 0.50]) + jit[0], _tilt_rotation(+0.31 * sgn, 0.03 * sgn, 0.02), [
            (0.38, np.array([0.0, 0.0, -2.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (1.00, np.array([0.0, 0.0, -1.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (0.69, np.array([0.0, 0.0,  0.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (1.00, np.array([0.0, 0.0, +1.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (0.38, np.array([0.0, 0.0, +2.0 * sep]), np.array([0.56, 0.070, 0.019])),
        ]),
        (np.array([0.50, 0.500, 0.50]) + jit[1], _tilt_rotation(-0.31 * sgn, -0.03 * sgn, -0.02), [
            (0.69, np.array([0.0, 0.0, -2.5 * sep]), np.array([0.56, 0.074, 0.019])),
            (1.00, np.array([0.0, 0.0, -1.5 * sep]), np.array([0.56, 0.074, 0.019])),
            (0.38, np.array([0.0, 0.0, -0.5 * sep]), np.array([0.56, 0.074, 0.019])),
            (1.00, np.array([0.0, 0.0, +0.5 * sep]), np.array([0.56, 0.074, 0.019])),
            (0.69, np.array([0.0, 0.0, +1.5 * sep]), np.array([0.56, 0.074, 0.019])),
            (0.38, np.array([0.0, 0.0, +2.5 * sep]), np.array([0.56, 0.074, 0.019])),
        ]),
        (np.array([0.50, 0.650, 0.50]) + jit[2], _tilt_rotation(+0.31 * sgn, 0.03 * sgn, 0.02), [
            (0.69, np.array([0.0, 0.0, -2.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (1.00, np.array([0.0, 0.0, -1.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (1.00, np.array([0.0, 0.0,  0.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (0.69, np.array([0.0, 0.0, +1.0 * sep]), np.array([0.56, 0.070, 0.019])),
            (0.38, np.array([0.0, 0.0, +2.0 * sep]), np.array([0.56, 0.070, 0.019])),
        ]),
    ]

    beta = 36.0
    acc = np.zeros((n, n, n), dtype=np.float64) if need_k else None
    peak_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    basin_id_grid = np.zeros((n, n, n), dtype=np.int32) if need_meta else None
    best_core = np.full((n, n, n), 0.44, dtype=np.float64) if need_meta else None
    best_basin = np.full((n, n, n), 0.12, dtype=np.float64) if need_meta else None
    peak_specs = []
    close_pairs = []

    sid = 0
    for c_idx, (c_center, rot, members) in enumerate(clusters, start=1):
        cluster_pids = []
        for s_idx, (alpha, loc_off, radii) in enumerate(members, start=1):
            sid += 1
            cluster_pids.append((sid, tier_lookup_by_alpha[alpha]))
            w_center = c_center + rot @ loc_off
            phi = _ellipsoid_profile(coords, w_center, radii, rot, power=1.95)
            if need_k:
                acc += np.expm1(beta * (alpha * mod * phi))
            if need_meta:
                m_core = phi > best_core
                peak_id_grid[m_core] = sid
                best_core[m_core] = phi[m_core]
                m_bas = phi > best_basin
                basin_id_grid[m_bas] = sid
                best_basin[m_bas] = phi[m_bas]
                peak_specs.append({
                    "peak_id": sid,
                    "name": f"cluster{c_idx}_streak{s_idx}",
                    "tier": tier_lookup_by_alpha[alpha],
                    "alpha": float(alpha),
                    "role": "anisotropic_streak",
                    "parent_peak_id": None,
                })
        if need_meta:
            for idx_pair in range(len(cluster_pids) - 1):
                pa, ta = cluster_pids[idx_pair]
                pb, tb = cluster_pids[idx_pair + 1]
                close_pairs.append({
                    "pair": (pa, pb),
                    "relation": "adjacent_cluster_streaks",
                    "tiers": (ta, tb),
                })

    k = None
    if need_k:
        s_struct = np.log1p(acc) / beta
        s = bg + (1.0 - bg) * s_struct
        k = np.ascontiguousarray(np.exp(np.log(contrast) * s), dtype=np.float64)

    meta = None
    if need_meta:
        meta = _finalize_meta(n, "multilevel_anisotropic_streaks", contrast, seed,
                              peak_id_grid, basin_id_grid, peak_specs, close_pairs, k=k)
    return k, meta


_BUILDERS = {
    "multilevel_nested_lobes": _build_nested_lobes,
    "multilevel_braided_channels": _build_braided_channels,
    "multilevel_layered_barriers": _build_layered_barriers,
    "multilevel_anisotropic_streaks": _build_anisotropic_streaks,
}


def geometry(n: int, family: str, contrast: float = DEFAULT_CONTRAST,
             seed: int = DEFAULT_SEED, return_meta: bool = False):
    """Return positive cubic float64 permeability ``k[n, n, n]`` on ``[0, 1]^3``.

    Parameters
    ----------
    n : int
        Cubic grid size along each axis (``n >= 8``).
    family : str
        One of :data:`FAMILIES`.
    contrast : float, default 1e6
        Nominal continuous physical contrast parameter (``contrast > 1.0``).
    seed : int, default 20261102
        Nonnegative integer realization seed (grid-independent).
    return_meta : bool, default False
        If ``True``, return ``(k, meta)`` including planted peak core masks,
        tier labels, basin assignments, and close-pair relationships.
    """
    n, contrast, seed, return_meta = _validate_arguments(n, family, contrast, seed, return_meta)
    with threadpool_limits(limits=1):
        k, meta = _BUILDERS[family](n, contrast, seed, need_k=True, need_meta=return_meta)
    if k.shape != (n, n, n) or k.dtype != np.float64 or not np.all(np.isfinite(k)) or np.any(k < 1.0):
        raise ValueError("invalid permeability array generated by multilevel_continuous_media")
    if return_meta:
        return k, meta
    return k


def reference_peak_masks(n: int, family: str, contrast: float = DEFAULT_CONTRAST,
                         seed: int = DEFAULT_SEED) -> dict:
    """Return planted peak core masks, tier labels, and close-pair metadata."""
    n, contrast, seed, _ = _validate_arguments(n, family, contrast, seed, False)
    with threadpool_limits(limits=1):
        _, meta = _BUILDERS[family](n, contrast, seed, need_k=False, need_meta=True)
    return meta


def describe(n: int, family: str, contrast: float = DEFAULT_CONTRAST,
             seed: int = DEFAULT_SEED, k: np.ndarray | None = None) -> dict:
    """Return small JSON-serializable provenance record (matching paper_media_v1 style)."""
    n, contrast, seed, _ = _validate_arguments(n, family, contrast, seed, False)
    record = {
        "schema": SCHEMA,
        "family": family,
        "n": n,
        "seed": seed,
        "physical_domain": "unit_cube",
        "sampling": "cell_centres_of_fixed_physical_function",
        "contrast_parameter": contrast,
        "contrast_parameter_kind": "continuous_multilevel_nominal_max_min",
        "nominal_tier_exponents": {str(t): v for t, v in FAMILY_TIERS[family].items()},
        "generator": "numpy.random.Generator(PCG64(SeedSequence))",
        "sample_extrema_normalized": False,
        "source_operator_boundary_flux_changed": False,
        "seed_role": (
            "tuning" if seed in TUNING_SEEDS else
            "heldout" if seed in HELDOUT_SEEDS else "undeclared"
        ),
        "component_label_caveat": (
            "Existing CP-MG uses k > min(k): >99.9% of cells belong to a single "
            "connected high mask (high_component_count == 1), merging all multi-level "
            "channels, nested cores, and barrier-separated layers."
        ),
    }
    if k is not None:
        arr = np.asarray(k, dtype=np.float64)
        if arr.shape != (n, n, n):
            raise ValueError("statistics array shape does not match descriptor")
        minimum, maximum = float(np.min(arr)), float(np.max(arr))
        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum <= 0.0:
            raise ValueError("statistics require finite positive permeability")
        record.update(
            sample_minimum=minimum,
            sample_maximum=maximum,
            sampled_contrast=maximum / minimum,
            sampled_high_fraction=float(np.mean(arr > minimum)),
        )
    return record


def manifest() -> dict:
    """Return JSON-serializable manifest of continuous multi-level media suite."""
    return {
        "schema": SCHEMA,
        "families": list(FAMILIES),
        "family_tiers": {fam: {str(t): v for t, v in tiers.items()} for fam, tiers in FAMILY_TIERS.items()},
        "default_contrast": DEFAULT_CONTRAST,
        "default_seed": DEFAULT_SEED,
        "tuning_seeds": list(TUNING_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "contrasts": list(CONTRASTS),
        "suggested_grids": list(SUGGESTED_GRIDS),
        "grid_independent_physical_realizations": True,
        "required_common_source": "darcy_core.sources(n)",
        "required_common_edges": "darcy_core.edges(k)",
    }


def unplanted_random(n: int = 64, contrast: float = 1e6, seed: int = 20261115) -> np.ndarray:
    """Anisotropically correlated Gaussian-filtered random field without planted channels."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal((n, n, n)), sigma=(4, 7, 2), mode="reflect")
    field = (field - field.min()) / (field.max() - field.min())
    return np.exp(np.log(contrast) * field)


def smooth_gaussian(n: int = 64, contrast: float = 1e6) -> np.ndarray:
    """Smooth unimodal Gaussian permeability hill (anti-onion-shell control)."""
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    field = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / 0.045)
    return np.exp(np.log(contrast) * field)
