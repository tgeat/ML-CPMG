"""Multi-Level Persistence Component-Separating Aggregation Multigrid (ML-CPMG).

This module generalizes component-preserving smoothed aggregation multigrid
(CP-MG) from binary permeability media (``k > min(k)``) to arbitrary continuous
multi-level heterogeneous, thin-layer anisotropic, and faulted corner-point
Darcy media while preserving exact binary-medium equivalence and strict SPD /
quotient-SPD admissibility for PCG.

Single-thread & frozen-parameter guarantees
-------------------------------------------
- Strictly single-threaded execution (``OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS = 1``,
  ``numba.set_num_threads(1)``, zero ``parallel=True``, zero ``prange``, zero
  ``fastmath``).
- All default algorithmic parameters in ``_BASE_CONTROLS`` are frozen
  (``max_basins=256, stall_ratio=0.33, prolong_omega=0.45, tau_barrier=8.0,
  theta_pers=16.0, bg_alpha=0.22, coarse_size=512, max_direct=2048``).
- All internal comparison baselines (``ML-CPMG``, ``Min-threshold``,
  ``Single-median``, ``Algebraic-only``) share the exact same fine assembly,
  1D dominant-axis column tridiagonal ``LDL^T`` line smoother (zero-transpose
  for Cartesian ``dom_ax in {0, 1, 2}`` and permutation-tridiagonal for
  ``ActiveGraph``), anisotropic geometric semi-coarsening (``ratio >= 4.0``),
  intra-tag box connectivity splitting, repaired M-matrix coarse Jacobi damping,
  and zero-allocation fused ``@njit`` CSR V-cycle kernels.
"""
from __future__ import annotations

import os
import sys
for _env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_env_name] = "1"
if "numba" not in sys.modules:
    os.environ["NUMBA_NUM_THREADS"] = "1"

from math import prod
import time

import numba as nb
nb.set_num_threads(1)

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import label
from scipy.sparse import _sparsetools
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg, splu
from threadpoolctl import threadpool_limits

from .grid import ActiveGraph, _csr_bytes, _fine_csr, _project, action, edges, metrics


# ============================================================================
# 0. Configuration Registry, Validation & Solve Base
# ============================================================================

PHASE_MODES = (
    "ml_cpmg",
    "cliff_only",
    "naive_multibin",
    "single_median",
    "legacy_k_gt_min",
    "algebraic_only",
)

PARAMETER_DOMAIN = {
    "phase_mode": PHASE_MODES,
    "fine_sweeps": (1, 2),
    "coarse_sweeps": (1, 2),
    "coarse_size": (64, 96, 192, 512),
    "max_direct": (512, 1024, 2048),
    "lump_max_relative": "None (exact Galerkin) or a float in [0, 1]",
    "lump_start_level": (1, 2, 3),
    "cycle_repeats": (1, 2),
    "theta_pers": "float > 1.0 (persistence saddle prominence ratio)",
    "tau_barrier": "float > 1.0 (geometry-normalized barrier ratio)",
    "tau_cliff": "float > 1.0 (local cliff ratio for cliff_only mode)",
    "bg_alpha": "float in (0.0, 0.8) (log-range background fraction)",
    "min_peak_cells": "positive int (base minimum basin core volume)",
    "pedestal_ratio": "float >= 1.0 (nested lobe-to-core pedestal volume ratio)",
    "max_basins": "None or positive int (prominence-ranked basin cap)",
    "prolong_omega": "float in (0, 0.5) (rank-preserving interpolation damping)",
    "stall_ratio": "float in (0.15, 0.99] (geometric-to-algebraic switch ratio)",
    "prolong_filter": "float in [0.0, 0.5] (coarse prolongation strength cutoff)",
    "prolong_filter_start_level": (0, 1, 2, 3),
    "max_alg_group": (4, 6, 8, 12, 16),
}

_BASE_CONTROLS = {
    "fine_sweeps": 1,
    "coarse_sweeps": 1,
    "coarse_size": 512,
    "max_direct": 2048,
    "lump_max_relative": 1.0,
    "lump_start_level": 1,
    "cycle_repeats": 1,
    "theta_pers": 16.0,
    "tau_barrier": 8.0,
    "tau_cliff": 1.0e30,
    "bg_alpha": 0.22,
    "min_peak_cells": 6,
    "pedestal_ratio": 2.0,
    "max_basins": 256,
    "prolong_omega": 0.45,
    "stall_ratio": 0.33,
    "prolong_filter": 0.04,
    "prolong_filter_start_level": 1,
    "max_alg_group": 8,
}

CONFIGS = {
    "ml_cpmg": dict(
        _BASE_CONTROLS,
        phase_mode="ml_cpmg",
    ),
    "cliff_only": dict(
        _BASE_CONTROLS,
        phase_mode="cliff_only",
        tau_cliff=24.0,
    ),
    "naive_multibin": dict(
        _BASE_CONTROLS,
        phase_mode="naive_multibin",
    ),
    "single_median": dict(
        _BASE_CONTROLS,
        phase_mode="single_median",
    ),
    "legacy_k_gt_min": dict(
        _BASE_CONTROLS,
        phase_mode="legacy_k_gt_min",
        prolong_filter=0.0,
        stall_ratio=0.95,
    ),
    "algebraic_only": dict(
        _BASE_CONTROLS,
        phase_mode="algebraic_only",
    ),
}

DEFAULT_CONFIG = "ml_cpmg"
DEFAULT_SMOOTHER_SCHEME = "line_plus_jacobi"
_JIT_WARMED = False


def _resolve_config(config):
    """Validate and resolve a named preset or parameter override dictionary."""
    if config is None:
        name = DEFAULT_CONFIG
        cfg = dict(CONFIGS[name])
    elif isinstance(config, str):
        if config not in CONFIGS:
            raise ValueError(f"unknown multigrid configuration: {config!r}")
        name = config
        cfg = dict(CONFIGS[name])
    elif isinstance(config, dict):
        supplied = dict(config)
        unknown = set(supplied) - set(PARAMETER_DOMAIN)
        if unknown:
            raise ValueError("unknown multigrid parameter(s): " + ", ".join(sorted(unknown)))
        if not supplied:
            name = DEFAULT_CONFIG
            cfg = dict(CONFIGS[name])
        else:
            base_mode = supplied.get("phase_mode", DEFAULT_CONFIG)
            base_preset = CONFIGS.get(base_mode, CONFIGS[DEFAULT_CONFIG])
            cfg = dict(base_preset)
            cfg.update(supplied)
            matched = [k for k, v in CONFIGS.items() if v == cfg]
            name = matched[0] if len(matched) == 1 else "custom"
    else:
        raise TypeError("config must be None, a preset name, or a parameter dict")

    if cfg["phase_mode"] not in PHASE_MODES:
        raise ValueError(f"unsupported phase_mode: {cfg['phase_mode']!r}")
    for key in (
        "fine_sweeps",
        "coarse_sweeps",
        "coarse_size",
        "max_direct",
        "lump_start_level",
        "cycle_repeats",
        "prolong_filter_start_level",
        "max_alg_group",
    ):
        val = cfg[key]
        if isinstance(val, (bool, np.bool_)) or val not in PARAMETER_DOMAIN[key]:
            raise ValueError(f"unsupported {key}: {val!r}")
        cfg[key] = int(val)
    if cfg["max_direct"] < cfg["coarse_size"] or cfg["max_direct"] > 2048:
        raise ValueError("max_direct must satisfy coarse_size <= max_direct <= 2048")

    cutoff = cfg["lump_max_relative"]
    if cutoff is not None:
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or not (0.0 <= cutoff <= 1.0):
            raise ValueError("lump_max_relative must be None or lie in [0, 1]")
        cfg["lump_max_relative"] = cutoff

    for key, lo in (("theta_pers", 1.0), ("tau_barrier", 1.0), ("tau_cliff", 1.0)):
        val = float(cfg[key])
        if not np.isfinite(val) or val <= lo:
            raise ValueError(f"{key} must be finite and > {lo}")
        cfg[key] = val

    bg_alpha = float(cfg["bg_alpha"])
    if not np.isfinite(bg_alpha) or not (0.0 < bg_alpha < 0.8):
        raise ValueError("bg_alpha must lie in (0.0, 0.8)")
    cfg["bg_alpha"] = bg_alpha

    ped = float(cfg["pedestal_ratio"])
    if not np.isfinite(ped) or ped < 1.0:
        raise ValueError("pedestal_ratio must be >= 1.0")
    cfg["pedestal_ratio"] = ped

    pw = float(cfg["prolong_omega"])
    if not np.isfinite(pw) or not (0.0 < pw < 0.5):
        raise ValueError("prolong_omega must lie in (0, 0.5)")
    cfg["prolong_omega"] = pw

    sr = float(cfg["stall_ratio"])
    if not np.isfinite(sr) or not (0.15 < sr <= 0.99):
        raise ValueError("stall_ratio must lie in (0.15, 0.99]")
    cfg["stall_ratio"] = sr

    pf = float(cfg["prolong_filter"])
    if not np.isfinite(pf) or not (0.0 <= pf <= 0.5):
        raise ValueError("prolong_filter must lie in [0.0, 0.5]")
    cfg["prolong_filter"] = pf

    mpc = cfg["min_peak_cells"]
    if isinstance(mpc, (bool, np.bool_)) or not isinstance(mpc, (int, np.integer)) or int(mpc) < 1:
        raise ValueError("min_peak_cells must be a positive integer")
    cfg["min_peak_cells"] = int(mpc)

    mb = cfg["max_basins"]
    if mb is not None:
        if isinstance(mb, (bool, np.bool_)) or not isinstance(mb, (int, np.integer)) or int(mb) < 1:
            raise ValueError("max_basins must be None or a positive integer")
        cfg["max_basins"] = int(mb)

    return name, cfg


def _check_deadline(deadline, message="multigrid deadline exceeded"):
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
    tail, head, _ = edge
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


def is_binary_field(k_flat):
    """Fast O(N) exact check for uniform or two-value permeability fields."""
    if k_flat.size == 0:
        return True
    kmin = float(np.min(k_flat))
    kmax = float(np.max(k_flat))
    if kmax <= kmin:
        return True
    sample = k_flat[: min(512, k_flat.size)]
    if not np.all((sample == kmin) | (sample == kmax)):
        return False
    return bool(np.all((k_flat == kmin) | (k_flat == kmax)))


class _Stopped(Exception):
    def __init__(self, x, reason):
        self.x = x.copy()
        self.reason = reason


class _SolveBase:
    """Minimal V-cycle and PCG wrapper inherited by Multigrid."""

    def _cycle(self, index, rhs):
        level = self.levels[index]
        rhs = _project(rhs)
        if "factor" in level:
            x = np.zeros_like(rhs)
            if level["factor"] is not None:
                x[1:] = level["factor"].solve(rhs[1:])
            return _project(x)
        matrix, diagonal = level["A"], level["di"]
        x = diagonal * rhs
        for _ in range(self.smoothing_steps - 1):
            x += diagonal * (rhs - matrix @ x)
        residual = _project(rhs - matrix @ x)
        x += level["P"] @ self._cycle(index + 1, level["R"] @ residual)
        for _ in range(self.smoothing_steps):
            x += diagonal * (rhs - matrix @ x)
        return _project(x)

    def apply_preconditioner(self, rhs):
        """Apply the same fixed, symmetric V-cycle to any residual."""
        return self._cycle(0, np.asarray(rhs, dtype=np.float64).ravel())

    def solve(
        self,
        b,
        deadline=None,
        target_tol=1e-9,
        x0=None,
        precisiontail=True,
        maxiter=1200,
    ):
        """Solve with optional explicit restart pressure and strict graph gates."""
        began = time.perf_counter()
        b = np.asarray(b, dtype=np.float64).ravel()
        if len(b) != self.A.shape[0] or not np.all(np.isfinite(b)):
            raise ValueError("expected finite source for every cell")
        if abs(float(np.sum(b))) > 1e-12 * max(1.0, float(np.sum(abs(b)))):
            raise ValueError("Neumann source must have zero integral")
        if not (0 < target_tol < 1):
            raise ValueError("target_tol must lie between zero and one")
        b = _project(b)
        p = (
            np.zeros_like(b)
            if x0 is None
            else _project(np.asarray(x0, dtype=np.float64).ravel().copy())
        )
        if p.shape != b.shape or not np.all(np.isfinite(p)):
            raise ValueError("x0 must contain one finite pressure per cell")
        iterations, passes, tail_iterations = 0, 0, 0
        reason, info = "krylov_finished", 0
        values = metrics(p, b, self.edge)
        accepted = values["rho"] <= target_tol and values["eta"] <= target_tol
        main_iteration_limit = int(maxiter)
        with threadpool_limits(limits=1):
            for pass_index in range(4 if precisiontail else 1):
                if accepted:
                    break
                if deadline is not None and time.perf_counter() >= deadline - 0.002:
                    reason = "deadline"
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
                        raise _Stopped(correction, "deadline")
                    if local_iterations == 1 or local_iterations % 8 == 0:
                        check = metrics(_project(p + correction), b, self.edge)
                        if check["rho"] <= target_tol * 0.1 and check["eta"] <= target_tol * 0.1:
                            raise _Stopped(correction, "true_residual_converged")

                try:
                    correction, info = cg(
                        operator,
                        residual,
                        M=self.preconditioner,
                        rtol=min(1e-12, target_tol * 0.01),
                        atol=0.0,
                        maxiter=main_iteration_limit,
                        callback=callback,
                    )
                except _Stopped as stopped:
                    correction = stopped.x
                    reason = stopped.reason
                p = _project(p + correction)
                values = metrics(p, b, self.edge)
                accepted = values["rho"] <= target_tol and values["eta"] <= target_tol
                if reason == "deadline":
                    break
        p = np.asarray(_project(p), dtype=np.float64).ravel()
        values = metrics(p, b, self.edge)
        accepted = values["rho"] <= target_tol and values["eta"] <= target_tol
        self.last_pressure = p.copy()
        ended = time.perf_counter()
        before_deadline = deadline is None or ended < deadline
        meta = {
            **self.setup_meta,
            "solve_seconds": ended - began,
            "total_seconds": self.setup_meta["setup_seconds"] + ended - began,
            "iterations": iterations,
            "precision_tail_iterations": tail_iterations,
            "residual_replacement_passes": passes,
            "status": (
                "completed"
                if accepted and before_deadline
                else (
                    "deadline"
                    if reason == "deadline" or not before_deadline
                    else "not_converged"
                )
            ),
            "conservation_compliant": bool(accepted),
            "completed_before_deadline": bool(before_deadline),
            "checkpoint_completed_perf_counter": ended if accepted and before_deadline else None,
            "stop_reason": reason,
            "info": int(info),
            "target_tol": float(target_tol),
            "setup_reused": self.solve_calls > 0,
            "setup_accounting": "total_seconds includes the full original construction time on every call; solve_seconds is this call only",
            "mechanism": "linear-time phase/geometric aggregation; CSR-scaled smoothed prolongation; symmetric Galerkin V-cycle; CSR PCG with strict graph residual tail",
            **values,
        }
        self.last_meta = dict(meta)
        self.solve_calls += 1
        return p, meta


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
            f"coarsest matrix size {n_curr} exceeds direct-solver guard {hard_limit}"
        )
    if grounded:
        if n_curr <= 512 and not prefer_splu:
            dense = ((matrix + matrix.T) * 0.5).toarray()
            try:
                return _CholeskySolver(dense)
            except np.linalg.LinAlgError as exc:
                if verify_spd:
                    raise ValueError("coarsest grounded matrix failed Cholesky SPD check") from exc
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


def _pair_consecutive_within_flow(flow_tags, n):
    """Fallback deterministic pairing within identical flow_tags when edges vanish."""
    if flow_tags is None:
        inv = np.arange(n, dtype=np.int64) // 2
        return inv, int(inv[-1]) + 1, None
    order = np.argsort(flow_tags, kind="stable")
    ft = flow_tags[order]
    same_prev = np.zeros(n, dtype=bool)
    same_prev[1:] = ft[1:] == ft[:-1]
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


# ============================================================================
# 1. Numba Kernels: Input Validation, Graph Components, and Fine CSR Assembly
# ============================================================================


@nb.njit(cache=True)
def _validate_and_assemble_cartesian_nb(
    nx: int,
    ny: int,
    nz: int,
    k_flat: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
    trans: np.ndarray,
    has_d: bool,
    d_arr: np.ndarray,
):
    size = nx * ny * nz
    for i in range(size):
        kv = k_flat[i]
        if not np.isfinite(kv) or kv <= 0.0:
            return (
                1,
                np.empty(1, dtype=np.int32),
                np.empty(1, dtype=np.int32),
                np.empty(1, dtype=np.float64),
                np.empty(1, dtype=np.float64),
                0.0,
            )

    c0 = (nx - 1) * ny * nz
    c1 = nx * (ny - 1) * nz
    c2 = nx * ny * (nz - 1)
    expected = c0 + c1 + c2
    if tail.size != expected or head.size != expected or trans.size != expected:
        return (
            3,
            np.empty(1, dtype=np.int32),
            np.empty(1, dtype=np.int32),
            np.empty(1, dtype=np.float64),
            np.empty(1, dtype=np.float64),
            0.0,
        )

    nyz = ny * nz
    pos = 0
    for ix in range(nx - 1):
        base_x = ix * nyz
        for iy in range(ny):
            base_xy = base_x + iy * nz
            for iz in range(nz):
                u = base_xy + iz
                tv = trans[pos]
                if not np.isfinite(tv) or tv <= 0.0:
                    return (
                        2,
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.float64),
                        np.empty(1, dtype=np.float64),
                        0.0,
                    )
                if tail[pos] != u or head[pos] != u + nyz:
                    return (
                        3,
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.float64),
                        np.empty(1, dtype=np.float64),
                        0.0,
                    )
                pos += 1

    for ix in range(nx):
        base_x = ix * nyz
        for iy in range(ny - 1):
            base_xy = base_x + iy * nz
            for iz in range(nz):
                u = base_xy + iz
                tv = trans[pos]
                if not np.isfinite(tv) or tv <= 0.0:
                    return (
                        2,
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.float64),
                        np.empty(1, dtype=np.float64),
                        0.0,
                    )
                if tail[pos] != u or head[pos] != u + nz:
                    return (
                        3,
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.float64),
                        np.empty(1, dtype=np.float64),
                        0.0,
                    )
                pos += 1

    for ix in range(nx):
        base_x = ix * nyz
        for iy in range(ny):
            base_xy = base_x + iy * nz
            for iz in range(nz - 1):
                u = base_xy + iz
                tv = trans[pos]
                if not np.isfinite(tv) or tv <= 0.0:
                    return (
                        2,
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.float64),
                        np.empty(1, dtype=np.float64),
                        0.0,
                    )
                if tail[pos] != u or head[pos] != u + 1:
                    return (
                        3,
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.int32),
                        np.empty(1, dtype=np.float64),
                        np.empty(1, dtype=np.float64),
                        0.0,
                    )
                pos += 1

    nnz = size + 2 * expected
    indptr = np.empty(size + 1, dtype=np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    diag = np.empty(size, dtype=np.float64)

    off1 = c0
    off2 = c0 + c1
    ny1_nz = (ny - 1) * nz
    nz1 = nz - 1

    indptr[0] = 0
    wpos = 0
    bound = 0.0

    for ix in range(nx):
        base_x = ix * nyz
        for iy in range(ny):
            base_xy = base_x + iy * nz
            for iz in range(nz):
                u = base_xy + iz
                t_mx = trans[(ix - 1) * nyz + iy * nz + iz] if ix > 0 else 0.0
                t_my = trans[off1 + ix * ny1_nz + (iy - 1) * nz + iz] if iy > 0 else 0.0
                t_mz = trans[off2 + (ix * ny + iy) * nz1 + (iz - 1)] if iz > 0 else 0.0
                t_pz = trans[off2 + (ix * ny + iy) * nz1 + iz] if iz + 1 < nz else 0.0
                t_py = trans[off1 + ix * ny1_nz + iy * nz + iz] if iy + 1 < ny else 0.0
                t_px = trans[ix * nyz + iy * nz + iz] if ix + 1 < nx else 0.0

                d_tail = (t_px + t_py) + t_pz
                d_head = (t_mx + t_my) + t_mz
                d_u = d_tail + d_head
                if has_d:
                    d_u += d_arr[u]
                diag[u] = d_u

                row_start = wpos
                if ix > 0:
                    indices[wpos] = u - nyz
                    data[wpos] = -t_mx
                    wpos += 1
                if iy > 0:
                    indices[wpos] = u - nz
                    data[wpos] = -t_my
                    wpos += 1
                if iz > 0:
                    indices[wpos] = u - 1
                    data[wpos] = -t_mz
                    wpos += 1

                indices[wpos] = u
                data[wpos] = d_u
                wpos += 1

                if iz + 1 < nz:
                    indices[wpos] = u + 1
                    data[wpos] = -t_pz
                    wpos += 1
                if iy + 1 < ny:
                    indices[wpos] = u + nz
                    data[wpos] = -t_py
                    wpos += 1
                if ix + 1 < nx:
                    indices[wpos] = u + nyz
                    data[wpos] = -t_px
                    wpos += 1

                indptr[u + 1] = wpos

                abs_sum = 0.0
                for p in range(row_start, wpos):
                    abs_sum += abs(data[p])
                if d_u > 0.0:
                    r = abs_sum * (1.0 / d_u)
                    if r > bound:
                        bound = r

    return 0, indptr, indices, data, diag, bound


@nb.njit(cache=True)
def _uf_find(parent: np.ndarray, x: int) -> int:
    r = x
    while parent[r] != r:
        r = parent[r]
    curr = x
    while curr != r:
        nxt = parent[curr]
        parent[curr] = r
        curr = nxt
    return r


@nb.njit(cache=True)
def _undirected_components_nb(n: int, u_arr: np.ndarray, v_arr: np.ndarray):
    parent = np.arange(n, dtype=np.int32)
    m_edges = u_arr.size
    for e in range(m_edges):
        ru = _uf_find(parent, int(u_arr[e]))
        rv = _uf_find(parent, int(v_arr[e]))
        if ru != rv:
            if ru < rv:
                parent[rv] = ru
            else:
                parent[ru] = rv

    comp = np.empty(n, dtype=np.int64)
    root_label = np.full(n, -1, dtype=np.int32)
    num = 0
    for i in range(n):
        r = _uf_find(parent, i)
        lbl = root_label[r]
        if lbl < 0:
            lbl = num
            root_label[r] = lbl
            num += 1
        comp[i] = lbl
    return num, comp


@nb.njit(cache=True)
def _stable_sort_col_val_nb(cols: np.ndarray, vals: np.ndarray, s: int, e: int) -> None:
    cnt = e - s
    if cnt <= 32:
        for u in range(s + 1, e):
            kc = cols[u]
            kv = vals[u]
            v = u - 1
            while v >= s and cols[v] > kc:
                cols[v + 1] = cols[v]
                vals[v + 1] = vals[v]
                v -= 1
            cols[v + 1] = kc
            vals[v + 1] = kv
        return

    tc = np.empty(cnt, dtype=cols.dtype)
    tv = np.empty(cnt, dtype=vals.dtype)
    width = 1
    while width < cnt:
        for left in range(0, cnt, 2 * width):
            mid = left + width
            if mid > cnt:
                mid = cnt
            right = left + 2 * width
            if right > cnt:
                right = cnt
            i = left
            j = mid
            k = left
            while i < mid and j < right:
                if cols[s + i] <= cols[s + j]:
                    tc[k] = cols[s + i]
                    tv[k] = vals[s + i]
                    i += 1
                else:
                    tc[k] = cols[s + j]
                    tv[k] = vals[s + j]
                    j += 1
                k += 1
            while i < mid:
                tc[k] = cols[s + i]
                tv[k] = vals[s + i]
                i += 1
                k += 1
            while j < right:
                tc[k] = cols[s + j]
                tv[k] = vals[s + j]
                j += 1
                k += 1
        for k in range(cnt):
            cols[s + k] = tc[k]
            vals[s + k] = tv[k]
        width *= 2


@nb.njit(cache=True)
def _assemble_general_edges_nb(
    n: int,
    tail: np.ndarray,
    head: np.ndarray,
    trans: np.ndarray,
    has_d: bool,
    d_arr: np.ndarray,
):
    """Byte-for-byte SHA-256 identical assembly matching symmetric COO->CSR deduplication."""
    m_edges = trans.size
    raw_deg = np.ones(n, dtype=np.int32)
    diag_acc = np.zeros(n, dtype=np.float64)

    for e in range(m_edges):
        u = int(tail[e])
        raw_deg[u] += 1
        diag_acc[u] += trans[e]
    for e in range(m_edges):
        v = int(head[e])
        raw_deg[v] += 1
        diag_acc[v] += trans[e]
    if has_d:
        for i in range(n):
            diag_acc[i] += d_arr[i]

    raw_ptr = np.empty(n + 1, dtype=np.int32)
    raw_ptr[0] = 0
    for i in range(n):
        raw_ptr[i + 1] = raw_ptr[i] + raw_deg[i]

    tot_raw = raw_ptr[n]
    cols = np.empty(tot_raw, dtype=np.int32)
    vals = np.empty(tot_raw, dtype=np.float64)
    cur = raw_ptr[:-1].copy()

    for i in range(n):
        p = cur[i]
        cols[p] = i
        vals[p] = diag_acc[i]
        cur[i] = p + 1

    for e in range(m_edges):
        u = int(tail[e])
        pu = cur[u]
        cols[pu] = int(head[e])
        vals[pu] = -trans[e]
        cur[u] = pu + 1
    for e in range(m_edges):
        v = int(head[e])
        pv = cur[v]
        cols[pv] = int(tail[e])
        vals[pv] = -trans[e]
        cur[v] = pv + 1

    indptr = np.empty(n + 1, dtype=np.int32)
    indptr[0] = 0
    wpos = 0
    upper_cnt = 0
    bound = 0.0

    for i in range(n):
        s = raw_ptr[i]
        e = raw_ptr[i + 1]
        _stable_sort_col_val_nb(cols, vals, s, e)

        row_start = wpos
        p = s
        d_i = 0.0
        while p < e:
            c = cols[p]
            sv = vals[p]
            p += 1
            while p < e and cols[p] == c:
                sv += vals[p]
                p += 1
            if c == i:
                d_i = sv
            if sv != 0.0:
                cols[wpos] = c
                vals[wpos] = sv
                wpos += 1
                if c > i and sv < 0.0:
                    upper_cnt += 1
        indptr[i + 1] = wpos
        diag_acc[i] = d_i

        abs_sum = 0.0
        for q in range(row_start, wpos):
            abs_sum += abs(vals[q])
        if d_i > 0.0:
            r = abs_sum * (1.0 / d_i)
            if r > bound:
                bound = r

    indices_out = cols[:wpos].copy()
    data_out = vals[:wpos].copy()

    eff_u = np.empty(upper_cnt, dtype=np.int64)
    eff_v = np.empty(upper_cnt, dtype=np.int64)
    eff_t = np.empty(upper_cnt, dtype=np.float64)
    upos = 0
    for i in range(n):
        for q in range(indptr[i], indptr[i + 1]):
            c = indices_out[q]
            v = data_out[q]
            if c > i and v < 0.0:
                eff_u[upos] = i
                eff_v[upos] = c
                eff_t[upos] = -v
                upos += 1

    return indptr, indices_out, data_out, diag_acc, bound, eff_u, eff_v, eff_t


# ============================================================================
# 2. Numba Kernels: Normalized Conductivities, Barrier Masks & Watershed
# ============================================================================


@nb.njit(cache=True)
def _sample_cartesian_section_ratios_nb(
    k_flat: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
    trans: np.ndarray,
    start: int,
    stop: int,
    step: int,
    tiny: float,
) -> np.ndarray:
    n_sub = (stop - start + step - 1) // step
    out = np.empty(n_sub, dtype=np.float64)
    idx = 0
    for e in range(start, stop, step):
        ku = k_flat[tail[e]]
        kv = k_flat[head[e]]
        harm = (2.0 * ku * kv) / (ku + kv)
        denom = harm if harm > tiny else tiny
        out[idx] = trans[e] / denom
        idx += 1
    return out


@nb.njit(cache=True)
def _fill_cartesian_c_edge_nb(
    trans: np.ndarray,
    c_edge: np.ndarray,
    start: int,
    stop: int,
    g_denom: float,
) -> None:
    for e in range(start, stop):
        c_edge[e] = trans[e] / g_denom


@nb.njit(cache=True)
def _classify_active_graph_edges_nb(
    k_flat: np.ndarray,
    xyz: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
    trans: np.ndarray,
    tiny: float,
):
    m_edges = trans.size
    ratio = np.empty(m_edges, dtype=np.float64)
    cat = np.empty(m_edges, dtype=np.uint8)
    counts = np.zeros(4, dtype=np.int64)
    for e in range(m_edges):
        u = tail[e]
        v = head[e]
        ku = k_flat[u]
        kv = k_flat[v]
        harm = (2.0 * ku * kv) / (ku + kv)
        denom = harm if harm > tiny else tiny
        ratio[e] = trans[e] / denom

        dx = abs(xyz[u, 0] - xyz[v, 0])
        dy = abs(xyz[u, 1] - xyz[v, 1])
        dz = abs(xyz[u, 2] - xyz[v, 2])
        l1 = dx + dy + dz
        if l1 == 1:
            if dx == 1:
                c = 0
            elif dy == 1:
                c = 1
            else:
                c = 2
        else:
            c = 3
        cat[e] = c
        counts[c] += 1
    return ratio, cat, counts


@nb.njit(cache=True)
def _gather_cat_strided_nb(
    ratio: np.ndarray, cat: np.ndarray, target_cat: int, total_in_cat: int
) -> np.ndarray:
    step = total_in_cat // 16384
    if step < 1:
        step = 1
    n_sub = (total_in_cat + step - 1) // step
    out = np.empty(n_sub, dtype=np.float64)
    seen = 0
    out_idx = 0
    for e in range(ratio.size):
        if cat[e] == target_cat:
            if seen % step == 0:
                out[out_idx] = ratio[e]
                out_idx += 1
            seen += 1
    return out[:out_idx]


@nb.njit(cache=True)
def _fill_cat_c_edge_nb(trans: np.ndarray, cat: np.ndarray, g_denom: np.ndarray) -> np.ndarray:
    m_edges = trans.size
    c_edge = np.empty(m_edges, dtype=np.float64)
    for e in range(m_edges):
        c_edge[e] = trans[e] / g_denom[cat[e]]
    return c_edge


def normalized_conductivities(k_flat, xyz, edge, cartesian_shape=None):
    tail, head, trans = edge
    if trans.size == 0:
        return k_flat, np.empty(0, dtype=np.float64)
    tiny = float(np.finfo(np.float64).tiny)

    if cartesian_shape is not None:
        c_edge = np.empty_like(trans, dtype=np.float64)
        for section in _sections(cartesian_shape):
            cnt = section.stop - section.start
            if cnt > 0:
                step = max(1, cnt // 16384)
                r_sub = _sample_cartesian_section_ratios_nb(
                    k_flat, tail, head, trans, section.start, section.stop, step, tiny
                )
                g_ax = float(np.median(r_sub))
                _fill_cartesian_c_edge_nb(
                    trans, c_edge, section.start, section.stop, max(g_ax, tiny)
                )
        return k_flat, c_edge

    if xyz is not None:
        ratio, cat, counts = _classify_active_graph_edges_nb(k_flat, xyz, tail, head, trans, tiny)
        g_denom = np.ones(4, dtype=np.float64)
        for ax in range(3):
            cnt_ax = int(counts[ax])
            if cnt_ax > 0:
                r_sub = _gather_cat_strided_nb(ratio, cat, ax, cnt_ax)
                g_ax = float(np.median(r_sub))
                g_denom[ax] = max(g_ax, tiny)
        if counts[3] > 0:
            step = max(1, ratio.size // 16384)
            g_all = float(np.median(ratio[::step]))
            g_denom[3] = max(g_all, tiny)
        c_edge = _fill_cat_c_edge_nb(trans, cat, g_denom)
        return k_flat, c_edge

    r_sub_all = _sample_cartesian_section_ratios_nb(
        k_flat, tail, head, trans, 0, trans.size, max(1, trans.size // 16384), tiny
    )
    g_all = float(np.median(r_sub_all))
    c_edge = trans / max(g_all, tiny)
    return k_flat, c_edge


@nb.njit(cache=True)
def _node_stats_and_binary_nb(s_node: np.ndarray):
    n = s_node.size
    if n == 0:
        return 0.0, 0.0, True
    smin = s_node[0]
    smax = s_node[0]
    for i in range(1, n):
        v = s_node[i]
        if v < smin:
            smin = v
        elif v > smax:
            smax = v
    if smax <= smin:
        return smin, smax, True
    for i in range(n):
        v = s_node[i]
        if v != smin and v != smax:
            return smin, smax, False
    return smin, smax, True


@nb.njit(cache=True)
def _edge_barrier_scan_nb(
    s_node: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
    c_edge: np.ndarray,
    tau_bar: float,
    use_cliff: bool,
    tau_clf: float,
):
    m_edges = c_edge.size
    keep_edge = np.empty(m_edges, dtype=np.uint8)
    has_bar = False
    all_keep = True
    for e in range(m_edges):
        su = s_node[tail[e]]
        sv = s_node[head[e]]
        lo = su if su < sv else sv
        ce = c_edge[e]
        bar = lo > tau_bar * ce
        if bar:
            has_bar = True
        if use_cliff:
            hi = su if su > sv else sv
            drop = bar or (hi > tau_clf * ce)
        else:
            drop = bar
        if drop:
            keep_edge[e] = 0
            all_keep = False
        else:
            keep_edge[e] = 1
    return keep_edge, has_bar, all_keep


@nb.njit(cache=True)
def _graph_mask_components_nb(
    n: int,
    tail: np.ndarray,
    head: np.ndarray,
    node_mask: np.ndarray,
    has_edge_mask: bool,
    edge_mask: np.ndarray,
):
    inv_map = np.full(n, -1, dtype=np.int32)
    m = 0
    for i in range(n):
        if node_mask[i]:
            inv_map[i] = m
            m += 1
    if m == 0:
        return np.zeros(n, dtype=np.int64), 0

    ids = np.empty(m, dtype=np.int32)
    pos = 0
    for i in range(n):
        if inv_map[i] >= 0:
            ids[pos] = i
            pos += 1

    parent = np.arange(m, dtype=np.int32)
    m_edges = tail.size
    for e in range(m_edges):
        if has_edge_mask and not edge_mask[e]:
            continue
        iu = inv_map[tail[e]]
        if iu < 0:
            continue
        iv = inv_map[head[e]]
        if iv < 0:
            continue
        ru = _uf_find(parent, iu)
        rv = _uf_find(parent, iv)
        if ru != rv:
            if ru < rv:
                parent[rv] = ru
            else:
                parent[ru] = rv

    tags = np.zeros(n, dtype=np.int64)
    root_lbl = np.full(m, -1, dtype=np.int32)
    num = 0
    for k in range(m):
        r = _uf_find(parent, k)
        lbl = root_lbl[r]
        if lbl < 0:
            num += 1
            lbl = num
            root_lbl[r] = lbl
        tags[ids[k]] = lbl
    return tags, num


@nb.njit(cache=True)
def _partition_with_background_zero_nb(
    n: int, tail: np.ndarray, head: np.ndarray, keep_edge: np.ndarray
):
    parent = np.arange(n, dtype=np.int32)
    for e in range(tail.size):
        if keep_edge[e] != 0:
            ru = _uf_find(parent, int(tail[e]))
            rv = _uf_find(parent, int(head[e]))
            if ru != rv:
                if ru < rv:
                    parent[rv] = ru
                else:
                    parent[ru] = rv

    comp = np.empty(n, dtype=np.int32)
    root_lbl = np.full(n, -1, dtype=np.int32)
    num = 0
    for i in range(n):
        r = _uf_find(parent, i)
        lbl = root_lbl[r]
        if lbl < 0:
            lbl = num
            root_lbl[r] = lbl
            num += 1
        comp[i] = lbl

    if num <= 1:
        return np.zeros(n, dtype=np.int64), 0

    counts = np.zeros(num, dtype=np.int64)
    for i in range(n):
        counts[comp[i]] += 1
    bg_c = 0
    best_cnt = counts[0]
    for c in range(1, num):
        if counts[c] > best_cnt:
            best_cnt = counts[c]
            bg_c = c

    remap = np.empty(num, dtype=np.int64)
    remap[bg_c] = 0
    nxt = 1
    for c in range(num):
        if c != bg_c:
            remap[c] = nxt
            nxt += 1

    tags = np.empty(n, dtype=np.int64)
    for i in range(n):
        tags[i] = remap[comp[i]]
    return tags, num - 1


@nb.njit(cache=True)
def _watershed_core_nb(
    ell: np.ndarray,
    ell_bg: float,
    exp_bg: float,
    tail: np.ndarray,
    head: np.ndarray,
    c_edge: np.ndarray,
    keep_edge: np.ndarray,
    taus: np.ndarray,
    log_theta: float,
    pedestal_ratio: float,
    eff_min_cells: int,
):
    n = ell.size
    node_to_act = np.full(n, -1, dtype=np.int32)
    m = 0
    for i in range(n):
        if ell[i] > ell_bg:
            node_to_act[i] = m
            m += 1

    if m == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            0,
        )

    active_ids = np.empty(m, dtype=np.int32)
    ell_act = np.empty(m, dtype=np.float64)
    pos = 0
    for i in range(n):
        if node_to_act[i] >= 0:
            active_ids[pos] = i
            ell_act[pos] = ell[i]
            pos += 1

    n_steps = taus.size
    neg_taus = np.empty(n_steps, dtype=np.float64)
    for s in range(n_steps):
        neg_taus[s] = -taus[s]

    node_bucket = np.empty(m, dtype=np.int32)
    nb_counts = np.zeros(n_steps, dtype=np.int64)
    for u in range(m):
        x = -ell_act[u]
        lo = 0
        hi = n_steps
        while lo < hi:
            mid = (lo + hi) >> 1
            if neg_taus[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        if lo >= n_steps:
            lo = n_steps - 1
        node_bucket[u] = lo
        nb_counts[lo] += 1

    node_splits = np.empty(n_steps + 1, dtype=np.int64)
    node_splits[0] = 0
    for s in range(n_steps):
        node_splits[s + 1] = node_splits[s] + nb_counts[s]
    node_order = np.empty(m, dtype=np.int32)
    ncur = node_splits[:-1].copy()
    for u in range(m):
        b = node_bucket[u]
        dst = ncur[b]
        node_order[dst] = u
        ncur[b] = dst + 1

    m_edges_all = tail.size
    n_valid_e = 0
    for e in range(m_edges_all):
        if keep_edge[e] != 0 and c_edge[e] > exp_bg:
            iu = node_to_act[tail[e]]
            if iu >= 0 and node_to_act[head[e]] >= 0:
                n_valid_e += 1

    eu = np.empty(n_valid_e, dtype=np.int32)
    ev = np.empty(n_valid_e, dtype=np.int32)
    ec = np.empty(n_valid_e, dtype=np.float64)
    edge_bucket = np.empty(n_valid_e, dtype=np.int32)
    eb_counts = np.zeros(n_steps, dtype=np.int64)

    epos = 0
    for e in range(m_edges_all):
        if keep_edge[e] != 0 and c_edge[e] > exp_bg:
            iu = node_to_act[tail[e]]
            if iu >= 0:
                iv = node_to_act[head[e]]
                if iv >= 0:
                    w = np.log(c_edge[e])
                    eu[epos] = iu
                    ev[epos] = iv
                    ec[epos] = w
                    x = -w
                    lo = 0
                    hi = n_steps
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if neg_taus[mid] <= x:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo >= n_steps:
                        lo = n_steps - 1
                    bu = node_bucket[iu]
                    bv = node_bucket[iv]
                    bmax = bu if bu >= bv else bv
                    if lo > bmax:
                        bmax = lo
                    edge_bucket[epos] = bmax
                    eb_counts[bmax] += 1
                    epos += 1

    edge_splits = np.empty(n_steps + 1, dtype=np.int64)
    edge_splits[0] = 0
    max_bucket_e = 0
    for s in range(n_steps):
        cnt_s = eb_counts[s]
        if cnt_s > max_bucket_e:
            max_bucket_e = int(cnt_s)
        edge_splits[s + 1] = edge_splits[s] + cnt_s

    edge_order = np.empty(n_valid_e, dtype=np.int32)
    ecur = edge_splits[:-1].copy()
    for e in range(n_valid_e):
        b = edge_bucket[e]
        dst = ecur[b]
        edge_order[dst] = e
        ecur[b] = dst + 1

    comp = np.full(m, -1, dtype=np.int32)
    max_comps = m + 1
    c_parent = np.empty(max_comps, dtype=np.int32)
    c_peak = np.zeros(max_comps, dtype=np.float64)
    c_size = np.zeros(max_comps, dtype=np.int32)
    c_pedestal = np.zeros(max_comps, dtype=np.uint8)

    scratch_min = np.empty(max_comps, dtype=np.int32)
    scratch_max = np.empty(max_comps, dtype=np.int32)
    best_w = np.empty(max_comps, dtype=np.float64)
    best_mat = np.full(max_comps, -1, dtype=np.int32)
    seen_stamp = np.zeros(max_comps, dtype=np.int32)
    touched_imm = np.empty(max_comps, dtype=np.int32)

    buf_alloc = max(1, max_bucket_e)
    buf_u = np.empty(buf_alloc, dtype=np.int32)
    buf_v = np.empty(buf_alloc, dtype=np.int32)
    buf_w = np.empty(buf_alloc, dtype=np.float64)
    sv_u = np.empty(buf_alloc, dtype=np.int32)
    sv_v = np.empty(buf_alloc, dtype=np.int32)
    sv_ru = np.empty(buf_alloc, dtype=np.int32)
    sv_rv = np.empty(buf_alloc, dtype=np.int32)
    sv_tmp = np.empty(max_comps, dtype=np.int32)

    n_comp = 0
    stamp = 0

    for b_idx in range(n_steps):
        tau = taus[b_idx]
        ns = node_splits[b_idx]
        ne = node_splits[b_idx + 1]
        for p in range(ns, ne):
            nd = node_order[p]
            cid = n_comp
            n_comp += 1
            comp[nd] = cid
            c_parent[cid] = cid
            c_peak[cid] = ell_act[nd]
            c_size[cid] = 1
            c_pedestal[cid] = 0

        es = edge_splits[b_idx]
        ee = edge_splits[b_idx + 1]
        if es == ee:
            continue

        k = 0
        n_imm = 0
        stamp += 1
        n_sv_touched = 0
        for p in range(es, ee):
            ei = edge_order[p]
            ru = _uf_find(c_parent, comp[eu[ei]])
            rv = _uf_find(c_parent, comp[ev[ei]])
            if ru != rv:
                buf_u[k] = ru
                buf_v[k] = rv
                buf_w[k] = ec[ei]
                mu = (c_peak[ru] - tau >= log_theta) and (c_size[ru] >= eff_min_cells)
                mv = (c_peak[rv] - tau >= log_theta) and (c_size[rv] >= eff_min_cells)
                if (not mu) and (not mv):
                    sv_u[n_imm] = ru
                    sv_v[n_imm] = rv
                    n_imm += 1
                    if seen_stamp[ru] != stamp:
                        seen_stamp[ru] = stamp
                        touched_imm[n_sv_touched] = ru
                        n_sv_touched += 1
                    if seen_stamp[rv] != stamp:
                        seen_stamp[rv] = stamp
                        touched_imm[n_sv_touched] = rv
                        n_sv_touched += 1
                k += 1

        if k == 0:
            continue

        if n_imm > 0:
            cur_m = n_imm
            while cur_m > 0:
                m_next = 0
                for idx in range(cur_m):
                    uu = sv_u[idx]
                    vv = sv_v[idx]
                    ru = c_parent[uu]
                    rv = c_parent[vv]
                    if ru != rv:
                        sv_u[m_next] = uu
                        sv_v[m_next] = vv
                        sv_ru[m_next] = ru
                        sv_rv[m_next] = rv
                        m_next += 1
                if m_next == 0:
                    break
                for idx in range(m_next):
                    ru = sv_ru[idx]
                    rv = sv_rv[idx]
                    hi = ru if ru > rv else rv
                    lo = rv if ru > rv else ru
                    if lo < c_parent[hi]:
                        c_parent[hi] = lo
                cur_m = m_next
                for t in range(n_sv_touched):
                    nd_t = touched_imm[t]
                    sv_tmp[t] = c_parent[c_parent[nd_t]]
                for t in range(n_sv_touched):
                    c_parent[touched_imm[t]] = sv_tmp[t]

            for t in range(n_sv_touched):
                nd_t = touched_imm[t]
                r = c_parent[nd_t]
                while c_parent[r] != r:
                    r = c_parent[r]
                sv_tmp[t] = r
            for t in range(n_sv_touched):
                nd_t = touched_imm[t]
                r = sv_tmp[t]
                c_parent[nd_t] = r
                if nd_t != r:
                    c_size[r] += c_size[nd_t]
                    if c_peak[nd_t] > c_peak[r]:
                        c_peak[r] = c_peak[nd_t]
                    if c_pedestal[nd_t] != 0:
                        c_pedestal[r] = 1

            k2 = 0
            for idx in range(k):
                ru = _uf_find(c_parent, buf_u[idx])
                rv = _uf_find(c_parent, buf_v[idx])
                if ru != rv:
                    buf_u[k2] = ru
                    buf_v[k2] = rv
                    buf_w[k2] = buf_w[idx]
                    k2 += 1
            k = k2
            if k == 0:
                continue

        n_one = 0
        for idx in range(k):
            ru = buf_u[idx]
            rv = buf_v[idx]
            mu = (c_peak[ru] - tau >= log_theta) and (c_size[ru] >= eff_min_cells)
            mv = (c_peak[rv] - tau >= log_theta) and (c_size[rv] >= eff_min_cells)
            if mu != mv:
                if mu:
                    buf_u[n_one] = rv
                    buf_v[n_one] = ru
                else:
                    buf_u[n_one] = ru
                    buf_v[n_one] = rv
                buf_w[n_one] = buf_w[idx]
                n_one += 1

        if n_one > 0:
            stamp += 1
            n_touched = 0
            for idx in range(n_one):
                ic = buf_u[idx]
                mc = buf_v[idx]
                if seen_stamp[ic] != stamp:
                    seen_stamp[ic] = stamp
                    touched_imm[n_touched] = ic
                    n_touched += 1
                    scratch_min[ic] = mc
                    scratch_max[ic] = mc
                    best_w[ic] = -1.0e308
                    best_mat[ic] = -1
                else:
                    if mc < scratch_min[ic]:
                        scratch_min[ic] = mc
                    if mc > scratch_max[ic]:
                        scratch_max[ic] = mc

            two_min = 2 * eff_min_cells
            for idx in range(n_one):
                ic = buf_u[idx]
                mc = buf_v[idx]
                touches_multi = (scratch_min[ic] < scratch_max[ic]) and (c_size[ic] >= two_min)
                base_sz = c_size[mc] if c_size[mc] >= eff_min_cells else eff_min_cells
                if ((c_size[ic] > pedestal_ratio * base_sz) or touches_multi) and (
                    c_peak[mc] - c_peak[ic] >= log_theta
                ):
                    c_pedestal[ic] = 1

            for idx in range(n_one):
                ic = buf_u[idx]
                mc = buf_v[idx]
                wc = buf_w[idx]
                blocked = (c_pedestal[ic] != 0) and (c_peak[mc] - c_peak[ic] >= log_theta)
                if not blocked:
                    if wc >= best_w[ic]:
                        best_w[ic] = wc
                        best_mat[ic] = mc

            for t in range(n_touched):
                ic = touched_imm[t]
                mc = best_mat[ic]
                if mc >= 0:
                    c_parent[ic] = mc
                    c_size[mc] += c_size[ic]
                    if c_peak[ic] > c_peak[mc]:
                        c_peak[mc] = c_peak[ic]

    if n_comp == 0:
        return (
            active_ids,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            0,
        )

    final_roots = np.empty(m, dtype=np.int32)
    for u in range(m):
        final_roots[u] = _uf_find(c_parent, comp[u])

    return active_ids, final_roots, c_peak[:n_comp], n_comp


def _extract_components_with_keep_edge(k_flat, xyz, edge, cartesian_shape, cfg):
    """Return ``(tags, count, rule, keep_edge_or_none)``."""
    n = k_flat.size
    mode = cfg["phase_mode"]
    tail, head, _ = edge

    if mode == "algebraic_only":
        return np.zeros(n, dtype=np.int64), 0, "algebraic_only", None

    if mode == "legacy_k_gt_min":
        high = k_flat > float(np.min(k_flat))
        if not np.any(high):
            return np.zeros(n, dtype=np.int64), 0, "Kx > min(Kx)", None
        if cartesian_shape is not None:
            comp_grid, num = label(high.reshape(cartesian_shape))
            return comp_grid.ravel().astype(np.int64, copy=False), int(num), "Kx > min(Kx)", None
        tags, num = _graph_mask_components_nb(
            n, tail, head, high, False, np.empty(0, dtype=np.bool_)
        )
        return tags, int(num), "Kx > min(Kx)", None

    if mode == "single_median":
        ell = np.log(k_flat)
        med = float(np.median(ell))
        high = ell > med
        if not np.any(high):
            return np.zeros(n, dtype=np.int64), 0, "log(Kx) > median(log(Kx))", None
        if cartesian_shape is not None:
            comp_grid, num = label(high.reshape(cartesian_shape))
            return (
                comp_grid.ravel().astype(np.int64, copy=False),
                int(num),
                "log(Kx) > median(log(Kx))",
                None,
            )
        tags, num = _graph_mask_components_nb(
            n, tail, head, high, False, np.empty(0, dtype=np.bool_)
        )
        return tags, int(num), "log(Kx) > median(log(Kx))", None

    if mode == "naive_multibin":
        ell = np.log(k_flat)
        emin, emax = float(np.min(ell)), float(np.max(ell))
        if emax <= emin + 1e-12:
            return np.zeros(n, dtype=np.int64), 0, "naive_4bin_log_quantization", None
        bins = np.clip(np.floor(4.0 * (ell - emin) / (emax - emin)).astype(np.int32), 0, 3)
        active = bins > 0
        if not np.any(active):
            return np.zeros(n, dtype=np.int64), 0, "naive_4bin_log_quantization", None
        same_bin = active[tail] & active[head] & (bins[tail] == bins[head])
        tags, num = _graph_mask_components_nb(n, tail, head, active, True, same_bin)
        return tags, int(num), "naive_4bin_log_quantization", None

    s_node, c_edge = normalized_conductivities(k_flat, xyz, edge, cartesian_shape)
    tau_bar = float(cfg["tau_barrier"])
    tau_clf = float(cfg["tau_cliff"])
    use_cliff = tau_clf < 1.0e20 and tail.size > 0
    if tail.size > 0:
        keep_edge, has_bar, all_keep = _edge_barrier_scan_nb(
            s_node, tail, head, c_edge, tau_bar, use_cliff, tau_clf
        )
    else:
        keep_edge = np.zeros(0, dtype=np.uint8)
        has_bar = False
        all_keep = True

    eff_keep_out = None if all_keep else keep_edge
    smin, smax, is_bin = _node_stats_and_binary_nb(s_node)

    if is_bin:
        high = k_flat > smin
        if smax <= smin and not has_bar:
            return np.zeros(n, dtype=np.int64), 0, "binary_exact_fastpath", None
        if smax > smin and not has_bar and cartesian_shape is not None:
            comp_grid, num = label(high.reshape(cartesian_shape))
            return (
                comp_grid.ravel().astype(np.int64, copy=False),
                int(num),
                "binary_exact_fastpath",
                None,
            )
        if smax > smin:
            if use_cliff:
                keep_bar_only, _, _ = _edge_barrier_scan_nb(
                    s_node, tail, head, c_edge, tau_bar, False, tau_clf
                )
            else:
                keep_bar_only = keep_edge
            tags, num = _graph_mask_components_nb(
                n, tail, head, high, True, keep_bar_only.view(np.bool_)
            )
            return tags, int(num), "binary_barrier_fastpath", keep_bar_only
        if use_cliff:
            keep_bar_only, _, _ = _edge_barrier_scan_nb(
                s_node, tail, head, c_edge, tau_bar, False, tau_clf
            )
        else:
            keep_bar_only = keep_edge
        tags, num = _partition_with_background_zero_nb(n, tail, head, keep_bar_only)
        return tags, int(num), "uniform_anisotropic_barrier_partition", keep_bar_only

    if smax <= 4.0 * smin:
        if all_keep:
            return np.zeros(n, dtype=np.int64), 0, "near_homogeneous_single_basin", None
        tags, num = _partition_with_background_zero_nb(n, tail, head, keep_edge)
        return tags, int(num), "near_homogeneous_barrier_partition", eff_keep_out

    ell = np.log(s_node)
    step_q = max(1, n // 32768)
    ell_lo = float(np.quantile(ell[::step_q], 0.05))
    ell_hi = float(np.log(smax))
    if ell_hi <= ell_lo + 1e-9:
        ell_lo = float(np.log(smin))
    ell_bg = float(ell_lo + cfg["bg_alpha"] * (ell_hi - ell_lo))
    exp_bg = float(np.exp(ell_bg))

    eff_min_cells = max(4, min(cfg["min_peak_cells"], max(4, n // 8192)))
    max_basins = cfg["max_basins"]

    if mode == "cliff_only":
        active_mask = ell > ell_bg
        if not np.any(active_mask):
            return np.zeros(n, dtype=np.int64), 0, f"{mode}_empty_foreground", eff_keep_out
        valid_e = (keep_edge != 0) & (c_edge > exp_bg)
        raw_tags, num = _graph_mask_components_nb(n, tail, head, active_mask, True, valid_e)
        if num == 0:
            return raw_tags, 0, "cliff_only_superlevel", eff_keep_out
        sizes = np.bincount(raw_tags, minlength=num + 1)[1:]
        kept = np.flatnonzero(sizes >= eff_min_cells)
        if max_basins is not None and len(kept) > max_basins:
            top = np.argsort(sizes[kept])[::-1][:max_basins]
            kept = np.sort(kept[top])
        remap = np.zeros(num + 1, dtype=np.int64)
        remap[kept + 1] = np.arange(1, len(kept) + 1, dtype=np.int64)
        return remap[raw_tags], int(len(kept)), "cliff_only_superlevel", eff_keep_out

    log_theta = float(np.log(cfg["theta_pers"]))
    pedestal_ratio = float(cfg["pedestal_ratio"])
    delta = max(0.35, 0.25 * log_theta)
    n_steps = int(np.clip(np.ceil((ell_hi - ell_bg) / delta), 4, 20))
    taus = np.linspace(ell_hi - 0.5 * delta, ell_bg, n_steps)

    active_ids, final_roots, c_peak, n_comp = _watershed_core_nb(
        ell,
        ell_bg,
        exp_bg,
        tail,
        head,
        c_edge,
        keep_edge,
        taus,
        log_theta,
        pedestal_ratio,
        int(eff_min_cells),
    )

    if active_ids.size == 0:
        return np.zeros(n, dtype=np.int64), 0, f"{mode}_empty_foreground", eff_keep_out
    if n_comp == 0:
        return np.zeros(n, dtype=np.int64), 0, "ml_persistence_watershed", eff_keep_out

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
    return tags, int(len(kept_idx)), "ml_persistence_watershed", eff_keep_out


def _extract_components_core(k_flat, xyz, edge, cartesian_shape, cfg):
    tags, count, rule, _ = _extract_components_with_keep_edge(
        k_flat, xyz, edge, cartesian_shape, cfg
    )
    return tags, count, rule


def extract_multilevel_components(k, edge=None, xyz=None, config=None):
    t0 = time.perf_counter()
    cfg_name, cfg = _resolve_config(config)
    k_arr = np.asarray(k, dtype=np.float64)
    if k_arr.size == 0 or not np.all(np.isfinite(k_arr)) or np.any(k_arr <= 0):
        raise ValueError("k must be nonempty, finite, and strictly positive")

    if k_arr.ndim == 3:
        cartesian_shape = k_arr.shape
        if edge is None:
            if len(set(cartesian_shape)) != 1:
                raise ValueError("edge is required for non-cubic 3D permeability arrays")
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
            raise ValueError("1D active-cell permeability requires an explicit edge tuple")
        edge_tup = (
            np.asarray(edge[0]),
            np.asarray(edge[1]),
            np.asarray(edge[2], dtype=np.float64),
        )
        xyz_arr = None if xyz is None else np.asarray(xyz)
        tags_flat, count, rule = _extract_components_core(k_arr, xyz_arr, edge_tup, None, cfg)
        labels_out = tags_flat
    else:
        raise ValueError("k must be a 3D Cartesian array or 1D active-cell vector")

    elapsed = time.perf_counter() - t0
    info = {
        "config_name": cfg_name,
        "phase_mode": cfg["phase_mode"],
        "phase_rule": rule,
        "high_component_count": int(count),
        "high_fraction": float(np.count_nonzero(tags_flat > 0) / max(1, tags_flat.size)),
        "component_label_seconds": elapsed,
    }
    return labels_out, int(count), info


# ============================================================================
# 3. Barrier CSR Masking, Anisotropy Detection, Semi-Coarsening,
#    Intra-Tag Box Splitting & ActiveGraph 1D Column Line LDL^T Smoother
# ============================================================================


@nb.njit(cache=True)
def _build_csr_allow_mask_from_edges_nb(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    eu: np.ndarray,
    ev: np.ndarray,
    keep_edge: np.ndarray,
) -> np.ndarray:
    """Map undirected non-barrier edges (keep_edge[e] != 0) onto sorted CSR entries."""
    nnz = indices.size
    allow = np.zeros(nnz, dtype=np.uint8)
    for e in range(keep_edge.size):
        if keep_edge[e] == 0:
            continue
        u = int(eu[e])
        v = int(ev[e])
        lo = indptr[u]
        hi = indptr[u + 1]
        while lo < hi:
            mid = (lo + hi) >> 1
            c = indices[mid]
            if c < v:
                lo = mid + 1
            elif c > v:
                hi = mid
            else:
                if data[mid] < 0.0:
                    allow[mid] = 1
                break
        lo = indptr[v]
        hi = indptr[v + 1]
        while lo < hi:
            mid = (lo + hi) >> 1
            c = indices[mid]
            if c < u:
                lo = mid + 1
            elif c > u:
                hi = mid
            else:
                if data[mid] < 0.0:
                    allow[mid] = 1
                break
    return allow


@nb.njit(cache=True)
def _project_allow_mask_to_coarse_nb(
    n_fine: int,
    n_coarse: int,
    f_indptr: np.ndarray,
    f_indices: np.ndarray,
    f_allow: np.ndarray,
    inv: np.ndarray,
    c_indptr: np.ndarray,
    c_indices: np.ndarray,
    c_data: np.ndarray,
) -> np.ndarray:
    """Project fine non-barrier adjacency E_keep onto coarse CSR matrix in O(nnz) time."""
    counts = np.zeros(n_coarse, dtype=np.int32)
    for u in range(n_fine):
        counts[inv[u]] += 1
    agg_ptr = np.empty(n_coarse + 1, dtype=np.int32)
    agg_ptr[0] = 0
    for c in range(n_coarse):
        agg_ptr[c + 1] = agg_ptr[c] + counts[c]
    agg_nodes = np.empty(n_fine, dtype=np.int32)
    cur = agg_ptr[:-1].copy()
    for u in range(n_fine):
        c = int(inv[u])
        dst = cur[c]
        agg_nodes[dst] = u
        cur[c] = dst + 1

    c_allow = np.zeros(c_indices.size, dtype=np.uint8)
    stamp = np.full(n_coarse, -1, dtype=np.int32)
    for cu in range(n_coarse):
        for p in range(agg_ptr[cu], agg_ptr[cu + 1]):
            u = agg_nodes[p]
            for q in range(f_indptr[u], f_indptr[u + 1]):
                if f_allow[q] != 0:
                    cv = int(inv[f_indices[q]])
                    if cv != cu:
                        stamp[cv] = cu
        for q in range(c_indptr[cu], c_indptr[cu + 1]):
            cv = c_indices[q]
            if cv != cu and c_data[q] < 0.0 and stamp[cv] == cu:
                c_allow[q] = 1
    return c_allow


@nb.njit(cache=True)
def _xyz_spans_and_max_nb(xyz: np.ndarray):
    n = xyz.shape[0]
    if n == 0:
        return 0, 0, 0, 0, 0, 0
    mn0 = mx0 = xyz[0, 0]
    mn1 = mx1 = xyz[0, 1]
    mn2 = mx2 = xyz[0, 2]
    for i in range(1, n):
        v0 = xyz[i, 0]
        v1 = xyz[i, 1]
        v2 = xyz[i, 2]
        if v0 < mn0:
            mn0 = v0
        elif v0 > mx0:
            mx0 = v0
        if v1 < mn1:
            mn1 = v1
        elif v1 > mx1:
            mx1 = v1
        if v2 < mn2:
            mn2 = v2
        elif v2 > mx2:
            mx2 = v2
    return mx0, mx1, mx2, mx0 - mn0, mx1 - mn1, mx2 - mn2


@nb.njit(cache=True)
def _gather_csr_axis_trans_nb(
    n: int,
    xyz: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
):
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    for u in range(n):
        xu = xyz[u, 0]
        yu = xyz[u, 1]
        zu = xyz[u, 2]
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            if v > u and data[p] < 0.0:
                dx = abs(xu - xyz[v, 0])
                dy = abs(yu - xyz[v, 1])
                dz = abs(zu - xyz[v, 2])
                if dx + dy + dz == 1:
                    if dx == 1:
                        cnt0 += 1
                    elif dy == 1:
                        cnt1 += 1
                    else:
                        cnt2 += 1

    st0 = max(1, cnt0 // 8192)
    st1 = max(1, cnt1 // 8192)
    st2 = max(1, cnt2 // 8192)
    a0 = np.empty((cnt0 + st0 - 1) // st0 if cnt0 > 0 else 0, dtype=np.float64)
    a1 = np.empty((cnt1 + st1 - 1) // st1 if cnt1 > 0 else 0, dtype=np.float64)
    a2 = np.empty((cnt2 + st2 - 1) // st2 if cnt2 > 0 else 0, dtype=np.float64)

    s0 = s1 = s2 = 0
    o0 = o1 = o2 = 0
    for u in range(n):
        xu = xyz[u, 0]
        yu = xyz[u, 1]
        zu = xyz[u, 2]
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            if v > u and data[p] < 0.0:
                dx = abs(xu - xyz[v, 0])
                dy = abs(yu - xyz[v, 1])
                dz = abs(zu - xyz[v, 2])
                if dx + dy + dz == 1:
                    t = -data[p]
                    if dx == 1:
                        if s0 % st0 == 0:
                            a0[o0] = t
                            o0 += 1
                        s0 += 1
                    elif dy == 1:
                        if s1 % st1 == 0:
                            a1[o1] = t
                            o1 += 1
                        s1 += 1
                    else:
                        if s2 % st2 == 0:
                            a2[o2] = t
                            o2 += 1
                        s2 += 1
    return a0[:o0], a1[:o1], a2[:o2]


def _detect_dominant_axis_csr(
    xyz: np.ndarray, matrix: sparse.csr_matrix, ratio_threshold: float = 4.0
):
    """Return ``(dom_ax_or_none, medians_tuple, aniso_ratio)`` using the >= 4.0 criterion."""
    n = xyz.shape[0]
    _, _, _, sp0, sp1, sp2 = _xyz_spans_and_max_nb(xyz)
    spans = (int(sp0), int(sp1), int(sp2))
    a0, a1, a2 = _gather_csr_axis_trans_nb(n, xyz, matrix.indptr, matrix.indices, matrix.data)
    m0 = float(np.median(a0)) if (a0.size > 0 and spans[0] >= 1) else 0.0
    m1 = float(np.median(a1)) if (a1.size > 0 and spans[1] >= 1) else 0.0
    m2 = float(np.median(a2)) if (a2.size > 0 and spans[2] >= 1) else 0.0
    meds = (m0, m1, m2)
    dom_ax = int(np.argmax(meds))
    if spans[dom_ax] < 1 or meds[dom_ax] <= 0.0:
        return None, meds, 1.0
    other_max = max(meds[i] for i in range(3) if i != dom_ax)
    tiny = float(np.finfo(np.float64).tiny)
    ratio = meds[dom_ax] / max(other_max, tiny)
    if ratio >= ratio_threshold:
        return dom_ax, meds, ratio
    return None, meds, ratio


@nb.njit(cache=True)
def _make_cartesian_xyz_nb(nx: int, ny: int, nz: int) -> np.ndarray:
    n = nx * ny * nz
    xyz = np.empty((n, 3), dtype=np.int64)
    pos = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                xyz[pos, 0] = ix
                xyz[pos, 1] = iy
                xyz[pos, 2] = iz
                pos += 1
    return xyz


@nb.njit(cache=True)
def _aggregate_geom_connected_nb(
    xyz: np.ndarray,
    tags: np.ndarray,
    sh0: int,
    sh1: int,
    sh2: int,
    s0: int,
    s1: int,
    s2: int,
    split_disconnected: bool,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    has_allow_mask: bool,
    allow_mask: np.ndarray,
):
    """Component-preserving geometric box aggregation with intra-(box, tag) connectivity."""
    n = tags.size
    n_boxes = s0 * s1 * s2
    s12 = s1 * s2

    box_head = np.full(n_boxes, -1, dtype=np.int32)
    box_next = np.empty(n, dtype=np.int32)
    for i in range(n - 1, -1, -1):
        bx = xyz[i, 0] >> sh0
        by = xyz[i, 1] >> sh1
        bz = xyz[i, 2] >> sh2
        b = (bx * s1 + by) * s2 + bz
        box_next[i] = box_head[b]
        box_head[b] = i

    inv = np.empty(n, dtype=np.int64)
    c_xyz = np.empty((n, 3), dtype=np.int64)
    c_tags = np.empty(n, dtype=np.int64)
    in_run_slot = np.full(n, -1, dtype=np.int32)

    buf_cap = 64
    loc_i = np.empty(buf_cap, dtype=np.int32)
    loc_t = np.empty(buf_cap, dtype=np.int64)
    loc_cc = np.empty(buf_cap, dtype=np.int32)
    uf_loc = np.empty(buf_cap, dtype=np.int32)
    nc = 0

    for b in range(n_boxes):
        cur = box_head[b]
        if cur < 0:
            continue
        cnt = 0
        while cur >= 0:
            if cnt >= buf_cap:
                new_cap = buf_cap * 2
                ni = np.empty(new_cap, dtype=np.int32)
                nt = np.empty(new_cap, dtype=np.int64)
                loc_cc = np.empty(new_cap, dtype=np.int32)
                uf_loc = np.empty(new_cap, dtype=np.int32)
                ni[:cnt] = loc_i[:cnt]
                nt[:cnt] = loc_t[:cnt]
                loc_i = ni
                loc_t = nt
                buf_cap = new_cap
            loc_i[cnt] = cur
            loc_t[cnt] = tags[cur]
            cnt += 1
            cur = box_next[cur]

        bx = b // s12
        rem = b - bx * s12
        by = rem // s2
        bz = rem - by * s2

        if cnt == 1:
            cid = nc
            c_xyz[cid, 0] = bx
            c_xyz[cid, 1] = by
            c_xyz[cid, 2] = bz
            c_tags[cid] = loc_t[0]
            inv[loc_i[0]] = cid
            nc += 1
            continue

        if split_disconnected:
            for q in range(cnt):
                in_run_slot[loc_i[q]] = q
                uf_loc[q] = q
            rem_cc = cnt
            for q in range(cnt):
                u_node = loc_i[q]
                tq = loc_t[q]
                for p in range(indptr[u_node], indptr[u_node + 1]):
                    if has_allow_mask:
                        if allow_mask[p] == 0:
                            continue
                    else:
                        if indices[p] == u_node or data[p] >= 0.0:
                            continue
                    sv = in_run_slot[indices[p]]
                    if sv > q and loc_t[sv] == tq:
                        rq = _uf_find(uf_loc, q)
                        rv = _uf_find(uf_loc, sv)
                        if rq != rv:
                            if rq < rv:
                                uf_loc[rv] = rq
                            else:
                                uf_loc[rq] = rv
                            rem_cc -= 1
                            if rem_cc == 1:
                                break
                if rem_cc == 1:
                    break
            for q in range(cnt):
                in_run_slot[loc_i[q]] = -1
                loc_cc[q] = 0 if rem_cc == 1 else _uf_find(uf_loc, q)
        else:
            for q in range(cnt):
                loc_cc[q] = 0

        for u in range(1, cnt):
            kt = loc_t[u]
            kcc = loc_cc[u]
            ki = loc_i[u]
            v = u - 1
            while v >= 0 and (loc_t[v] > kt or (loc_t[v] == kt and loc_cc[v] > kcc)):
                loc_t[v + 1] = loc_t[v]
                loc_cc[v + 1] = loc_cc[v]
                loc_i[v + 1] = loc_i[v]
                v -= 1
            loc_t[v + 1] = kt
            loc_cc[v + 1] = kcc
            loc_i[v + 1] = ki

        prev_cc = loc_cc[0]
        prev_t = loc_t[0]
        cid = nc
        c_xyz[cid, 0] = bx
        c_xyz[cid, 1] = by
        c_xyz[cid, 2] = bz
        c_tags[cid] = prev_t
        inv[loc_i[0]] = cid
        nc += 1

        for u in range(1, cnt):
            cc_u = loc_cc[u]
            t_u = loc_t[u]
            if cc_u != prev_cc or t_u != prev_t:
                prev_cc = cc_u
                prev_t = t_u
                cid = nc
                c_xyz[cid, 0] = bx
                c_xyz[cid, 1] = by
                c_xyz[cid, 2] = bz
                c_tags[cid] = t_u
                nc += 1
            inv[loc_i[u]] = cid

    return inv, c_xyz[:nc].copy(), c_tags[:nc].copy()


@nb.njit(cache=True)
def _build_active_graph_tridiag_nb(
    n: int,
    xyz: np.ndarray,
    ax_a: int,
    ax_b: int,
    dom_ax: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    diag: np.ndarray,
):
    """Order ActiveGraph along (ax_a, ax_b, dom_ax) and factor contiguous non-NNC lines."""
    mn0 = mn1 = mn2 = 0
    mx0 = mx1 = mx2 = 0
    for i in range(n):
        ca = xyz[i, ax_a]
        cb = xyz[i, ax_b]
        cd = xyz[i, dom_ax]
        if i == 0:
            mn0 = mx0 = ca
            mn1 = mx1 = cb
            mn2 = mx2 = cd
        else:
            if ca < mn0:
                mn0 = ca
            elif ca > mx0:
                mx0 = ca
            if cb < mn1:
                mn1 = cb
            elif cb > mx1:
                mx1 = cb
            if cd < mn2:
                mn2 = cd
            elif cd > mx2:
                mx2 = cd

    sb = (mx1 - mn1) + 1
    sd = (mx2 - mn2) + 1
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        ka = xyz[i, ax_a] - mn0
        kb = xyz[i, ax_b] - mn1
        kd = xyz[i, dom_ax] - mn2
        keys[i] = (ka * sb + kb) * sd + kd

    perm = np.argsort(keys).astype(np.int32)
    l_sub = np.zeros(max(1, n - 1), dtype=np.float64)
    d_fac = np.empty(n, dtype=np.float64)
    for p in range(n):
        d_fac[p] = diag[perm[p]]

    for p in range(n - 1):
        u = perm[p]
        v = perm[p + 1]
        w_uv = 0.0
        if (
            xyz[u, ax_a] == xyz[v, ax_a]
            and xyz[u, ax_b] == xyz[v, ax_b]
            and xyz[v, dom_ax] == xyz[u, dom_ax] + 1
        ):
            lo = indptr[u]
            hi = indptr[u + 1]
            while lo < hi:
                mid = (lo + hi) >> 1
                c = indices[mid]
                if c < v:
                    lo = mid + 1
                elif c > v:
                    hi = mid
                else:
                    if data[mid] < 0.0:
                        w_uv = data[mid]
                    break
        dk = d_fac[p]
        if dk <= 0.0 or not np.isfinite(dk):
            return False, perm, l_sub, d_fac
        if w_uv != 0.0:
            mult = w_uv / dk
            l_sub[p] = mult
            d_fac[p + 1] -= mult * w_uv

    if d_fac[n - 1] <= 0.0 or not np.isfinite(d_fac[n - 1]):
        return False, perm, l_sub, d_fac

    inv_d_fac = np.empty(n, dtype=np.float64)
    for p in range(n):
        df = d_fac[p]
        if not np.isfinite(df) or df <= 0.0:
            return False, perm, l_sub, d_fac
        inv_d_fac[p] = 1.0 / df
    return True, perm, l_sub, inv_d_fac


@nb.njit(cache=True)
def _apply_perm_line_smoother_into_nb(
    perm: np.ndarray,
    l_sub: np.ndarray,
    inv_d_fac: np.ndarray,
    omega: float,
    rhs: np.ndarray,
    out: np.ndarray,
    add_to_out: bool,
    work_y: np.ndarray,
) -> None:
    """Strictly SPD 1D tridiagonal LDL^T solve omega * Pi^T (L D L^T)^-1 Pi rhs."""
    n = perm.size
    for p in range(n):
        work_y[p] = rhs[perm[p]]
    for p in range(n - 1):
        ls = l_sub[p]
        if ls != 0.0:
            work_y[p + 1] -= ls * work_y[p]
    for p in range(n):
        work_y[p] *= inv_d_fac[p]
    for p in range(n - 2, -1, -1):
        ls = l_sub[p]
        if ls != 0.0:
            work_y[p] -= ls * work_y[p + 1]
    if add_to_out:
        for p in range(n):
            out[perm[p]] += omega * work_y[p]
    else:
        for p in range(n):
            out[perm[p]] = omega * work_y[p]


def _build_active_graph_line_smoother(
    xyz: np.ndarray,
    matrix: sparse.csr_matrix,
    diag: np.ndarray,
    omega: float = 2.0 / 3.0,
    forced_axis: int | None = None,
):
    """Build exact symmetric 1D column line-Jacobi LDL^T smoother on ActiveGraph."""
    n = xyz.shape[0]
    if n < 2:
        return None
    if forced_axis is None:
        dom_ax, _, _ = _detect_dominant_axis_csr(xyz, matrix, ratio_threshold=4.0)
        if dom_ax is None:
            return None
    else:
        dom_ax = int(forced_axis)
        if dom_ax not in (0, 1, 2):
            raise ValueError("forced line axis must be 0, 1 or 2")

    orth = [ax for ax in range(3) if ax != dom_ax]
    ok, perm, l_sub, inv_d_fac = _build_active_graph_tridiag_nb(
        n, xyz, orth[0], orth[1], dom_ax, matrix.indptr, matrix.indices, matrix.data, diag
    )
    if not ok:
        return None

    w_val = float(omega)
    work_y = np.empty(n, dtype=np.float64)

    def apply_into(rhs_vec: np.ndarray, out_vec: np.ndarray, add_to_out: bool) -> None:
        _apply_perm_line_smoother_into_nb(
            perm, l_sub, inv_d_fac, w_val, rhs_vec, out_vec, add_to_out, work_y
        )

    def apply_line(rhs_vec):
        out = np.empty(n, dtype=np.float64)
        _apply_perm_line_smoother_into_nb(
            perm,
            l_sub,
            inv_d_fac,
            w_val,
            np.asarray(rhs_vec, dtype=np.float64).ravel(),
            out,
            False,
            work_y,
        )
        return out

    apply_line._into = apply_into
    return apply_line


# ============================================================================
# 4. Algebraic Coarsening, Prolongation, Galerkin & Zero-Copy Cartesian Line Smoothers
# ============================================================================


@nb.njit(cache=True)
def _count_unique_nonneg_tags_nb(tags: np.ndarray) -> int:
    n = tags.size
    if n == 0:
        return 0
    mx = tags[0]
    mn = tags[0]
    for i in range(1, n):
        v = tags[i]
        if v > mx:
            mx = v
        elif v < mn:
            mn = v
    span = int(mx - mn) + 1
    if span <= max(4096, 2 * n):
        seen = np.zeros(span, dtype=np.uint8)
        cnt = 0
        for i in range(n):
            idx = int(tags[i] - mn)
            if seen[idx] == 0:
                seen[idx] = 1
                cnt += 1
        return cnt
    s = tags.copy()
    s.sort()
    cnt = 1
    for i in range(1, n):
        if s[i] != s[i - 1]:
            cnt += 1
    return cnt


@nb.njit(cache=True)
def _coarsen_algebraic_nb(
    n: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    diag: np.ndarray,
    has_ft: bool,
    flow_tags: np.ndarray,
    has_allow: bool,
    allow_mask: np.ndarray,
    max_group: int,
):
    best_nbr = np.empty(n, dtype=np.int32)
    any_valid_edge = False

    for i in range(n):
        s = indptr[i]
        e = indptr[i + 1]
        di = diag[i]
        fti = flow_tags[i] if has_ft else 0
        max_s = -1.0
        row_has = False
        for p in range(s, e):
            if has_allow and allow_mask[p] == 0:
                continue
            j = indices[p]
            val = data[p]
            if j != i and val < 0.0:
                if has_ft and flow_tags[j] != fti:
                    continue
                sc = (-val) / np.sqrt(di * diag[j])
                if sc > max_s:
                    max_s = sc
                row_has = True

        if not row_has:
            best_nbr[i] = i
        else:
            any_valid_edge = True
            thresh = max_s * (1.0 - 1.0e-15)
            min_c = n + 1
            for p in range(s, e):
                if has_allow and allow_mask[p] == 0:
                    continue
                j = indices[p]
                val = data[p]
                if j != i and val < 0.0:
                    if has_ft and flow_tags[j] != fti:
                        continue
                    sc = (-val) / np.sqrt(di * diag[j])
                    if sc >= thresh and j < min_c:
                        min_c = j
            best_nbr[i] = min_c if min_c <= n else i

    if not any_valid_edge:
        return False, np.empty(0, dtype=np.int64), 0

    uf = np.arange(n, dtype=np.int32)
    for i in range(n):
        nb_i = best_nbr[i]
        if nb_i != i:
            ru = _uf_find(uf, i)
            rv = _uf_find(uf, nb_i)
            if ru != rv:
                if ru < rv:
                    uf[rv] = ru
                else:
                    uf[ru] = rv

    parent = np.empty(n, dtype=np.int32)
    root_lbl = np.full(n, -1, dtype=np.int32)
    num_cc = 0
    for i in range(n):
        r = _uf_find(uf, i)
        lbl = root_lbl[r]
        if lbl < 0:
            lbl = num_cc
            root_lbl[r] = lbl
            num_cc += 1
        parent[i] = lbl

    cc_counts = np.zeros(num_cc, dtype=np.int32)
    for i in range(n):
        cc_counts[parent[i]] += 1
    cc_ptr = np.empty(num_cc + 1, dtype=np.int32)
    cc_ptr[0] = 0
    for c in range(num_cc):
        cc_ptr[c + 1] = cc_ptr[c] + cc_counts[c]
    order_p = np.empty(n, dtype=np.int32)
    cur = cc_ptr[:-1].copy()
    for i in range(n):
        c = parent[i]
        dst = cur[c]
        order_p[dst] = i
        cur[c] = dst + 1

    inv = np.empty(n, dtype=np.int64)
    nc = 0
    for c in range(num_cc):
        s = cc_ptr[c]
        e = cc_ptr[c + 1]
        pos_in_grp = 0
        for p in range(s, e):
            if pos_in_grp % max_group == 0:
                cid = nc
                nc += 1
            inv[order_p[p]] = cid
            pos_in_grp += 1

    return True, inv, nc


def _coarsen_algebraic(matrix, diag, flow_tags=None, allow_mask=None, max_group=8):
    n = matrix.shape[0]
    has_ft = flow_tags is not None
    ft_arr = np.asarray(flow_tags, dtype=np.int64) if has_ft else np.empty(0, dtype=np.int64)
    has_allow = allow_mask is not None
    al_arr = allow_mask if has_allow else np.empty(0, dtype=np.uint8)

    ok, inv, nc = _coarsen_algebraic_nb(
        n,
        matrix.indptr,
        matrix.indices,
        matrix.data,
        diag,
        has_ft,
        ft_arr,
        has_allow,
        al_arr,
        int(max_group),
    )
    if not ok:
        return _pair_consecutive_within_flow(flow_tags, n)

    c_ft = None
    if has_ft:
        c_ft = np.empty(nc, dtype=flow_tags.dtype)
        c_ft[inv] = flow_tags

    if nc > int(0.85 * n):
        agg_counts = np.bincount(inv, minlength=nc)
        singletons = np.flatnonzero(agg_counts == 1)
        min_possible = 1 if c_ft is None else int(_count_unique_nonneg_tags_nb(c_ft))
        if len(singletons) >= 2 and nc > min_possible:
            s_ft = None if c_ft is None else c_ft[singletons]
            s_inv, s_nc, _ = _pair_consecutive_within_flow(s_ft, len(singletons))
            non_sing = np.flatnonzero(agg_counts > 1)
            remap_agg = np.empty(nc, dtype=np.int64)
            remap_agg[non_sing] = np.arange(len(non_sing), dtype=np.int64)
            remap_agg[singletons] = len(non_sing) + s_inv
            inv = remap_agg[inv]
            nc = len(non_sing) + s_nc
            if has_ft:
                c_ft = np.empty(nc, dtype=flow_tags.dtype)
                c_ft[inv] = flow_tags

    return inv, nc, c_ft


@nb.njit(cache=True)
def _build_prolongation_nb(
    n_curr: int,
    nc: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    diag: np.ndarray,
    omega: float,
    di: np.ndarray,
    inv: np.ndarray,
    prolong_filter: float,
    has_allow_mask: bool,
    allow_mask: np.ndarray,
):
    use_w_branch = (prolong_filter > 0.0) or has_allow_mask
    if not use_w_branch:
        max_diag = 0.0
        for i in range(n_curr):
            if diag[i] > max_diag:
                max_diag = diag[i]
        tol_ground = 1.0e-11 * max_diag
        for i in range(n_curr):
            rsum_a = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                v = data[p]
                rsum_a += v
                if indices[p] != i and v > 0.0:
                    use_w_branch = True
                    break
            if use_w_branch or abs(rsum_a) > tol_ground:
                use_w_branch = True
                break

    max_p_nnz = indptr[n_curr] + n_curr
    p_indptr = np.empty(n_curr + 1, dtype=np.int32)
    p_indptr[0] = 0
    p_indices = np.empty(max_p_nnz, dtype=np.int32)
    p_data = np.empty(max_p_nnz, dtype=np.float64)

    marker = np.full(nc, -1, dtype=np.int32)
    acc = np.empty(nc, dtype=np.float64)
    row_cols_cap = 64
    row_cols = np.empty(row_cols_cap, dtype=np.int32)

    wpos = 0
    bad_row = False

    if use_w_branch:
        has_pf = prolong_filter > 0.0
        for i in range(n_curr):
            s = indptr[i]
            e = indptr[i + 1]
            max_deg = (e - s) + 1
            if max_deg > row_cols_cap:
                row_cols_cap = max(row_cols_cap * 2, max_deg)
                row_cols = np.empty(row_cols_cap, dtype=np.int32)

            thresh = prolong_filter * diag[i] if has_pf else 0.0
            df_i = 0.0
            rcnt = 0
            for p in range(s, e):
                if has_allow_mask and allow_mask[p] == 0:
                    continue
                j = indices[p]
                if j == i:
                    continue
                nv = -data[p]
                keep = (nv >= thresh) if has_pf else (nv > 0.0)
                if keep and nv != 0.0:
                    df_i += nv
                    cj = int(inv[j])
                    if marker[cj] != i:
                        marker[cj] = i
                        acc[cj] = nv
                        row_cols[rcnt] = cj
                        rcnt += 1
                    else:
                        acc[cj] += nv

            denom = diag[i] if diag[i] >= df_i else df_i
            alpha_i = omega / denom
            for k in range(rcnt):
                c = row_cols[k]
                acc[c] *= alpha_i

            ci = int(inv[i])
            z_diag = 1.0 - alpha_i * df_i
            if z_diag != 0.0:
                if marker[ci] != i:
                    marker[ci] = i
                    acc[ci] = z_diag
                    row_cols[rcnt] = ci
                    rcnt += 1
                else:
                    acc[ci] += z_diag

            if rcnt > 1:
                if rcnt <= 32:
                    for u in range(1, rcnt):
                        kc = row_cols[u]
                        v = u - 1
                        while v >= 0 and row_cols[v] > kc:
                            row_cols[v + 1] = row_cols[v]
                            v -= 1
                        row_cols[v + 1] = kc
                else:
                    row_cols[:rcnt].sort()

            row_start = wpos
            rsum = 0.0
            for k in range(rcnt):
                c = row_cols[k]
                val = acc[c]
                if val != 0.0:
                    p_indices[wpos] = c
                    p_data[wpos] = val
                    wpos += 1
                    rsum += val

            p_indptr[i + 1] = wpos
            if not np.isfinite(rsum) or rsum <= 1.0e-14:
                bad_row = True
                break
            inv_rsum = 1.0 / rsum
            for q in range(row_start, wpos):
                p_data[q] *= inv_rsum
    else:
        for i in range(n_curr):
            s = indptr[i]
            e = indptr[i + 1]
            max_deg = (e - s) + 1
            if max_deg > row_cols_cap:
                row_cols_cap = max(row_cols_cap * 2, max_deg)
                row_cols = np.empty(row_cols_cap, dtype=np.int32)

            rcnt = 0
            for p in range(s, e):
                j = indices[p]
                v = data[p]
                cj = int(inv[j])
                if marker[cj] != i:
                    marker[cj] = i
                    acc[cj] = v
                    row_cols[rcnt] = cj
                    rcnt += 1
                else:
                    acc[cj] += v

            ci = int(inv[i])
            if marker[ci] != i:
                marker[ci] = i
                acc[ci] = 0.0
                row_cols[rcnt] = ci
                rcnt += 1

            di_i = di[i]
            for k in range(rcnt):
                c = row_cols[k]
                az = acc[c] * di_i
                acc[c] = (1.0 - az) if c == ci else (-az)

            if rcnt > 1:
                if rcnt <= 32:
                    for u in range(1, rcnt):
                        kc = row_cols[u]
                        v = u - 1
                        while v >= 0 and row_cols[v] > kc:
                            row_cols[v + 1] = row_cols[v]
                            v -= 1
                        row_cols[v + 1] = kc
                else:
                    row_cols[:rcnt].sort()

            row_start = wpos
            rsum = 0.0
            for k in range(rcnt):
                c = row_cols[k]
                val = acc[c]
                if val != 0.0:
                    p_indices[wpos] = c
                    p_data[wpos] = val
                    wpos += 1
                    rsum += val

            p_indptr[i + 1] = wpos
            if not np.isfinite(rsum) or rsum <= 1.0e-14:
                bad_row = True
                break
            inv_rsum = 1.0 / rsum
            for q in range(row_start, wpos):
                p_data[q] *= inv_rsum

    return bad_row, p_indptr, p_indices[:wpos].copy(), p_data[:wpos].copy()


def _build_prolongation(matrix, diag, omega, di, inv, nc, prolong_filter, allow_mask=None):
    n_curr = matrix.shape[0]
    has_allow = allow_mask is not None
    al_arr = allow_mask if has_allow else np.empty(0, dtype=np.uint8)

    bad, p_indptr, p_indices, p_data = _build_prolongation_nb(
        n_curr,
        int(nc),
        matrix.indptr,
        matrix.indices,
        matrix.data,
        diag,
        float(omega),
        di,
        inv,
        float(prolong_filter),
        has_allow,
        al_arr,
    )
    if bad:
        raise ValueError("invalid prolongation row sum before normalization")

    P = sparse.csr_matrix((p_data, p_indices, p_indptr), shape=(n_curr, nc), copy=False)
    P._has_sorted_indices = True
    P._has_canonical_format = True

    nnz_p = p_data.size
    r_indptr = np.empty(nc + 1, dtype=np.int32)
    r_indices = np.empty(nnz_p, dtype=np.int32)
    r_data = np.empty(nnz_p, dtype=np.float64)
    _sparsetools.csr_tocsc(n_curr, nc, p_indptr, p_indices, p_data, r_indptr, r_indices, r_data)
    R = sparse.csr_matrix((r_data, r_indices, r_indptr), shape=(nc, n_curr), copy=False)
    R._has_sorted_indices = True
    R._has_canonical_format = True
    return P, R


@nb.njit(cache=True)
def _diag_and_bound_csr_nb(m: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray):
    diag = np.zeros(m, dtype=np.float64)
    bound = 0.0
    valid = True
    for i in range(m):
        d_i = 0.0
        abs_sum = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            v = data[p]
            abs_sum += abs(v)
            if indices[p] == i:
                d_i = v
        diag[i] = d_i
        if not np.isfinite(d_i) or d_i <= 0.0:
            valid = False
        else:
            r = abs_sum * (1.0 / d_i)
            if r > bound:
                bound = r
    return diag, bound, valid


@nb.njit(cache=True)
def _sym_filter_and_bound_nb(
    m: int,
    cs_indptr: np.ndarray,
    cs_indices: np.ndarray,
    cs_data: np.ndarray,
    ct_indptr: np.ndarray,
    ct_indices: np.ndarray,
    ct_data: np.ndarray,
    has_cutoff: bool,
    cutoff: float,
    grounded: bool,
):
    nnz = cs_indptr[m]
    strict_sym = True
    if ct_indptr[m] != nnz:
        strict_sym = False
    else:
        for i in range(m + 1):
            if cs_indptr[i] != ct_indptr[i]:
                strict_sym = False
                break
        if strict_sym:
            for p in range(nnz):
                if cs_indices[p] != ct_indices[p]:
                    strict_sym = False
                    break

    if strict_sym:
        s_indptr = cs_indptr
        s_indices = cs_indices
        s_data = cs_data
        for p in range(nnz):
            s_data[p] = 0.5 * (cs_data[p] + ct_data[p])
    else:
        s_indptr = np.empty(m + 1, dtype=np.int32)
        s_indptr[0] = 0
        for i in range(m):
            p1, e1 = cs_indptr[i], cs_indptr[i + 1]
            p2, e2 = ct_indptr[i], ct_indptr[i + 1]
            cnt = 0
            while p1 < e1 and p2 < e2:
                j1, j2 = cs_indices[p1], ct_indices[p2]
                if j1 == j2:
                    p1 += 1
                    p2 += 1
                elif j1 < j2:
                    p1 += 1
                else:
                    p2 += 1
                cnt += 1
            cnt += (e1 - p1) + (e2 - p2)
            s_indptr[i + 1] = s_indptr[i] + cnt

        tot = s_indptr[m]
        s_indices = np.empty(tot, dtype=np.int32)
        s_data = np.empty(tot, dtype=np.float64)
        for i in range(m):
            p1, e1 = cs_indptr[i], cs_indptr[i + 1]
            p2, e2 = ct_indptr[i], ct_indptr[i + 1]
            dst = s_indptr[i]
            while p1 < e1 and p2 < e2:
                j1, j2 = cs_indices[p1], ct_indices[p2]
                if j1 == j2:
                    s_indices[dst] = j1
                    s_data[dst] = 0.5 * (cs_data[p1] + ct_data[p2])
                    p1 += 1
                    p2 += 1
                elif j1 < j2:
                    s_indices[dst] = j1
                    s_data[dst] = 0.5 * cs_data[p1]
                    p1 += 1
                else:
                    s_indices[dst] = j2
                    s_data[dst] = 0.5 * ct_data[p2]
                    p2 += 1
                dst += 1
            while p1 < e1:
                s_indices[dst] = cs_indices[p1]
                s_data[dst] = 0.5 * cs_data[p1]
                p1 += 1
                dst += 1
            while p2 < e2:
                s_indices[dst] = ct_indices[p2]
                s_data[dst] = 0.5 * ct_data[p2]
                p2 += 1
                dst += 1

    galerkin_nnz = 0
    diag = np.zeros(m, dtype=np.float64)
    drift = np.zeros(m, dtype=np.float64)
    for i in range(m):
        rs = 0.0
        for p in range(s_indptr[i], s_indptr[i + 1]):
            v = s_data[p]
            if v != 0.0:
                galerkin_nnz += 1
            rs += v
            if s_indices[p] == i:
                diag[i] = v
        drift[i] = rs

    pos_count = 0
    removed = 0
    if has_cutoff:
        for i in range(m):
            di_val = diag[i]
            add = 0.0
            for p in range(s_indptr[i], s_indptr[i + 1]):
                j = s_indices[p]
                v = s_data[p]
                if j != i and v > 0.0:
                    pos_count += 1
                    if v <= cutoff * np.sqrt(di_val * diag[j]):
                        add += v
                        s_data[p] = 0.0
                        removed += 1
            diag[i] += add

    if not grounded:
        for i in range(m):
            diag[i] -= drift[i]

    out_indptr = np.empty(m + 1, dtype=np.int32)
    out_indptr[0] = 0
    write_pos = 0
    bound = 0.0
    valid = True

    for i in range(m):
        d_i = diag[i]
        abs_sum = 0.0
        s_i = s_indptr[i]
        e_i = s_indptr[i + 1]
        for p in range(s_i, e_i):
            j = s_indices[p]
            v = d_i if j == i else s_data[p]
            if v != 0.0:
                s_indices[write_pos] = j
                s_data[write_pos] = v
                write_pos += 1
                abs_sum += abs(v)
        out_indptr[i + 1] = write_pos
        if not np.isfinite(d_i) or d_i <= 0.0:
            valid = False
        else:
            r = abs_sum * (1.0 / d_i)
            if r > bound:
                bound = r

    return (
        out_indptr,
        s_indices[:write_pos].copy(),
        s_data[:write_pos].copy(),
        diag,
        bound,
        valid,
        galerkin_nnz,
        removed,
        pos_count,
    )


def _galerkin_coarse_and_lump(R, matrix, P, cutoff, grounded):
    m, k = R.shape
    n = P.shape[1]
    ap_indptr = np.empty(k + 1, dtype=np.int32)
    ap_nnz = _sparsetools.csr_matmat_maxnnz(
        k, n, matrix.indptr, matrix.indices, P.indptr, P.indices
    )
    ap_indices = np.empty(ap_nnz, dtype=np.int32)
    ap_data = np.empty(ap_nnz, dtype=np.float64)
    _sparsetools.csr_matmat(
        k,
        n,
        matrix.indptr,
        matrix.indices,
        matrix.data,
        P.indptr,
        P.indices,
        P.data,
        ap_indptr,
        ap_indices,
        ap_data,
    )

    c_indptr = np.empty(m + 1, dtype=np.int32)
    c_nnz = _sparsetools.csr_matmat_maxnnz(m, n, R.indptr, R.indices, ap_indptr, ap_indices)
    c_indices = np.empty(c_nnz, dtype=np.int32)
    c_data = np.empty(c_nnz, dtype=np.float64)
    _sparsetools.csr_matmat(
        m,
        n,
        R.indptr,
        R.indices,
        R.data,
        ap_indptr,
        ap_indices,
        ap_data,
        c_indptr,
        c_indices,
        c_data,
    )
    del ap_indptr, ap_indices, ap_data

    ct_indptr = np.empty(m + 1, dtype=np.int32)
    ct_indices = np.empty(c_nnz, dtype=np.int32)
    ct_data = np.empty(c_nnz, dtype=np.float64)
    _sparsetools.csr_tocsc(m, m, c_indptr, c_indices, c_data, ct_indptr, ct_indices, ct_data)
    _sparsetools.csr_tocsc(m, m, ct_indptr, ct_indices, ct_data, c_indptr, c_indices, c_data)

    has_cutoff = cutoff is not None
    cutoff_val = float(cutoff) if has_cutoff else 0.0
    (
        out_indptr,
        out_indices,
        out_data,
        next_diag,
        next_bound,
        _,
        galerkin_nnz,
        removed,
        pos_count,
    ) = _sym_filter_and_bound_nb(
        m,
        c_indptr,
        c_indices,
        c_data,
        ct_indptr,
        ct_indices,
        ct_data,
        has_cutoff,
        cutoff_val,
        bool(grounded),
    )
    coarse_mat = sparse.csr_matrix((out_data, out_indices, out_indptr), shape=(m, m), copy=False)
    coarse_mat._has_sorted_indices = True
    coarse_mat._has_canonical_format = True
    return coarse_mat, next_diag, next_bound, int(galerkin_nnz), int(removed), int(pos_count)


# ---- Zero-copy Cartesian 1D column tridiagonal LDL^T smoothers (axes 0, 1, 2) ----


@nb.njit(cache=True)
def _factor_tridiagonal_lines_nb(tz: np.ndarray, d2: np.ndarray):
    n_line = d2.shape[0]
    n_cols = d2.shape[1]
    l_sub = np.empty((n_line - 1, n_cols), dtype=np.float64)
    for kk in range(n_line - 1):
        for c in range(n_cols):
            dk = d2[kk, c]
            if dk <= 0.0 or not np.isfinite(dk):
                return False, l_sub, d2
            w_col = -tz[kk, c]
            mult = w_col / dk
            l_sub[kk, c] = mult
            d2[kk + 1, c] -= mult * w_col
    last = n_line - 1
    for c in range(n_cols):
        dl = d2[last, c]
        if dl <= 0.0 or not np.isfinite(dl):
            return False, l_sub, d2
    inv_d2 = np.empty_like(d2)
    for kk in range(n_line):
        for c in range(n_cols):
            inv_d2[kk, c] = 1.0 / d2[kk, c]
    return True, l_sub, inv_d2


@nb.njit(cache=True)
def _apply_cart_line_ax0_nb(
    l_sub: np.ndarray,
    inv_d2: np.ndarray,
    omega: float,
    rhs_vec: np.ndarray,
    out_vec: np.ndarray,
    add_to_out: bool,
    work_f: np.ndarray,
    prev_y: np.ndarray,
) -> None:
    """Two-pass fused sequential AVX2 row-streaming tridiagonal solve along axis 0 (bit-exact)."""
    n_line = inv_d2.shape[0]
    n_cols = inv_d2.shape[1]
    rhs2 = rhs_vec.reshape(n_line, n_cols)
    out2 = out_vec.reshape(n_line, n_cols)
    wf = work_f.reshape(n_line, n_cols) if add_to_out else out2
    for c in range(n_cols):
        wf[0, c] = rhs2[0, c]
    for kk in range(n_line - 1):
        for c in range(n_cols):
            wf[kk + 1, c] = rhs2[kk + 1, c] - l_sub[kk, c] * wf[kk, c]
    last = n_line - 1
    if add_to_out:
        for c in range(n_cols):
            yk = wf[last, c] * inv_d2[last, c]
            prev_y[c] = yk
            out2[last, c] += omega * yk
        for kk in range(n_line - 2, -1, -1):
            for c in range(n_cols):
                yk = wf[kk, c] * inv_d2[kk, c] - l_sub[kk, c] * prev_y[c]
                prev_y[c] = yk
                out2[kk, c] += omega * yk
    else:
        for c in range(n_cols):
            yk = wf[last, c] * inv_d2[last, c]
            prev_y[c] = yk
            out2[last, c] = omega * yk
        for kk in range(n_line - 2, -1, -1):
            for c in range(n_cols):
                yk = wf[kk, c] * inv_d2[kk, c] - l_sub[kk, c] * prev_y[c]
                prev_y[c] = yk
                out2[kk, c] = omega * yk


@nb.njit(cache=True)
def _factor_tridiagonal_ax1_nb(
    t_sec: np.ndarray, diag: np.ndarray, n0: int, n1: int, n2: int
):
    l_sub = np.empty((n0, n1 - 1, n2), dtype=np.float64)
    inv_d = np.empty((n0, n1, n2), dtype=np.float64)
    t3 = t_sec.reshape(n0, n1 - 1, n2)
    d3 = diag.reshape(n0, n1, n2)
    dcur = np.empty(n2, dtype=np.float64)
    for i0 in range(n0):
        for i2 in range(n2):
            dk = d3[i0, 0, i2]
            if dk <= 0.0 or not np.isfinite(dk):
                return False, l_sub, inv_d
            dcur[i2] = dk
            inv_d[i0, 0, i2] = 1.0 / dk
        for kk in range(n1 - 1):
            for i2 in range(n2):
                dk = dcur[i2]
                w_col = -t3[i0, kk, i2]
                mult = w_col / dk
                l_sub[i0, kk, i2] = mult
                dnxt = d3[i0, kk + 1, i2] - mult * w_col
                if dnxt <= 0.0 or not np.isfinite(dnxt):
                    return False, l_sub, inv_d
                dcur[i2] = dnxt
                inv_d[i0, kk + 1, i2] = 1.0 / dnxt
    return True, l_sub, inv_d


@nb.njit(cache=True)
def _apply_cart_line_ax1_nb(
    l_sub: np.ndarray,
    inv_d: np.ndarray,
    omega: float,
    rhs_vec: np.ndarray,
    out_vec: np.ndarray,
    add_to_out: bool,
    n0: int,
    n1: int,
    n2: int,
) -> None:
    """Zero-transpose tiled tridiagonal solve along middle axis 1."""
    rhs3 = rhs_vec.reshape(n0, n1, n2)
    out3 = out_vec.reshape(n0, n1, n2)
    tile = 64
    buf = np.empty((n1, tile), dtype=np.float64)
    for i0 in range(n0):
        for c0 in range(0, n2, tile):
            cw = n2 - c0
            if cw > tile:
                cw = tile
            for c in range(cw):
                buf[0, c] = rhs3[i0, 0, c0 + c]
            for kk in range(n1 - 1):
                for c in range(cw):
                    buf[kk + 1, c] = rhs3[i0, kk + 1, c0 + c] - l_sub[i0, kk, c0 + c] * buf[kk, c]
            for kk in range(n1):
                for c in range(cw):
                    buf[kk, c] *= inv_d[i0, kk, c0 + c]
            for kk in range(n1 - 2, -1, -1):
                for c in range(cw):
                    buf[kk, c] -= l_sub[i0, kk, c0 + c] * buf[kk + 1, c]
            if add_to_out:
                for kk in range(n1):
                    for c in range(cw):
                        out3[i0, kk, c0 + c] += omega * buf[kk, c]
            else:
                for kk in range(n1):
                    for c in range(cw):
                        out3[i0, kk, c0 + c] = omega * buf[kk, c]


@nb.njit(cache=True)
def _factor_tridiagonal_ax2_nb(
    t_sec: np.ndarray, diag: np.ndarray, n_batch: int, n_line: int
):
    l_sub = np.empty(n_batch * (n_line - 1), dtype=np.float64)
    inv_d = np.empty(n_batch * n_line, dtype=np.float64)
    for b in range(n_batch):
        dbase = b * n_line
        tbase = b * (n_line - 1)
        dk = diag[dbase]
        if dk <= 0.0 or not np.isfinite(dk):
            return False, l_sub, inv_d
        inv_d[dbase] = 1.0 / dk
        for kk in range(n_line - 1):
            w_col = -t_sec[tbase + kk]
            mult = w_col / dk
            l_sub[tbase + kk] = mult
            dk = diag[dbase + kk + 1] - mult * w_col
            if dk <= 0.0 or not np.isfinite(dk):
                return False, l_sub, inv_d
            inv_d[dbase + kk + 1] = 1.0 / dk
    return True, l_sub, inv_d


@nb.njit(cache=True)
def _apply_cart_line_ax2_nb(
    l_sub: np.ndarray,
    inv_d: np.ndarray,
    omega: float,
    rhs_vec: np.ndarray,
    out_vec: np.ndarray,
    add_to_out: bool,
    n_batch: int,
    n_line: int,
) -> None:
    """Zero-transpose contiguous column sweep along innermost axis 2 (IEEE-754 bit-exact)."""
    buf = np.empty(n_line, dtype=np.float64)
    for b in range(n_batch):
        dbase = b * n_line
        tbase = b * (n_line - 1)
        buf[0] = rhs_vec[dbase]
        for kk in range(n_line - 1):
            buf[kk + 1] = rhs_vec[dbase + kk + 1] - l_sub[tbase + kk] * buf[kk]
        for kk in range(n_line):
            buf[kk] *= inv_d[dbase + kk]
        for kk in range(n_line - 2, -1, -1):
            buf[kk] -= l_sub[tbase + kk] * buf[kk + 1]
        if add_to_out:
            for kk in range(n_line):
                out_vec[dbase + kk] += omega * buf[kk]
        else:
            for kk in range(n_line):
                out_vec[dbase + kk] = omega * buf[kk]


def _build_anisotropic_line_smoother(shape, edge, diag, omega=2.0 / 3.0, forced_axis=None):
    """Build exact symmetric zero-transpose 1D line-Jacobi LDL^T smoother if one axis >= 4x."""
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
        raise ValueError("forced line axis must be 0, 1 or 2")
    other_max = max(medians[i] for i in range(3) if i != dom_ax)
    if forced_axis is None and medians[dom_ax] < 4.0 * max(other_max, np.finfo(np.float64).tiny):
        return None

    n0, n1, n2 = map(int, shape)
    total = n0 * n1 * n2
    n_line = shape[dom_ax]
    n_cols = total // n_line
    if n_cols < 2:
        return None

    t_sec = np.ascontiguousarray(trans[sections[dom_ax]], dtype=np.float64)
    diag_arr = np.ascontiguousarray(diag, dtype=np.float64)
    w_val = float(omega)

    if dom_ax == 0:
        tz = t_sec.reshape(n0 - 1, n_cols)
        d2 = diag_arr.reshape(n0, n_cols).copy()
        ok, l_sub, inv_d2 = _factor_tridiagonal_lines_nb(tz, d2)
        if not ok:
            return None
        work_f = np.empty(total, dtype=np.float64)
        prev_y = np.empty(n_cols, dtype=np.float64)

        def apply_into(rhs_vec: np.ndarray, out_vec: np.ndarray, add_to_out: bool) -> None:
            _apply_cart_line_ax0_nb(
                l_sub, inv_d2, w_val, rhs_vec, out_vec, add_to_out, work_f, prev_y
            )

        def apply_line(rhs_vec):
            out = np.empty(total, dtype=np.float64)
            _apply_cart_line_ax0_nb(
                l_sub,
                inv_d2,
                w_val,
                np.asarray(rhs_vec, dtype=np.float64).ravel(),
                out,
                False,
                work_f,
                prev_y,
            )
            return out

        apply_line._into = apply_into
        return apply_line

    if dom_ax == 2:
        ok, l_sub, inv_d = _factor_tridiagonal_ax2_nb(t_sec, diag_arr, n_cols, n2)
        if not ok:
            return None

        def apply_into(rhs_vec: np.ndarray, out_vec: np.ndarray, add_to_out: bool) -> None:
            _apply_cart_line_ax2_nb(l_sub, inv_d, w_val, rhs_vec, out_vec, add_to_out, n_cols, n2)

        def apply_line(rhs_vec):
            out = np.empty(total, dtype=np.float64)
            _apply_cart_line_ax2_nb(
                l_sub,
                inv_d,
                w_val,
                np.asarray(rhs_vec, dtype=np.float64).ravel(),
                out,
                False,
                n_cols,
                n2,
            )
            return out

        apply_line._into = apply_into
        return apply_line

    # dom_ax == 1
    ok, l_sub, inv_d = _factor_tridiagonal_ax1_nb(t_sec, diag_arr, n0, n1, n2)
    if not ok:
        return None

    def apply_into(rhs_vec: np.ndarray, out_vec: np.ndarray, add_to_out: bool) -> None:
        _apply_cart_line_ax1_nb(l_sub, inv_d, w_val, rhs_vec, out_vec, add_to_out, n0, n1, n2)

    def apply_line(rhs_vec):
        out = np.empty(total, dtype=np.float64)
        _apply_cart_line_ax1_nb(
            l_sub,
            inv_d,
            w_val,
            np.asarray(rhs_vec, dtype=np.float64).ravel(),
            out,
            False,
            n0,
            n1,
            n2,
        )
        return out

    apply_line._into = apply_into
    return apply_line


# ---- Optional @njit Symmetric Gauss-Seidel Smoother ----


@nb.njit(cache=True)
def _find_csr_diag_ptrs_nb(n: int, indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    diag_ptr = np.empty(n, dtype=np.int32)
    for i in range(n):
        lo = indptr[i]
        hi = indptr[i + 1]
        found = lo
        while lo < hi:
            mid = (lo + hi) >> 1
            c = indices[mid]
            if c < i:
                lo = mid + 1
            elif c > i:
                hi = mid
            else:
                found = mid
                break
        diag_ptr[i] = found
    return diag_ptr


@nb.njit(cache=True)
def _apply_sgs_csr_into_nb(
    n: int,
    indptr: np.ndarray,
    diag_ptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    inv_diag: np.ndarray,
    rhs: np.ndarray,
    out: np.ndarray,
    add_to_out: bool,
    dz_buf: np.ndarray,
    x_tmp: np.ndarray,
) -> None:
    """Exact zero-parameter SPD Symmetric Gauss-Seidel W^-1 rhs, W = (D+L) D^-1 (D+L^T)."""
    target = x_tmp if add_to_out else out
    for i in range(n):
        s = rhs[i]
        dp = diag_ptr[i]
        for p in range(indptr[i], dp):
            s -= data[p] * target[indices[p]]
        dz_buf[i] = s
        target[i] = s * inv_diag[i]

    for i in range(n - 1, -1, -1):
        s = dz_buf[i]
        for p in range(diag_ptr[i] + 1, indptr[i + 1]):
            s -= data[p] * target[indices[p]]
        target[i] = s * inv_diag[i]

    if add_to_out:
        for i in range(n):
            out[i] += target[i]


def _build_sgs_smoother(matrix: sparse.csr_matrix, diag: np.ndarray):
    if not matrix.has_sorted_indices:
        matrix.sort_indices()
    n = matrix.shape[0]
    diag_ptr = _find_csr_diag_ptrs_nb(n, matrix.indptr, matrix.indices)
    inv_diag = np.ascontiguousarray(1.0 / diag, dtype=np.float64)
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    dz_buf = np.empty(n, dtype=np.float64)
    x_tmp = np.empty(n, dtype=np.float64)

    def apply_into(rhs_vec: np.ndarray, out_vec: np.ndarray, add_to_out: bool) -> None:
        _apply_sgs_csr_into_nb(
            n, indptr, diag_ptr, indices, data, inv_diag, rhs_vec, out_vec, add_to_out, dz_buf, x_tmp
        )

    def apply_sgs(rhs_vec):
        out = np.empty(n, dtype=np.float64)
        _apply_sgs_csr_into_nb(
            n,
            indptr,
            diag_ptr,
            indices,
            data,
            inv_diag,
            np.asarray(rhs_vec, dtype=np.float64).ravel(),
            out,
            False,
            dz_buf,
            x_tmp,
        )
        return out

    apply_sgs._into = apply_into
    return apply_sgs


# ============================================================================
# 5. Zero-Allocation Fused @njit CSR V-Cycle Kernels & Hierarchy Construction
# ============================================================================


@nb.njit(cache=True)
def _vec_mul_into_nb(n: int, a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    for i in range(n):
        out[i] = a[i] * b[i]


@nb.njit(cache=True)
def _sub_scalar_inplace_nb(n: int, vec: np.ndarray, val: float) -> None:
    for i in range(n):
        vec[i] -= val


@nb.njit(cache=True)
def _copy_minus_scalar_nb(n: int, src: np.ndarray, val: float, dst: np.ndarray) -> None:
    for i in range(n):
        dst[i] = src[i] - val


@nb.njit(cache=True)
def _csr_residual_into_nb(
    n: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    rhs: np.ndarray,
    x: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fused out = rhs - A @ x in exact CSR row accumulation order."""
    for i in range(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * x[indices[p]]
        out[i] = rhs[i] - s


@nb.njit(cache=True)
def _csr_jacobi_sweep_nb(
    n: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    di: np.ndarray,
    rhs: np.ndarray,
    x: np.ndarray,
    tmp: np.ndarray,
) -> None:
    """Fused x += di * (rhs - A @ x) in exact CSR row accumulation order without heap alloc."""
    for i in range(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * x[indices[p]]
        tmp[i] = di[i] * (rhs[i] - s)
    for i in range(n):
        x[i] += tmp[i]


@nb.njit(cache=True)
def _csr_matvec_into_nb(
    m: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    x: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fused out = R @ x in exact CSR row accumulation order."""
    for i in range(m):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * x[indices[p]]
        out[i] = s


@nb.njit(cache=True)
def _csr_add_matvec_inplace_nb(
    n: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    e_c: np.ndarray,
    x: np.ndarray,
) -> None:
    """Fused x += P @ e_c in exact CSR row accumulation order."""
    for i in range(n):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * e_c[indices[p]]
        x[i] += s


@nb.njit(cache=True)
def _build_semi_axis_prolong_mask_nb(
    n: int,
    xyz: np.ndarray,
    ax_a: int,
    ax_b: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    has_allow: bool,
    allow_mask: np.ndarray,
) -> np.ndarray:
    """Restrict 1D semi-coarsening prolongation smoothing strictly along the coarsened pillar."""
    out = np.zeros(indices.size, dtype=np.uint8)
    for u in range(n):
        xa = xyz[u, ax_a]
        xb = xyz[u, ax_b]
        for p in range(indptr[u], indptr[u + 1]):
            if has_allow and allow_mask[p] == 0:
                continue
            v = indices[p]
            if v != u and data[p] < 0.0:
                if xyz[v, ax_a] == xa and xyz[v, ax_b] == xb:
                    out[p] = 1
    return out


def _make_cycle_callable(levels, *, grounded=False, cycle_repeats=1, fine_A=None):
    """Return fixed symmetric zero-temporary V-cycle matvec callable (bit-exact CSR matvec)."""
    n_levels = len(levels)
    x_bufs = [np.empty(lev["A"].shape[0], dtype=np.float64) for lev in levels]
    res_bufs = [np.empty(lev["A"].shape[0], dtype=np.float64) for lev in levels]
    rhs_bufs = [np.empty(lev["A"].shape[0], dtype=np.float64) for lev in levels]
    ax_bufs = [np.empty(lev["A"].shape[0], dtype=np.float64) for lev in levels]

    def _residual_into(
        n_curr: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        rhs: np.ndarray,
        x: np.ndarray,
        ax_b: np.ndarray,
        out: np.ndarray,
    ) -> None:
        ax_b.fill(0.0)
        _sparsetools.csr_matvec(n_curr, n_curr, indptr, indices, data, x, ax_b)
        np.subtract(rhs, ax_b, out=out)

    def _jacobi_sweep(
        n_curr: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        di: np.ndarray,
        rhs: np.ndarray,
        x: np.ndarray,
        ax_b: np.ndarray,
        rb: np.ndarray,
    ) -> None:
        ax_b.fill(0.0)
        _sparsetools.csr_matvec(n_curr, n_curr, indptr, indices, data, x, ax_b)
        np.subtract(rhs, ax_b, out=rb)
        np.multiply(di, rb, out=rb)
        x += rb

    if grounded:
        def _vcycle_spd_buf(index: int, rhs_in: np.ndarray) -> np.ndarray:
            lev = levels[index]
            xb = x_bufs[index]
            if "factor" in lev:
                xb[:] = lev["factor"].solve(rhs_in)
                return xb

            mat = lev["A"]
            n_curr = mat.shape[0]
            indptr, indices, data = mat.indptr, mat.indices, mat.data
            sw = lev["sweeps"]
            rb = res_bufs[index]
            axb = ax_bufs[index]
            sm_fn = lev.get("line_smoother") or lev.get("sgs_smoother")

            if sm_fn is not None:
                sm_into = getattr(sm_fn, "_into", None)
                if sm_into is not None:
                    sm_into(rhs_in, xb, False)
                    for _ in range(sw - 1):
                        _residual_into(n_curr, indptr, indices, data, rhs_in, xb, axb, rb)
                        sm_into(rb, xb, True)
                else:
                    xb[:] = sm_fn(rhs_in)
                    for _ in range(sw - 1):
                        _residual_into(n_curr, indptr, indices, data, rhs_in, xb, axb, rb)
                        xb += sm_fn(rb)
            else:
                di = lev["di"]
                np.multiply(di, rhs_in, out=xb)
                for _ in range(sw - 1):
                    _jacobi_sweep(n_curr, indptr, indices, data, di, rhs_in, xb, axb, rb)

            _residual_into(n_curr, indptr, indices, data, rhs_in, xb, axb, rb)
            R = lev["R"]
            rc_buf = rhs_bufs[index + 1]
            rc_buf.fill(0.0)
            _sparsetools.csr_matvec(R.shape[0], R.shape[1], R.indptr, R.indices, R.data, rb, rc_buf)
            ec_buf = _vcycle_spd_buf(index + 1, rc_buf)
            P = lev["P"]
            axb.fill(0.0)
            _sparsetools.csr_matvec(P.shape[0], P.shape[1], P.indptr, P.indices, P.data, ec_buf, axb)
            xb += axb

            if sm_fn is not None:
                sm_into = getattr(sm_fn, "_into", None)
                if sm_into is not None:
                    for _ in range(sw):
                        _residual_into(n_curr, indptr, indices, data, rhs_in, xb, axb, rb)
                        sm_into(rb, xb, True)
                else:
                    for _ in range(sw):
                        _residual_into(n_curr, indptr, indices, data, rhs_in, xb, axb, rb)
                        xb += sm_fn(rb)
            else:
                di = lev["di"]
                for _ in range(sw):
                    _jacobi_sweep(n_curr, indptr, indices, data, di, rhs_in, xb, axb, rb)
            return xb

        def _vcycle_spd(index, rhs):
            return _vcycle_spd_buf(index, np.asarray(rhs, dtype=np.float64).ravel()).copy()

        def apply_spd(rhs):
            rhs_arr = np.asarray(rhs, dtype=np.float64).ravel()
            out = _vcycle_spd_buf(0, rhs_arr).copy()
            if cycle_repeats == 2:
                fa = fine_A if fine_A is not None else levels[0]["A"]
                _residual_into(
                    fa.shape[0], fa.indptr, fa.indices, fa.data, rhs_arr, out, ax_bufs[0], rhs_bufs[0]
                )
                out += _vcycle_spd_buf(0, rhs_bufs[0])
            return out

        return apply_spd, _vcycle_spd

    def _vcycle_neumann_buf(index: int, rhs_in: np.ndarray) -> np.ndarray:
        lev = levels[index]
        xb = x_bufs[index]
        n_curr = xb.size
        rhs_b = rhs_bufs[index]
        np.subtract(rhs_in, float(np.mean(rhs_in)), out=rhs_b)

        if "factor" in lev:
            xb.fill(0.0)
            if lev["factor"] is not None:
                xb[1:] = lev["factor"].solve(rhs_b[1:])
            xb -= float(np.mean(xb))
            return xb

        mat = lev["A"]
        indptr, indices, data = mat.indptr, mat.indices, mat.data
        sw = lev["sweeps"]
        rb = res_bufs[index]
        axb = ax_bufs[index]
        sm_fn = lev.get("line_smoother") or lev.get("sgs_smoother")

        if sm_fn is not None:
            sm_into = getattr(sm_fn, "_into", None)
            if sm_into is not None:
                sm_into(rhs_b, xb, False)
                for _ in range(sw - 1):
                    _residual_into(n_curr, indptr, indices, data, rhs_b, xb, axb, rb)
                    sm_into(rb, xb, True)
            else:
                xb[:] = sm_fn(rhs_b)
                for _ in range(sw - 1):
                    _residual_into(n_curr, indptr, indices, data, rhs_b, xb, axb, rb)
                    xb += sm_fn(rb)
        else:
            di = lev["di"]
            np.multiply(di, rhs_b, out=xb)
            for _ in range(sw - 1):
                _jacobi_sweep(n_curr, indptr, indices, data, di, rhs_b, xb, axb, rb)

        _residual_into(n_curr, indptr, indices, data, rhs_b, xb, axb, rb)
        rb -= float(np.mean(rb))
        R = lev["R"]
        rc_next = res_bufs[index + 1] if (index + 1 < n_levels) else np.empty(R.shape[0], dtype=np.float64)
        rc_next.fill(0.0)
        _sparsetools.csr_matvec(R.shape[0], R.shape[1], R.indptr, R.indices, R.data, rb, rc_next)
        ec_buf = _vcycle_neumann_buf(index + 1, rc_next)
        P = lev["P"]
        axb.fill(0.0)
        _sparsetools.csr_matvec(P.shape[0], P.shape[1], P.indptr, P.indices, P.data, ec_buf, axb)
        xb += axb

        if sm_fn is not None:
            sm_into = getattr(sm_fn, "_into", None)
            if sm_into is not None:
                for _ in range(sw):
                    _residual_into(n_curr, indptr, indices, data, rhs_b, xb, axb, rb)
                    sm_into(rb, xb, True)
            else:
                for _ in range(sw):
                    _residual_into(n_curr, indptr, indices, data, rhs_b, xb, axb, rb)
                    xb += sm_fn(rb)
        else:
            di = lev["di"]
            for _ in range(sw):
                _jacobi_sweep(n_curr, indptr, indices, data, di, rhs_b, xb, axb, rb)

        xb -= float(np.mean(xb))
        return xb

    def _vcycle_neumann(index, rhs):
        return _vcycle_neumann_buf(index, np.asarray(rhs, dtype=np.float64).ravel()).copy()

    def apply_neumann(rhs):
        rhs_arr = np.asarray(rhs, dtype=np.float64).ravel()
        out = _vcycle_neumann_buf(0, rhs_arr).copy()
        if cycle_repeats == 2:
            fa = fine_A if fine_A is not None else levels[0]["A"]
            _residual_into(
                fa.shape[0], fa.indptr, fa.indices, fa.data, rhs_arr, out, ax_bufs[0], res_bufs[0]
            )
            out += _vcycle_neumann_buf(0, res_bufs[0])
            out -= float(np.mean(out))
        return out

    return apply_neumann, _vcycle_neumann


def _construct_hierarchy(
    A,
    xyz,
    tags,
    cfg,
    *,
    grounded=False,
    flow_tags=None,
    cartesian_fine=True,
    cartesian_shape=None,
    precomputed_diag=None,
    precomputed_bound=None,
    initial_tag_count=None,
    initial_allow_mask=None,
    split_boxes_by_connectivity=True,
    enable_semi_coarsening=True,
    smoother_scheme=DEFAULT_SMOOTHER_SCHEME,
    rank_safe_damping=False,
    verify_coarse_spd=False,
    binary_exact=False,
    deadline=None,
):
    """Build anisotropic semi-coarsened + barrier-respecting symmetric Galerkin hierarchy."""
    levels = []
    sizes, nnzs, pnnzs = [], [], []
    galerkin_nnzs, lumped_counts, positive_counts = [], [], []
    sweeps_per_level, tag_counts, modes_used = [], [], []
    geom_schedule = []

    matrix = A.tocsr()
    if not matrix.has_sorted_indices:
        matrix.sort_indices()

    if xyz is None and cartesian_shape is not None and tags is not None:
        curr_xyz = _make_cartesian_xyz_nb(
            int(cartesian_shape[0]), int(cartesian_shape[1]), int(cartesian_shape[2])
        )
    else:
        curr_xyz = None if xyz is None else np.ascontiguousarray(xyz, dtype=np.int64)

    curr_tags = tags
    curr_ft = flow_tags
    curr_allow = initial_allow_mask
    alg_mode = curr_xyz is None

    coarse_size = cfg["coarse_size"]
    max_direct = cfg["max_direct"]
    stall_ratio = cfg["stall_ratio"]
    semi_stall_ratio = float(stall_ratio) ** (1.0 / 3.0)
    pf = cfg["prolong_filter"]
    pf_start = cfg["prolong_filter_start_level"]
    max_group = cfg["max_alg_group"]

    cached_diag = precomputed_diag
    cached_bound = precomputed_bound
    semi_phase_active = bool(enable_semi_coarsening)

    while True:
        _check_deadline(deadline, "deadline during multigrid hierarchy construction")
        matrix = matrix.tocsr()
        if not matrix.has_sorted_indices:
            matrix.sort_indices()
        n_curr = matrix.shape[0]

        if cached_diag is not None and cached_bound is not None:
            diag = cached_diag
            bound = float(cached_bound)
            diag_ok = bool(np.all(np.isfinite(diag)) and np.all(diag > 0.0))
            cached_diag = None
            cached_bound = None
        else:
            diag, bound, diag_ok = _diag_and_bound_csr_nb(
                n_curr, matrix.indptr, matrix.indices, matrix.data
            )

        if (grounded or n_curr > 1) and not diag_ok:
            raise ValueError("nonpositive or nonfinite multigrid diagonal")

        if n_curr == 1 and not grounded:
            omega = 0.0
            di = np.zeros(1, dtype=np.float64)
            bound = 0.0
        else:
            inv_d = 1.0 / diag
            if not levels and cartesian_fine and not grounded and not rank_safe_damping:
                omega = 2.0 / 3.0
            elif rank_safe_damping:
                omega = min(2.0 / 3.0, 0.9 / bound)
            else:
                omega = min(2.0 / 3.0, 4.0 / (3.0 * bound))
            di = omega * inv_d

        sweeps = cfg["fine_sweeps"] if not levels else cfg["coarse_sweeps"]
        level = {"A": matrix, "di": di, "sweeps": sweeps, "omega": omega, "bound": bound}
        levels.append(level)
        sizes.append(int(n_curr))
        nnzs.append(int(matrix.nnz))
        sweeps_per_level.append(int(sweeps))

        if len(levels) == 1 and initial_tag_count is not None:
            tag_counts.append(int(initial_tag_count))
        elif curr_tags is not None and len(curr_tags) == n_curr:
            tag_counts.append(int(_count_unique_nonneg_tags_nb(curr_tags)))
        else:
            tag_counts.append(1 if curr_ft is None else int(_count_unique_nonneg_tags_nb(curr_ft)))

        if n_curr <= coarse_size:
            level["factor"] = _factor_coarsest(
                matrix, grounded, max_direct, verify_spd=verify_coarse_spd, prefer_splu=binary_exact
            )
            break

        if smoother_scheme == "sgs" or (smoother_scheme == "line_plus_sgs" and len(levels) > 1):
            level["sgs_smoother"] = _build_sgs_smoother(matrix, diag)

        step_mode = "geometric"
        accepted_geom = False
        semi_prolong_mask = None

        if not alg_mode and curr_xyz is not None:
            mx0, mx1, mx2, sp0, sp1, sp2 = _xyz_spans_and_max_nb(curr_xyz)
            if max(sp0, sp1, sp2) > 0:
                has_al = curr_allow is not None
                al_arr = curr_allow if has_al else np.empty(0, dtype=np.uint8)

                if semi_phase_active:
                    dom_ax, _, aniso_ratio = _detect_dominant_axis_csr(
                        curr_xyz, matrix, ratio_threshold=4.0
                    )
                    if dom_ax is not None:
                        sh0 = 1 if dom_ax == 0 else 0
                        sh1 = 1 if dom_ax == 1 else 0
                        sh2 = 1 if dom_ax == 2 else 0
                        s0 = int(mx0 >> sh0) + 1
                        s1 = int(mx1 >> sh1) + 1
                        s2 = int(mx2 >> sh2) + 1
                        if s0 * s1 * s2 <= max(65536, 16 * n_curr):
                            inv_s, nxyz_s, ntags_s = _aggregate_geom_connected_nb(
                                curr_xyz,
                                curr_tags,
                                sh0,
                                sh1,
                                sh2,
                                s0,
                                s1,
                                s2,
                                bool(split_boxes_by_connectivity),
                                matrix.indptr,
                                matrix.indices,
                                matrix.data,
                                has_al,
                                al_arr,
                            )
                            nc_s = int(len(ntags_s))
                            if nc_s < n_curr and (
                                nc_s <= coarse_size or nc_s <= int(semi_stall_ratio * n_curr)
                            ):
                                inv, next_xyz, next_tags, nc = inv_s, nxyz_s, ntags_s, nc_s
                                accepted_geom = True
                                orth = [ax for ax in range(3) if ax != dom_ax]
                                semi_prolong_mask = _build_semi_axis_prolong_mask_nb(
                                    n_curr,
                                    curr_xyz,
                                    orth[0],
                                    orth[1],
                                    matrix.indptr,
                                    matrix.indices,
                                    matrix.data,
                                    has_al,
                                    al_arr,
                                )
                                geom_schedule.append(f"semi_axis_{dom_ax}_r{aniso_ratio:.2f}")
                    if not accepted_geom:
                        semi_phase_active = False

                if not accepted_geom:
                    s0 = int(mx0 >> 1) + 1
                    s1 = int(mx1 >> 1) + 1
                    s2 = int(mx2 >> 1) + 1
                    if s0 * s1 * s2 <= max(65536, 16 * n_curr):
                        inv_f, nxyz_f, ntags_f = _aggregate_geom_connected_nb(
                            curr_xyz,
                            curr_tags,
                            1,
                            1,
                            1,
                            s0,
                            s1,
                            s2,
                            bool(split_boxes_by_connectivity),
                            matrix.indptr,
                            matrix.indices,
                            matrix.data,
                            has_al,
                            al_arr,
                        )
                        nc_f = int(len(ntags_f))
                        if nc_f < n_curr and (
                            nc_f <= coarse_size or nc_f <= int(stall_ratio * n_curr)
                        ):
                            inv, next_xyz, next_tags, nc = inv_f, nxyz_f, ntags_f, nc_f
                            accepted_geom = True
                            geom_schedule.append("full_2x2x2")

            if accepted_geom:
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
            step_mode = "algebraic"
            geom_schedule.append("algebraic_heavy_edge")
            inv, nc, curr_ft = _coarsen_algebraic(
                matrix, diag, flow_tags=curr_ft, allow_mask=curr_allow, max_group=max_group
            )
            if nc >= n_curr or len(levels) >= 20:
                level.pop("sgs_smoother", None)
                level["factor"] = _factor_coarsest(
                    matrix, grounded, max_direct, verify_spd=verify_coarse_spd
                )
                break

        modes_used.append(step_mode)
        cur_lvl_idx = len(levels) - 1
        eff_pf = pf if cur_lvl_idx >= pf_start else 0.0
        po = cfg["prolong_omega"]
        prolong_mask = semi_prolong_mask if semi_prolong_mask is not None else curr_allow
        P, R = _build_prolongation(
            matrix, diag, po, po / diag, inv, nc, eff_pf, allow_mask=prolong_mask
        )
        level["P"], level["R"] = P, R
        pnnzs.append(int(P.nnz))

        prev_matrix = matrix
        prev_allow = curr_allow
        cutoff = cfg["lump_max_relative"] if len(levels) >= cfg["lump_start_level"] else None
        (
            matrix,
            cached_diag,
            cached_bound,
            gnnz,
            removed,
            positives,
        ) = _galerkin_coarse_and_lump(R, prev_matrix, P, cutoff, grounded)
        galerkin_nnzs.append(gnnz)
        lumped_counts.append(removed)
        positive_counts.append(positives)

        if prev_allow is not None and nc > coarse_size:
            curr_allow = _project_allow_mask_to_coarse_nb(
                prev_matrix.shape[0],
                int(nc),
                prev_matrix.indptr,
                prev_matrix.indices,
                prev_allow,
                inv,
                matrix.indptr,
                matrix.indices,
                matrix.data,
            )
        else:
            curr_allow = None

    meta = {
        "level_sizes": sizes,
        "level_nnz": nnzs,
        "interpolation_nnz": pnnzs,
        "galerkin_nnz_before_lump": galerkin_nnzs,
        "positive_offdiagonals_lumped": lumped_counts,
        "positive_offdiagonals_before_filter": positive_counts,
        "smoothing_steps_per_level_each_side": sweeps_per_level,
        "component_tag_count_per_level": tag_counts,
        "coarsening_modes": modes_used,
        "geometric_schedule": geom_schedule,
        "smoother_scheme": smoother_scheme,
        "operator_complexity": float(sum(nnzs) / max(1, nnzs[0])),
    }
    return levels, meta


# ============================================================================
# 6. Public Solver Builders (100% Fair Shared Base Across All Internal Methods)
# ============================================================================


def assemble_cartesian_fast(shape, edge, d=None):
    """Sorted direct 7-point CSR assembly for arbitrary 3D rectangular shape."""
    nx, ny, nz = map(int, shape)
    tail = np.asarray(edge[0])
    head = np.asarray(edge[1])
    trans = np.asarray(edge[2], dtype=np.float64)
    has_d = d is not None
    d_arr = np.asarray(d, dtype=np.float64) if has_d else np.empty(0, dtype=np.float64)
    dummy_k = np.ones(nx * ny * nz, dtype=np.float64)
    err, indptr, indices, data, _, _ = _validate_and_assemble_cartesian_nb(
        nx, ny, nz, dummy_k, tail, head, trans, has_d, d_arr
    )
    if err != 0:
        mat = _assemble_general_edges(nx * ny * nz, (tail, head, trans), well_diagonal=d)
        return mat
    mat = sparse.csr_matrix((data, indices, indptr), shape=(nx * ny * nz, nx * ny * nz), copy=False)
    mat._has_sorted_indices = True
    mat._has_canonical_format = True
    return mat


def _assemble_general_edges(size, edge, well_diagonal=None):
    """Assemble symmetric Laplacian (plus optional nonnegative grounding d) from edges."""
    tail = np.asarray(edge[0])
    head = np.asarray(edge[1])
    trans = np.asarray(edge[2], dtype=np.float64)
    has_d = well_diagonal is not None
    d_arr = np.asarray(well_diagonal, dtype=np.float64) if has_d else np.empty(0, dtype=np.float64)
    indptr, indices, data, _, _, _, _, _ = _assemble_general_edges_nb(
        int(size), tail, head, trans, has_d, d_arr
    )
    mat = sparse.csr_matrix((data, indices, indptr), shape=(int(size), int(size)), copy=False)
    mat._has_sorted_indices = True
    mat._has_canonical_format = True
    return mat


class Multigrid(_SolveBase):
    """Cubic Cartesian zero-mean Neumann ML-CPMG state compatible with extension_multigrid_v2."""

    krylov = "cg"
    coordinate_system = "physical zero-mean pressure quotient"

    def __init__(
        self,
        k,
        config=None,
        deadline=None,
        edge=None,
        smoother_scheme=DEFAULT_SMOOTHER_SCHEME,
    ):
        started = time.perf_counter()
        _check_deadline(deadline, "deadline before ML-CPMG setup")
        self.config_name, self.config = _resolve_config(config)
        self.k = np.asarray(k, dtype=np.float64)
        if self.k.ndim != 3 or len(set(self.k.shape)) != 1:
            raise ValueError("expected cubic permeability")

        with threadpool_limits(limits=1):
            nx, ny, nz = self.k.shape
            size = self.k.size
            k_flat = np.ascontiguousarray(self.k.ravel())
            if np.any(k_flat <= 0) or not np.all(np.isfinite(k_flat)):
                raise ValueError("permeability must be finite and positive")

            if edge is None:
                self.edge = edges(self.k)
            else:
                tail_in, head_in, trans_in = (
                    np.asarray(edge[0]),
                    np.asarray(edge[1]),
                    np.asarray(edge[2], dtype=np.float64),
                )
                if (
                    tail_in.ndim != 1
                    or head_in.shape != tail_in.shape
                    or trans_in.shape != tail_in.shape
                    or tail_in.dtype.kind not in "iu"
                    or head_in.dtype.kind not in "iu"
                    or np.any(tail_in < 0)
                    or np.any(head_in < 0)
                    or np.any(tail_in >= size)
                    or np.any(head_in >= size)
                    or np.any(trans_in <= 0)
                    or not np.all(np.isfinite(trans_in))
                ):
                    raise ValueError("invalid serialized edge graph")
                self.edge = (tail_in, head_in, trans_in)

            tail, head, trans = self.edge
            err, indptr, indices, data, diag0, bound0 = _validate_and_assemble_cartesian_nb(
                nx, ny, nz, k_flat, tail, head, trans, False, np.empty(0, dtype=np.float64)
            )
            if err == 0:
                std_cart = True
                self.A = sparse.csr_matrix((data, indices, indptr), shape=(size, size), copy=False)
                self.A._has_sorted_indices = True
                self.A._has_canonical_format = True
            else:
                std_cart = False
                indptr, indices, data, diag0, bound0, _, _, _ = _assemble_general_edges_nb(
                    size, tail, head, trans, False, np.empty(0, dtype=np.float64)
                )
                self.A = sparse.csr_matrix((data, indices, indptr), shape=(size, size), copy=False)
                self.A._has_sorted_indices = True
                self.A._has_canonical_format = True

            assembled = time.perf_counter()
            _check_deadline(deadline, "deadline after fine assembly")

            xyz = None if std_cart else np.indices(self.k.shape).reshape(3, -1).T
            tags, high_count, phase_rule, keep_edge = _extract_components_with_keep_edge(
                k_flat,
                xyz,
                self.edge,
                self.k.shape if std_cart else None,
                self.config,
            )
            self.component_labels = tags
            self.high_component_count = int(high_count)
            allow0 = (
                _build_csr_allow_mask_from_edges_nb(
                    self.A.indptr, self.A.indices, self.A.data, tail, head, keep_edge
                )
                if (not std_cart and keep_edge is not None)
                else None
            )
            labelled = time.perf_counter()

            self.cycle_repeats = self.config["cycle_repeats"]
            self.solve_calls = 0
            self.last_pressure = None
            self.last_meta = None

            init_tags = int(_count_unique_nonneg_tags_nb(tags))
            is_alg_only = self.config["phase_mode"] == "algebraic_only"

            self.levels, hier_meta = _construct_hierarchy(
                self.A,
                None if is_alg_only else xyz,
                None if is_alg_only else tags,
                self.config,
                grounded=False,
                cartesian_fine=std_cart,
                cartesian_shape=(self.k.shape if (std_cart and not is_alg_only) else None),
                precomputed_diag=diag0,
                precomputed_bound=bound0,
                initial_tag_count=1 if is_alg_only else init_tags,
                initial_allow_mask=allow0,
                split_boxes_by_connectivity=(not std_cart),
                smoother_scheme=smoother_scheme,
                binary_exact=(phase_rule == "binary_exact_fastpath" and high_count <= 256),
                deadline=deadline,
            )
            if std_cart and edge is not None and smoother_scheme != "sgs":
                line_fn = _build_anisotropic_line_smoother(
                    self.k.shape, self.edge, diag0, omega=self.levels[0]["omega"]
                )
                if line_fn is not None:
                    self.levels[0]["line_smoother"] = line_fn
                elif smoother_scheme == "line_plus_sgs" and "factor" not in self.levels[0]:
                    self.levels[0]["sgs_smoother"] = _build_sgs_smoother(self.A, diag0)
            elif smoother_scheme == "line_plus_sgs" and "factor" not in self.levels[0]:
                self.levels[0]["sgs_smoother"] = _build_sgs_smoother(self.A, diag0)

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
            _csr_bytes(lev["A"])
            + lev["di"].nbytes
            + (_csr_bytes(lev["P"]) + _csr_bytes(lev["R"]) if "P" in lev else 0)
            for lev in self.levels
        )
        finished = time.perf_counter()
        self.setup_meta = {
            "config_name": self.config_name,
            "config": dict(self.config),
            "phase_mode": self.config["phase_mode"],
            "phase_rule": phase_rule,
            "assembly_seconds": assembled - started,
            "component_label_seconds": labelled - assembled,
            "hierarchy_seconds": finished - labelled,
            "setup_seconds": finished - started,
            **hier_meta,
            "cycle_repeats": self.cycle_repeats,
            "high_component_count": int(high_count),
            "high_fraction": float(np.count_nonzero(tags > 0) / max(1, tags.size)),
            "hierarchy_array_bytes": int(storage),
            "edge_array_bytes": int(sum(arr.nbytes for arr in self.edge)),
            "array_bytes_note": "persistent CSR/diagonal/edge arrays only; RSS includes LU, temporaries and Python",
            "fine_operator_unchanged": True,
            "preconditioner_is_fixed_linear_symmetric": True,
            "coordinate_system": self.coordinate_system,
            "mechanism": (
                f"ML-CPMG ({self.config['phase_mode']}: {phase_rule}); "
                "hybrid component-geometric + heavy-edge algebraic SA; "
                "symmetric Galerkin + PSD positive-coupling lump; fixed symmetric V-cycle"
            ),
        }

    def _cycle(self, index, rhs):
        return self._vcycle_fn(index, rhs)

    def apply_preconditioner(self, rhs):
        return self._apply_fn(rhs)

    def solve(self, *args, **kwargs):
        p, meta = super().solve(*args, **kwargs)
        meta["mechanism"] = self.setup_meta["mechanism"]
        self.last_meta = dict(meta)
        return p, meta


def build(k, config=None, deadline=None, edge=None, smoother_scheme=DEFAULT_SMOOTHER_SCHEME):
    """Build cubic Cartesian Neumann ML-CPMG state."""
    return Multigrid(k, config=config, deadline=deadline, edge=edge, smoother_scheme=smoother_scheme)


def build_from_context(ctx, **parameters):
    """Context builder compatible with extension_worker_v2."""
    return Multigrid(
        ctx["k"],
        config=parameters if parameters else None,
        deadline=ctx.get("deadline"),
        edge=ctx.get("edge"),
    )


def make_precision_setup(config=None):
    """Return a setup(ctx) callback compatible with extension_precision_v2.run."""
    def setup(ctx):
        state = Multigrid(
            ctx["k"],
            config=config,
            deadline=ctx.get("deadline"),
            edge=ctx.get("edge"),
        )
        return {
            "A": state.A,
            "M": state.preconditioner,
            "krylov": "cg",
            "metadata": state.setup_meta,
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
            "iterations": 0,
            "setup_seconds": elapsed,
            "solve_seconds": 0.0,
            "total_seconds": elapsed,
            "status": "deadline_in_setup",
        }
    p, meta = state.solve(b, deadline=deadline, **kwargs)
    meta["total_seconds"] = time.perf_counter() - started
    return p, meta


def build_rectangular(
    k_tag,
    edge,
    d=None,
    config=None,
    deadline=None,
    line_axis=None,
    smoother_scheme=DEFAULT_SMOOTHER_SCHEME,
):
    """Build ML-CPMG or internal ablation baseline on rectangular anisotropic grids."""
    start = time.perf_counter()
    _check_deadline(deadline, "SPE10 rectangular solver deadline")
    cfg_name, cfg = _resolve_config(config)
    k_tag = np.asarray(k_tag, dtype=np.float64)
    if k_tag.ndim != 3 or k_tag.size < 2:
        raise ValueError("expected three nonempty Kx axes and finite positive permeability")

    tail = np.asarray(edge[0])
    head = np.asarray(edge[1])
    trans = np.asarray(edge[2], dtype=np.float64)
    if (
        tail.ndim != 1
        or tail.shape != head.shape
        or tail.shape != trans.shape
        or tail.dtype.kind not in "iu"
        or head.dtype.kind not in "iu"
    ):
        raise ValueError("expected integer directed endpoints and positive float64 transmissibilities")
    edge_tup = (tail, head, trans)

    grounded = d is not None
    if grounded:
        d_arr = np.asarray(d, dtype=np.float64)
        if (
            d_arr.shape != (k_tag.size,)
            or not np.all(np.isfinite(d_arr))
            or np.any(d_arr < 0.0)
            or not np.any(d_arr > 0.0)
        ):
            raise ValueError("d must be finite, nonnegative, shape (N,), with positive grounding")
    else:
        d_arr = np.empty(0, dtype=np.float64)

    with threadpool_limits(1):
        nx, ny, nz = map(int, k_tag.shape)
        k_flat = np.ascontiguousarray(k_tag.ravel())
        err, indptr, indices, data, diag0, bound0 = _validate_and_assemble_cartesian_nb(
            nx, ny, nz, k_flat, tail, head, trans, grounded, d_arr
        )
        if err == 1:
            raise ValueError("expected three nonempty Kx axes and finite positive permeability")
        if err == 2:
            raise ValueError("expected integer directed endpoints and positive float64 transmissibilities")
        if err == 3:
            raise ValueError("face ordering must be axis 0 then 1 then 2, C order in each")

        A = sparse.csr_matrix(
            (data, indices, indptr), shape=(k_flat.size, k_flat.size), copy=False
        )
        A._has_sorted_indices = True
        A._has_canonical_format = True
        assembled = time.perf_counter()

        _check_deadline(deadline, "SPE10 rectangular solver deadline")
        is_alg_only = cfg["phase_mode"] == "algebraic_only"
        if is_alg_only:
            tags = None
            high_count = 0
            phase_rule = "algebraic_only"
            init_tags = 1
        else:
            tags, high_count, phase_rule, _ = _extract_components_with_keep_edge(
                k_flat, None, edge_tup, k_tag.shape, cfg
            )
            init_tags = int(_count_unique_nonneg_tags_nb(tags))
        labelled = time.perf_counter()

        levels, hier_meta = _construct_hierarchy(
            A,
            None,
            tags,
            cfg,
            grounded=grounded,
            cartesian_fine=(not grounded),
            cartesian_shape=(None if is_alg_only else k_tag.shape),
            precomputed_diag=diag0,
            precomputed_bound=bound0,
            initial_tag_count=init_tags,
            initial_allow_mask=None,
            split_boxes_by_connectivity=False,
            smoother_scheme=smoother_scheme,
            deadline=deadline,
        )

        line_used = False
        if smoother_scheme != "sgs":
            line_fn = _build_anisotropic_line_smoother(
                k_tag.shape, edge_tup, diag0, omega=levels[0]["omega"], forced_axis=line_axis
            )
            if line_fn is not None:
                levels[0]["line_smoother"] = line_fn
                line_used = True
            elif smoother_scheme == "line_plus_sgs" and "factor" not in levels[0]:
                levels[0]["sgs_smoother"] = _build_sgs_smoother(A, diag0)
        hier_meta["anisotropic_line_smoother"] = line_used
        hier_meta["forced_line_axis"] = line_axis

        apply_fn, _ = _make_cycle_callable(
            levels, grounded=grounded, cycle_repeats=cfg["cycle_repeats"], fine_A=A
        )
        M = LinearOperator(A.shape, matvec=apply_fn, dtype=np.float64)

    end = time.perf_counter()
    info = {
        "name": f"ml_cpmg_{cfg_name}",
        "config_name": cfg_name,
        "config": dict(cfg),
        "phase_mode": cfg["phase_mode"],
        "phase_rule": phase_rule,
        "grounded": bool(grounded),
        "assembly": "assemble_cartesian_fast",
        "assembly_seconds": assembled - start,
        "component_label_seconds": (0.0 if is_alg_only else labelled - assembled),
        "hierarchy_seconds": end - labelled,
        "setup_seconds": end - start,
        "high_component_count": int(high_count),
        "high_fraction": (
            0.0 if tags is None else float(np.count_nonzero(tags > 0) / max(1, tags.size))
        ),
        **hier_meta,
    }
    return {"A": A, "M": M, "krylov": "cg", "metadata": info}


@nb.njit(cache=True)
def _validate_active_graph_arrays_nb(
    n: int,
    k_arr: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
    trans: np.ndarray,
):
    for i in range(n):
        kv = k_arr[i]
        if not np.isfinite(kv) or kv <= 0.0:
            return 1
    for e in range(trans.size):
        u = tail[e]
        v = head[e]
        tv = trans[e]
        if u < 0 or v < 0 or u >= n or v >= n or u == v:
            return 2
        if not np.isfinite(tv) or tv <= 0.0:
            return 2
    return 0


def build_active_graph(
    graph,
    k,
    d=None,
    config=None,
    coarse_size=512,
    max_direct=512,
    deadline=None,
    line_axis=None,
    smoother_scheme=DEFAULT_SMOOTHER_SCHEME,
    *,
    _return_levels=False,
):
    """Build ML-CPMG or internal ablation baseline on a faulted corner-point ActiveGraph.

    100% shared multigrid base across ``ML-CPMG``, ``Min-threshold``,
    ``Single-median``, and ``Algebraic-only`` (``phase_mode='algebraic_only'``).
    """
    started = time.perf_counter()
    _check_deadline(deadline, "active-graph solver deadline")
    for attr in ("xyz", "tail", "head", "trans", "shape"):
        if not hasattr(graph, attr):
            raise TypeError(f"graph is missing required ActiveGraph attribute {attr!r}")

    if coarse_size < 1 or max_direct < coarse_size or max_direct > 2048:
        raise ValueError("require 1 <= coarse_size <= max_direct <= 2048")

    cfg_name, cfg = _resolve_config(config)
    cfg = dict(cfg)
    cfg["coarse_size"] = int(coarse_size)
    cfg["max_direct"] = int(max_direct)

    xyz = np.ascontiguousarray(graph.xyz, dtype=np.int64)
    n = len(xyz)
    k_arr = np.asarray(k, dtype=np.float64)
    if k_arr.shape != (n,):
        raise ValueError("one finite positive permeability per active cell required")

    tail = np.asarray(graph.tail, dtype=np.int64)
    head = np.asarray(graph.head, dtype=np.int64)
    trans = np.asarray(graph.trans, dtype=np.float64)
    if tail.ndim != 1 or head.shape != tail.shape or trans.shape != tail.shape:
        raise ValueError("invalid active-graph edge arrays")

    err = _validate_active_graph_arrays_nb(n, k_arr, tail, head, trans)
    if err == 1:
        raise ValueError("one finite positive permeability per active cell required")
    if err == 2:
        raise ValueError("invalid active-graph edge arrays")

    with threadpool_limits(1):
        flow_count, flow_comp = _undirected_components_nb(n, tail, head)
        flow_count = int(flow_count)

        grounded = d is not None
        if grounded:
            d_arr = np.asarray(d, dtype=np.float64)
            if d_arr.shape != (n,) or not np.all(np.isfinite(d_arr)) or np.any(d_arr < 0.0):
                raise ValueError("well diagonal must be finite, nonnegative, and one per active cell")
            grounding = np.bincount(flow_comp, weights=d_arr, minlength=flow_count)
            if np.any(grounding <= 0.0) or not np.all(np.isfinite(grounding)):
                raise ValueError("every flow component must be pressure grounded")
        else:
            d_arr = np.empty(0, dtype=np.float64)
            grounding = None
            if flow_count != 1:
                raise ValueError("ungrounded pure-Neumann active graph must be connected")

        indptr, indices, data, diag0, bound0, eff_u, eff_v, eff_t = _assemble_general_edges_nb(
            n, tail, head, trans, grounded, d_arr
        )
        A = sparse.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
        A._has_sorted_indices = True
        A._has_canonical_format = True
        assembled = time.perf_counter()

        is_alg_only = cfg["phase_mode"] == "algebraic_only"
        if is_alg_only:
            high_tags = None
            comb_tags = None
            high_count = 0
            phase_rule = "algebraic_only"
            allow0 = None
            init_comb_tags = flow_count
        else:
            eff_edge = (eff_u, eff_v, eff_t)
            high_tags, high_count, phase_rule, keep_edge = _extract_components_with_keep_edge(
                k_arr, xyz, eff_edge, None, cfg
            )
            allow0 = (
                _build_csr_allow_mask_from_edges_nb(
                    A.indptr, A.indices, A.data, eff_u, eff_v, keep_edge
                )
                if keep_edge is not None
                else None
            )
            if flow_count > 1:
                packed = flow_comp * (int(high_count) + 1) + high_tags
                _, comb_tags = np.unique(packed, return_inverse=True)
                comb_tags = comb_tags.astype(np.int64, copy=False)
                init_comb_tags = int(np.max(comb_tags)) + 1 if comb_tags.size else 0
            else:
                comb_tags = high_tags
                init_comb_tags = int(_count_unique_nonneg_tags_nb(comb_tags))
        labelled = time.perf_counter()

        levels, hier_meta = _construct_hierarchy(
            A,
            None if is_alg_only else xyz,
            comb_tags,
            cfg,
            grounded=grounded,
            flow_tags=flow_comp if flow_count > 1 else None,
            cartesian_fine=False,
            precomputed_diag=diag0,
            precomputed_bound=bound0,
            initial_tag_count=init_comb_tags,
            initial_allow_mask=allow0,
            smoother_scheme=smoother_scheme,
            rank_safe_damping=False,
            verify_coarse_spd=grounded,
            deadline=deadline,
        )

        line_used = False
        if smoother_scheme != "sgs":
            line_fn = _build_active_graph_line_smoother(
                xyz, A, diag0, omega=2.0 / 3.0, forced_axis=line_axis
            )
            if line_fn is not None:
                levels[0]["line_smoother"] = line_fn
                line_used = True
            elif smoother_scheme == "line_plus_sgs" and "factor" not in levels[0]:
                levels[0]["sgs_smoother"] = _build_sgs_smoother(A, diag0)
        hier_meta["anisotropic_line_smoother"] = line_used
        hier_meta["forced_line_axis"] = line_axis

        apply_fn, _ = _make_cycle_callable(
            levels, grounded=grounded, cycle_repeats=cfg["cycle_repeats"], fine_A=A
        )
        M = LinearOperator(A.shape, matvec=apply_fn, dtype=np.float64)

    finished = time.perf_counter()
    description = {
        "method": f"ml_cpmg_{cfg['phase_mode']}",
        "config_name": cfg_name,
        "config": dict(cfg),
        "phase_mode": cfg["phase_mode"],
        "phase_rule": phase_rule,
        "grounded": bool(grounded),
        "flow_components": flow_count,
        "grounding_by_component": grounding.tolist() if grounding is not None else None,
        "high_component_count": int(high_count),
        "high_fraction": (
            0.0 if high_tags is None else float(np.count_nonzero(high_tags > 0) / max(1, n))
        ),
        "initial_combined_tags": int(init_comb_tags),
        "assembly_seconds": assembled - started,
        "component_label_seconds": (0.0 if is_alg_only else labelled - assembled),
        "hierarchy_seconds": finished - labelled,
        "setup_seconds": finished - started,
        "levels": [
            {
                "size": int(s),
                "nnz": int(nz),
                "tags": int(tc),
            }
            for s, nz, tc in zip(
                hier_meta["level_sizes"],
                hier_meta["level_nnz"],
                hier_meta["component_tag_count_per_level"],
            )
        ],
        **hier_meta,
    }
    if _return_levels:
        return A, M, description, levels
    return A, M, description


def build_algebraic_only_active_graph(
    graph,
    k=None,
    d=None,
    config=None,
    cap: int = 256,
    stall: float = 0.33,
    coarse_size: int = 512,
    max_direct: int = 512,
    deadline=None,
    smoother_scheme: str = DEFAULT_SMOOTHER_SCHEME,
):
    """Build Algebraic-only on ActiveGraph using the exact same shared smoother base."""
    n = len(graph.xyz)
    if k is None:
        k_use = np.ones(n, dtype=np.float64)
        d_use = d
    else:
        arr = np.asarray(k, dtype=np.float64)
        if d is None and config is not None and np.any(arr == 0.0):
            k_use = np.ones(n, dtype=np.float64)
            d_use = arr
        else:
            k_use = arr if np.all(arr > 0.0) else np.ones(n, dtype=np.float64)
            d_use = d
    if config is not None:
        _, base_cfg = _resolve_config(config)
        cfg = dict(base_cfg, phase_mode="algebraic_only")
    else:
        cfg = dict(
            _BASE_CONTROLS,
            phase_mode="algebraic_only",
            max_basins=int(cap),
            stall_ratio=float(stall),
            prolong_omega=0.45,
        )
    return build_active_graph(
        graph,
        k_use,
        d=d_use,
        config=cfg,
        coarse_size=coarse_size,
        max_direct=max_direct,
        deadline=deadline,
        smoother_scheme=smoother_scheme,
    )


def warmup_jit(force: bool = False) -> float:
    """Compile all @njit specializations (int32/int64 Cartesian, axes 0/1/2, ActiveGraph) once."""
    global _JIT_WARMED
    if _JIT_WARMED and not force:
        return 0.0

    t0 = time.perf_counter()
    rng = np.random.default_rng(20261120)

    with threadpool_limits(1):
        # 1. Cartesian int64 AND int32 edges across dom_ax in {0, 1, 2}, pure-Neumann & grounded
        k_cart = np.exp(rng.uniform(0.0, 12.0, size=(10, 10, 10)))
        e64 = edges(k_cart)
        e32 = (e64[0].astype(np.int32), e64[1].astype(np.int32), e64[2].copy())
        d_cart = np.ones(k_cart.size, dtype=np.float64) * 1.0e-2
        for e_use in (e64, e32):
            for ax in (0, 1, 2, None):
                for d_use in (None, d_cart):
                    st = build_rectangular(
                        k_cart,
                        e_use,
                        d=d_use,
                        config=dict(_BASE_CONTROLS, coarse_size=64, max_direct=512),
                        line_axis=ax,
                    )
                    _ = st["M"] @ np.ones(k_cart.size, dtype=np.float64)
            st_alg = build_rectangular(
                k_cart,
                e_use,
                d=None,
                config=dict(_BASE_CONTROLS, phase_mode="algebraic_only", coarse_size=64, max_direct=512),
                line_axis=0,
            )
            _ = st_alg["M"] @ np.ones(k_cart.size, dtype=np.float64)

        # 2. ActiveGraph with thin-layer z-anisotropy, fault barrier, and NNC across all 4 internal modes
        xyz = np.ascontiguousarray(np.indices((12, 12, 8)).reshape(3, -1).T, dtype=np.int64)
        k_ag = np.exp(rng.uniform(0.0, 12.0, size=xyz.shape[0]))
        u_list, v_list, t_list = [], [], []
        idx_grid = np.arange(12 * 12 * 8, dtype=np.int64).reshape(12, 12, 8)
        for ax in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[ax], hi[ax] = slice(None, -1), slice(1, None)
            ua = idx_grid[tuple(lo)].ravel()
            va = idx_grid[tuple(hi)].ravel()
            ta = np.full(ua.size, 25.0 if ax == 2 else 1.0, dtype=np.float64)
            if ax == 1:
                ta[:24] *= 1.0e-4
            u_list.append(ua)
            v_list.append(va)
            t_list.append(ta)
        u_arr = np.concatenate(u_list + [np.array([0], dtype=np.int64)])
        v_arr = np.concatenate(v_list + [np.array([12 * 12 * 8 - 1], dtype=np.int64)])
        t_arr = np.concatenate(t_list + [np.array([1.0e-4], dtype=np.float64)])
        ag = ActiveGraph.from_arrays(xyz, u_arr, v_arr, t_arr, (12, 12, 8))
        d_ag = np.zeros(len(k_ag), dtype=np.float64)
        d_ag[0] = 1.0
        d_ag[-1] = 1.0
        for mode in ("ml_cpmg", "legacy_k_gt_min", "single_median", "algebraic_only"):
            _, M_ag, _ = build_active_graph(
                ag,
                k_ag,
                d=d_ag,
                config=dict(_BASE_CONTROLS, phase_mode=mode),
                coarse_size=16,
                max_direct=512,
            )
            _ = M_ag @ np.ones(len(k_ag), dtype=np.float64)

    _JIT_WARMED = True
    return time.perf_counter() - t0
