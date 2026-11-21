"""Shared PCG-Lanczos solver, double-double conservation closure, and I/O helpers."""
from __future__ import annotations

import os
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_var] = "1"

import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
from scipy import sparse
from scipy.linalg import eigvalsh_tridiagonal
from scipy.sparse.linalg import LinearOperator
from threadpoolctl import threadpool_limits

from . import double_double as gp


def check_resources(
    *,
    min_avail_gib: float = 10.0,
    max_rss_gib: float = 5.0,
    context: str = "",
) -> dict[str, Any]:
    """Enforce workspace RAM availability and process RSS limits."""
    gc.collect()
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    avail_bytes = int(vm.available)
    rss_bytes = int(proc.memory_info().rss)
    avail_gib = avail_bytes / (1024.0 ** 3)
    rss_gib = rss_bytes / (1024.0 ** 3)
    prefix = f"[{context}] " if context else ""
    if avail_gib < min_avail_gib:
        raise MemoryError(f"{prefix}Available RAM {avail_gib:.2f} GiB < required {min_avail_gib:.2f} GiB")
    if rss_gib > max_rss_gib:
        raise MemoryError(f"{prefix}Process RSS {rss_gib:.2f} GiB > limit {max_rss_gib:.2f} GiB")
    return {
        "available_bytes": avail_bytes,
        "available_gib": round(avail_gib, 4),
        "rss_bytes": rss_bytes,
        "rss_gib": round(rss_gib, 4),
    }


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_csv(path: Path | str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def project_zero_mean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x - float(np.mean(x))


def lanczos_condition_estimate(alphas: list[float], betas: list[float]) -> dict[str, float | None]:
    """Compute Ritz extremal eigenvalues and condition number kappa(MA) from CG scalars."""
    m = len(alphas)
    if m == 0:
        return {"lambda_min_est": None, "lambda_max_est": None, "kappa_est": None}
    a = np.asarray(alphas, dtype=np.float64)
    if np.any(a <= 0.0) or not np.all(np.isfinite(a)):
        return {"lambda_min_est": None, "lambda_max_est": None, "kappa_est": None}
    if m == 1:
        val = float(1.0 / a[0])
        return {"lambda_min_est": val, "lambda_max_est": val, "kappa_est": 1.0}
    b = np.asarray(betas[: m - 1], dtype=np.float64)
    if np.any(b < 0.0) or not np.all(np.isfinite(b)):
        return {"lambda_min_est": None, "lambda_max_est": None, "kappa_est": None}
    diag = np.empty(m, dtype=np.float64)
    diag[0] = 1.0 / a[0]
    diag[1:] = (1.0 / a[1:]) + (b / a[:-1])
    offdiag = np.sqrt(b) / a[:-1]
    eigs = eigvalsh_tridiagonal(diag, offdiag, check_finite=True)
    lmin = float(eigs[0])
    lmax = float(eigs[-1])
    if lmin <= 0.0 or not np.isfinite(lmin) or not np.isfinite(lmax):
        return {
            "lambda_min_est": lmin if np.isfinite(lmin) else None,
            "lambda_max_est": lmax if np.isfinite(lmax) else None,
            "kappa_est": None,
        }
    return {"lambda_min_est": lmin, "lambda_max_est": lmax, "kappa_est": float(lmax / lmin)}


def pcg_lanczos_solve(
    A: sparse.spmatrix | LinearOperator,
    b: np.ndarray,
    M: LinearOperator | Callable[[np.ndarray], np.ndarray] | None,
    *,
    quotient: bool = True,
    rtol: float = 1e-10,
    maxiter: int = 2500,
    x0: np.ndarray | None = None,
    record_trace: bool = True,
    deadline: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Float64 Preconditioned Conjugate Gradient with Lanczos Ritz spectrum estimation."""
    t0 = time.perf_counter()
    rhs = np.asarray(b, dtype=np.float64).ravel()
    if quotient:
        rhs = project_zero_mean(rhs)
    norm_b = float(np.linalg.norm(rhs))
    if norm_b == 0.0:
        return np.zeros_like(rhs), {
            "status": 0,
            "converged": True,
            "iterations": 0,
            "solve_seconds": 0.0,
            "rel_res_l2": 0.0,
            "lambda_min_est": 1.0,
            "lambda_max_est": 1.0,
            "kappa_est": 1.0,
            "trace_rel_res": [0.0] if record_trace else [],
        }

    apply_M = (lambda v: v.copy()) if M is None else (M.matvec if hasattr(M, "matvec") else M)
    apply_A = A.matvec if hasattr(A, "matvec") else (lambda v: A @ v)

    if x0 is None:
        x = np.zeros_like(rhs)
        r = rhs.copy()
    else:
        x = np.asarray(x0, dtype=np.float64).ravel().copy()
        if quotient:
            x = project_zero_mean(x)
        r = rhs - np.asarray(apply_A(x), dtype=np.float64).ravel()
        if quotient:
            r = project_zero_mean(r)

    tol_abs = rtol * norm_b
    rnorm = float(np.linalg.norm(r))
    trace = [rnorm / norm_b] if record_trace else []
    if rnorm <= tol_abs:
        return x, {
            "status": 0,
            "converged": True,
            "iterations": 0,
            "solve_seconds": time.perf_counter() - t0,
            "rel_res_l2": rnorm / norm_b,
            "lambda_min_est": None,
            "lambda_max_est": None,
            "kappa_est": None,
            "trace_rel_res": trace,
        }

    z = np.asarray(apply_M(r), dtype=np.float64).ravel()
    if quotient:
        z = project_zero_mean(z)
    p = z.copy()
    rz = float(np.dot(r, z))
    if not np.isfinite(rz) or rz <= 0.0:
        raise ValueError(f"Non-SPD preconditioner detected at iter 0: <r, M r> = {rz!r}")

    alphas: list[float] = []
    betas: list[float] = []
    iters = 0
    converged = False
    status = maxiter

    with threadpool_limits(limits=1):
        for k in range(1, maxiter + 1):
            if deadline is not None and (k & 7) == 0 and time.perf_counter() >= deadline:
                raise TimeoutError(f"PCG exceeded deadline at iteration {k}")
            Ap = np.asarray(apply_A(p), dtype=np.float64).ravel()
            if quotient:
                Ap = project_zero_mean(Ap)
            pAp = float(np.dot(p, Ap))
            if not np.isfinite(pAp) or pAp <= 0.0:
                raise ValueError(f"Non-SPD operator detected at iter {k}: <p, A p> = {pAp!r}")
            alpha = rz / pAp
            alphas.append(alpha)
            x += alpha * p
            r -= alpha * Ap
            if quotient and (k % 32 == 0):
                x = project_zero_mean(x)
                r = project_zero_mean(r)
            rnorm = float(np.linalg.norm(r))
            if record_trace:
                trace.append(rnorm / norm_b)
            iters = k
            if rnorm <= tol_abs:
                converged = True
                status = 0
                break
            z = np.asarray(apply_M(r), dtype=np.float64).ravel()
            if quotient:
                z = project_zero_mean(z)
            rz_new = float(np.dot(r, z))
            if not np.isfinite(rz_new) or rz_new <= 0.0:
                raise ValueError(f"Non-SPD preconditioner detected at iter {k}: <r, M r> = {rz_new!r}")
            beta = rz_new / rz
            betas.append(beta)
            p = z + beta * p
            rz = rz_new

    if quotient:
        x = project_zero_mean(x)
    solve_seconds = time.perf_counter() - t0
    spec = lanczos_condition_estimate(alphas, betas)
    return x, {
        "status": int(status),
        "converged": bool(converged),
        "iterations": int(iters),
        "solve_seconds": float(solve_seconds),
        "rel_res_l2": float(rnorm / norm_b),
        **spec,
        "trace_rel_res": trace,
    }


def flux_and_residual(
    hi: np.ndarray,
    lo: np.ndarray,
    edge: tuple[np.ndarray, np.ndarray, np.ndarray],
    well_diagonal: np.ndarray,
    rhs: np.ndarray,
    chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual of actual returned binary64 face and producer fluxes."""
    i, j, weights = edge
    q = np.empty(len(weights), dtype=np.float64)
    for begin in range(0, q.size, chunk):
        end = min(begin + chunk, q.size)
        q[begin:end] = gp.graph_flux(hi, lo, (i[begin:end], j[begin:end], weights[begin:end]))
    d = np.asarray(well_diagonal, dtype=np.float64).ravel()
    wells = np.flatnonzero(d)
    produced = np.zeros_like(rhs)
    ph, pl = gp._two_product(d[wells], hi[wells])
    produced[wells] = ph + (pl + d[wells] * lo[wells])
    residual = rhs - (
        np.bincount(i, weights=q, minlength=rhs.size)
        - np.bincount(j, weights=q, minlength=rhs.size)
        + produced
    )
    return q, produced, residual


def dd_refine_neumann(
    A: sparse.spmatrix,
    b: np.ndarray,
    edge: tuple[np.ndarray, np.ndarray, np.ndarray],
    M: LinearOperator | Callable[[np.ndarray], np.ndarray],
    p_init: np.ndarray,
    *,
    target_gate: float = 1e-9,
    inner_rtol: float = 1e-8,
    max_outer: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Double-double outer correction for pure-Neumann graph-flux conservation <= 1e-9."""
    t0 = time.perf_counter()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_b = float(np.linalg.norm(b))
    max_b = float(np.max(np.abs(b)))
    i, j, t = edge
    size = b.size

    hi, lo = gp.pair_project(project_zero_mean(p_init), np.zeros(size, dtype=np.float64))
    outer_iters = 0
    extra_cg_iters = 0

    def eval_flux(h: np.ndarray, l: np.ndarray):
        q = np.empty(len(t), dtype=np.float64)
        chunk = 65536
        for start in range(0, len(t), chunk):
            stop = min(start + chunk, len(t))
            q[start:stop] = gp.graph_flux(h, l, (i[start:stop], j[start:stop], t[start:stop]))
        res = b - (np.bincount(i, weights=q, minlength=size) - np.bincount(j, weights=q, minlength=size))
        rho = float(np.linalg.norm(res) / (norm_b if norm_b > 0 else 1.0))
        eta = float(np.max(np.abs(res)) / (max_b if max_b > 0 else 1.0))
        return q, res, rho, eta

    q, res, rho, eta = eval_flux(hi, lo)
    while max(rho, eta) > target_gate and outer_iters < max_outer:
        outer_iters += 1
        corr, cmeta = pcg_lanczos_solve(
            A, res, M, quotient=True, rtol=inner_rtol, maxiter=1000, record_trace=False
        )
        extra_cg_iters += cmeta["iterations"]
        hi, lo = gp.pair_update(hi, lo, project_zero_mean(corr))
        hi, lo = gp.pair_project(hi, lo)
        q, res, rho, eta = eval_flux(hi, lo)

    return hi, lo, q, {
        "dd_outer_corrections": int(outer_iters),
        "dd_extra_cg_iterations": int(extra_cg_iters),
        "dd_refine_seconds": float(time.perf_counter() - t0),
        "rho_dd": float(rho),
        "eta_dd": float(eta),
        "gate_1e9_pass": bool(max(rho, eta) <= target_gate),
    }


def dd_refine_grounded(
    A: sparse.spmatrix,
    b: np.ndarray,
    edge: tuple[np.ndarray, np.ndarray, np.ndarray],
    well_diagonal: np.ndarray,
    M: LinearOperator | Callable[[np.ndarray], np.ndarray],
    p_init: np.ndarray,
    *,
    target_gate: float = 1e-9,
    inner_rtol: float = 1e-7,
    max_outer: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Double-double outer correction for well-grounded SPD systems <= 1e-9."""
    started = time.perf_counter()
    hi = p_init.copy()
    lo = np.zeros_like(p_init)
    extra = 0
    outer = 0
    rho = eta = float("inf")
    for outer in range(max_outer + 1):
        _, _, r = flux_and_residual(hi, lo, edge, well_diagonal, b)
        rho = float(np.linalg.norm(r) / np.linalg.norm(b))
        eta = float(np.max(abs(r)) / np.max(abs(b)))
        if max(rho, eta) <= target_gate or outer == max_outer:
            break
        delta, meta = pcg_lanczos_solve(
            A, r, M, quotient=False, rtol=inner_rtol, maxiter=1500, record_trace=False
        )
        extra += meta["iterations"]
        hi, lo = gp.pair_update(hi, lo, delta)
    return hi, lo, {
        "dd_outer_corrections": int(outer),
        "dd_extra_cg_iterations": int(extra),
        "dd_refine_seconds": float(time.perf_counter() - started),
        "rho_dd": float(rho),
        "eta_dd": float(eta),
        "gate_1e9_pass": bool(max(rho, eta) <= target_gate),
    }


def extract_hierarchy_metrics(meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize hierarchy metadata across internal and external AMG preconditioners."""
    level_sizes = list(meta.get("level_sizes") or [])
    level_nnz = list(meta.get("level_nnz") or [])
    if not level_sizes and "levels" in meta and isinstance(meta["levels"], list):
        level_sizes = [int(lv["size"]) for lv in meta["levels"] if "size" in lv]
        level_nnz = [int(lv["nnz"]) for lv in meta["levels"] if "nnz" in lv]

    op_comp = meta.get("operator_complexity")
    if op_comp is None and level_nnz and level_nnz[0] > 0:
        op_comp = float(sum(level_nnz) / level_nnz[0])
    grid_comp = meta.get("grid_complexity")
    if grid_comp is None and level_sizes and level_sizes[0] > 0:
        grid_comp = float(sum(level_sizes) / level_sizes[0])

    eff_components = None
    for key in (
        "effective_component_count",
        "separated_basin_count",
        "basin_count",
        "multilevel_component_count",
        "component_count",
        "high_component_count",
        "num_basins",
        "num_components",
    ):
        if key in meta and meta[key] is not None:
            eff_components = int(meta[key])
            break
    if eff_components is None and "tag_counts_per_level" in meta and meta["tag_counts_per_level"]:
        eff_components = max(0, int(meta["tag_counts_per_level"][0]) - 1)

    return {
        "num_levels": len(level_sizes),
        "level_sizes": [int(x) for x in level_sizes],
        "level_nnz": [int(x) for x in level_nnz],
        "l1_coarse_size": int(level_sizes[1]) if len(level_sizes) > 1 else (int(level_sizes[0]) if level_sizes else None),
        "coarsest_size": int(level_sizes[-1]) if level_sizes else None,
        "operator_complexity": float(op_comp) if op_comp is not None else None,
        "grid_complexity": float(grid_comp) if grid_comp is not None else None,
        "effective_components": eff_components,
    }
