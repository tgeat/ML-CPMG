"""Shared solver protocol, external AMG wrappers, and timing harness for v18 experiments."""
from __future__ import annotations

import os
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_key] = "1"

import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import psutil
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits

from ml_cpmg import grid, krylov, reservoirs, solver as mg, synthetic_media as media
from ml_cpmg.paths import OUTPUTS_DIR, RESULTS_DIR, ensure_vendor_on_path

ensure_vendor_on_path()
import pyamg  # noqa: E402

CALIBRATION_METHODS = ["ML-CPMG", "Min-threshold", "Single-median", "SA-0", "SA-0.04", "SA-0.1", "RS-0.25"]
PRIMARY_METHODS = ["ML-CPMG", "Min-threshold", "Single-median", "Algebraic-only", "SA-0.04", "RS-0.25"]
OLYMPUS_METHODS = CALIBRATION_METHODS + ["Algebraic-only"]
BRUGGE_METHODS = OLYMPUS_METHODS


def pin_single_cpu(cpu: int | None = None) -> int | None:
    """Pin the current process to a single logical CPU when supported."""
    proc = psutil.Process()
    if not hasattr(proc, "cpu_affinity"):
        return None
    allowed = proc.cpu_affinity()
    target = cpu if cpu is not None else (19 if 19 in allowed else allowed[0])
    if target not in allowed:
        raise ValueError(f"CPU {target} unavailable; allowed CPUs: {allowed}")
    proc.cpu_affinity([target])
    return target


def selected_ml_params() -> tuple[int, float]:
    """Load the frozen (max_basins, stall_ratio) selected by Experiment 1."""
    cfg_path = RESULTS_DIR / "selected_config.json"
    if cfg_path.exists():
        sel = json.loads(cfg_path.read_text(encoding="utf-8"))
        return int(sel["max_basins"]), float(sel["stall_ratio"])
    return 256, 0.33


def internal_config(method: str, cap: int = 256, stall: float = 0.33) -> dict:
    mode_map = {
        "ML-CPMG": "ml_cpmg",
        "Min-threshold": "legacy_k_gt_min",
        "Single-median": "single_median",
        "Cliff-only": "cliff_only",
        "Multi-bin": "naive_multibin",
        "Algebraic-only": "algebraic_only",
    }
    return dict(mg._BASE_CONTROLS, phase_mode=mode_map[method], max_basins=cap, stall_ratio=stall, prolong_omega=0.45)


def build_external_amg(A: sparse.csr_matrix, method: str, quotient: bool):
    """Build PyAMG SA (theta in {0, 0.04, 0.1}) or RS (theta=0.25) with exact coarse solve."""
    smooth = ("gauss_seidel", {"sweep": "symmetric", "iterations": 1})
    common = dict(
        presmoother=smooth,
        postsmoother=smooth,
        max_levels=20,
        max_coarse=10 if method.startswith("SA") else 512,
        keep=False,
    )
    np.random.seed(1729)
    if method.startswith("SA"):
        theta = float(method.split("-")[1])
        h = pyamg.smoothed_aggregation_solver(
            sparse.csr_array(A),
            B=np.ones((A.shape[0], 1)),
            symmetry="hermitian",
            strength=("symmetric", {"theta": theta}),
            **common,
        )
    else:
        h = pyamg.ruge_stuben_solver(
            sparse.csr_array(A),
            strength=("classical", {"theta": 0.25}),
            **common,
        )
    C = h.levels[-1].A
    if quotient:
        z = np.asarray(h.levels[-1].B[:, 0]).ravel() if hasattr(h.levels[-1], "B") else np.ones(C.shape[0])
        U = linalg.null_space(z[None, :])
        cf = linalg.cho_factor(U.T @ C.toarray() @ U, lower=True, check_finite=False)

        def coarse(_, b):
            return U @ linalg.cho_solve(cf, U.T @ b, check_finite=False)
    else:
        cf = splu(C.tocsc())

        def coarse(_, b):
            return cf.solve(b)

    h.coarse_solver = pyamg.coarse_grid_solver(coarse)
    raw = h.aspreconditioner(cycle="V")
    if quotient:
        def apply(x):
            y = np.asarray(raw @ (x - np.mean(x))).ravel()
            return y - np.mean(y)
        M = LinearOperator(A.shape, matvec=apply, dtype=float)
    else:
        M = raw
    meta = dict(
        level_sizes=[v.A.shape[0] for v in h.levels],
        level_nnz=[v.A.nnz for v in h.levels],
        library_version=pyamg.__version__,
    )
    return M, meta


def build_preconditioner(case: dict, method: str, cap: int = 256, stall: float = 0.33, force_line: bool = False):
    """Assemble A and construct the preconditioner M under the unified protocol."""
    start = time.perf_counter()
    if method == "Algebraic-only":
        cfg_alg = internal_config("Algebraic-only", cap, stall)
        if case["kind"] == "cartesian":
            st = mg.build_rectangular(
                case["k"],
                case["edge"],
                d=case["d"],
                config=cfg_alg,
                line_axis=0 if force_line else None,
            )
            A, M, meta = st["A"], st["M"], st["metadata"]
        else:
            A, M, meta = mg.build_algebraic_only_active_graph(
                case["graph"],
                case["k"],
                d=case["d"],
                config=cfg_alg,
                coarse_size=512,
                max_direct=512,
            )
    elif method in ("ML-CPMG", "Min-threshold", "Single-median", "Cliff-only", "Multi-bin"):
        if case["kind"] == "cartesian":
            st = mg.build_rectangular(
                case["k"],
                case["edge"],
                d=case["d"],
                config=internal_config(method, cap, stall),
                line_axis=0 if force_line else None,
            )
            A, M, meta = st["A"], st["M"], st["metadata"]
        else:
            A, M, meta = mg.build_active_graph(
                case["graph"],
                case["k"],
                case["d"],
                config=internal_config(method, cap, stall),
                coarse_size=512,
                max_direct=512,
            )
    else:
        A = (
            mg.assemble_cartesian_fast(case["k"].shape, case["edge"], d=case["d"])
            if case["kind"] == "cartesian"
            else mg._assemble_general_edges(case["k"].size, case["edge"], well_diagonal=case["d"])
        )
        M, meta = build_external_amg(A, method, quotient=case["d"] is None)
    return A, M, meta, time.perf_counter() - start


def iter_cases(suite: str, *, case_filter: str | None = None) -> Iterator[dict]:
    """Yield benchmark problem dictionaries for 'static', 'dynamic', 'olympus', or 'synthetic'."""
    if suite == "static":
        k, b, edge = reservoirs.load_spe10_static()
        yield dict(id="SPE10-static", kind="cartesian", k=k, edge=edge, d=None, b=b)
    elif suite == "dynamic":
        model = reservoirs.SPE10DynamicModel.load()
        for day in (0.0, 520.04, 2000.0):
            case_id = f"SPE10-{day:g}d"
            if case_filter and case_id != case_filter:
                continue
            S = model.load_snapshot_saturation(day)
            kt, edge, d, b = model.pressure_system(S)
            yield dict(id=case_id, kind="cartesian", k=kt, edge=edge, d=d, b=b, S=S)
    elif suite == "olympus":
        yield from reservoirs.load_olympus_cases()
    elif suite == "brugge":
        inp = reservoirs.load_brugge_inputs()
        for day in (0.0, 1000.0, 2000.0):
            S = reservoirs.load_brugge_snapshot_saturation(day)
            g, k, d, b, _ = reservoirs.brugge_dynamic_system(inp, S)
            yield dict(id=f"Brugge-{day:g}d", kind="graph", graph=g, k=k, edge=(g.tail, g.head, g.trans), d=d, b=b)
    elif suite == "synthetic":
        b = grid.sources(64)
        for fam in media.FAMILIES:
            k, ref = media.geometry(64, fam, contrast=1e6, seed=20261102, return_meta=True)
            yield dict(id="Synthetic-" + fam, kind="cartesian", k=k, edge=grid.edges(k), d=None, b=b, ref=ref)
        k_rand = media.unplanted_random(64, contrast=1e6, seed=20261115)
        yield dict(id="Synthetic-unplanted_random", kind="cartesian", k=k_rand, edge=grid.edges(k_rand), d=None, b=b)
        k_gauss = media.smooth_gaussian(64, contrast=1e6)
        yield dict(id="Synthetic-smooth_gaussian", kind="cartesian", k=k_gauss, edge=grid.edges(k_gauss), d=None, b=b)
    else:
        raise ValueError(f"Unknown suite: {suite}")


def run_single(
    case: dict,
    method: str,
    rep: int,
    *,
    cap: int = 256,
    stall: float = 0.33,
    force_line: bool = False,
    reliable: bool = True,
    save_dir: Path | None = None,
) -> dict:
    """Execute one timed pressure solve with optional double-double 1e-9 conservation closure."""
    gc.collect()
    start = time.perf_counter()
    A, M, meta, setup = build_preconditioner(case, method, cap=cap, stall=stall, force_line=force_line)
    x, cg_meta = krylov.pcg_lanczos_solve(
        A, case["b"], M, quotient=case["d"] is None, rtol=1e-10, maxiter=2000, deadline=start + 600.0
    )
    true_res = float(np.linalg.norm(case["b"] - A @ x) / np.linalg.norm(case["b"]))
    row = dict(
        case=case["id"],
        method=method,
        rep=rep,
        cap=cap,
        stall=stall,
        setup_seconds=setup,
        total_seconds=setup + cg_meta["solve_seconds"],
        true_residual=true_res,
        line_smoother=meta.get("anisotropic_line_smoother", False),
        **cg_meta,
        **krylov.extract_hierarchy_metrics(meta),
    )
    hi = lo = None
    if reliable:
        if case["d"] is None:
            hi, lo, _, quality = krylov.dd_refine_neumann(
                A, case["b"], case["edge"], M, x, target_gate=1e-9, inner_rtol=1e-7, max_outer=4
            )
        else:
            hi, lo, quality = krylov.dd_refine_grounded(
                A, case["b"], case["edge"], case["d"], M, x, target_gate=1e-9, inner_rtol=1e-7, max_outer=4
            )
        row.update(quality)
        row["converged"] = bool(cg_meta["converged"] and quality["gate_1e9_pass"])
        row["pcg_initial_iterations"] = cg_meta["iterations"]
        row["pcg_initial_seconds"] = cg_meta["solve_seconds"]
        row["iterations"] = cg_meta["iterations"] + quality["dd_extra_cg_iterations"]
        row["solve_seconds"] = cg_meta["solve_seconds"] + quality["dd_refine_seconds"]
        row["total_seconds"] = setup + row["solve_seconds"]

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if reliable and hi is not None and lo is not None:
            np.savez_compressed(save_dir / f"{case['id']}_{method}_reliable.npz", hi=hi, lo=lo, x=hi + lo)
        else:
            np.savez_compressed(save_dir / f"{case['id']}_{method}_pressure.npz", x=x)

    if "ref" in case and rep == 0 and method in ("ML-CPMG", "Min-threshold", "Single-median"):
        labels, _, _ = mg._extract_components_core(
            case["k"].ravel(), None, case["edge"], case["k"].shape, internal_config(method, cap, stall)
        )
        ref = case["ref"]
        dom, coverage, purity = {}, [], []
        for pid in range(1, ref["num_peaks"] + 1):
            v = labels.reshape(case["k"].shape)[ref["core_masks"][pid]]
            u, c = np.unique(v[v > 0], return_counts=True)
            dom[pid] = int(u[np.argmax(c)]) if len(u) else 0
            coverage.append(float(np.count_nonzero(v) / max(1, len(v))))
            purity.append(float(max(c) / max(1, len(v))) if len(c) else 0.0)
        row.update(
            planted_peaks=ref["num_peaks"],
            distinct_cores=len(set(dom.values()) - {0}),
            close_pairs=len(ref["close_pairs"]),
            separated_pairs=sum(
                dom[a] > 0 and dom[b] > 0 and dom[a] != dom[b] for a, b in (z["pair"] for z in ref["close_pairs"])
            ),
            mean_core_coverage=float(np.mean(coverage)),
            mean_core_purity=float(np.mean(purity)),
        )

    print(
        f"[{case['id']}] {method:14s} rep={rep}: it={row['iterations']:4d}, "
        f"cold={row['total_seconds']:.3f}s, Cop={row['operator_complexity']:.3f}, "
        f"gate_1e9={row.get('gate_1e9_pass', 'N/A')}",
        flush=True,
    )
    return row


def run_suite(
    suite: str,
    *,
    methods: list[str] | None = None,
    case_filter: str | None = None,
    repeats: int = 3,
    force_line: bool = False,
    reliable: bool = True,
    out_dir: Path | None = None,
    cpu: int | None = None,
) -> list[dict]:
    """Run an entire benchmark suite sequentially on one pinned CPU and save to out_dir."""
    pin_single_cpu(cpu)
    cap, stall = selected_ml_params()
    default_methods = OLYMPUS_METHODS if suite in ("olympus", "brugge") else PRIMARY_METHODS
    method_list = methods or default_methods
    dest = out_dir or (OUTPUTS_DIR / "reruns" / suite)
    dest.mkdir(parents=True, exist_ok=True)
    write_frozen_reliable = (
        reliable
        and case_filter is None
        and list(method_list) == list(default_methods)
        and dest.resolve() == RESULTS_DIR.resolve()
    )
    rows: list[dict] = []
    matched = 0
    with threadpool_limits(1):
        mg.warmup_jit()
        for case in iter_cases(suite, case_filter=case_filter):
            if case_filter and case["id"] != case_filter:
                continue
            matched += 1
            for rep in range(repeats):
                order = list(method_list)
                random.Random(20261115 + rep).shuffle(order)
                for method in order:
                    row = run_single(
                        case,
                        method,
                        rep,
                        cap=cap,
                        stall=stall,
                        force_line=force_line,
                        reliable=reliable,
                        save_dir=dest if rep == 0 else None,
                    )
                    rows.append(row)
                    krylov.save_json(dest / f"{suite}_measurements.json", rows)
                    if write_frozen_reliable:
                        krylov.save_json(dest / f"{suite}_reliable.json", rows)
    if matched == 0:
        raise SystemExit(f"No case matched filter {case_filter!r} in suite {suite!r}.")
    return rows
