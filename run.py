"""Unified command-line entry point for ML-CPMG and the six v20 manuscript experiments.

All generated outputs go to outputs/. Archived frozen measurements under data/ are never overwritten.
"""
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

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_smoke() -> None:
    """Fast self-contained check on Cartesian Neumann (16^3) and grounded NNC graph (10x9x8)."""
    import numpy as np
    from scipy.sparse.linalg import cg
    from threadpoolctl import threadpool_limits
    from ml_cpmg import ActiveGraph, edges, krylov, solver as mg
    from ml_cpmg.paths import OUTPUTS_DIR

    cfg = dict(mg._BASE_CONTROLS, max_basins=256, stall_ratio=0.33, prolong_omega=0.45)
    reports = []
    with threadpool_limits(1):
        mg.warmup_jit()
        for graph_case in (False, True):
            shape = (10, 9, 8) if graph_case else (16, 16, 16)
            xyz = np.indices(shape).reshape(3, -1).T
            n = len(xyz)
            coords = xyz / np.asarray(shape)
            k = np.exp(4.0 * np.exp(-np.sum((coords - 0.5) ** 2, axis=1) / 0.06)).reshape(shape)
            e = edges(k)
            if graph_case:
                e = (np.r_[e[0], 0], np.r_[e[1], n - 1], np.r_[e[2], 0.7])
                g = ActiveGraph.from_arrays(xyz, *e, shape)
                d = np.zeros(n)
                d[0] = 1.0
                A, M, _ = mg.build_active_graph(g, k.ravel(), d, config=cfg, coarse_size=32, max_direct=512)
            else:
                built = mg.build_rectangular(k, e, config=cfg)
                A, M = built["A"], built["M"]
            exact = np.sin(2 * np.pi * coords[:, 0]) + np.cos(2 * np.pi * coords[:, 1])
            exact -= exact.mean()
            b = A @ exact
            count = [0]
            x, info = cg(A, b, M=M, rtol=1e-11, atol=0, maxiter=500, callback=lambda _: count.__setitem__(0, count[0] + 1))
            if not graph_case:
                x -= x.mean()
            delta = x - exact
            res = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
            err = float(np.sqrt(max(0.0, delta @ (A @ delta)) / (exact @ (A @ exact))))
            assert info == 0 and res < 1e-9 and err < 1e-7, (info, res, err)
            reports.append(
                dict(
                    case="grounded_graph_with_NNC" if graph_case else "Cartesian_Neumann",
                    unknowns=n,
                    iterations=count[0],
                    relative_residual=res,
                    relative_energy_error=err,
                    passed=True,
                )
            )
    out = OUTPUTS_DIR / "smoke.json"
    krylov.save_json(out, reports)
    print(f"[PASS] Smoke tests passed -> {out}")
    print(json.dumps(reports, indent=2))


def cmd_check_inputs() -> None:
    """Verify all 13 pressure systems across Experiments 1-5 load cleanly from data/inputs/."""
    from experiments.common import iter_cases
    total = 0
    for suite in ("static", "dynamic", "olympus", "synthetic"):
        ids = [c["id"] for c in iter_cases(suite)]
        total += len(ids)
        print(f"  [{suite:9s}] ({len(ids)} cases): {ids}")
    print(f"[PASS] All {total} pressure systems loaded cleanly.")


def main() -> None:
    p = argparse.ArgumentParser(description="ML-CPMG unified runner for manuscript experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="Fast Cartesian Neumann + faulted NNC graph verification")
    sub.add_parser("check-inputs", help="Verify all 13 benchmark cases load cleanly")

    p1 = sub.add_parser("exp1", help="Experiment 1: SPE10 static parameter calibration (Table 2, Fig. 3)")
    p1.add_argument("--cpu", type=int, default=None)

    p2 = sub.add_parser("exp2", help="Experiment 2: Six 64^3 continuous synthetic fields (Tables 3 & A.1, Figs. 4-6)")
    p2.add_argument("--case", default=None)
    p2.add_argument("--method", default=None)
    p2.add_argument("--repeats", type=int, default=3)
    p2.add_argument("--cpu", type=int, default=None)

    p3 = sub.add_parser("exp3", help="Experiment 3: 1.12M SPE10 static five-spot (Table 4, Figs. 7-8)")
    p3.add_argument("--method", default=None)
    p3.add_argument("--repeats", type=int, default=3)
    p3.add_argument("--cpu", type=int, default=None)

    p4 = sub.add_parser("exp4", help="Experiment 4: SPE10 dynamic mobility states (Table 5, Fig. 10a)")
    p4.add_argument("--case", default=None)
    p4.add_argument("--method", default=None)
    p4.add_argument("--repeats", type=int, default=3)
    p4.add_argument("--cpu", type=int, default=None)

    p5 = sub.add_parser("exp5", help="Experiment 5: Faulted OLYMPUS_1 active graph (Table 6, Figs. 9 & 10b)")
    p5.add_argument("--case", default=None)
    p5.add_argument("--method", default=None)
    p5.add_argument("--repeats", type=int, default=3)
    p5.add_argument("--cpu", type=int, default=None)

    sub.add_parser("exp6", help="Experiment 6: 16 explicit small-system algebraic & spectral checks (Table A.2)")

    pc = sub.add_parser("certify", help="Offline IEEE interval & SuperLU accuracy certification")
    pc.add_argument("--suites", nargs="+", default=["static", "dynamic", "olympus", "synthetic"])
    pc.add_argument("--cpu", type=int, default=None)

    sub.add_parser("tables", help="Regenerate all 8 manuscript LaTeX tables into outputs/tables/")

    pf = sub.add_parser("figures", help="Regenerate publication figures into outputs/figures/")
    pf.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=["stencil", "block", "fields", "segmentation", "performance", "spe10", "reservoir", "pareto", "olympus"],
    )

    args = p.parse_args()
    if args.cmd == "smoke":
        cmd_smoke()
    elif args.cmd == "check-inputs":
        cmd_check_inputs()
    elif args.cmd == "exp1":
        from experiments import exp1_calibration
        exp1_calibration.run(cpu=args.cpu)
    elif args.cmd == "exp2":
        from experiments import exp2_synthetic
        exp2_synthetic.run(case=args.case, method=args.method, repeats=args.repeats, cpu=args.cpu)
    elif args.cmd == "exp3":
        from experiments import exp3_spe10_static
        exp3_spe10_static.run(method=args.method, repeats=args.repeats, cpu=args.cpu)
    elif args.cmd == "exp4":
        from experiments import exp4_spe10_dynamic
        exp4_spe10_dynamic.run(case=args.case, method=args.method, repeats=args.repeats, cpu=args.cpu)
    elif args.cmd == "exp5":
        from experiments import exp5_olympus
        exp5_olympus.run(case=args.case, method=args.method, repeats=args.repeats, cpu=args.cpu)
    elif args.cmd == "exp6":
        from experiments import exp6_algebra_checks
        exp6_algebra_checks.run()
    elif args.cmd == "certify":
        from experiments import certify
        certify.run(suites=tuple(args.suites), cpu=args.cpu)
    elif args.cmd == "tables":
        from experiments import make_tables
        make_tables.run()
    elif args.cmd == "figures":
        from experiments import make_figures
        sys.argv = ["make_figures.py"] + (["--only", *args.only] if args.only else [])
        make_figures.main()


if __name__ == "__main__":
    main()
