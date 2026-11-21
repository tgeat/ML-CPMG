"""Experiment 1 (Section 4.2, Table 2, Fig. 3): SPE10 static complexity-convergence calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits

from .common import CALIBRATION_METHODS, iter_cases, pin_single_cpu, run_single
from ml_cpmg import krylov
from ml_cpmg.paths import OUTPUTS_DIR


def run(out_dir: Path | None = None, cpu: int | None = None) -> dict:
    pin_single_cpu(cpu)
    dest = out_dir or (OUTPUTS_DIR / "exp1_calibration")
    dest.mkdir(parents=True, exist_ok=True)
    case = next(iter_cases("static"))
    rows: list[dict] = []
    with threadpool_limits(1):
        # Scan ML-CPMG basin caps and algebraic fallback thresholds (automatic smoother, pre-closure)
        for stall in (0.33, 0.20):
            for cap in (32, 64, 128, 256, 512):
                rows.append(run_single(case, "ML-CPMG", 0, cap=cap, stall=stall, force_line=False, reliable=False))
                krylov.save_json(dest / "calibrate.json", rows)
        # Scan internal controls and PyAMG SA/RS thresholds
        for method in CALIBRATION_METHODS[1:]:
            rows.append(run_single(case, method, 0, cap=256, stall=0.33, force_line=False, reliable=False))
            krylov.save_json(dest / "calibrate.json", rows)

    feasible = [r for r in rows if r.get("converged") and r["method"] == "ML-CPMG" and r["operator_complexity"] <= 3.0]
    fastest = min(r["total_seconds"] for r in feasible)
    acceptable = [r for r in feasible if r["total_seconds"] <= 1.05 * fastest]
    chosen = min(acceptable, key=lambda r: r["operator_complexity"])
    selected = dict(max_basins=chosen["cap"], stall_ratio=chosen["stall"], prolong_omega=0.45, primary_SA_theta=0.04)
    krylov.save_json(dest / "selected_config.json", selected)
    krylov.save_csv(
        dest / "calibrate_clean.csv",
        [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows if "error" not in r],
    )
    print("Selected calibration:", json.dumps(selected, indent=2), flush=True)
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cpu", type=int, default=None)
    args = p.parse_args()
    run(out_dir=args.out_dir, cpu=args.cpu)


if __name__ == "__main__":
    main()
