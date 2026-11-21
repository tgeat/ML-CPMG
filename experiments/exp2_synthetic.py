"""Experiment 2 (Section 4.3, Tables 3 & A.1, Figs. 4-6): Six 64^3 continuous synthetic fields."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import PRIMARY_METHODS, run_suite
from ml_cpmg.paths import OUTPUTS_DIR


def run(
    *,
    case: str | None = None,
    method: str | None = None,
    repeats: int = 3,
    out_dir: Path | None = None,
    cpu: int | None = None,
) -> list[dict]:
    return run_suite(
        "synthetic",
        methods=[method] if method else PRIMARY_METHODS,
        case_filter=case,
        repeats=repeats,
        force_line=False,
        reliable=True,
        out_dir=out_dir or (OUTPUTS_DIR / "exp2_synthetic"),
        cpu=cpu,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default=None, help="Optional case ID (e.g. Synthetic-smooth_gaussian)")
    p.add_argument("--method", default=None, choices=PRIMARY_METHODS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cpu", type=int, default=None)
    args = p.parse_args()
    run(case=args.case, method=args.method, repeats=args.repeats, out_dir=args.out_dir, cpu=args.cpu)


if __name__ == "__main__":
    main()
