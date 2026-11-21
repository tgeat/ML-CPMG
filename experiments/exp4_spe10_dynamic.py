"""Experiment 4 (Section 5.2, Table 5, Fig. 10a): SPE10 evolving-mobility states (0, 520.04, 2000 d)."""
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
        "dynamic",
        methods=[method] if method else PRIMARY_METHODS,
        case_filter=case,
        repeats=repeats,
        force_line=True,
        reliable=True,
        out_dir=out_dir or (OUTPUTS_DIR / "exp4_spe10_dynamic"),
        cpu=cpu,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default=None, help="Optional case ID (SPE10-0d, SPE10-520.04d, SPE10-2000d)")
    p.add_argument("--method", default=None, choices=PRIMARY_METHODS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cpu", type=int, default=None)
    args = p.parse_args()
    run(case=args.case, method=args.method, repeats=args.repeats, out_dir=args.out_dir, cpu=args.cpu)


if __name__ == "__main__":
    main()
