"""Experiment 5 (Section 5.3, Table 6, Figs. 9 & 10b): Faulted OLYMPUS_1 active graph."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import OLYMPUS_METHODS, run_suite
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
        "olympus",
        methods=[method] if method else OLYMPUS_METHODS,
        case_filter=case,
        repeats=repeats,
        force_line=False,
        reliable=True,
        out_dir=out_dir or (OUTPUTS_DIR / "reruns" / "olympus"),
        cpu=cpu,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        default=None,
        help="Optional case ID (OLYMPUS-open, OLYMPUS-baffle, OLYMPUS-sealing-wet)",
    )
    p.add_argument("--method", default=None, choices=OLYMPUS_METHODS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cpu", type=int, default=None)
    args = p.parse_args()
    run(case=args.case, method=args.method, repeats=args.repeats, out_dir=args.out_dir, cpu=args.cpu)


if __name__ == "__main__":
    main()
