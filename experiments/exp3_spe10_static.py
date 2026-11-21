"""Experiment 3 (Section 5.1, Table 4, Figs. 7-8): 1.12M-cell SPE10 static five-spot pressure."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import PRIMARY_METHODS, run_suite
from ml_cpmg.paths import OUTPUTS_DIR


def run(
    *,
    method: str | None = None,
    repeats: int = 3,
    out_dir: Path | None = None,
    cpu: int | None = None,
) -> list[dict]:
    return run_suite(
        "static",
        methods=[method] if method else PRIMARY_METHODS,
        repeats=repeats,
        force_line=True,
        reliable=True,
        out_dir=out_dir or (OUTPUTS_DIR / "exp3_spe10_static"),
        cpu=cpu,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default=None, choices=PRIMARY_METHODS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cpu", type=int, default=None)
    args = p.parse_args()
    run(method=args.method, repeats=args.repeats, out_dir=args.out_dir, cpu=args.cpu)


if __name__ == "__main__":
    main()
