"""Independent offline accuracy certification (IEEE interval bounds & OLYMPUS SuperLU check)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .common import (
    OLYMPUS_METHODS,
    PRIMARY_METHODS,
    build_preconditioner,
    iter_cases,
    pin_single_cpu,
    selected_ml_params,
)
from ml_cpmg import krylov, solver as mg
from ml_cpmg.certification import grounded_cert as da, neumann_cert as sa
from ml_cpmg.double_double import load_checkpoint
from ml_cpmg.paths import OUTPUTS_DIR, RESULTS_DIR, SPE10_INPUTS_DIR


def _compact(q: dict) -> dict:
    keys = (
        "pressure_relative_upper",
        "flux_relative_upper",
        "accuracy_pass",
        "pressure_true_residual",
        "flux_true_residual",
        "rho_upper",
        "eta_upper",
        "reason",
    )
    return {k: q.get(k) for k in keys if k in q}


def run(
    *,
    data_dir: Path | None = None,
    out_file: Path | None = None,
    suites: tuple[str, ...] = ("static", "dynamic", "olympus", "synthetic"),
    cpu: int | None = None,
) -> list[dict]:
    """Audit stored first-repeat reliable pressure solutions without touching timing files."""
    pin_single_cpu(cpu)
    src_dir = data_dir or RESULTS_DIR
    out_path = out_file or (OUTPUTS_DIR / "certification" / "accuracy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap, stall = selected_ml_params()
    records = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []

    with threadpool_limits(1):
        mg.warmup_jit()
        for suite in suites:
            for case in iter_cases(suite):
                cid = case["id"]
                if any(r.get("case") == cid and r.get("complete_case") for r in records):
                    continue
                print("Assessing stored pressures:", cid, flush=True)
                names = OLYMPUS_METHODS if suite in ("olympus", "brugge") else PRIMARY_METHODS
                if suite in ("olympus", "brugge"):
                    A = mg._assemble_general_edges(
                        case["k"].size, case["edge"], well_diagonal=case["d"]
                    )
                    ref = splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(case["b"])
                    en = float(ref @ (A @ ref))
                    for name in names:
                        f = src_dir / f"{cid}_{name}_reliable.npz"
                        if not f.exists():
                            continue
                        x = np.load(f)["x"]
                        delta = x - ref
                        rec = dict(
                            case=cid,
                            method=name,
                            relative_energy_difference=float(
                                np.sqrt(max(0.0, delta @ (A @ delta)) / en)
                            ),
                            true_residual=float(
                                np.linalg.norm(case["b"] - A @ x) / np.linalg.norm(case["b"])
                            ),
                            comparison="direct SuperLU (MMD_AT_PLUS_A); numerical difference, not interval bound",
                        )
                        records.append(rec)
                        print(
                            f"  {name}: energy_diff={rec['relative_energy_difference']:.3e}, "
                            f"true_res={rec['true_residual']:.3e}",
                            flush=True,
                        )
                else:
                    if suite == "static":
                        z = np.load(SPE10_INPUTS_DIR / "static_reference.npz")
                        rh, rl = z["hi"], z["lo"]
                        cert = json.loads(
                            (SPE10_INPUTS_DIR / "static_reference.json").read_text(encoding="utf-8")
                        )
                    elif suite == "dynamic" and cid != "SPE10-0d":
                        day = float(cid[6:-1])
                        z = load_checkpoint(
                            SPE10_INPUTS_DIR / f"snapshot_{day:09.2f}.reference.npz.lz4"
                        )
                        rh, rl = z["hi"], z["lo"]
                        raw = json.loads(
                            (SPE10_INPUTS_DIR / f"snapshot_{day:09.2f}.reference.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        cert = raw.get("certificate", raw)
                    else:
                        ref_npz = src_dir / f"{cid}_reference.npz"
                        ref_json = src_dir / f"{cid}_reference.json"
                        if ref_npz.exists() and ref_json.exists():
                            z = np.load(ref_npz)
                            rh, rl = z["hi"], z["lo"]
                            cert = json.loads(ref_json.read_text(encoding="utf-8"))
                        else:
                            saved = np.load(src_dir / f"{cid}_ML-CPMG_reliable.npz")
                            x = (saved["hi"], saved["lo"])
                            A, M, _, _ = build_preconditioner(
                                case, "ML-CPMG", cap, stall, force_line=(suite == "dynamic")
                            )
                            if case["d"] is None:
                                rh, rl, cert = sa.make_reference(
                                    case["k"].shape,
                                    case["b"],
                                    case["edge"],
                                    A,
                                    preconditioner=M,
                                    initial_pressure=x,
                                    relative_target=5e-9,
                                )
                            else:
                                rh, rl, cert = da.make_reference(
                                    case["k"].shape,
                                    case["b"],
                                    case["edge"],
                                    case["d"],
                                    A,
                                    preconditioner=M,
                                    initial_pressure=x,
                                    relative_target=5e-9,
                                )
                            np.savez_compressed(
                                out_path.parent / f"{cid}_reference.npz", hi=rh, lo=rl
                            )
                            krylov.save_json(out_path.parent / f"{cid}_reference.json", cert)

                    for name in names:
                        f = src_dir / f"{cid}_{name}_reliable.npz"
                        if not f.exists():
                            continue
                        saved = np.load(f)
                        hi, lo = saved["hi"], saved["lo"]
                        if case["d"] is None:
                            q = sa.assess(
                                case["k"].shape, case["b"], case["edge"], hi, lo, rh, rl, cert
                            )
                        else:
                            q = da.assess(
                                case["k"].shape,
                                case["b"],
                                case["edge"],
                                case["d"],
                                hi,
                                lo,
                                rh,
                                rl,
                                cert,
                            )
                        records.append(dict(case=cid, method=name, **_compact(q)))
                        print(f"  {name}: {_compact(q)}", flush=True)

                records.append(dict(case=cid, complete_case=True))
                krylov.save_json(out_path, records)

    print("Certification complete ->", out_path, flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-file", type=Path, default=None)
    p.add_argument(
        "--suites", nargs="+", default=["static", "dynamic", "olympus", "synthetic"]
    )
    p.add_argument("--cpu", type=int, default=None)
    args = p.parse_args()
    run(data_dir=args.data_dir, out_file=args.out_file, suites=tuple(args.suites), cpu=args.cpu)


if __name__ == "__main__":
    main()
