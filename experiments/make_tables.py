"""Deterministically generate all 8 manuscript LaTeX tables from data/results/."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_cpmg.paths import OUTPUTS_DIR, RESULTS_DIR

PRIMARY = ["ML-CPMG", "Min-threshold", "Single-median", "Algebraic-only", "SA-0.04", "RS-0.25"]
ALL = PRIMARY[:4] + ["SA-0", "SA-0.04", "SA-0.1", "RS-0.25"]
FAMILIES = [
    "multilevel_nested_lobes",
    "multilevel_braided_channels",
    "multilevel_layered_barriers",
    "multilevel_anisotropic_streaks",
    "unplanted_random",
    "smooth_gaussian",
]
NAMES = dict(
    zip(
        FAMILIES,
        ["Nested lobes", "Braided channels", "Layered barriers", "Anisotropic streaks", "Unplanted random", "Gaussian hill"],
    )
)


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _dump(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _csvwrite(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def _sci(x):
    if x is None:
        return "--"
    a, b = f"{x:.2e}".split("e")
    return rf"${a}\times10^{{{int(b)}}}$"


def _num(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def _write_table(tab_dir: Path, name: str, caption: str, headers: list[str], rows: list[list], note: str = "", wide: bool = True, placement: str = "!tbp"):
    env = "table*" if wide else "table"
    s = [
        f"\\begin{{{env}}}[pos={placement}]\\centering\\small",
        f"\\caption{{{caption}}}\\label{{tab:{name}}}",
        "\\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r"\\",
        r"\midrule",
    ]
    s += [" & ".join(str(x) for x in row) + r"\\" for row in rows]
    s += [r"\bottomrule", r"\end{tabular}"]
    if note:
        s += [r"\par\smallskip\footnotesize " + note]
    s += [f"\\end{{{env}}}"]
    if not wide:
        s[0] = r"\par\medskip\noindent\begin{minipage}{\linewidth}\centering\sffamily\small"
        s[1] = s[1].replace(r"\caption{", r"\captionof{table}{")
        s[-1] = r"\end{minipage}\par\medskip"
    (tab_dir / f"{name}.tex").write_text("\n".join(s) + "\n", encoding="utf-8")
    return dict(name=name, caption=caption, headers=headers, rows=rows, note=note)


def run(data_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    src = data_dir or RESULTS_DIR
    out = out_dir or (OUTPUTS_DIR / "tables")
    out.mkdir(parents=True, exist_ok=True)

    raw, summaries = [], []
    for suite, expected in [("static", 18), ("olympus", 72), ("synthetic", 108), ("dynamic", 54)]:
        rows = _read(src / f"{suite}_reliable.json")
        bad = [r for r in rows if "error" in r or not r.get("converged")]
        assert not bad, (suite, bad)
        assert len(rows) == expected, (suite, len(rows), expected)
        grouped = collections.defaultdict(list)
        for r in rows:
            r = dict(r, suite=suite)
            raw.append(r)
            grouped[r["case"], r["method"]].append(r)
        for (case, method), rr in grouped.items():
            assert sorted(r["rep"] for r in rr) == [0, 1, 2]
            first = next(r for r in rr if r["rep"] == 0)
            s = {k: v for k, v in first.items() if not isinstance(v, (list, dict)) and k not in ("rep", "trace_rel_res")}
            for k in ("setup_seconds", "solve_seconds", "total_seconds", "pcg_initial_seconds", "dd_refine_seconds"):
                s[k] = statistics.median(r[k] for r in rr)
                s[k + "_min"] = min(r[k] for r in rr)
                s[k + "_max"] = max(r[k] for r in rr)
            s["repeats"] = 3
            s["rho_dd"] = max(r["rho_dd"] for r in rr)
            s["eta_dd"] = max(r["eta_dd"] for r in rr)
            s["iterations_min"] = min(r["iterations"] for r in rr)
            s["iterations_max"] = max(r["iterations"] for r in rr)
            summaries.append(s)

    _dump(out / "summary.json", summaries)
    _csvwrite(out / "summary.csv", summaries)
    _csvwrite(out / "all_reliable_runs.csv", [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in raw])

    S = {(r["case"], r["method"]): r for r in summaries}
    acc = {(r["case"], r["method"]): r for r in _read(src / "accuracy.json") if "method" in r}
    assert len(acc) == 84, ("first-repeat assessments", len(acc))
    certified = [q for q in acc.values() if "accuracy_pass" in q]
    assert len(certified) == 60 and all(q["accuracy_pass"] for q in certified)

    tables = []
    # Table 1: fixed configuration table
    cfg_tex = src / "configuration.tex"
    if src.resolve() == RESULTS_DIR.resolve() or not cfg_tex.is_file():
        cfg_tex = Path(__file__).with_name("configuration.tex")
    shutil.copy2(cfg_tex, out / "configuration.tex")

    # Table 2: Experiment 1 calibration
    calib = [r for r in _read(src / "calibrate.json") if "error" not in r]
    calib.sort(key=lambda r: (0 if r["method"] == "ML-CPMG" else 1, r["method"], r["stall"], r["cap"]))
    rows = [
        [
            r["method"],
            r["cap"] if r["method"] == "ML-CPMG" else "--",
            r["stall"] if r["method"] == "ML-CPMG" else "--",
            r["l1_coarse_size"],
            _num(r["operator_complexity"], 3),
            r["iterations"],
            _num(r["setup_seconds"]),
            _num(r["solve_seconds"]),
            _num(r["total_seconds"]),
        ]
        for r in calib
    ]
    tables.append(
        _write_table(
            out,
            "calibration",
            "Exploratory SPE10 static calibration with the automatic fine-smoother rule.",
            ["Method", r"$B_{\max}$", r"$\eta_{\rm stall}$", r"$n_1$", r"$C_{\rm op}$", "PCG", "Setup (s)", "PCG (s)", "Cold (s)"],
            rows,
            "One run per configuration, before the common conservation closure. The automatic line trigger is inactive. These measurements support parameter exploration; validation timings include closure and use matched vertical line relaxation for internal methods.",
        )
    )

    # Table 3: Experiment 2 separation
    rows = []
    for fam in FAMILIES[:4]:
        for method in PRIMARY[:3]:
            r = S["Synthetic-" + fam, method]
            rows.append(
                [
                    NAMES[fam],
                    method,
                    f"{r['distinct_cores']}/{r['planted_peaks']}",
                    f"{r['separated_pairs']}/{r['close_pairs']}",
                    _num(r["mean_core_coverage"], 3),
                    _num(r["mean_core_purity"], 3),
                    r["effective_components"],
                ]
            )
    tables.append(
        _write_table(
            out,
            "separation",
            "Structural recovery on the four planted continuous fields.",
            ["Field", "Label rule", "Distinct cores", "Separated pairs", "Coverage", "Purity", "Labels"],
            rows,
            "Distinct-core resolution and pair separation use positive dominant labels. Coverage and purity are averaged over the prescribed core masks. The fields were constructed with identifiable cores and barriers.",
        )
    )

    def perfrows(cases, methods):
        out_rows = []
        for cid, label in cases:
            for method in methods:
                r = S[cid, method]
                q = acc[cid, method]
                error = q.get("pressure_relative_upper", q.get("relative_energy_difference"))
                out_rows.append(
                    [
                        label,
                        method,
                        r["iterations"],
                        _num(r["operator_complexity"], 3),
                        _num(r["setup_seconds"], 3),
                        _num(r["solve_seconds"], 3),
                        _num(r["total_seconds"], 3),
                        _sci(error),
                    ]
                )
        return out_rows

    hdr = ["Case", "Method", "Iterations", r"$C_{\rm op}$", "Setup (s)", "Solve (s)", "Cold (s)", "Pressure error"]
    note = "Times are medians of three repetitions. Solve time includes physical-residual checks and double-double defect closure; iterations include correction solves. Error entries are first-repeat outward-rounded relative energy bounds."
    tables.append(_write_table(out, "static", "SPE10 static pressure with identical vertical line relaxation in all internal methods.", hdr, perfrows([("SPE10-static", "Static")], PRIMARY), note))
    tables.append(_write_table(out, "dynamic", "SPE10 pressure solves at identical fixed saturation states.", hdr, perfrows([("SPE10-0d", "0 d"), ("SPE10-520.04d", "520.04 d"), ("SPE10-2000d", "2000 d")], PRIMARY), note + " All solvers are evaluated on identical prescribed saturation states to isolate linear-solver performance from transport-trajectory drift."))
    tables.append(_write_table(out, "olympus", r"Pressure-system comparisons on the faulted OLYMPUS\_1 corner-point active graph.", hdr, perfrows([("OLYMPUS-open", "Open"), ("OLYMPUS-baffle", "Baffle"), ("OLYMPUS-sealing-wet", "Sealing-wet")], ALL), "Times are medians of three repetitions and include common conservation closure. All four internal methods use vertical pillar-line relaxation. Only the three geometric variants semi-coarsen along pillars. The error column gives the relative energy difference from a floating-point SuperLU solution. All SA strength thresholds are retained."))
    tables.append(_write_table(out, "synthetic_full", "Complete solver comparison on the six continuous synthetic fields.", hdr, perfrows([("Synthetic-" + f, NAMES[f]) for f in FAMILIES], PRIMARY), note, wide=False, placement="H"))

    def _space_label(r: dict) -> str:
        if r.get("domain") == "active_graph_faulted":
            ftm = float(r["ftm"])
            return "AG (FTM=1)" if abs(ftm - 1.0) < 1e-12 else "AG (FTM=1e-4)"
        return "Grounded" if r["grounded"] else "Neumann"

    checks = _read(src / "algebra_checks.json")
    rows = [
        [
            _space_label(r),
            r["seed"],
            "Line" if r["line"] else "Point",
            _num(r["min_preconditioned_eigenvalue"], 4),
            _num(r["max_preconditioned_eigenvalue"], 8),
            _num(r["energy_contraction"], 4),
            _sci(r["symmetry_defect"]),
        ]
        for r in checks
    ]
    tables.append(
        _write_table(
            out,
            "algebra_checks",
            "Small-system regression checks of the algebraic construction.",
            ["Space", "Seed", "Smoother", r"$\lambda_{\min}$", r"$\lambda_{\max}$", r"$\|E\|_A$", "Symmetry defect"],
            rows,
            r"The first twelve cases have 216 fine unknowns on $6^3$ Cartesian grids; the last four have 288 fine unknowns on grounded thin-layer faulted active graphs exercising vertical semi-coarsening, pillar-line $LDL^\top$ relaxation, and intra-box barrier splitting. Every prolongator has full column rank, and coarse energy increments are positive semidefinite to rounding accuracy; the general result is Proposition 2.",
            wide=False,
            placement="H",
        )
    )

    facts = dict(
        runs=len(raw),
        cases=len(set(r["case"] for r in raw)),
        all_conservation_pass=all(r["gate_1e9_pass"] for r in raw),
        max_rho=max(r["rho_dd"] for r in raw),
        max_eta=max(r["eta_dd"] for r in raw),
        certified_first_repeats=sum("accuracy_pass" in q for q in acc.values()),
        interval_passes=sum(bool(q.get("accuracy_pass")) for q in acc.values()),
        max_pressure_error=max(q["pressure_relative_upper"] for q in acc.values() if "pressure_relative_upper" in q),
        max_flux_error=max(q["flux_relative_upper"] for q in acc.values() if "flux_relative_upper" in q),
        max_olympus_difference=max(q["relative_energy_difference"] for q in acc.values() if "relative_energy_difference" in q),
    )
    _dump(out / "tables.json", tables)
    _dump(out / "facts.json", facts)
    print(f"[PASS] Generated 8 LaTeX tables & summary facts -> {out}")
    print(json.dumps(facts, indent=2))
    return facts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    run(data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
