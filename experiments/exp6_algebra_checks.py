"""Experiment 6 (Appendix A.2, Table A.2): 16 explicit small-system algebraic & spectral checks.

Covers:
  - 12 Cartesian 6x6x6 checks (pure-Neumann & well-grounded x seeds {7, 11, 29} x {Point, Line});
  - 4 grounded thin-layer faulted ActiveGraph checks (seeds {7, 11} x FTM in {1.0, 1e-4}),
    verifying anisotropic z-axis semi-coarsening, 1D ActiveGraph column LDL^T line smoothing,
    and intra-tag barrier-respecting box splitting simultaneously.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from .common import PROJECT_ROOT  # noqa: F401 (ensures src/ is on sys.path)
from ml_cpmg import grid, krylov, solver as mg
from ml_cpmg.paths import OUTPUTS_DIR


def _verify_levels_and_spectrum(A, levels, apply_fn, grounded: bool) -> tuple[list[dict], float, np.ndarray]:
    """Verify full-rank P, partition of unity, PSD energy increments, symmetry & spectrum in (0, 1]."""
    n = A.shape[0]
    details = []
    for lv, nxt in zip(levels[:-1], levels[1:]):
        P = lv["P"].toarray()
        G = P.T @ lv["A"].toarray() @ P
        C = nxt["A"].toarray()
        rank = int(np.linalg.matrix_rank(P))
        diff = linalg.eigvalsh(C - G)
        assert rank == P.shape[1], f"Prolongator rank deficient: {rank} != {P.shape[1]}"
        assert np.max(abs(P.sum(axis=1) - 1.0)) < 1e-12
        assert diff.min() > -1e-9 * max(1.0, linalg.norm(C, 2))
        details.append(
            dict(
                n=lv["A"].shape[0],
                nc=P.shape[1],
                rank=rank,
                min_energy_increment=float(diff.min()),
                lumped=bool(np.max(abs(C - G)) > 1e-10),
            )
        )

    eye = np.eye(n)
    B = np.column_stack([apply_fn(eye[:, i]) for i in range(n)])
    sym_defect = float(linalg.norm(B - B.T) / linalg.norm(B))
    assert sym_defect < 1e-11, f"Asymmetry defect {sym_defect}"
    U = np.eye(n) if grounded else linalg.null_space(np.ones((1, n)))
    Ar = U.T @ A.toarray() @ U
    Br = U.T @ B @ U
    w, V = linalg.eigh(Ar)
    Ah = (V * np.sqrt(w)) @ V.T
    spec = linalg.eigvalsh(Ah @ Br @ Ah)
    assert spec.min() > 0.0 and spec.max() < 1.0 + 1e-10, (spec.min(), spec.max())
    return details, sym_defect, spec


def check_one(grounded: bool, seed: int, line: bool = False) -> dict:
    """Explicitly form V-cycle operator B on a 6x6x6 Cartesian grid and verify Propositions 1 & 2."""
    rng = np.random.default_rng(seed)
    shape = (6, 6, 6)
    n = int(np.prod(shape))
    k = np.exp(rng.uniform(0.0, np.log(100.0), shape))
    edge = grid.edges(k)
    d = np.linspace(0.1, 0.3, n) if grounded else None
    A = mg.assemble_cartesian_fast(shape, edge, d=d)
    xyz = np.indices(shape).reshape(3, -1).T
    tags = (k.ravel() > np.median(k)).astype(np.int64)
    cfg = dict(mg._BASE_CONTROLS, coarse_size=16, max_direct=512)
    levels, _ = mg._construct_hierarchy(
        A,
        xyz,
        tags,
        cfg,
        grounded=grounded,
        cartesian_fine=not grounded,
        split_boxes_by_connectivity=False,
    )
    if line:
        levels[0]["line_smoother"] = mg._build_anisotropic_line_smoother(
            shape, edge, A.diagonal(), omega=levels[0]["omega"], forced_axis=0
        )

    apply_fn, _ = mg._make_cycle_callable(levels, grounded=grounded, cycle_repeats=1, fine_A=A)
    details, sym_defect, spec = _verify_levels_and_spectrum(A, levels, apply_fn, grounded)
    return dict(
        domain="cartesian_6x6x6",
        grounded=grounded,
        seed=seed,
        line=line,
        ftm=None,
        n_fine=n,
        levels=details,
        symmetry_defect=sym_defect,
        min_preconditioned_eigenvalue=float(spec.min()),
        max_preconditioned_eigenvalue=float(spec.max()),
        energy_contraction=float(1.0 - spec.min()),
    )


def check_active_graph_one(seed: int, ftm: float) -> dict:
    """Explicitly form V-cycle operator B on a grounded thin-layer faulted ActiveGraph (288 DOFs)."""
    rng = np.random.default_rng(seed)
    shape = (6, 6, 8)
    nx, ny, nz = shape
    n = int(np.prod(shape))
    xyz = np.ascontiguousarray(np.indices(shape, dtype=np.int64).reshape(3, -1).T)
    ids = np.arange(n, dtype=np.int64).reshape(shape)

    k = np.exp(rng.uniform(0.0, np.log(8.0), shape))
    k[:, 1:3, :] *= 200.0
    k[:, 4:5, :6] *= 800.0
    k_flat = k.ravel()

    tails, heads, trans_list = [], [], []
    # Axis 0 (x): fault plane between x=2 and x=3 (straddling 2x2x2 box bx=1: x in {2, 3}!)
    for ix in range(nx - 1):
        for iy in range(ny):
            for iz in range(nz):
                u = int(ids[ix, iy, iz])
                if ix == 2:
                    # Displaced fault NNC with vertical throw dz = +2 and multiplier FTM
                    v = int(ids[3, iy, (iz + 2) % nz])
                    mult = float(ftm)
                else:
                    v = int(ids[ix + 1, iy, iz])
                    mult = 1.0
                ku, kv = k_flat[u], k_flat[v]
                tails.append(u)
                heads.append(v)
                trans_list.append(mult * (2.0 * ku * kv / (ku + kv)))

    # Axis 1 (y): regular horizontal connections (scale 1.0)
    for ix in range(nx):
        for iy in range(ny - 1):
            for iz in range(nz):
                u = int(ids[ix, iy, iz])
                v = int(ids[ix, iy + 1, iz])
                ku, kv = k_flat[u], k_flat[v]
                tails.append(u)
                heads.append(v)
                trans_list.append(2.0 * ku * kv / (ku + kv))

    # Axis 2 (z): thin-pancake vertical connections (20x geometric aspect -> triggers semi-coarsening & line LDL^T)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz - 1):
                u = int(ids[ix, iy, iz])
                v = int(ids[ix, iy, iz + 1])
                ku, kv = k_flat[u], k_flat[v]
                tails.append(u)
                heads.append(v)
                trans_list.append(20.0 * (2.0 * ku * kv / (ku + kv)))

    g = grid.ActiveGraph.from_arrays(
        xyz,
        np.asarray(tails, dtype=np.int64),
        np.asarray(heads, dtype=np.int64),
        np.asarray(trans_list, dtype=np.float64),
        shape,
    )
    # Ground both fault compartments (hanging wall x=0 and footwall x=5)
    d = np.zeros(n, dtype=np.float64)
    d[ids[0, :, :].ravel()] = 0.20
    d[ids[nx - 1, :, :].ravel()] = 0.20

    A, M, meta, levels = mg.build_active_graph(
        g,
        k_flat,
        d=d,
        config=dict(mg._BASE_CONTROLS),
        coarse_size=16,
        max_direct=512,
        _return_levels=True,
    )
    assert meta["anisotropic_line_smoother"] is True, "Expected ActiveGraph 1D line smoother to trigger"
    assert any(s.startswith("semi_axis_2") for s in meta["geometric_schedule"]), (
        f"Expected z-axis semi-coarsening in schedule: {meta['geometric_schedule']}"
    )
    assert len(levels) >= 3, f"Expected >= 3 levels, got {len(levels)}"

    details, sym_defect, spec = _verify_levels_and_spectrum(
        A, levels, lambda v: np.asarray(M @ v, dtype=np.float64).ravel(), grounded=True
    )
    return dict(
        domain="active_graph_faulted",
        grounded=True,
        seed=seed,
        line=True,
        ftm=float(ftm),
        n_fine=n,
        high_component_count=int(meta["high_component_count"]),
        geometric_schedule=meta["geometric_schedule"],
        levels=details,
        symmetry_defect=sym_defect,
        min_preconditioned_eigenvalue=float(spec.min()),
        max_preconditioned_eigenvalue=float(spec.max()),
        energy_contraction=float(1.0 - spec.min()),
    )


def run(out_dir: Path | None = None) -> list[dict]:
    dest = out_dir or (OUTPUTS_DIR / "exp6_algebra")
    dest.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        mg.warmup_jit()
        rows = [
            check_one(g, s, line)
            for g in (False, True)
            for s in (7, 11, 29)
            for line in (False, True)
        ]
        for seed in (7, 11):
            for ftm in (1.0, 1.0e-4):
                rows.append(check_active_graph_one(seed, ftm))
    krylov.save_json(dest / "algebra_checks.json", rows)
    print(
        f"[PASS] Experiment 6: {len(rows)}/16 explicit algebraic & spectral checks passed -> "
        f"{dest / 'algebra_checks.json'}"
    )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    rows = run(out_dir=args.out_dir)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "domain",
                        "grounded",
                        "seed",
                        "line",
                        "ftm",
                        "min_preconditioned_eigenvalue",
                        "max_preconditioned_eigenvalue",
                        "energy_contraction",
                    )
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
