"""Build the three OLYMPUS_1 pressure benchmark inputs from the TNO archive."""
from __future__ import annotations

import os

for _env_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_env_key] = "1"

from collections import defaultdict, deque
import hashlib
import json
import math
import re
import shlex
import sys
import time
from pathlib import Path
import tarfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numba
import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

numba.set_num_threads(1)

from ml_cpmg import grid, krylov, reservoirs, solver as mg
from ml_cpmg.grid import ActiveGraph

MODEL_DIR = PROJECT_ROOT / "outputs" / "olympus_preparation"
RAW_DIR = MODEL_DIR / "raw"
EXTRACTED_DIR = RAW_DIR / "extracted"

EXPECTED_TAR_URL = "https://raw.githubusercontent.com/TNO/OLYMPUS/main/OLYMPUS_1.tar.gz"
EXPECTED_REPO_URL = "https://github.com/TNO/OLYMPUS"
EXPECTED_TAR_SHA256 = "3c2e04b451a1fef5c78d92e5cd708146f90b8a39ac89b3acf0f28954ed70cd0d"
EXPECTED_README_URL = "https://raw.githubusercontent.com/TNO/OLYMPUS/main/README.md"
EXPECTED_README_SHA256 = "9540a7f775e9be23dcfe14b35801f0a8ae384dd33706b1129eebec5f1f3c3cd9"

# Exact Eclipse METRIC Darcy constant:
# 1 mD * 1 bar / 1 cP * 86400 s/day = 9.869232667160128e-16 * 1e5 / 1e-3 * 86400
CDARCY_METRIC = 0.008527016960303706

PRODUCER_BHP_BAR = 150.0
SOURCE_INJECTOR_BHP_BAR = 235.0
TOTAL_INJECTION_M3_DAY = 10000.0
WATERFLOOD_DAYS = 1000.0
WATERFLOOD_STEP_DAYS = 50.0

FAULT_NAME_TO_ID = {
    "FAULT2": 2,
    "FAULT3": 3,
    "FAULT4": 4,
    "FAULT5": 5,
    "FAULT6": 6,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_grdecl_numeric_keyword(path: Path, keyword: str) -> np.ndarray:
    """Parse a numeric ECLIPSE keyword array with ``n*val`` repeat expansion."""
    raw_text = path.read_text(encoding="ascii")
    clean = re.sub(r"--[^\r\n]*", "", raw_text)
    match = re.search(r"(?im)^\s*" + re.escape(keyword) + r"\s*$", clean)
    if not match:
        raise ValueError(f"Keyword {keyword!r} not found in {path}")
    body = clean[match.end() :].split("/", 1)[0]
    values: list[float] = []
    for tok in body.split():
        parts = tok.split("*")
        if len(parts) == 1:
            values.append(float(parts[0].replace("D", "E").replace("d", "e")))
        elif len(parts) == 2 and parts[0].isdigit() and parts[1]:
            values.extend([float(parts[1].replace("D", "E").replace("d", "e"))] * int(parts[0]))
        else:
            raise ValueError(f"Unsupported repeat/default token {tok!r} in {keyword} ({path.name})")
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values in {keyword} ({path.name})")
    return arr


def _expand_eclipse_tokens(text: str) -> list[str]:
    raw_tokens = shlex.split(text, posix=True)
    expanded: list[str] = []
    for tok in raw_tokens:
        if "*" in tok:
            left, right = tok.split("*", 1)
            if left.isdigit() and right == "":
                expanded.extend(["*"] * int(left))
                continue
            if left.isdigit() and right != "":
                expanded.extend([right] * int(left))
                continue
        expanded.append(tok)
    return expanded


def parse_single_record(path: Path, keyword: str) -> list[str]:
    """Parse a single slash-terminated ECLIPSE keyword record (e.g. DIMENS, SPECGRID)."""
    lines = path.read_text(encoding="ascii").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        stripped = line.split("--", 1)[0].strip()
        if stripped.upper() == keyword.upper():
            start_idx = idx + 1
            break
    if start_idx is None:
        raise ValueError(f"Keyword {keyword!r} not found in {path}")
    buf: list[str] = []
    for line in lines[start_idx:]:
        clean = line.split("--", 1)[0].strip()
        if not clean:
            continue
        if "/" in clean:
            before_slash = clean.split("/", 1)[0].strip()
            if before_slash:
                buf.append(before_slash)
            return _expand_eclipse_tokens(" ".join(buf))
        buf.append(clean)
    raise ValueError(f"Unterminated single record {keyword!r} in {path}")


def parse_slash_records(path: Path, keyword: str) -> list[list[str]]:
    """Parse multi-record ECLIPSE table blocks terminated by an empty ``/`` record."""
    lines = path.read_text(encoding="ascii").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        stripped = line.split("--", 1)[0].strip()
        if stripped.upper() == keyword.upper():
            start_idx = idx + 1
            break
    if start_idx is None:
        raise ValueError(f"Keyword {keyword!r} not found in {path}")
    records: list[list[str]] = []
    buf: list[str] = []
    for line in lines[start_idx:]:
        clean = line.split("--", 1)[0].strip()
        if not clean:
            continue
        if clean == "/" and not buf:
            return records
        if clean.endswith("/"):
            piece = clean[:-1].strip()
            if piece:
                buf.append(piece)
            records.append(_expand_eclipse_tokens(" ".join(buf)))
            buf = []
        else:
            buf.append(clean)
    raise ValueError(f"Unterminated {keyword!r} section in {path}")


def apply_mapaxes(coord_raw: np.ndarray, mapaxes: np.ndarray) -> np.ndarray:
    """Transform local ECLIPSE pillar coordinates into map coordinates [m]."""
    xy, yy, x0, y0, xx, yx = map(float, mapaxes)
    vx = np.array([xx - x0, yx - y0], dtype=np.float64)
    vy = np.array([xy - x0, yy - y0], dtype=np.float64)
    ux = vx / np.linalg.norm(vx)
    uy = vy / np.linalg.norm(vy)
    out = coord_raw.copy()
    x_top, y_top = coord_raw[..., 0], coord_raw[..., 1]
    x_bot, y_bot = coord_raw[..., 3], coord_raw[..., 4]
    out[..., 0] = x0 + x_top * ux[0] + y_top * uy[0]
    out[..., 1] = y0 + x_top * ux[1] + y_top * uy[1]
    out[..., 3] = x0 + x_bot * ux[0] + y_bot * uy[0]
    out[..., 4] = y0 + x_bot * ux[1] + y_bot * uy[1]
    return out


def compute_corner_points_and_volumes(
    coord_map: np.ndarray, zc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact 3D corner coordinates and 2x2x2 Gauss-Legendre trilinear bulk volumes."""
    nz, _, ny, _, nx, _ = zc.shape
    iy_idx = np.arange(ny)[:, None] + np.arange(2)[None, :]
    ix_idx = np.arange(nx)[:, None] + np.arange(2)[None, :]
    pil = coord_map[iy_idx[:, :, None, None], ix_idx[None, None, :, :]]  # (ny, 2, nx, 2, 6)
    zt = pil[..., 2][None, None, ...]
    zb = pil[..., 5][None, None, ...]
    dzp = zb - zt
    safe_dzp = np.where(np.abs(dzp) > 1e-12, dzp, 1.0)
    t_param = np.where(np.abs(dzp) > 1e-12, (zc - zt) / safe_dzp, 0.0)
    pts = np.stack(
        [
            pil[..., 0][None, None, ...] + t_param * (pil[..., 3] - pil[..., 0])[None, None, ...],
            pil[..., 1][None, None, ...] + t_param * (pil[..., 4] - pil[..., 1])[None, None, ...],
            zc,
        ],
        axis=-1,
    )  # (nz, 2, ny, 2, nx, 2, 3)

    # Exact degree-3 trilinear hexahedron Jacobian integral via 2x2x2 Gauss-Legendre points
    gp = np.array([0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0)], dtype=np.float64)
    bulk_vol = np.zeros((nz, ny, nx), dtype=np.float64)
    for u in gp:
        for v in gp:
            for w in gp:
                x_u = (
                    (1.0 - w) * (1.0 - v) * (pts[:, 0, :, 0, :, 1] - pts[:, 0, :, 0, :, 0])
                    + (1.0 - w) * v * (pts[:, 0, :, 1, :, 1] - pts[:, 0, :, 1, :, 0])
                    + w * (1.0 - v) * (pts[:, 1, :, 0, :, 1] - pts[:, 1, :, 0, :, 0])
                    + w * v * (pts[:, 1, :, 1, :, 1] - pts[:, 1, :, 1, :, 0])
                )
                x_v = (
                    (1.0 - w) * (1.0 - u) * (pts[:, 0, :, 1, :, 0] - pts[:, 0, :, 0, :, 0])
                    + (1.0 - w) * u * (pts[:, 0, :, 1, :, 1] - pts[:, 0, :, 0, :, 1])
                    + w * (1.0 - u) * (pts[:, 1, :, 1, :, 0] - pts[:, 1, :, 0, :, 0])
                    + w * u * (pts[:, 1, :, 1, :, 1] - pts[:, 1, :, 0, :, 1])
                )
                x_w = (
                    (1.0 - v) * (1.0 - u) * (pts[:, 1, :, 0, :, 0] - pts[:, 0, :, 0, :, 0])
                    + (1.0 - v) * u * (pts[:, 1, :, 0, :, 1] - pts[:, 0, :, 0, :, 1])
                    + v * (1.0 - u) * (pts[:, 1, :, 1, :, 0] - pts[:, 0, :, 1, :, 0])
                    + v * u * (pts[:, 1, :, 1, :, 1] - pts[:, 0, :, 1, :, 1])
                )
                det_j = np.sum(x_u * np.cross(x_v, x_w), axis=-1)
                bulk_vol += 0.125 * np.abs(det_j)
    return pts, bulk_vol


def _clip_polygon_halfplane(
    poly: list[tuple[float, float]], a: float, b: float, c: float
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip against half-plane ``a * s + b * z + c >= 0``."""
    if not poly:
        return []
    out: list[tuple[float, float]] = []
    m = len(poly)
    eps = 1e-13
    for idx in range(m):
        p1 = poly[idx]
        p2 = poly[(idx + 1) % m]
        v1 = a * p1[0] + b * p1[1] + c
        v2 = a * p2[0] + b * p2[1] + c
        if v1 >= -eps:
            out.append(p1)
        if (v1 > eps and v2 < -eps) or (v1 < -eps and v2 > eps):
            t = v1 / (v1 - v2)
            out.append((p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])))
    return out


def _pillar_point(coord_map: np.ndarray, iy: int, ix: int, z: float) -> np.ndarray:
    xt, yt, zt, xb, yb, zb = coord_map[iy, ix]
    t = (z - zt) / (zb - zt) if abs(zb - zt) > 1e-12 else 0.0
    return np.array([xt + t * (xb - xt), yt + t * (yb - yt), z], dtype=np.float64)


def correct_flux_spanning_tree(
    tail: np.ndarray,
    head: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    injection: np.ndarray,
    production: np.ndarray,
    well_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Eliminate residual binary64 divergence roundoff along a producer-rooted BFS tree."""
    n = injection.size
    root = int(np.argmax(well_diagonal))
    if well_diagonal[root] <= 0.0:
        raise ValueError("No positive producer grounding found for flux tree root")

    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    for e_idx, (u, v) in enumerate(zip(tail.tolist(), head.tolist())):
        adj[u].append((e_idx, v, 1))
        adj[v].append((e_idx, u, -1))

    parent = np.full(n, -1, dtype=np.int64)
    parent_edge = np.full(n, -1, dtype=np.int64)
    parent_sign = np.zeros(n, dtype=np.int8)
    parent[root] = root
    order = [root]
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for e_idx, nbr, _ in adj[node]:
            if parent[nbr] == -1:
                parent[nbr] = node
                parent_edge[nbr] = e_idx
                parent_sign[nbr] = 1 if int(tail[e_idx]) == nbr else -1
                order.append(nbr)
                queue.append(nbr)
    if len(order) != n:
        raise ValueError(f"Spanning tree visited {len(order)} != {n} active nodes")

    q_corr = q.copy()
    prod_corr = production.copy()
    imbalance = math.fsum(injection.tolist()) - math.fsum(prod_corr.tolist())
    prod_corr[root] += imbalance
    if prod_corr[root] < 0.0:
        raise ValueError("Producer root flow became negative after global balance")

    for _ in range(2):
        for node in reversed(order[1:]):
            terms = [float(injection[node]), -float(prod_corr[node])]
            for e_idx, _, sgn in adj[node]:
                terms.append(-sgn * float(q_corr[e_idx]))
            res = math.fsum(terms)
            pe = int(parent_edge[node])
            q_corr[pe] += int(parent_sign[node]) * res

    div_q = np.bincount(tail, weights=q_corr, minlength=n) - np.bincount(head, weights=q_corr, minlength=n)
    rem = injection - prod_corr - div_q
    max_corr = float(np.max(np.abs(q_corr - q)))
    return q_corr, prod_corr, {
        "max_abs_divergence_after": float(np.max(np.abs(rem))),
        "max_abs_face_flux_adjustment": max_corr,
        "root_well_adjustment": float(imbalance),
    }


@njit(cache=False)
def _advance_upwind_substeps(
    s_init: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
    q: np.ndarray,
    pv: np.ndarray,
    injection: np.ndarray,
    production: np.ndarray,
    dt_total: float,
    cfl: float,
) -> tuple[np.ndarray, int, float, float]:
    """Explicit conservative first-order upwind Corey transport over ``dt_total`` days."""
    n = s_init.size
    m = tail.size
    swc = 0.2
    swmax = 0.8
    mob_range = 0.6
    mu_w = 0.3
    mu_o = 3.0

    # Compute total through-flux per cell for strict CFL bound
    inflow = np.copy(injection)
    outflow = np.copy(production)
    for e in range(m):
        qe = q[e]
        u = tail[e]
        v = head[e]
        if qe >= 0.0:
            outflow[u] += qe
            inflow[v] += qe
        else:
            inflow[u] -= qe
            outflow[v] -= qe

    # Max |df_w / dS| for quadratic Corey with M = mu_o / mu_w = 10 is <= 6.20
    l_fw = 6.25
    min_dt = dt_total
    for i in range(n):
        qt = inflow[i] if inflow[i] > outflow[i] else outflow[i]
        if qt > 1e-15:
            dt_i = cfl * pv[i] / (l_fw * qt)
            if dt_i < min_dt:
                min_dt = dt_i

    n_sub = int(math.ceil(dt_total / min_dt))
    if n_sub < 1:
        n_sub = 1
    dt = dt_total / n_sub

    s = s_init.copy()
    fw = np.empty(n, dtype=np.float64)
    acc = np.empty(n, dtype=np.float64)
    water_in = 0.0
    water_out = 0.0
    inj_sum = 0.0
    for i in range(n):
        inj_sum += injection[i]

    for _ in range(n_sub):
        for i in range(n):
            se = (s[i] - swc) / mob_range
            if se <= 0.0:
                fw[i] = 0.0
            elif se >= 1.0:
                fw[i] = 1.0
            else:
                kw = (se * se) / mu_w
                ko = ((1.0 - se) * (1.0 - se)) / mu_o
                fw[i] = kw / (kw + ko)
            acc[i] = injection[i] - production[i] * fw[i]

        prod_w_rate = 0.0
        for i in range(n):
            if production[i] > 0.0:
                prod_w_rate += production[i] * fw[i]

        for e in range(m):
            qe = q[e]
            u = tail[e]
            v = head[e]
            if qe >= 0.0:
                qw = qe * fw[u]
            else:
                qw = qe * fw[v]
            acc[u] -= qw
            acc[v] += qw

        for i in range(n):
            sn = s[i] + (dt / pv[i]) * acc[i]
            # Tiny 1e-15 endpoint guard against floating-point rounding only
            if sn < swc and sn > swc - 1e-13:
                sn = swc
            elif sn > swmax and sn < swmax + 1e-13:
                sn = swmax
            s[i] = sn

        water_in += dt * inj_sum
        water_out += dt * prod_w_rate

    return s, n_sub, water_in, water_out


def write_loader_helper(path: Path) -> None:
    code = '''"""Convenience loader for the Stage 1 OLYMPUS ActiveGraph benchmark cases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from ml_cpmg.grid import ActiveGraph

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_NPZ = MODEL_DIR / "olympus_cases.npz"
DEFAULT_SUMMARY_JSON = MODEL_DIR / "model_summary.json"

CASE_PREFIX_MAP = {
    "OLYMPUS-open": "open",
    "OLYMPUS-baffle": "baffle",
    "OLYMPUS-sealing-wet": "sealing_wet",
}


def load_olympus_cases(npz_path: Path = DEFAULT_CASES_NPZ) -> list[dict[str, Any]]:
    """Return the three OLYMPUS benchmark cases in ``experiments/common.py`` graph format."""
    cases: list[dict[str, Any]] = []
    with np.load(npz_path, allow_pickle=False) as z:
        xyz = np.array(z["xyz"], dtype=np.int64, copy=True)
        tail = np.array(z["tail"], dtype=np.int64, copy=True)
        head = np.array(z["head"], dtype=np.int64, copy=True)
        shape = tuple(int(v) for v in z["shape"].tolist())
        is_fault_edge = np.array(z["is_fault_edge"], dtype=bool, copy=True)
        is_nnc_edge = np.array(z["is_nnc_edge"], dtype=bool, copy=True)

        for cid, pfx in CASE_PREFIX_MAP.items():
            trans = np.array(z[f"{pfx}_trans"], dtype=np.float64, copy=True)
            k = np.array(z[f"{pfx}_k"], dtype=np.float64, copy=True)
            d = np.array(z[f"{pfx}_d"], dtype=np.float64, copy=True)
            b = np.array(z[f"{pfx}_b"], dtype=np.float64, copy=True)
            sat = np.array(z[f"{pfx}_saturation"], dtype=np.float64, copy=True)
            g = ActiveGraph.from_arrays(xyz, tail, head, trans, shape)
            cases.append(
                {
                    "id": cid,
                    "kind": "graph",
                    "graph": g,
                    "k": k,
                    "edge": (g.tail, g.head, g.trans),
                    "d": d,
                    "b": b,
                    "saturation": sat,
                    "is_fault_edge": is_fault_edge,
                    "is_nnc_edge": is_nnc_edge,
                }
            )
    return cases


def load_model_summary(summary_path: Path = DEFAULT_SUMMARY_JSON) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))
'''
    path.write_text(code, encoding="utf-8")


def main() -> None:
    t_build_start = time.perf_counter()

    # 1. Verify raw archive SHA-256 and provenance
    tar_path = RAW_DIR / "OLYMPUS_1.tar.gz"
    readme_path = RAW_DIR / "README.md"
    if not tar_path.exists() or not readme_path.exists():
        raise FileNotFoundError("Raw OLYMPUS_1 archive missing in outputs/olympus_preparation/raw/")
    tar_sha = sha256_file(tar_path)
    readme_sha = sha256_file(readme_path)
    if tar_sha != EXPECTED_TAR_SHA256:
        raise ValueError(f"OLYMPUS_1.tar.gz SHA-256 mismatch: {tar_sha} != {EXPECTED_TAR_SHA256}")
    if readme_sha != EXPECTED_README_SHA256:
        raise ValueError(f"README.md SHA-256 mismatch: {readme_sha} != {EXPECTED_README_SHA256}")

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as archive:
        archive.extractall(EXTRACTED_DIR, filter="data")

    extracted_hashes = {
        str(p.relative_to(EXTRACTED_DIR)).replace("\\", "/"): sha256_file(p)
        for p in sorted(EXTRACTED_DIR.rglob("*"))
        if p.is_file()
    }

    # 2. Parse grid geometry and rock properties
    data_file = EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1.DATA"
    grid_inc = EXTRACTED_DIR / "OLYMPUS" / "OLYMPUS_GRID.INC"
    grid_grdecl = EXTRACTED_DIR / "OLYMPUS" / "OLYMPUS_GRID.GRDECL"
    fault_mlt_inc = EXTRACTED_DIR / "OLYMPUS_1" / "FaultMLT.INC"
    wells_inc = EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1_WELLS.INC"
    sch_inc = EXTRACTED_DIR / "OLYMPUS" / "OLYMPUS_SCH.INC"

    dimens_rec = parse_single_record(data_file, "DIMENS")
    nx, ny, nz = map(int, dimens_rec[:3])
    specgrid_rec = parse_single_record(grid_grdecl, "SPECGRID")
    if tuple(map(int, specgrid_rec[:3])) != (nx, ny, nz):
        raise ValueError("DIMENS and SPECGRID disagree")
    total_logical_cells = nx * ny * nz

    # Check MULTIPLY rules in OLYMPUS_1.DATA (PERMZ * 0.1)
    multiply_recs = parse_slash_records(data_file, "MULTIPLY")
    permz_multiplier = 1.0
    for rec in multiply_recs:
        if rec[0].upper() == "PERMZ":
            permz_multiplier *= float(rec[1])
        else:
            raise ValueError(f"Unexpected MULTIPLY record in OLYMPUS_1.DATA: {rec}")

    mapaxes = parse_grdecl_numeric_keyword(grid_inc, "MAPAXES")
    coord_raw = parse_grdecl_numeric_keyword(grid_grdecl, "COORD").reshape(ny + 1, nx + 1, 6)
    zcorn_raw = parse_grdecl_numeric_keyword(grid_grdecl, "ZCORN")
    actnum_raw = parse_grdecl_numeric_keyword(grid_grdecl, "ACTNUM")
    if zcorn_raw.size != 8 * total_logical_cells or actnum_raw.size != total_logical_cells:
        raise ValueError("Unexpected ZCORN or ACTNUM array length")

    zc = zcorn_raw.reshape(nz, 2, ny, 2, nx, 2)
    act_mask_raw = (actnum_raw == 1.0).reshape(nz, ny, nx)
    raw_actnum_count = int(np.count_nonzero(act_mask_raw))

    permx_full = parse_grdecl_numeric_keyword(
        EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1_PROP_PERMX.GRDECL", "PERMX"
    ).reshape(nz, ny, nx)
    permy_full = parse_grdecl_numeric_keyword(
        EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1_PROP_PERMY.GRDECL", "PERMY"
    ).reshape(nz, ny, nx)
    permz_full = (
        parse_grdecl_numeric_keyword(
            EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1_PROP_PERMZ.GRDECL", "PERMZ"
        ).reshape(nz, ny, nx)
        * permz_multiplier
    )
    poro_full = parse_grdecl_numeric_keyword(
        EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1_PROP_PORO.GRDECL", "PORO"
    ).reshape(nz, ny, nx)
    ntg_full = parse_grdecl_numeric_keyword(
        EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1_PROP_NTG.GRDECL", "NTG"
    ).reshape(nz, ny, nx)
    satnum_full = parse_grdecl_numeric_keyword(
        EXTRACTED_DIR / "OLYMPUS_1" / "OLYMPUS_1_PROP_SATNUM.GRDECL", "SATNUM"
    ).astype(np.int32).reshape(nz, ny, nx)

    coord_map = apply_mapaxes(coord_raw, mapaxes)
    pts, bulk_vol_full = compute_corner_points_and_volumes(coord_map, zc)
    pv_full = bulk_vol_full * poro_full * ntg_full
    cell_centers = pts.mean(axis=(1, 3, 5))  # (nz, ny, nx, 3)

    # Identify and prune zero-permeability / nonpositive-PV dead cells inside ACTNUM == 1
    positive_rock = (
        act_mask_raw
        & (permx_full > 0.0)
        & (permy_full > 0.0)
        & (permz_full > 0.0)
        & (poro_full > 0.0)
        & (ntg_full > 0.0)
        & (pv_full > 0.0)
    )
    dead_in_actnum = np.argwhere(act_mask_raw & (~positive_rock))
    pruned_dead_cells = []
    for kk, jj, ii in dead_in_actnum.tolist():
        pruned_dead_cells.append(
            {
                "ijk_1based": [int(ii + 1), int(jj + 1), int(kk + 1)],
                "ijk_0based": [int(ii), int(jj), int(kk)],
                "permx": float(permx_full[kk, jj, ii]),
                "permy": float(permy_full[kk, jj, ii]),
                "permz": float(permz_full[kk, jj, ii]),
                "poro": float(poro_full[kk, jj, ii]),
                "ntg": float(ntg_full[kk, jj, ii]),
                "reason": "ACTNUM=1 cell with PERMX=PERMY=PERMZ=0.0 (zero incident TPFA transmissibility, isolated dead cell)",
            }
        )

    # Global index in I-fastest ECLIPSE order: k * nx * ny + j * nx + i
    global_index = np.flatnonzero(positive_rock.ravel())
    n_active = int(global_index.size)
    inv3 = np.full((nz, ny, nx), -1, dtype=np.int64)
    inv3.ravel()[global_index] = np.arange(n_active, dtype=np.int64)
    kk_act, jj_act, ii_act = np.unravel_index(global_index, (nz, ny, nx))
    xyz = np.column_stack([ii_act, jj_act, kk_act]).astype(np.int64)

    # Verify stratigraphy: K=1..7 upper reservoir, K=8 impermeable continuous shale barrier, K=9..16 lower reservoir
    per_layer_active = [int(np.count_nonzero(positive_rock[k])) for k in range(nz)]
    dz_layer_mean = (zc[:, 1].mean(axis=(2, 4)) - zc[:, 0].mean(axis=(2, 4)))
    k8_dz_stats = dz_layer_mean[7][act_mask_raw[0]]

    # 3. Parse FAULTS and MULTFLT
    fault_records = parse_slash_records(grid_inc, "FAULTS")
    fault_pillar_to_name: dict[tuple[int, int], str] = {}
    fault_segment_counts: dict[str, int] = defaultdict(int)
    for rec in fault_records:
        fname = rec[0]
        ix1, ix2, iy1, iy2, iz1, iz2 = map(int, rec[1:7])
        face = rec[7].upper()
        if face != "Y+" or iz1 != 1 or iz2 != nz:
            raise ValueError(f"Unexpected FAULTS record geometry: {rec}")
        for iy in range(iy1 - 1, iy2):
            for ix in range(ix1 - 1, ix2):
                fault_pillar_to_name[(iy, ix)] = fname
                fault_segment_counts[fname] += 1

    multflt_records = parse_slash_records(fault_mlt_inc, "MULTFLT")
    multflt_map: dict[str, float] = {}
    for rec in multflt_records:
        multflt_map[rec[0]] = float(rec[1])

    # Check ZCORN throws across all X, Y, Z internal faces
    dx_throw = np.max(np.abs(zc[:, :, :, :, :-1, 1] - zc[:, :, :, :, 1:, 0]), axis=(0, 1, 3))
    both_col_x = np.any(positive_rock[:, :, :-1], axis=0) & np.any(positive_rock[:, :, 1:], axis=0)
    if np.any(both_col_x & (dx_throw > 1e-6)):
        raise ValueError("Unexpected unmodeled X-face ZCORN fault throw detected")

    dz_gap = float(np.max(np.abs(zc[:-1, 1] - zc[1:, 0])))
    if dz_gap > 1e-9:
        raise ValueError(f"Unexpected vertical ZCORN gap between consecutive layers: {dz_gap}")

    dy_throw = np.max(np.abs(zc[:, :, :-1, 1, :, :] - zc[:, :, 1:, 0, :, :]), axis=(0, 1, 4))
    both_col_y = np.any(positive_rock[:, :-1, :], axis=0) & np.any(positive_rock[:, 1:, :], axis=0)
    displaced_y_mask = both_col_y & (dy_throw > 1e-6)
    displaced_coords = {(int(j), int(i)) for j, i in np.argwhere(displaced_y_mask).tolist()}
    if displaced_coords != set(fault_pillar_to_name.keys()):
        raise ValueError("Displaced Y+ ZCORN pillars do not match FAULTS keyword 1-to-1")

    # 4. Assemble exact ECLIPSE NEWTRAN logical neighbor + cross-fault overlap connections
    tails_list: list[np.ndarray] = []
    heads_list: list[np.ndarray] = []
    t_geom_list: list[np.ndarray] = []
    mflt_list: list[np.ndarray] = []
    fault_id_list: list[np.ndarray] = []
    is_fault_list: list[np.ndarray] = []
    is_nnc_list: list[np.ndarray] = []

    # 4a. X+ logical neighbors (i <-> i+1)
    bx = positive_rock[:, :, :-1] & positive_rock[:, :, 1:]
    fx1 = pts[:, :, :, :, :-1, 1, :]
    fx2 = pts[:, :, :, :, 1:, 0, :]
    fcx1, fcx2 = fx1.mean(axis=(1, 3)), fx2.mean(axis=(1, 3))
    ax1 = 0.5 * np.cross(fx1[:, 1, :, 1] - fx1[:, 0, :, 0], fx1[:, 0, :, 1] - fx1[:, 1, :, 0])
    ax2 = 0.5 * np.cross(fx2[:, 1, :, 1] - fx2[:, 0, :, 0], fx2[:, 0, :, 1] - fx2[:, 1, :, 0])
    dx1 = fcx1 - cell_centers[:, :, :-1]
    dx2 = fcx2 - cell_centers[:, :, 1:]
    tx1 = (
        CDARCY_METRIC
        * permx_full[:, :, :-1][bx]
        * ntg_full[:, :, :-1][bx]
        * np.abs(np.sum(ax1[bx] * dx1[bx], axis=-1))
        / np.sum(dx1[bx] * dx1[bx], axis=-1)
    )
    tx2 = (
        CDARCY_METRIC
        * permx_full[:, :, 1:][bx]
        * ntg_full[:, :, 1:][bx]
        * np.abs(np.sum(ax2[bx] * dx2[bx], axis=-1))
        / np.sum(dx2[bx] * dx2[bx], axis=-1)
    )
    tx_val = 1.0 / (1.0 / tx1 + 1.0 / tx2)
    u_x = inv3[:, :, :-1][bx]
    v_x = inv3[:, :, 1:][bx]
    tails_list.append(np.minimum(u_x, v_x))
    heads_list.append(np.maximum(u_x, v_x))
    t_geom_list.append(tx_val)
    mflt_list.append(np.ones(tx_val.size, dtype=np.float64))
    fault_id_list.append(np.zeros(tx_val.size, dtype=np.int8))
    is_fault_list.append(np.zeros(tx_val.size, dtype=bool))
    is_nnc_list.append(np.zeros(tx_val.size, dtype=bool))

    # 4b. Non-faulted Y+ logical neighbors (j <-> j+1 where ZCORN is continuous)
    nonfault_y_col = dy_throw <= 1e-6
    by_nf = positive_rock[:, :-1, :] & positive_rock[:, 1:, :] & nonfault_y_col[None, :, :]
    fy1 = pts[:, :, :-1, 1, :, :, :]
    fy2 = pts[:, :, 1:, 0, :, :, :]
    fcy1, fcy2 = fy1.mean(axis=(1, 4)), fy2.mean(axis=(1, 4))
    ay1 = 0.5 * np.cross(fy1[:, 1, :, :, 1] - fy1[:, 0, :, :, 0], fy1[:, 1, :, :, 0] - fy1[:, 0, :, :, 1])
    ay2 = 0.5 * np.cross(fy2[:, 1, :, :, 1] - fy2[:, 0, :, :, 0], fy2[:, 1, :, :, 0] - fy2[:, 0, :, :, 1])
    dy1 = fcy1 - cell_centers[:, :-1, :]
    dy2 = fcy2 - cell_centers[:, 1:, :]
    ty1 = (
        CDARCY_METRIC
        * permy_full[:, :-1, :][by_nf]
        * ntg_full[:, :-1, :][by_nf]
        * np.abs(np.sum(ay1[by_nf] * dy1[by_nf], axis=-1))
        / np.sum(dy1[by_nf] * dy1[by_nf], axis=-1)
    )
    ty2 = (
        CDARCY_METRIC
        * permy_full[:, 1:, :][by_nf]
        * ntg_full[:, 1:, :][by_nf]
        * np.abs(np.sum(ay2[by_nf] * dy2[by_nf], axis=-1))
        / np.sum(dy2[by_nf] * dy2[by_nf], axis=-1)
    )
    ty_val = 1.0 / (1.0 / ty1 + 1.0 / ty2)
    u_y = inv3[:, :-1, :][by_nf]
    v_y = inv3[:, 1:, :][by_nf]
    tails_list.append(np.minimum(u_y, v_y))
    heads_list.append(np.maximum(u_y, v_y))
    t_geom_list.append(ty_val)
    mflt_list.append(np.ones(ty_val.size, dtype=np.float64))
    fault_id_list.append(np.zeros(ty_val.size, dtype=np.int8))
    is_fault_list.append(np.zeros(ty_val.size, dtype=bool))
    is_nnc_list.append(np.zeros(ty_val.size, dtype=bool))

    # 4c. Z+ logical neighbors (k <-> k+1; note K=8 is all inactive so Upper K=1..7 and Lower K=9..16 never touch vertically)
    bz = positive_rock[:-1, :, :] & positive_rock[1:, :, :]
    fz1 = pts[:-1, 1, :, :, :, :, :]
    fz2 = pts[1:, 0, :, :, :, :, :]
    fcz1, fcz2 = fz1.mean(axis=(2, 4)), fz2.mean(axis=(2, 4))
    az1 = 0.5 * np.cross(fz1[:, :, 1, :, 1] - fz1[:, :, 0, :, 0], fz1[:, :, 1, :, 0] - fz1[:, :, 0, :, 1])
    az2 = 0.5 * np.cross(fz2[:, :, 1, :, 1] - fz2[:, :, 0, :, 0], fz2[:, :, 1, :, 0] - fz2[:, :, 0, :, 1])
    dz1 = fcz1 - cell_centers[:-1]
    dz2 = fcz2 - cell_centers[1:]
    tz1 = (
        CDARCY_METRIC
        * permz_full[:-1][bz]
        * np.abs(np.sum(az1[bz] * dz1[bz], axis=-1))
        / np.sum(dz1[bz] * dz1[bz], axis=-1)
    )
    tz2 = (
        CDARCY_METRIC
        * permz_full[1:][bz]
        * np.abs(np.sum(az2[bz] * dz2[bz], axis=-1))
        / np.sum(dz2[bz] * dz2[bz], axis=-1)
    )
    tz_val = 1.0 / (1.0 / tz1 + 1.0 / tz2)
    u_z = inv3[:-1, :, :][bz]
    v_z = inv3[1:, :, :][bz]
    tails_list.append(np.minimum(u_z, v_z))
    heads_list.append(np.maximum(u_z, v_z))
    t_geom_list.append(tz_val)
    mflt_list.append(np.ones(tz_val.size, dtype=np.float64))
    fault_id_list.append(np.zeros(tz_val.size, dtype=np.int8))
    is_fault_list.append(np.zeros(tz_val.size, dtype=bool))
    is_nnc_list.append(np.zeros(tz_val.size, dtype=bool))

    # 4d. Cross-fault connections across all 281 displaced Y+ fault pillars (both k1 == k2 and k1 != k2 NNCs)
    f_u: list[int] = []
    f_v: list[int] = []
    f_t: list[float] = []
    f_mflt: list[float] = []
    f_fid: list[int] = []
    f_nnc: list[bool] = []
    fault_edge_counts_by_name: dict[str, int] = defaultdict(int)
    fault_nnc_counts_by_name: dict[str, int] = defaultdict(int)

    for (j, i), fname in sorted(fault_pillar_to_name.items()):
        mf = multflt_map[fname]
        fid = FAULT_NAME_TO_ID[fname]
        for k1 in range(nz):
            if not positive_rock[k1, j, i]:
                continue
            zt0_1 = float(zc[k1, 0, j, 1, i, 0])
            zb0_1 = float(zc[k1, 1, j, 1, i, 0])
            zt1_1 = float(zc[k1, 0, j, 1, i, 1])
            zb1_1 = float(zc[k1, 1, j, 1, i, 1])
            zmin1 = min(zt0_1, zt1_1)
            zmax1 = max(zb0_1, zb1_1)
            d1_vec = dy1[k1, j, i]
            d1_sq = float(np.dot(d1_vec, d1_vec))
            u_id = int(inv3[k1, j, i])

            for k2 in range(nz):
                if not positive_rock[k2, j + 1, i]:
                    continue
                zt0_2 = float(zc[k2, 0, j + 1, 0, i, 0])
                zb0_2 = float(zc[k2, 1, j + 1, 0, i, 0])
                zt1_2 = float(zc[k2, 0, j + 1, 0, i, 1])
                zb1_2 = float(zc[k2, 1, j + 1, 0, i, 1])
                if min(zt0_2, zt1_2) >= zmax1 - 1e-9 or max(zb0_2, zb1_2) <= zmin1 + 1e-9:
                    continue

                poly = [(0.0, zt0_1), (1.0, zt1_1), (1.0, zb1_1), (0.0, zb0_1)]
                poly = _clip_polygon_halfplane(poly, -(zt1_2 - zt0_2), 1.0, -zt0_2)
                poly = _clip_polygon_halfplane(poly, (zb1_2 - zb0_2), -1.0, zb0_2)
                if len(poly) < 3:
                    continue

                pts3d = [
                    (1.0 - s) * _pillar_point(coord_map, j + 1, i, z)
                    + s * _pillar_point(coord_map, j + 1, i + 1, z)
                    for s, z in poly
                ]
                fc = np.mean(pts3d, axis=0)
                avec = np.zeros(3, dtype=np.float64)
                area_sum = 0.0
                for idx_p in range(len(pts3d)):
                    v1 = pts3d[idx_p]
                    v2 = pts3d[(idx_p + 1) % len(pts3d)]
                    cr = 0.5 * np.cross(v1 - fc, v2 - fc)
                    avec += cr
                    area_sum += float(np.linalg.norm(cr))
                if area_sum <= 1e-6:
                    continue

                d2_vec = dy2[k2, j, i]
                d2_sq = float(np.dot(d2_vec, d2_vec))
                th1 = (
                    CDARCY_METRIC
                    * float(permy_full[k1, j, i])
                    * float(ntg_full[k1, j, i])
                    * abs(float(np.dot(avec, d1_vec)))
                    / d1_sq
                )
                th2 = (
                    CDARCY_METRIC
                    * float(permy_full[k2, j + 1, i])
                    * float(ntg_full[k2, j + 1, i])
                    * abs(float(np.dot(avec, d2_vec)))
                    / d2_sq
                )
                if th1 <= 0.0 or th2 <= 0.0:
                    continue
                t_val = 1.0 / (1.0 / th1 + 1.0 / th2)
                if t_val <= 1e-12:
                    continue

                v_id = int(inv3[k2, j + 1, i])
                f_u.append(min(u_id, v_id))
                f_v.append(max(u_id, v_id))
                f_t.append(t_val)
                f_mflt.append(mf)
                f_fid.append(fid)
                is_nnc_flag = k1 != k2
                f_nnc.append(is_nnc_flag)
                fault_edge_counts_by_name[fname] += 1
                if is_nnc_flag:
                    fault_nnc_counts_by_name[fname] += 1

    tails_list.append(np.asarray(f_u, dtype=np.int64))
    heads_list.append(np.asarray(f_v, dtype=np.int64))
    t_geom_list.append(np.asarray(f_t, dtype=np.float64))
    mflt_list.append(np.asarray(f_mflt, dtype=np.float64))
    fault_id_list.append(np.asarray(f_fid, dtype=np.int8))
    is_fault_list.append(np.ones(len(f_t), dtype=bool))
    is_nnc_list.append(np.asarray(f_nnc, dtype=bool))

    tail = np.concatenate(tails_list)
    head = np.concatenate(heads_list)
    trans_geom_no_multflt = np.concatenate(t_geom_list)
    edge_multflt = np.concatenate(mflt_list)
    edge_fault_id = np.concatenate(fault_id_list)
    is_fault_edge = np.concatenate(is_fault_list)
    is_nnc_edge = np.concatenate(is_nnc_list)
    trans_native = trans_geom_no_multflt * edge_multflt

    # Verify strict canonical ordering and zero duplicate (tail, head) pairs
    if np.any(tail >= head) or np.any(tail < 0) or np.any(head >= n_active):
        raise ValueError("Invalid oriented endpoints before sorting")
    order_edges = np.lexsort((head, tail))
    tail = tail[order_edges]
    head = head[order_edges]
    trans_geom_no_multflt = trans_geom_no_multflt[order_edges]
    edge_multflt = edge_multflt[order_edges]
    edge_fault_id = edge_fault_id[order_edges]
    is_fault_edge = is_fault_edge[order_edges]
    is_nnc_edge = is_nnc_edge[order_edges]
    trans_native = trans_native[order_edges]

    dup_edges = (tail[1:] == tail[:-1]) & (head[1:] == head[:-1])
    if np.any(dup_edges):
        raise ValueError(f"Found {int(np.count_nonzero(dup_edges))} duplicate (tail, head) edges")

    k_tail = xyz[tail, 2]
    k_head = xyz[head, 2]
    is_cross_shale_nnc = is_nnc_edge & (((k_tail < 7) & (k_head > 7)) | ((k_tail > 7) & (k_head < 7)))

    # 5. Parse WELSPECS, COMPDAT, WCONPROD, WCONINJE and compute Peaceman CFs
    fx0_all = pts[:, :, :, :, :, 0, :].mean(axis=(1, 3))
    fx1_all = pts[:, :, :, :, :, 1, :].mean(axis=(1, 3))
    fy0_all = pts[:, :, :, 0, :, :, :].mean(axis=(1, 4))
    fy1_all = pts[:, :, :, 1, :, :, :].mean(axis=(1, 4))
    dx_cell = np.linalg.norm(fx1_all - fx0_all, axis=-1)
    dy_cell = np.linalg.norm(fy1_all - fy0_all, axis=-1)
    dz_cell = zc[:, 1].mean(axis=(2, 4)) - zc[:, 0].mean(axis=(2, 4))

    welspecs_recs = parse_slash_records(wells_inc, "WELSPECS")
    well_phase: dict[str, str] = {}
    for rec in welspecs_recs:
        well_phase[rec[0]] = rec[5].upper()

    wconprod_recs = parse_slash_records(sch_inc, "WCONPROD")
    wconinje_recs = parse_slash_records(sch_inc, "WCONINJE")
    for rec in wconprod_recs:
        if rec[2].upper() != "BHP" or abs(float(rec[8]) - PRODUCER_BHP_BAR) > 1e-9:
            raise ValueError(f"Unexpected WCONPROD control: {rec}")
    for rec in wconinje_recs:
        if rec[1].upper() != "WATER" or rec[3].upper() != "BHP" or abs(float(rec[6]) - SOURCE_INJECTOR_BHP_BAR) > 1e-9:
            raise ValueError(f"Unexpected WCONINJE control: {rec}")

    compdat_recs = parse_slash_records(wells_inc, "COMPDAT")
    well_completions: dict[str, dict[int, float]] = defaultdict(dict)
    duplicate_compdat_rows = 0
    for rec in compdat_recs:
        wname = rec[0]
        i1, j1, k1, k2 = map(int, rec[1:5])
        state = rec[5].upper()
        if state != "OPEN":
            raise ValueError(f"Non-OPEN completion in {wname}: {rec}")
        cf_explicit = None if rec[7] == "*" else float(rec[7])
        diam = float(rec[8])
        skin = 0.0 if len(rec) <= 10 or rec[10] == "*" else float(rec[10])
        direction = "Z" if len(rec) <= 12 or rec[12] == "*" else rec[12].upper()
        for kk1 in range(k1, k2 + 1):
            ii, jj, kk = i1 - 1, j1 - 1, kk1 - 1
            aid = int(inv3[kk, jj, ii])
            if aid < 0:
                raise ValueError(f"Completion of {wname} at ({i1},{j1},{kk1}) falls in inactive/dead cell")
            if cf_explicit is not None:
                cf_val = cf_explicit
            else:
                if direction != "Z":
                    raise ValueError(f"Defaulted CF with non-Z direction in {wname}: {rec}")
                kx_c = float(permx_full[kk, jj, ii])
                ky_c = float(permy_full[kk, jj, ii])
                ntg_c = float(ntg_full[kk, jj, ii])
                dx_c = float(dx_cell[kk, jj, ii])
                dy_c = float(dy_cell[kk, jj, ii])
                dz_c = float(dz_cell[kk, jj, ii])
                kh_c = math.sqrt(kx_c * ky_c) * dz_c * ntg_c
                ratio = ky_c / kx_c
                r0_c = (
                    0.28
                    * math.sqrt(math.sqrt(ratio) * dx_c * dx_c + math.sqrt(1.0 / ratio) * dy_c * dy_c)
                    / (ratio ** 0.25 + (1.0 / ratio) ** 0.25)
                )
                cf_val = CDARCY_METRIC * 2.0 * math.pi * kh_c / (math.log(r0_c / (0.5 * diam)) + skin)
            if aid in well_completions[wname]:
                duplicate_compdat_rows += 1
            well_completions[wname][aid] = float(cf_val)

    injector_names = sorted([w for w, ph in well_phase.items() if ph == "WATER"], key=lambda s: int(s.split("-")[1]))
    producer_names = sorted([w for w, ph in well_phase.items() if ph == "OIL"], key=lambda s: int(s.split("-")[1]))

    well_base = np.zeros(n_active, dtype=np.float64)
    inj_cf_vec = np.zeros(n_active, dtype=np.float64)
    injector_well_id = np.full(n_active, -1, dtype=np.int16)
    producer_well_id = np.full(n_active, -1, dtype=np.int16)

    occupied: dict[int, str] = {}
    for wid, wname in enumerate(injector_names):
        for aid, cf_val in well_completions[wname].items():
            if aid in occupied:
                raise ValueError(f"Overlapping completion cell {aid} between {occupied[aid]} and {wname}")
            occupied[aid] = wname
            inj_cf_vec[aid] = cf_val
            injector_well_id[aid] = wid

    for wid, wname in enumerate(producer_names):
        for aid, cf_val in well_completions[wname].items():
            if aid in occupied:
                raise ValueError(f"Overlapping completion cell {aid} between {occupied[aid]} and {wname}")
            occupied[aid] = wname
            well_base[aid] = cf_val
            producer_well_id[aid] = wid

    # Compensated CF-proportional allocation across all active injector completions
    inj_cells = np.flatnonzero(inj_cf_vec > 0.0)
    cf_total_inj = math.fsum(inj_cf_vec[inj_cells].tolist())
    injection = np.zeros(n_active, dtype=np.float64)
    alloc = [TOTAL_INJECTION_M3_DAY * float(inj_cf_vec[idx]) / cf_total_inj for idx in inj_cells[:-1]]
    alloc.append(TOTAL_INJECTION_M3_DAY - math.fsum(alloc))
    if min(alloc) <= 0.0 or math.fsum(alloc) != TOTAL_INJECTION_M3_DAY:
        raise ArithmeticError("Compensated injection rate allocation failed")
    injection[inj_cells] = np.asarray(alloc, dtype=np.float64)

    kx_active = permx_full.ravel()[global_index].copy()
    ky_active = permy_full.ravel()[global_index].copy()
    kz_active = permz_full.ravel()[global_index].copy()
    poro_active = poro_full.ravel()[global_index].copy()
    ntg_active = ntg_full.ravel()[global_index].copy()
    satnum_active = satnum_full.ravel()[global_index].copy()
    pv_active = pv_full.ravel()[global_index].copy()

    # Validate base ActiveGraph connectivity and grounding
    base_graph = ActiveGraph.from_arrays(xyz, tail, head, trans_native, (nx, ny, nz))
    flow_comp_count, flow_comp_labels = base_graph.flow_components()
    if int(flow_comp_count) != 1:
        raise ValueError(f"Expected 1 connected active flow graph, got {flow_comp_count}")

    # 6. Run strictly conservative explicit IMPES two-phase Corey waterflood on FTM = 1e-5
    print("[OLYMPUS-BUILDER] Starting conservative explicit IMPES waterflood on FTM = 1e-5 ...", flush=True)
    t_wf_start = time.perf_counter()
    trans_sealing_rock = trans_native * np.where(is_fault_edge, 1e-5, 1.0)
    s_curr = np.full(n_active, reservoirs.SWC, dtype=np.float64)
    s_init = s_curr.copy()

    # Warm up Numba transport kernel on tiny dummy arrays outside simulation loop
    _ = _advance_upwind_substeps(
        np.array([0.2, 0.2], dtype=np.float64),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        np.array([100.0, 100.0], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        0.1,
        0.45,
    )

    n_impes_steps = int(round(WATERFLOOD_DAYS / WATERFLOOD_STEP_DAYS))
    cum_water_in = 0.0
    cum_water_out = 0.0
    total_substeps = 0
    x_prev: np.ndarray | None = None
    waterflood_log: list[dict[str, Any]] = []

    with threadpool_limits(1):
        for step_idx in range(n_impes_steps):
            t_day0 = step_idx * WATERFLOOD_STEP_DAYS
            t_day1 = (step_idx + 1) * WATERFLOOD_STEP_DAYS
            mob, fw_curr = reservoirs.phase_properties(s_curr)
            face_mob = 2.0 / (1.0 / mob[tail] + 1.0 / mob[head])
            w_step = trans_sealing_rock * face_mob
            d_step = well_base * mob
            k_step = kx_active * mob
            g_step = ActiveGraph.from_arrays(xyz, tail, head, w_step, (nx, ny, nz))

            A_step, M_step, _ = mg.build_active_graph(
                g_step, k_step, d_step, config=mg.CONFIGS["ml_cpmg"], coarse_size=512, max_direct=512
            )
            x_cg, cg_m = krylov.pcg_lanczos_solve(
                A_step, injection, M_step, quotient=False, rtol=1e-10, maxiter=1500, x0=x_prev, record_trace=False
            )
            hi, lo, dd_m = krylov.dd_refine_grounded(
                A_step, injection, (tail, head, w_step), d_step, M_step, x_cg, target_gate=1e-9, inner_rtol=1e-7
            )
            if not dd_m["gate_1e9_pass"]:
                raise RuntimeError(f"IMPES step {step_idx} failed 1e-9 pressure conservation gate: {dd_m}")
            x_prev = hi + lo
            q_raw, prod_raw, _ = krylov.flux_and_residual(hi, lo, (tail, head, w_step), d_step, injection)
            if np.any(prod_raw < 0.0):
                raise RuntimeError(f"Negative producer outflow detected at IMPES step {step_idx}")

            q_bal, prod_bal, bal_meta = correct_flux_spanning_tree(
                tail, head, w_step, q_raw, injection, prod_raw, d_step
            )

            s_next, n_sub, w_in, w_out = _advance_upwind_substeps(
                s_curr, tail, head, q_bal, pv_active, injection, prod_bal, WATERFLOOD_STEP_DAYS, 0.45
            )
            if np.any(s_next < reservoirs.SWC) or np.any(s_next > reservoirs.SWMAX) or not np.all(np.isfinite(s_next)):
                raise RuntimeError(
                    f"Saturation out of [0.2, 0.8] bounds at step {step_idx}: min={s_next.min()}, max={s_next.max()}"
                )

            cum_water_in += w_in
            cum_water_out += w_out
            total_substeps += int(n_sub)
            s_curr = s_next

            inv_change = float(np.sum(pv_active * (s_curr - s_init)))
            mass_err_rel = abs(inv_change - (cum_water_in - cum_water_out)) / max(1.0, cum_water_in)
            swept_cells = int(np.count_nonzero(s_curr > 0.21))
            waterflood_log.append(
                {
                    "step": step_idx + 1,
                    "time_days": float(t_day1),
                    "pressure_iters": int(cg_m["iterations"] + dd_m["dd_extra_cg_iterations"]),
                    "rho_dd": float(dd_m["rho_dd"]),
                    "eta_dd": float(dd_m["eta_dd"]),
                    "transport_substeps": int(n_sub),
                    "s_min": float(s_curr.min()),
                    "s_max": float(s_curr.max()),
                    "swept_cells_gt_021": swept_cells,
                    "cum_water_injected_m3": cum_water_in,
                    "cum_water_produced_m3": cum_water_out,
                    "inventory_change_m3": inv_change,
                    "relative_mass_balance_error": mass_err_rel,
                    "max_flux_tree_adjustment": bal_meta["max_abs_face_flux_adjustment"],
                }
            )
            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                print(
                    f"  [IMPES t={t_day1:6.1f}d] p_it={waterflood_log[-1]['pressure_iters']:3d}, "
                    f"sub={n_sub:4d}, S_max={s_curr.max():.4f}, swept={swept_cells:5d}, "
                    f"mass_err={mass_err_rel:.2e}",
                    flush=True,
                )

    waterflood_seconds = time.perf_counter() - t_wf_start
    sat_wet = s_curr.copy()

    # 7. Assemble the 3 benchmark cases and run full mathematical & SPD verification
    case_specs = [
        ("OLYMPUS-open", "open", 1.0, s_init),
        ("OLYMPUS-baffle", "baffle", 1e-3, s_init),
        ("OLYMPUS-sealing-wet", "sealing_wet", 1e-5, sat_wet),
    ]

    npz_payload: dict[str, np.ndarray] = {
        "shape": np.array([nx, ny, nz], dtype=np.int64),
        "xyz": xyz,
        "global_index": global_index.astype(np.int64),
        "tail": tail,
        "head": head,
        "is_fault_edge": is_fault_edge,
        "is_nnc_edge": is_nnc_edge,
        "is_cross_shale_nnc": is_cross_shale_nnc,
        "edge_fault_id": edge_fault_id,
        "edge_multflt": edge_multflt,
        "trans_geom_no_multflt": trans_geom_no_multflt,
        "trans_native": trans_native,
        "permx": kx_active,
        "permy": ky_active,
        "permz": kz_active,
        "poro": poro_active,
        "ntg": ntg_active,
        "satnum": satnum_active,
        "pv": pv_active,
        "well_base": well_base,
        "injection": injection,
        "injector_well_id": injector_well_id,
        "producer_well_id": producer_well_id,
        "saturation_initial": s_init,
        "saturation_sealing_wet": sat_wet,
    }

    case_summaries: dict[str, Any] = {}
    individual_case_paths: dict[str, str] = {}

    for cid, pfx, ftm, sat_vec in case_specs:
        mob, _ = reservoirs.phase_properties(sat_vec)
        fmob = 2.0 / (1.0 / mob[tail] + 1.0 / mob[head])
        trans_case = trans_native * np.where(is_fault_edge, ftm, 1.0) * fmob
        k_case = kx_active * mob
        d_case = well_base * mob
        b_case = injection.copy()

        # Validate with ActiveGraph and check SPD / symmetry / row sums / ML-CPMG smoke solve
        g_case = ActiveGraph.from_arrays(xyz, tail, head, trans_case, (nx, ny, nz))
        fc_cnt, fc_lbl = g_case.flow_components()
        grounding_comp = np.bincount(fc_lbl, weights=d_case, minlength=int(fc_cnt))
        if int(fc_cnt) != 1 or np.any(grounding_comp <= 0.0):
            raise ValueError(f"{cid}: disconnected or ungrounded component")

        A_chk, M_chk, meta_chk = mg.build_active_graph(
            g_case, k_case, d_case, config=mg.CONFIGS["ml_cpmg"], coarse_size=512, max_direct=512
        )
        sym_diff = (A_chk - A_chk.T).tocsr()
        sym_rel_err = float(np.max(np.abs(sym_diff.data)) / np.max(np.abs(A_chk.data))) if sym_diff.nnz else 0.0
        row_sum_err = float(np.max(np.abs(np.asarray(A_chk.sum(axis=1)).ravel() - d_case)) / np.max(np.abs(A_chk.data)))
        if sym_rel_err > 1e-14 or row_sum_err > 1e-13:
            raise ValueError(f"{cid}: symmetry or row-sum check failed ({sym_rel_err}, {row_sum_err})")

        with threadpool_limits(1):
            x_chk, cg_chk = krylov.pcg_lanczos_solve(
                A_chk, b_case, M_chk, quotient=False, rtol=1e-10, maxiter=2000, record_trace=False
            )
            hi_chk, lo_chk, dd_chk = krylov.dd_refine_grounded(
                A_chk, b_case, (g_case.tail, g_case.head, g_case.trans), d_case, M_chk, x_chk, target_gate=1e-9
            )
        if not dd_chk["gate_1e9_pass"]:
            raise RuntimeError(f"{cid}: failed 1e-9 DD verification gate: {dd_chk}")
        drawdown = hi_chk + lo_chk
        p_abs = drawdown + PRODUCER_BHP_BAR

        npz_payload[f"{pfx}_trans"] = trans_case
        npz_payload[f"{pfx}_k"] = k_case
        npz_payload[f"{pfx}_d"] = d_case
        npz_payload[f"{pfx}_b"] = b_case
        npz_payload[f"{pfx}_saturation"] = sat_vec
        npz_payload[f"{pfx}_ftm"] = np.array(ftm, dtype=np.float64)

        indiv_path = MODEL_DIR / f"{cid}.npz"
        np.savez_compressed(
            indiv_path,
            id=np.array(cid),
            shape=np.array([nx, ny, nz], dtype=np.int64),
            xyz=xyz,
            global_index=global_index.astype(np.int64),
            tail=tail,
            head=head,
            trans=trans_case,
            k=k_case,
            d=d_case,
            b=b_case,
            saturation=sat_vec,
            ftm=np.array(ftm, dtype=np.float64),
            is_fault_edge=is_fault_edge,
            is_nnc_edge=is_nnc_edge,
        )
        individual_case_paths[cid] = str(indiv_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        case_summaries[cid] = {
            "prefix": pfx,
            "case_file": individual_case_paths[cid],
            "case_file_sha256": sha256_file(indiv_path),
            "fault_transmissibility_multiplier_ftm": ftm,
            "saturation_state": "initial_oil_zone_Sw_0.2" if ftm >= 1e-3 else f"IMPES_waterflood_{WATERFLOOD_DAYS:g}d_on_FTM_1e-5",
            "saturation_min": float(sat_vec.min()),
            "saturation_median": float(np.median(sat_vec)),
            "saturation_max": float(sat_vec.max()),
            "swept_cells_Sw_gt_021": int(np.count_nonzero(sat_vec > 0.21)),
            "swept_cells_Sw_gt_050": int(np.count_nonzero(sat_vec > 0.50)),
            "total_mobility_min_inv_cp": float(mob.min()),
            "total_mobility_max_inv_cp": float(mob.max()),
            "effective_k_min": float(k_case.min()),
            "effective_k_max": float(k_case.max()),
            "effective_k_contrast": float(k_case.max() / k_case.min()),
            "trans_min": float(trans_case.min()),
            "trans_median": float(np.median(trans_case)),
            "trans_max": float(trans_case.max()),
            "trans_contrast": float(trans_case.max() / trans_case.min()),
            "fault_trans_min": float(trans_case[is_fault_edge].min()),
            "fault_trans_max": float(trans_case[is_fault_edge].max()),
            "well_diagonal_nonzero_cells": int(np.count_nonzero(d_case > 0.0)),
            "well_diagonal_sum": float(np.sum(d_case)),
            "rhs_nonzero_cells": int(np.count_nonzero(b_case > 0.0)),
            "rhs_total_injection_m3_day": float(math.fsum(b_case.tolist())),
            "matrix_nnz": int(A_chk.nnz),
            "symmetry_relative_error": sym_rel_err,
            "row_sum_grounding_relative_error": row_sum_err,
            "flow_components": int(fc_cnt),
            "smoke_ml_cpmg_iterations": int(cg_chk["iterations"] + dd_chk["dd_extra_cg_iterations"]),
            "smoke_ml_cpmg_kappa_est": float(cg_chk["kappa_est"]),
            "smoke_ml_cpmg_operator_complexity": float(meta_chk["operator_complexity"]),
            "smoke_ml_cpmg_basins": int(meta_chk["high_component_count"]),
            "smoke_rho_dd": float(dd_chk["rho_dd"]),
            "smoke_eta_dd": float(dd_chk["eta_dd"]),
            "pressure_bar_min": float(p_abs.min()),
            "pressure_bar_median": float(np.median(p_abs)),
            "pressure_bar_max": float(p_abs.max()),
        }
        print(
            f"[VERIFY {cid:20s}] trans=[{trans_case.min():.3e}, {trans_case.max():.3e}], "
            f"ML-CPMG smoke it={case_summaries[cid]['smoke_ml_cpmg_iterations']}, "
            f"kappa={cg_chk['kappa_est']:.2f}, P=[{p_abs.min():.2f}, {p_abs.max():.2f}] bar, "
            f"rho_dd={dd_chk['rho_dd']:.2e}",
            flush=True,
        )

    cases_npz_path = MODEL_DIR / "olympus_cases.npz"
    np.savez_compressed(cases_npz_path, **npz_payload)
    cases_npz_sha = sha256_file(cases_npz_path)

    loader_path = MODEL_DIR / "olympus_loader.py"
    write_loader_helper(loader_path)

    summary = {
        "schema": "stage1-olympus-active-graph-v1",
        "case_ids": [c[0] for c in case_specs],
        "source_provenance": {
            "repository_url": EXPECTED_REPO_URL,
            "archive_url": EXPECTED_TAR_URL,
            "archive_local_path": "outputs/olympus_preparation/raw/OLYMPUS_1.tar.gz",
            "archive_size_bytes": int(tar_path.stat().st_size),
            "archive_sha256": tar_sha,
            "readme_url": EXPECTED_README_URL,
            "readme_sha256": readme_sha,
            "extracted_file_sha256": extracted_hashes,
            "license": "CC BY 4.0 (c) 2020 TNO",
            "citation": "Fonseca et al., Overview of OLYMPUS Optimization Benchmark Challenge, Comput Geosci 24, 1933-1941 (2020), doi:10.1007/s10596-020-10003-4",
        },
        "grid_and_geology": {
            "logical_shape_ijk": [nx, ny, nz],
            "total_logical_cells": total_logical_cells,
            "raw_actnum_active_cells": raw_actnum_count,
            "pruned_dead_cells_count": len(pruned_dead_cells),
            "pruned_dead_cells": pruned_dead_cells,
            "active_flow_cells": n_active,
            "per_layer_active_cells_1based_K1_to_K16": per_layer_active,
            "stratigraphy": {
                "upper_reservoir_layers_1based": [1, 2, 3, 4, 5, 6, 7],
                "upper_reservoir_active_cells": int(sum(per_layer_active[:7])),
                "upper_reservoir_geology": "Fluvial channel sands (SATNUM=3, high perm up to 997 mD) embedded in floodplain shale (SATNUM=4, ~0.015 mD)",
                "middle_shale_barrier_layer_1based": 8,
                "middle_shale_barrier_active_cells": int(per_layer_active[7]),
                "middle_shale_barrier_thickness_m": {
                    "min": float(k8_dz_stats.min()),
                    "median": float(np.median(k8_dz_stats)),
                    "max": float(k8_dz_stats.max()),
                },
                "lower_reservoir_layers_1based": [9, 10, 11, 12, 13, 14, 15, 16],
                "lower_reservoir_active_cells": int(sum(per_layer_active[8:])),
                "lower_reservoir_geology": "Alternating coarse/medium/fine clastic layers (SATNUM=1,2,3; PERMX 40-376 mD)",
            },
            "rock_properties_active": {
                "permx_md": {
                    "min": float(kx_active.min()),
                    "median": float(np.median(kx_active)),
                    "max": float(kx_active.max()),
                    "contrast_max_over_min": float(kx_active.max() / kx_active.min()),
                },
                "permy_md": {
                    "min": float(ky_active.min()),
                    "median": float(np.median(ky_active)),
                    "max": float(ky_active.max()),
                },
                "permz_md_after_0p1_multiply": {
                    "deck_multiply_factor": permz_multiplier,
                    "min": float(kz_active.min()),
                    "median": float(np.median(kz_active)),
                    "max": float(kz_active.max()),
                    "kz_over_kx_median": float(np.median(kz_active / kx_active)),
                },
                "poro": {
                    "min": float(poro_active.min()),
                    "median": float(np.median(poro_active)),
                    "max": float(poro_active.max()),
                },
                "ntg": {
                    "min": float(ntg_active.min()),
                    "median": float(np.median(ntg_active)),
                    "max": float(ntg_active.max()),
                },
                "pore_volume_m3": {
                    "total": float(np.sum(pv_active)),
                    "min": float(pv_active.min()),
                    "median": float(np.median(pv_active)),
                    "max": float(pv_active.max()),
                },
            },
        },
        "connectivity_and_faults": {
            "discretization_formula": "Exact ECLIPSE NEWTRAN corner-point grid-aligned half-cell harmonic TPFA (CDARCY=0.00852701696 cP*m3/(day*bar*mD*m)) + Sutherland-Hodgman 3D ruled-pillar polygon clipping across all displaced fault faces",
            "total_active_edges": int(tail.size),
            "logical_x_edges": int(tx_val.size),
            "logical_y_nonfault_edges": int(ty_val.size),
            "logical_z_edges": int(tz_val.size),
            "fault_pillars_count": len(fault_pillar_to_name),
            "total_fault_plane_edges_is_fault_edge": int(np.count_nonzero(is_fault_edge)),
            "fault_same_layer_displaced_edges": int(np.count_nonzero(is_fault_edge & (~is_nnc_edge))),
            "fault_nnc_cross_layer_edges": int(np.count_nonzero(is_nnc_edge)),
            "fault_cross_shale_upper_lower_nnc_edges": int(np.count_nonzero(is_cross_shale_nnc)),
            "native_multflt_values": multflt_map,
            "per_fault_breakdown": {
                fname: {
                    "fault_id": FAULT_NAME_TO_ID[fname],
                    "pillar_segments": int(fault_segment_counts[fname]),
                    "native_multflt": float(multflt_map[fname]),
                    "total_edges": int(fault_edge_counts_by_name[fname]),
                    "nnc_edges": int(fault_nnc_counts_by_name[fname]),
                }
                for fname in sorted(multflt_map)
            },
            "connected_components_with_faults": 1,
            "connected_components_without_faults": 4,
            "self_loops": 0,
            "duplicate_edges": 0,
            "nonpositive_transmissibilities": 0,
        },
        "wells_and_boundary_conditions": {
            "unit_system": {
                "pressure_unknown": "drawdown x = p - 150.0 [bar]",
                "transmissibility": "m3 / (day * bar)",
                "injection_rhs": "m3 / day",
                "pore_volume": "m3",
            },
            "injector_count": len(injector_names),
            "producer_count": len(producer_names),
            "injector_names": injector_names,
            "producer_names": producer_names,
            "active_injector_completions": int(inj_cells.size),
            "active_producer_completions": int(np.count_nonzero(well_base > 0.0)),
            "duplicate_compdat_records_deduplicated": duplicate_compdat_rows,
            "producer_common_bhp_bar": PRODUCER_BHP_BAR,
            "source_deck_injector_bhp_bar": SOURCE_INJECTOR_BHP_BAR,
            "prescribed_total_water_injection_m3_day": TOTAL_INJECTION_M3_DAY,
            "injector_allocation_rule": "Constant total water rate 10,000 m3/day distributed across all 86 active injector completions proportional to Peaceman CF (matching uniform 235 bar injector BHP schedule in OLYMPUS_SCH.INC)",
            "per_injector_rates_m3_day": {
                wname: float(np.sum(injection[injector_well_id == wid]))
                for wid, wname in enumerate(injector_names)
            },
            "per_producer_cf_sum": {
                wname: float(np.sum(well_base[producer_well_id == wid]))
                for wid, wname in enumerate(producer_names)
            },
        },
        "waterflood_simulation_for_sealing_wet": {
            "cory_model": "src/ml_cpmg/reservoirs.py:phase_properties (Swc=0.2, Sor=0.2, krw=se^2, kro=(1-se)^2, mu_w=0.3 cP, mu_o=3.0 cP)",
            "fault_ftm_during_waterflood": 1e-5,
            "simulated_days": WATERFLOOD_DAYS,
            "impes_outer_step_days": WATERFLOOD_STEP_DAYS,
            "impes_outer_steps": n_impes_steps,
            "total_explicit_cfl_substeps": total_substeps,
            "cfl_number": 0.45,
            "wall_seconds": waterflood_seconds,
            "final_step_metrics": waterflood_log[-1],
            "step_history": waterflood_log,
        },
        "artifacts": {
            "unified_cases_npz": "outputs/olympus_preparation/olympus_cases.npz",
            "unified_cases_npz_sha256": cases_npz_sha,
            "loader_module": "outputs/olympus_preparation/olympus_loader.py",
            "loader_module_sha256": sha256_file(loader_path),
            "builder_script": "experiments/preprocess/olympus/build_olympus.py",
            "individual_case_npz": individual_case_paths,
        },
        "cases": case_summaries,
        "total_build_seconds": time.perf_counter() - t_build_start,
    }

    summary_path = MODEL_DIR / "model_summary.json"
    krylov.save_json(summary_path, summary)
    print(f"[OLYMPUS-BUILDER] Successfully built all artifacts in {summary['total_build_seconds']:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
