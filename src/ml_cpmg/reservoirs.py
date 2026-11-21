"""SPE10 Model 2 (static & evolving-mobility) and OLYMPUS_1 faulted active-graph case loaders."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from .double_double import load_checkpoint
from .grid import ActiveGraph
from .paths import BRUGGE_INPUTS_DIR, OLYMPUS_INPUTS_DIR, PROJECT_ROOT, SPE10_INPUTS_DIR

# SPE10 Model 2 geometry and physical unit constants
SPE10_SHAPE = (85, 220, 60)  # (K, J, I)
SPE10_DX, SPE10_DY, SPE10_DZ = 20.0, 10.0, 2.0  # ft
FT3_PER_BBL = 0.158987294928 / 0.028316846592
FLOW_FACTOR = 9.869233e-16 * 0.3048 * 6894.757293168 / 0.001 * 86400.0 / 0.028316846592
SPE10_BHP_PSI = 4000.0
SPE10_INJECTION_FT3_DAY = 5000.0 * FT3_PER_BBL

# Corey two-phase mobility constants
SWC = 0.2
SOR = 0.2
SWMAX = 1.0 - SOR
MOBILE_RANGE = 1.0 - SWC - SOR
MU_W = 0.3
MU_O = 3.0


def phase_properties(saturation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return total mobility [1/cP] and water fractional flow at each cell."""
    s = np.asarray(saturation, dtype=np.float64)
    if not np.isfinite(s).all() or np.any(s < SWC) or np.any(s > SWMAX):
        raise ValueError("Corey saturation must lie in [0.2, 0.8]")
    effective = (s - SWC) / MOBILE_RANGE
    water = effective * effective / MU_W
    oil = (1.0 - effective) * (1.0 - effective) / MU_O
    total = water + oil
    return total, water / total


def load_spe10_static() -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build the full 1,122,000-cell SPE10 Model 2 pure-Neumann five-spot problem."""
    kx, ky, kz = np.load(SPE10_INPUTS_DIR / "permeability.npy", allow_pickle=False)
    ids = np.arange(kx.size, dtype=np.int32).reshape(SPE10_SHAPE)
    tail, head, weights = [], [], []
    for axis, (field, area_over_spacing) in enumerate(
        ((kz, SPE10_DX * SPE10_DY / SPE10_DZ), (ky, SPE10_DX * SPE10_DZ / SPE10_DY), (kx, SPE10_DY * SPE10_DZ / SPE10_DX))
    ):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        tail.append(ids[a].ravel())
        head.append(ids[b].ravel())
        weights.append((2.0 * area_over_spacing / (1.0 / field[a] + 1.0 / field[b])).ravel())
    rhs = np.zeros(SPE10_SHAPE, dtype=np.float64)
    rhs[:, 109, 29] = 1.0
    for y in (0, SPE10_SHAPE[1] - 1):
        for x in (0, SPE10_SHAPE[2] - 1):
            rhs[:, y, x] = -0.25
    return kx, rhs.ravel(), tuple(np.concatenate(v) for v in (tail, head, weights))


class SPE10DynamicModel:
    """Well-grounded two-phase IMPES pressure operator for SPE10 Model 2."""

    def __init__(self, k: np.ndarray, phi: np.ndarray):
        self.k = k
        self.shape = tuple(k.shape[1:])
        self.pv = (np.asarray(phi) * SPE10_DX * SPE10_DY * SPE10_DZ).ravel()
        self.ids = np.arange(phi.size, dtype=np.int32).reshape(phi.shape)
        self.inj = np.zeros(phi.size, dtype=np.float64)
        nz, ny, nx = phi.shape
        self.inj[self.ids[:, (ny - 1) // 2, (nx - 1) // 2].ravel()] = SPE10_INJECTION_FT3_DAY / nz
        producer = np.zeros(phi.shape, dtype=np.float64)
        for y in (0, ny - 1):
            for x in (0, nx - 1):
                producer[:, y, x] = 1.0
        r1 = np.sqrt(9.0 * SPE10_DX**2 + SPE10_DY**2) / 2.0
        r2 = np.sqrt(SPE10_DX**2 + 9.0 * SPE10_DY**2) / 2.0
        wi = (SPE10_DX**-2 + SPE10_DY**-2) / (
            (2.0 / np.pi) * ((SPE10_DY / SPE10_DX) * np.log(r1) + (SPE10_DX / SPE10_DY) * np.log(r2)) - 1.0
        )
        self.well_base = (producer * k[0] * wi * SPE10_DX * SPE10_DY * SPE10_DZ * FLOW_FACTOR).ravel()
        self.ends, self.half_resistance = [], []
        for axis, (part, geom) in enumerate(
            ((k[2], SPE10_DX * SPE10_DY / SPE10_DZ), (k[1], SPE10_DX * SPE10_DZ / SPE10_DY), (k[0], SPE10_DY * SPE10_DZ / SPE10_DX))
        ):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            self.ends.append((self.ids[a].ravel(), self.ids[b].ravel()))
            self.half_resistance.append(
                (1.0 / (2.0 * geom * FLOW_FACTOR * part[a].ravel()), 1.0 / (2.0 * geom * FLOW_FACTOR * part[b].ravel()))
            )
        self.i = np.concatenate([v[0] for v in self.ends])
        self.j = np.concatenate([v[1] for v in self.ends])

    @classmethod
    def load(cls) -> "SPE10DynamicModel":
        k = np.load(SPE10_INPUTS_DIR / "permeability.npy")
        phi = np.load(SPE10_INPUTS_DIR / "porosity.npy")
        if (
            k.shape != (3, *SPE10_SHAPE)
            or phi.shape != SPE10_SHAPE
            or not np.isfinite(k).all()
            or not np.isfinite(phi).all()
            or np.min(k) <= 0
            or np.min(phi) < 0.05
        ):
            raise ValueError("invalid complete SPE10 fields")
        return cls(k, phi)

    def pressure_system(self, saturation: np.ndarray):
        lam, _ = phase_properties(saturation.ravel())
        weights = [1.0 / (ri / lam[i] + rj / lam[j]) for (i, j), (ri, rj) in zip(self.ends, self.half_resistance)]
        tag = self.k[0] * lam.reshape(self.shape)
        return tag, (self.i, self.j, np.concatenate(weights)), self.well_base * lam, self.inj

    def load_snapshot_saturation(self, day: float) -> np.ndarray:
        if day == 0.0:
            return np.full(self.shape, 0.2, dtype=np.float64)
        return load_checkpoint(SPE10_INPUTS_DIR / f"snapshot_{day:09.2f}.npz.lz4")["S"]


def load_olympus_cases(npz_path: Path | None = None) -> list[dict]:
    """Load the three audited 192,749-cell OLYMPUS_1 faulted active-graph benchmark cases."""
    src = npz_path or (OLYMPUS_INPUTS_DIR / "olympus_cases.npz")
    if not src.exists():
        raise FileNotFoundError(f"OLYMPUS input archive not found at {src}")
    prefix_map = (
        ("OLYMPUS-open", "open"),
        ("OLYMPUS-baffle", "baffle"),
        ("OLYMPUS-sealing-wet", "sealing_wet"),
    )
    cases: list[dict] = []
    with np.load(src, allow_pickle=False) as z:
        xyz = np.array(z["xyz"], dtype=np.int64, copy=True)
        tail = np.array(z["tail"], dtype=np.int64, copy=True)
        head = np.array(z["head"], dtype=np.int64, copy=True)
        shape = tuple(int(v) for v in z["shape"].tolist())
        for cid, pfx in prefix_map:
            trans = np.array(z[f"{pfx}_trans"], dtype=np.float64, copy=True)
            k = np.array(z[f"{pfx}_k"], dtype=np.float64, copy=True).ravel()
            d = np.array(z[f"{pfx}_d"], dtype=np.float64, copy=True).ravel()
            b = np.array(z[f"{pfx}_b"], dtype=np.float64, copy=True).ravel()
            g = ActiveGraph.from_arrays(xyz, tail, head, trans, shape)
            cases.append(
                dict(
                    id=cid,
                    kind="graph",
                    graph=g,
                    k=k,
                    edge=(g.tail, g.head, g.trans),
                    d=d,
                    b=b,
                )
            )
    return cases


def _resolve_legacy_brugge_dir() -> Path:
    candidates = (
        BRUGGE_INPUTS_DIR,
        PROJECT_ROOT / "archive_legacy" / "v18_frozen_results_and_brugge" / "brugge",
        PROJECT_ROOT / "archive_legacy" / "v18_frozen_results_and_brugge" / "inputs" / "brugge",
    )
    for cand in candidates:
        if (cand / "inputs.npz").exists():
            return cand
    raise FileNotFoundError(
        "Legacy Brugge inputs have been archived under archive_legacy/v18_frozen_results_and_brugge/; "
        "Experiment 5 now uses load_olympus_cases()."
    )


def load_brugge_inputs() -> dict:
    """Load the archived 43,474-cell Brugge faulted active graph and well definitions."""
    bdir = _resolve_legacy_brugge_dir()
    with np.load(bdir / "inputs.npz") as z:
        g = ActiveGraph.from_arrays(z["xyz"], z["tail"], z["head"], z["trans"], tuple(map(int, z["shape"])))
        return {
            "graph": g,
            "kx": z["kx"].copy(),
            "injection": z["injection"].copy(),
            "well_base": z["well_base"].copy(),
        }


def brugge_dynamic_system(inputs: dict, saturation: np.ndarray):
    """Assemble mobility-weighted Brugge active-graph system for a frozen saturation state."""
    graph = inputs["graph"]
    s = np.asarray(saturation, dtype=np.float64)
    if s.shape != (len(graph.xyz),) or not np.isfinite(s).all() or np.any((s < 0.2) | (s > 0.8)):
        raise ValueError("invalid saved saturation")
    start = time.perf_counter()
    mobility, _ = phase_properties(s)
    w = graph.trans * (2.0 / (1.0 / mobility[graph.tail] + 1.0 / mobility[graph.head]))
    d = np.asarray(inputs["well_base"]) * mobility
    k = np.asarray(inputs["kx"]) * mobility
    b = np.asarray(inputs["injection"], dtype=np.float64)
    dynamic = ActiveGraph.from_arrays(graph.xyz, graph.tail, graph.head, w, graph.shape)
    if any(v.shape != (len(s),) or not np.isfinite(v).all() for v in (d, k, b)):
        raise ValueError("invalid physical cell vectors")
    if np.any(d < 0) or np.any(k <= 0) or np.any(b < 0) or not np.any(b):
        raise ValueError("invalid wells, permeability or zero injection")
    return dynamic, k, d, b, time.perf_counter() - start


def load_brugge_snapshot_saturation(day: float) -> np.ndarray:
    bdir = _resolve_legacy_brugge_dir()
    with np.load(bdir / f"snapshot_day_{day:08.2f}.npz") as z:
        return z["saturation"].copy()
