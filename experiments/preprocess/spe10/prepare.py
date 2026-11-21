"""Convert the pinned public SPE10 Model 2 rock fields to the benchmark layout."""
from hashlib import sha256
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "outputs" / "spe10_preparation"
INPUTS = ROOT / "data" / "inputs" / "spe10"
SHAPE = (85, 220, 60)
REVISION = "2e810769abfc001ca2cf32d1b674c4bce7a918cb"
SOURCE = f"https://raw.githubusercontent.com/OPM/opm-data/{REVISION}/spe10model2"


def read_source(name: str, digest: str) -> bytes:
    source = RAW / name
    if source.is_file():
        data = source.read_bytes()
    else:
        with urlopen(f"{SOURCE}/{name}", timeout=60) as response:
            data = response.read()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"SPE10 source checksum mismatch: {name}")
    if not source.is_file():
        RAW.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
    return data


def prepare() -> None:
    perm_source = read_source(
        "SPE10MODEL2_PERM.INC", "1829e331b64d2efb416fc40f4d68388087a11e4f7ff5ac6866eb8f1a88f8993d"
    )
    poro_source = read_source(
        "SPE10MODEL2_PHI.INC", "1426c8b40c8a1ffd3cce6653aad264685f100f82018d212ca74c8ae6d0bd1295"
    )
    sections = re.split(rb"(?m)^PERM([XYZ])[ \t]*\r?$", perm_source)
    if sections[1::2] != [b"X", b"Y", b"Z"]:
        raise ValueError("Expected PERMX, PERMY and PERMZ in that order")
    permeability = np.stack([
        np.fromstring(section.split(b"/", 1)[0], sep=" ", dtype=np.float64).reshape(SHAPE)
        for section in sections[2::2]
    ])
    porosity_values = poro_source.split(b"PORO", 1)[1].split(b"/", 1)[0]
    porosity = np.fromstring(porosity_values, sep=" ", dtype=np.float64).reshape(SHAPE)
    if not np.isfinite(permeability).all() or permeability.min() <= 0:
        raise ValueError("Invalid SPE10 permeability")
    if not np.isfinite(porosity).all() or porosity.min() < 0 or porosity.max() > 1:
        raise ValueError("Invalid SPE10 porosity")

    INPUTS.mkdir(parents=True, exist_ok=True)
    np.save(INPUTS / "permeability.npy", permeability, allow_pickle=False)
    np.save(INPUTS / "porosity.npy", np.maximum(porosity, 0.05), allow_pickle=False)
    print(f"Prepared SPE10 permeability and porosity under {INPUTS}")


if __name__ == "__main__":
    prepare()
