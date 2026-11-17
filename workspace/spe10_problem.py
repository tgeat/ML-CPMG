"""Full SPE10 Model 2, steady single-phase, prescribed balanced five-well loads."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'spe10_model2_singlephase'
SHAPE = (85, 220, 60)  # K,J,I; Eclipse I is fastest
DX, DY, DZ = 20., 10., 2.  # ft
REV = '2e810769abfc001ca2cf32d1b674c4bce7a918cb'
NAME = 'SPE10MODEL2_PERM.INC'
URL = f'https://raw.githubusercontent.com/OPM/opm-data/{REV}/spe10model2/{NAME}'
CHECKSUM = '1829e331b64d2efb416fc40f4d68388087a11e4f7ff5ac6866eb8f1a88f8993d'


def prepare():
    """Cache the three component arrays, preserving every supplied value in mD."""
    from urllib.request import urlretrieve
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / NAME
    if not source.exists():
        urlretrieve(URL, source)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKSUM:
        raise ValueError('OPM permeability file checksum mismatch')
    sections = re.split(rb'(?m)^PERM([XYZ])[ \t]*\r?$', raw)
    if sections[1::2] != [b'X', b'Y', b'Z']:
        raise ValueError('missing or unordered PERM keywords')
    fields = []
    for section in sections[2::2]:
        body, suffix = section.strip().split(b'/')
        if suffix.strip():
            raise ValueError('unexpected Eclipse data after terminator')
        field = np.fromstring(body, sep=' ', dtype=np.float64).reshape(SHAPE)
        if not np.all(np.isfinite(field)) or np.min(field) <= 0:
            raise ValueError('invalid permeability data')
        fields.append(field)
    dest = OUT / 'permeability.npy'
    np.save(dest, np.stack(fields))
    metadata = {
        'provider': 'Open Porous Media, opm-data, spe10model2', 'url': URL,
        'revision': REV, 'source_sha256': CHECKSUM,
        'source_git_blob': hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
        'source_bytes': len(raw), 'shape_zyx': SHAPE, 'cell_ft_xyz': [DX,DY,DZ],
        'native_md_ranges': {name: [float(a.min()),float(a.max())]
                             for name,a in zip(('Kx','Ky','Kz'),fields)},
        'Kx_equals_Ky': bool(np.array_equal(fields[0],fields[1])),
        'tag_Kx_gt_min_fraction': float(np.mean(fields[0] > fields[0].min())),
        'converted_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
    }
    (OUT/'data.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def graph():
    kx, ky, kz = np.load(OUT/'permeability.npy', allow_pickle=False)
    ids = np.arange(kx.size, dtype=np.int32).reshape(SHAPE)
    tail, head, weights = [], [], []
    for axis, (field, area_over_spacing) in enumerate(
            ((kz, DX*DY/DZ), (ky, DX*DZ/DY), (kx, DY*DZ/DX))):
        a, b = [slice(None)]*3, [slice(None)]*3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        tail.append(ids[a].ravel())
        head.append(ids[b].ravel())
        weights.append((2*area_over_spacing/(1/field[a]+1/field[b])).ravel())
    # The reference pressure is a scaled pressure; mobility/field-unit conversion
    # is one global factor. Each layer has exactly compatible binary source rates.
    rhs = np.zeros(SHAPE)
    rhs[:, 109, 29] = 1.
    for y in (0, SHAPE[1]-1):
        for x in (0, SHAPE[2]-1):
            rhs[:, y, x] = -.25
    return kx, rhs.ravel(), tuple(np.concatenate(v) for v in (tail, head, weights))


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
