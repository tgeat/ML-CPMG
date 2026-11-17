"""Lossless immutable Darcy checkpoint I/O, without solution recomputation.

Saving is timed by the caller. Loading/decompression is an offline operation;
any future resume path must account for that I/O separately. ``np.savez`` uses
ZIP_STORED entries. LZ4 wraps that ZIP as one streaming level-0 frame and does
not build an uncompressed archive in memory before writing it.
"""
import hashlib
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np


def save_checkpoint(path, payload, *, compression='none'):
    """Exclusively create path, close all streams, hash it, and return its record.

    path must end in .npz for 'none', or .npz.lz4 for 'lz4'. Existing and
    incomplete files are never replaced. A failed write is retained for audit.
    Compression, file/frame close and SHA-256 all precede completed timestamp.
    The lz4.frame writer uses its single-threaded frame API, level 0.
    """
    begun = time.perf_counter()
    if compression not in ('none', 'lz4'):
        raise ValueError("compression must be 'none' or 'lz4'")
    path = Path(path).resolve()
    expected = '.npz.lz4' if compression == 'lz4' else '.npz'
    if not path.name.endswith(expected) or (compression == 'none' and path.name.endswith('.npz.lz4')):
        raise ValueError('checkpoint suffix must match compression: '+expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == 'lz4':
        import lz4.frame
    with path.open('xb') as destination:
        if compression == 'lz4':
            # LZ4FrameFile is unseekable when writing. ZipFile consequently
            # emits data descriptors, keeping this a sequential ZIP_STORED
            # stream rather than requesting an in-memory full NPZ buffer.
            with lz4.frame.open(destination, mode='wb', compression_level=0) as frame:
                np.savez(frame, **payload)
        else:
            np.savez(destination, **payload)
    closed = time.perf_counter()
    hashing = time.perf_counter()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    file_hash = digest.hexdigest()
    file_bytes = path.stat().st_size
    completed = time.perf_counter()
    return {'path': str(path), 'sha256': file_hash, 'bytes': int(file_bytes),
            'compression': compression,
            'compression_level': 0 if compression == 'lz4' else None,
            'compression_threads': 1,
            'file_closed_perf_counter': closed,
            'completed_perf_counter': completed,
            'write_and_close_seconds': closed-begun,
            'hash_seconds': completed-hashing,
            'save_seconds': completed-begun}


def load_checkpoint(path, *, allow_pickle=False):
    """Read .npz or .npz.lz4 into an ordinary dict of stored arrays.

    This helper only decodes bytes; it never recovers/recomputes pressure,
    flux, residuals or metrics. Compressed input is spooled with bounded RAM
    before ZipFile's random-access reads, avoiding repeated decompression.
    """
    path = Path(path)

    def read_archive(source):
        with np.load(source, allow_pickle=allow_pickle) as archive:
            return {name: archive[name].copy() for name in archive.files}

    if path.name.endswith('.npz.lz4'):
        import lz4.frame
        with lz4.frame.open(path, mode='rb') as source:
            with tempfile.SpooledTemporaryFile(max_size=64*1024*1024, mode='w+b') as decoded:
                shutil.copyfileobj(source, decoded, length=1024*1024)
                decoded.seek(0)
                return read_archive(decoded)
    if path.name.endswith('.npz'):
        return read_archive(path)
    raise ValueError('supported checkpoint suffixes are .npz and .npz.lz4')
