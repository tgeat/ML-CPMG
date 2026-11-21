"""Cartesian TPFA grid primitives and non-Cartesian faulted ActiveGraph."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _project(x: np.ndarray) -> np.ndarray:
    """Project onto the zero-mean subspace 1^perp."""
    return x - np.mean(x)


def _csr_bytes(matrix: sparse.csr_matrix) -> int:
    """Persistent byte count of a CSR matrix's three buffers."""
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def edges(k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oriented interior faces and harmonic TPFA transmissibilities on a unit cube."""
    n = k.shape[0]
    ids = np.arange(k.size).reshape(k.shape)
    tail, head, weights = [], [], []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(lo), tuple(hi)
        tail.append(ids[a].ravel())
        head.append(ids[b].ravel())
        weights.append((2.0 / n / (1.0 / k[a] + 1.0 / k[b])).ravel())
    return np.concatenate(tail), np.concatenate(head), np.concatenate(weights)


def sources(n: int) -> np.ndarray:
    """Balanced unit-rate injection/production source on an n^3 unit cube."""
    c = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    left = (x < 0.25) & (abs(y - 0.5) < 0.125) & (abs(z - 0.5) < 0.125)
    right = (x > 0.75) & (abs(y - 0.5) < 0.125) & (abs(z - 0.5) < 0.125)
    b = np.zeros((n, n, n), dtype=np.float64)
    b[left] = 1.0 / left.sum()
    b[right] = -1.0 / right.sum()
    return b.ravel()


def flux(p: np.ndarray, edge: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Conservative oriented face fluxes q = t * (p_tail - p_head)."""
    i, j, t = edge
    return t * (p[i] - p[j])


def divergence(
    q: np.ndarray,
    edge: tuple[np.ndarray, np.ndarray, np.ndarray],
    size: int,
) -> np.ndarray:
    """Discrete cellwise divergence of oriented face fluxes."""
    i, j, _ = edge
    return np.bincount(i, weights=q, minlength=size) - np.bincount(j, weights=q, minlength=size)


def action(p: np.ndarray, edge: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Matrix-free graph Laplacian action."""
    return divergence(flux(p, edge), edge, len(p))


def metrics(
    p: np.ndarray,
    b: np.ndarray,
    edge: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Normalized L2/Linf physical balance residuals and mean pressure."""
    r = b - action(p, edge)
    norm = np.linalg.norm(b)
    return {
        "rho": float(np.linalg.norm(r) / norm) if norm else float(np.linalg.norm(r)),
        "eta": float(np.max(abs(r)) / np.max(abs(b))) if norm else float(np.max(abs(r))),
        "mean_pressure": float(np.mean(p)),
    }


def _fine_csr(k: np.ndarray, edge: tuple[np.ndarray, np.ndarray, np.ndarray]) -> sparse.csr_matrix:
    """Fill sorted seven-point cubic CSR directly from oriented interior faces."""
    n = k.shape[0]
    total = k.size
    tail, head, weights = edge
    degree = np.bincount(tail, minlength=total) + np.bincount(head, minlength=total) + 1
    nonzeros = int(np.sum(degree))
    index_type = np.int32 if max(total, nonzeros) < np.iinfo(np.int32).max else np.int64
    indptr = np.empty(total + 1, dtype=index_type)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    indices = np.empty(nonzeros, dtype=index_type)
    data = np.empty(nonzeros, dtype=np.float64)
    cursor = indptr[:-1].copy()
    diagonal = (
        np.bincount(tail, weights=weights, minlength=total)
        + np.bincount(head, weights=weights, minlength=total)
    )
    faces_per_axis = (n - 1) * n * n

    def insert(rows, columns, values):
        positions = cursor[rows]
        indices[positions] = columns
        data[positions] = values
        cursor[rows] += 1

    for axis in range(3):
        section = slice(axis * faces_per_axis, (axis + 1) * faces_per_axis)
        insert(head[section], tail[section], -weights[section])
    cells = np.arange(total, dtype=index_type)
    insert(cells, cells, diagonal)
    for axis in (2, 1, 0):
        section = slice(axis * faces_per_axis, (axis + 1) * faces_per_axis)
        insert(tail[section], head[section], -weights[section])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(total, total))
    matrix.has_sorted_indices = True
    return matrix


@dataclass(frozen=True)
class ActiveGraph:
    """Immutable positive-transmissibility active cell graph with logical coordinates."""

    xyz: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    trans: np.ndarray
    shape: tuple[int, int, int]

    @classmethod
    def from_arrays(cls, xyz, tail, head, trans, shape) -> "ActiveGraph":
        xyz, tail, head = map(np.asarray, (xyz, tail, head))
        trans = np.asarray(trans, dtype=np.float64)
        shape = tuple(shape)
        if len(shape) != 3 or any(not isinstance(x, (int, np.integer)) or x <= 0 for x in shape):
            raise ValueError("shape must contain three positive integers in i,j,k order")
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0 or xyz.dtype.kind not in "iu":
            raise ValueError("xyz must be a nonempty integer array with shape (n,3)")
        if np.any(xyz < 0) or np.any(xyz >= np.asarray(shape)):
            raise ValueError("active logical coordinates outside model shape")
        if np.unique(xyz, axis=0).shape[0] != len(xyz):
            raise ValueError("duplicate active logical coordinate")
        if (
            tail.ndim != 1
            or head.ndim != 1
            or trans.ndim != 1
            or tail.shape != head.shape
            or tail.shape != trans.shape
            or tail.dtype.kind not in "iu"
            or head.dtype.kind not in "iu"
        ):
            raise ValueError("edge arrays must be equal-length vectors with integer endpoints")
        n = len(xyz)
        if np.any(tail < 0) or np.any(head < 0) or np.any(tail >= n) or np.any(head >= n):
            raise ValueError("edge endpoint outside active range")
        if np.any(tail == head):
            raise ValueError("self connection")
        if not np.isfinite(trans).all() or np.any(trans <= 0):
            raise ValueError("effective transmissibilities must be finite and strictly positive")
        values = [
            np.array(a, dtype=d, copy=True)
            for a, d in ((xyz, np.int64), (tail, np.int64), (head, np.int64), (trans, np.float64))
        ]
        for value in values:
            value.flags.writeable = False
        return cls(*values, tuple(map(int, shape)))

    def logical_to_active(self) -> dict[tuple[int, int, int], int]:
        return {tuple(coord): i for i, coord in enumerate(self.xyz.tolist())}

    def adjacency(self) -> sparse.csr_matrix:
        n = len(self.xyz)
        rows = np.concatenate((self.tail, self.head))
        cols = np.concatenate((self.head, self.tail))
        graph = sparse.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n)).tocsr()
        graph.data[:] = 1
        return graph

    def flow_components(self):
        return connected_components(self.adjacency(), directed=False, return_labels=True)

    def high_components(self, permeability: np.ndarray) -> tuple[np.ndarray, int]:
        k = np.asarray(permeability, dtype=np.float64)
        if k.shape != (len(self.xyz),) or not np.isfinite(k).all() or np.any(k <= 0):
            raise ValueError("one finite positive permeability per active cell required")
        high = k > np.min(k)
        tags = np.zeros(len(k), dtype=np.int64)
        ids = np.flatnonzero(high)
        if len(ids) == 0:
            return tags, 0
        count, component = connected_components(self.adjacency()[ids][:, ids], directed=False)
        tags[ids] = component.astype(np.int64) + 1
        return tags, int(count)
