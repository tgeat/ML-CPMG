"""Auditable activity graph and component keys for a corner-point adapter.

This module does not read a reservoir deck or construct its connections. Inputs
must come from a verified geometry/flow importer, and edges must include only
physically effective, positive-transmissibility connections. IDs are contiguous
zero-based active-cell indices; xyz are zero-based logical indices (i,j,k).
Multiple physical faces between the same cells are allowed and retained.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class ActiveGraph:
    xyz: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    trans: np.ndarray
    shape: tuple

    @classmethod
    def from_arrays(cls, xyz, tail, head, trans, shape):
        xyz, tail, head = map(np.asarray, (xyz, tail, head))
        trans = np.asarray(trans, dtype=np.float64)
        shape = tuple(shape)
        if (len(shape) != 3 or any(not isinstance(x, (int, np.integer)) or x <= 0 for x in shape)):
            raise ValueError('shape must contain three positive integers in i,j,k order')
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0 or xyz.dtype.kind not in 'iu':
            raise ValueError('xyz must be a nonempty integer array with shape (n,3)')
        if np.any(xyz < 0) or np.any(xyz >= np.asarray(shape)):
            raise ValueError('active logical coordinates outside model shape')
        if np.unique(xyz, axis=0).shape[0] != len(xyz):
            raise ValueError('duplicate active logical coordinate')
        if (tail.ndim != 1 or head.ndim != 1 or trans.ndim != 1 or
                tail.shape != head.shape or tail.shape != trans.shape or
                tail.dtype.kind not in 'iu' or head.dtype.kind not in 'iu'):
            raise ValueError('edge arrays must be equal-length vectors with integer endpoints')
        n = len(xyz)
        if np.any(tail < 0) or np.any(head < 0) or np.any(tail >= n) or np.any(head >= n):
            raise ValueError('edge endpoint outside active range')
        if np.any(tail == head):
            raise ValueError('self connection')
        if not np.isfinite(trans).all() or np.any(trans <= 0):
            raise ValueError('effective transmissibilities must be finite and strictly positive')
        # Copies prevent a caller from accidentally mutating the validated inputs.
        values = [np.array(a, dtype=d, copy=True) for a, d in
                  ((xyz, np.int64), (tail, np.int64), (head, np.int64), (trans, np.float64))]
        for value in values:
            value.flags.writeable = False
        return cls(*values, tuple(map(int, shape)))

    def logical_to_active(self):
        """Return dict mapping each active (i,j,k) to the corresponding row ID."""
        return {tuple(coord): i for i, coord in enumerate(self.xyz.tolist())}

    def adjacency(self):
        """Unweighted symmetric physical connectivity; duplicate faces are merged."""
        n = len(self.xyz)
        rows = np.concatenate((self.tail, self.head))
        cols = np.concatenate((self.head, self.tail))
        graph = sparse.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                                  shape=(n, n)).tocsr()
        graph.data[:] = 1
        return graph

    def flow_components(self):
        return connected_components(self.adjacency(), directed=False, return_labels=True)

    def high_components(self, permeability):
        """SPE10 rule Kx > min(Kx); identify high nodes via the physical graph.

        Low phase has tag 0, high components have tags 1..count. Only high-high
        physical edges contribute; a high node connected through low nodes does
        not join another high component. Isolated high cells are components.
        """
        k = np.asarray(permeability, dtype=np.float64)
        if k.shape != (len(self.xyz),) or not np.isfinite(k).all() or np.any(k <= 0):
            raise ValueError('one finite positive permeability per active cell required')
        high = k > np.min(k)
        tags = np.zeros(len(k), dtype=np.int64)
        ids = np.flatnonzero(high)
        if len(ids) == 0:
            return tags, 0
        count, component = connected_components(self.adjacency()[ids][:, ids], directed=False)
        tags[ids] = component.astype(np.int64) + 1
        return tags, int(count)

    def audit(self):
        count, labels = self.flow_components()
        pairs = np.sort(np.column_stack((self.tail, self.head)), axis=1)
        unique = len(np.unique(pairs, axis=0))
        steps = np.abs(self.xyz[self.tail] - self.xyz[self.head]).sum(axis=1)
        return {'active_cells': len(self.xyz), 'physical_edges': len(self.tail),
                'unique_connected_pairs': unique,
                'parallel_edge_count': len(self.tail) - unique,
                'non_face_logical_edges': int(np.count_nonzero(steps != 1)),
                'flow_components': int(count),
                'component_sizes': np.bincount(labels, minlength=count).tolist()}


def aggregate_components(xyz, tags):
    """Equivalent component-preserving 2x2x2 keys without integer packing.

    The returned inverse assigns exactly one parent to every current cell.
    Coordinates and inherited tags form the entire coarse key. This does not
    assert that disconnected low-phase cells in the same block are connected;
    it matches the original SPE10 rule and needs a separate physical audit.
    """
    xyz, tags = np.asarray(xyz), np.asarray(tags)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0 or
            xyz.dtype.kind not in 'iu' or np.any(xyz < 0) or
            tags.shape != (len(xyz),) or tags.dtype.kind not in 'iu' or np.any(tags < 0)):
        raise ValueError('invalid coarse coordinates or tags')
    keys = np.column_stack((xyz // 2, tags))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.astype(np.int64), unique[:, :3], unique[:, 3]
