"""Preliminary graph-pressure adaptation of the existing component V-cycle.

The caller must provide independently validated physical connections and
consistent units. This module does not import a Brugge deck or certify a
reservoir model. Every physical connected component must have positive total
grounding; pure Neumann components are intentionally rejected.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits
from extension_multigrid_v2 import _aggregate_components
from spe10_dynamic_pressure import _lump_positive
from active_graph import ActiveGraph


def _maxabs(a):
    return float(np.max(np.abs(a))) if np.size(a) else 0.0


def _symmetry_error(A):
    delta = (A-A.T).tocsr()
    return _maxabs(delta.data) / max(_maxabs(A.data), np.finfo(float).tiny)


def _check_matrix(A):
    if not np.isfinite(A.data).all() or not np.all(A.diagonal() > 0):
        raise ValueError('matrix has nonfinite values or nonpositive diagonal')
    sym = _symmetry_error(A)
    if sym > 1e-12:
        raise ValueError('matrix is not symmetric within tolerance')
    return sym


def assemble(graph, well_diagonal):
    """Assemble L + diag(d), and reject ungrounded flow components.

    Parallel physical faces are summed by COO conversion, not discarded.
    For exact arithmetic this grounded positive-weight graph is SPD. The
    caller is responsible for physically correct d, rhs and pressure units.
    """
    if not isinstance(graph, ActiveGraph):
        raise TypeError('graph must be a validated ActiveGraph')
    n = len(graph.xyz)
    d = np.asarray(well_diagonal, dtype=float)
    if d.shape != (n,) or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('well diagonal must be finite, nonnegative and one per active cell')
    count, component = graph.flow_components()
    grounding = np.bincount(component, weights=d, minlength=count)
    if np.any(grounding <= 0) or not np.isfinite(grounding).all():
        raise ValueError('every flow component must be pressure grounded')
    i, j, w = graph.tail, graph.head, graph.trans
    rows = np.concatenate((i, j, i, j, np.arange(n)))
    cols = np.concatenate((i, j, j, i, np.arange(n)))
    values = np.concatenate((w, w, -w, -w, d))
    A = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    _check_matrix(A)
    rowerr = _maxabs(np.asarray(A.sum(axis=1)).ravel()-d)
    rowscale = max(_maxabs(A.data), _maxabs(d), np.finfo(float).tiny)
    if rowerr / rowscale > 1e-12:
        raise ValueError('assembled graph row sums do not match grounding')
    return A, component, {'flow_components': int(count),
                         'grounding_by_component': grounding.tolist(),
                         'assembly_row_sum_absolute_error': rowerr}


def build(graph, permeability, well_diagonal, coarse_size=512, max_levels=10,
          max_direct=2048, damping='legacy'):
    """Build the inherited component-preserving symmetric Jacobi V-cycle.

    Activity coordinates and inherited high-phase labels use the original
    aggregator. A flow-component identity is additionally incorporated into
    the label to prevent aggregation across disconnected physical regions.
    This is a deliberate adaptation of the original Cartesian method.
    """
    if coarse_size < 1 or max_levels < 1 or max_direct < coarse_size:
        raise ValueError('invalid hierarchy limits')
    if damping not in ('legacy', 'rank_safe'):
        raise ValueError('damping must be legacy or rank_safe')
    with threadpool_limits(1):
        A, flow, info = assemble(graph, well_diagonal)
        high, high_count = graph.high_components(permeability)
        # Unique (flow region, high label) keys isolate low phase across regions.
        _, tags = np.unique(np.column_stack((flow, high)), axis=0, return_inverse=True)
        tags = tags.astype(np.int64)
        xyz = graph.xyz.copy()
        radix = int(tags.max()) + 1
        matrix = A.copy()
        levels, rows = [], []
        reason = None
        for level_no in range(max_levels):
            sym = _check_matrix(matrix)
            diagonal = matrix.diagonal()
            inverse = 1/diagonal
            bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()*inverse))
            if not np.isfinite(bound) or bound <= 0:
                raise ValueError('invalid Jacobi spectral bound')
            # The conservative alternative is an explicit algorithm change.
            # In exact arithmetic it guarantees ||omega D^-1 A||_infty <= .9.
            omega = (min(2/3, 4/(3*bound)) if damping == 'legacy'
                     else min(2/3, 0.9/bound))
            lev = {'A': matrix, 'di': omega*inverse}
            levels.append(lev)
            record = {'size': int(matrix.shape[0]), 'nnz': int(matrix.nnz),
                      'tags': int(np.unique(tags).size), 'symmetry_relative': sym,
                      'jacobi_bound': bound, 'jacobi_omega': omega,
                      'jacobi_bound_product': omega*bound,
                      'row_sum_min': float(np.min(np.asarray(matrix.sum(axis=1)))),
                      'row_sum_max': float(np.max(np.asarray(matrix.sum(axis=1))))}
            rows.append(record)
            if matrix.shape[0] <= coarse_size:
                reason = 'coarse_size'
                break
            if level_no + 1 == max_levels:
                reason = 'max_levels'
                break
            blocks = xyz//2
            side = int(np.max(blocks)) + 1
            # Existing packed integer keys must fit signed int64 before calling.
            if (side**3)*radix-1 > np.iinfo(np.int64).max:
                raise OverflowError('existing aggregate key may overflow int64')
            aggregate, next_xyz, next_tags = _aggregate_components(xyz, tags, radix)
            nc = len(next_tags)
            if (len(aggregate) != matrix.shape[0] or np.any(aggregate < 0) or
                    np.any(aggregate >= nc) or np.unique(aggregate).size != nc):
                raise ValueError('aggregation is not a complete nonempty partition')
            if nc >= matrix.shape[0]:
                reason = 'no_reduction'
                break
            Z = sparse.csr_matrix((np.ones(matrix.shape[0]), aggregate,
                                  np.arange(matrix.shape[0]+1)), shape=(matrix.shape[0], nc))
            AZ = (matrix@Z).tocsr()
            AZ.data *= np.repeat(lev['di'], np.diff(AZ.indptr))
            P = (Z-AZ).tocsr()
            P.eliminate_zeros()
            psum = np.asarray(P.sum(axis=1)).ravel()
            if not np.isfinite(psum).all() or np.any(np.abs(psum) < 1e-14):
                raise ValueError('invalid prolongation row sum before normalization')
            P.data *= np.repeat(1/psum, np.diff(P.indptr))
            if not np.isfinite(P.data).all():
                raise ValueError('nonfinite prolongation')
            perr = _maxabs(np.asarray(P.sum(axis=1)).ravel()-1)
            if perr > 1e-12:
                raise ValueError('prolongation row sums do not equal one')
            # Small cases get an explicit rank check; large cases still require
            # separate numerical SPD qualification before production use.
            if max(P.shape) <= 512 and np.linalg.matrix_rank(P.toarray()) != nc:
                raise ValueError('rank-deficient prolongation')
            R = P.T.tocsr()
            lev.update(P=P, R=R)
            coarse = (R@(matrix@P)).tocsr()
            coarse = ((coarse+coarse.T)*0.5).tocsr()
            before = np.asarray(coarse.sum(axis=1)).ravel()
            expected = np.asarray(R@np.asarray(matrix.sum(axis=1)).ravel()).ravel()
            galerkin_row_error = _maxabs(before-expected)
            if galerkin_row_error/max(_maxabs(coarse.data), np.finfo(float).tiny) > 1e-12:
                raise ValueError('Galerkin row sums do not preserve pressure grounding')
            coarse, lumped = _lump_positive(coarse)
            after = np.asarray(coarse.sum(axis=1)).ravel()
            err = _maxabs(before-after)
            scale = max(_maxabs(coarse.data), np.finfo(float).tiny)
            if err/scale > 1e-12:
                raise ValueError('positive lumping did not preserve row sums')
            _check_matrix(coarse)
            record.update(prolongation_row_error=perr,
                          galerkin_row_sum_error=galerkin_row_error,
                          positive_offdiagonals_lumped=int(lumped),
                          lumping_row_sum_error=err)
            matrix, xyz, tags = coarse, next_xyz, next_tags
        if matrix.shape[0] > max_direct:
            raise MemoryError('coarsest direct solve exceeds guard')
        # Coarse SPD checking is tractable for the intended <=512 direct level.
        if matrix.shape[0] <= 512:
            try:
                np.linalg.cholesky(matrix.toarray())
            except np.linalg.LinAlgError as exc:
                raise ValueError('coarsest matrix failed SPD check') from exc
        levels[-1]['factor'] = splu(matrix.tocsc())

    def cycle(index, rhs):
        lev = levels[index]
        if 'factor' in lev:
            return lev['factor'].solve(rhs)
        x = lev['di']*rhs
        x += lev['P']@cycle(index+1, lev['R']@(rhs-lev['A']@x))
        x += lev['di']*(rhs-lev['A']@x)
        return x

    M = LinearOperator(A.shape, matvec=lambda r: cycle(0, r), dtype=np.float64)
    info.update({'high_component_count': high_count, 'phase_rule': 'Kx > min(Kx)',
                 'levels': rows, 'termination': reason, 'coarse_size': coarse_size,
                 'damping': damping,
                 'adaptation': 'physical flow-region identity added to inherited aggregation tags',
                 'qualification': 'prototype; full-size SPD and physical inputs require independent validation'})
    return A, M, info
