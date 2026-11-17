"""Numerical core for a pre-registered single-phase pressure experiment.

Callers must validate inputs, control resources and freeze provenance. This
module does not read Brugge data, write files, or certify reference error.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import cg, splu
from threadpoolctl import threadpool_info, threadpool_limits

CODE = Path(__file__).resolve().parent.parent / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from active_graph import ActiveGraph
from pressure_graph import assemble, build


def _independent_matrix(graph, d):
    """B.T diag(w) B + diag(d), independently from COO graph assembly."""
    n, m = len(graph.xyz), len(graph.trans)
    rows = np.repeat(np.arange(m), 2)
    cols = np.column_stack((graph.tail, graph.head)).ravel()
    signs = np.tile([1., -1.], m)
    incidence = sparse.csr_matrix((signs, (rows, cols)), shape=(m, n))
    matrix = (incidence.T @ incidence.multiply(graph.trans[:, None])).tocsr()
    matrix = (matrix + sparse.diags(d, format='csr')).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _face_residual(graph, d, b, x):
    """Recompute fluxes and compensated per-cell balance from returned x.

    math.fsum controls accumulation roundoff in the rounded flux terms. The
    subtraction and multiplication that generate each float64 flux are not
    outward-rounded and this is not an interval or exact-residual certificate.
    """
    q = graph.trans * (x[graph.tail] - x[graph.head])
    qwell = d * x
    if not np.isfinite(q).all() or not np.isfinite(qwell).all():
        raise ValueError('nonfinite computed flux')
    terms = [[float(b[i]), -float(qwell[i])] for i in range(len(b))]
    for i, j, v in zip(graph.tail, graph.head, q):
        terms[int(i)].append(-float(v))
        terms[int(j)].append(float(v))
    residual = np.fromiter((math.fsum(v) for v in terms), dtype=float, count=len(b))
    return q, qwell, residual


def _describe_solution(graph, A, d, b, x):
    q, qw, r = _face_residual(graph, d, b, x)
    normb = float(np.linalg.norm(b))
    maxb = float(np.max(np.abs(b)))
    totalabsb = math.fsum(abs(float(v)) for v in b)
    if normb <= 0 or maxb <= 0 or totalabsb <= 0:
        raise ValueError('zero RHS requires a different pre-registered normalization')
    matrix_r = b - A @ x
    matrix_diff = r-matrix_r
    source = math.fsum(map(float, b))
    production = math.fsum(map(float, qw))
    metrics = {'rho_l2': float(np.linalg.norm(r)/normb),
               'eta_inf': float(np.max(np.abs(r))/maxb),
               'absolute_residual_linf': float(np.max(np.abs(r))),
               'balance_relative': float(abs(source-production)/totalabsb),
               'source_sum_m3_day': source,
               'well_outflow_sum_m3_day': production,
               'well_outflow_min_m3_day': float(np.min(qw[d > 0])),
               'well_outflow_nonnegative': bool(np.all(qw[d > 0] >= 0)),
               'max_abs_face_vs_matrix_residual_difference': float(np.max(np.abs(matrix_diff))),
               'face_vs_matrix_residual_difference_relative': float(np.max(np.abs(matrix_diff))/maxb),
               'face_vs_matrix_residual_agreement_pass': bool(np.max(np.abs(matrix_diff))/maxb <= 1e-9),
               'rho_pass': bool(np.linalg.norm(r)/normb <= 1e-9),
               'eta_pass': bool(np.max(np.abs(r))/maxb <= 1e-9),
               'balance_pass': bool(abs(source-production)/totalabsb <= 1e-9)}
    return q, qw, r, metrics


def _comparison(graph, d, reference, candidate, qref, qwref, q, qw):
    delta = candidate-reference
    energy_delta = math.fsum(float(v) for v in graph.trans*(delta[graph.tail]-delta[graph.head])**2)
    energy_delta += math.fsum(float(v) for v in d*delta**2)
    energy_ref = math.fsum(float(v) for v in graph.trans*(reference[graph.tail]-reference[graph.head])**2)
    energy_ref += math.fsum(float(v) for v in d*reference**2)
    mask = d > 0
    dq, dqw = q-qref, qw-qwref
    flux_delta = math.fsum(float(v) for v in dq*dq/graph.trans)
    flux_delta += math.fsum(float(v) for v in dqw[mask]*dqw[mask]/d[mask])
    flux_ref = math.fsum(float(v) for v in qref*qref/graph.trans)
    flux_ref += math.fsum(float(v) for v in qwref[mask]*qwref[mask]/d[mask])
    if energy_ref <= 0 or flux_ref <= 0:
        raise ValueError('zero reference energy or weighted flux norm')
    energy_relative = math.sqrt(max(energy_delta, 0.)/energy_ref)
    flux_relative = math.sqrt(max(flux_delta, 0.)/flux_ref)
    return {'reference_relative_energy_error': energy_relative,
            'reference_relative_weighted_flux_error': flux_relative,
            'energy_difference_pass': bool(energy_relative <= 1e-6),
            'weighted_flux_difference_pass': bool(flux_relative <= 1e-6),
            'face_flux_difference_l2': float(np.linalg.norm(dq)),
            'face_flux_difference_linf': float(np.max(np.abs(dq))) if len(dq) else 0.,
            'well_flux_difference_l2': float(np.linalg.norm(dqw)),
            'well_flux_difference_linf': float(np.max(np.abs(dqw))),
            'reference_energy_squared': energy_ref,
            'reference_weighted_flux_squared': flux_ref}


def solve_case(graph, permeability, d, b, *, damping='rank_safe', coarse_size=512,
               rtol=1e-12, maxiter=10000):
    """Return (vector_dict, JSON-compatible metrics) for the same graph system.

    All transmissibilities and well terms must already include mobility and be
    in consistent units. The direct and PCG systems receive the exact same
    inputs. Reference quality is not certified by this numerical comparison.
    """
    if not isinstance(graph, ActiveGraph):
        raise TypeError('graph must be ActiveGraph')
    n = len(graph.xyz)
    d, b, permeability = (np.asarray(v, dtype=np.float64) for v in (d, b, permeability))
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in (d, b, permeability)):
        raise ValueError('invalid cell vector shape or nonfinite values')
    if np.any(d < 0) or np.any(permeability <= 0) or not np.any(b):
        raise ValueError('invalid d, permeability, or zero RHS')
    if rtol <= 0 or rtol >= 1 or maxiter < 1:
        raise ValueError('invalid CG tolerance or iteration limit')
    if coarse_size < 1 or coarse_size > 512:
        raise ValueError('coarse_size must be between 1 and 512')
    if damping not in ('rank_safe', 'legacy'):
        raise ValueError('unsupported damping')
    if damping == 'legacy' and n > 512:
        raise ValueError('large legacy hierarchy has no demonstrated interpolation rank; use rank_safe')
    with threadpool_limits(limits=1):
        thread_info = threadpool_info()
        begin = time.perf_counter()
        Aref = _independent_matrix(graph, d)
        Acheck, _, assembly_info = assemble(graph, d)
        diff = (Aref-Acheck).tocsr()
        maxdiff = float(np.max(np.abs(diff.data))) if diff.nnz else 0.
        scale = float(np.max(np.abs(Aref.data)))
        if maxdiff/max(scale, np.finfo(float).tiny) > 1e-12:
            raise ValueError('independent incidence and graph matrix assemblies disagree')
        assembly_seconds = time.perf_counter()-begin
        ref_start = time.perf_counter()
        factor = splu(Aref.tocsc())
        reference = factor.solve(b)
        del factor
        reference_seconds = time.perf_counter()-ref_start
        if not np.isfinite(reference).all():
            raise ValueError('reference returned nonfinite pressure')
        hierarchy_start = time.perf_counter()
        A, M, hierarchy = build(graph, permeability, d, coarse_size=coarse_size,
                                max_direct=512, damping=damping)
        delta = (A-Aref).tocsr()
        delta_max = float(np.max(np.abs(delta.data))) if delta.nnz else 0.
        if delta_max/max(scale, np.finfo(float).tiny) > 1e-12:
            raise ValueError('candidate matrix does not match reference matrix')
        hierarchy_seconds = time.perf_counter()-hierarchy_start
        count = [0]
        def callback(_):
            count[0] += 1
        solve_start = time.perf_counter()
        candidate, status = cg(A, b, x0=np.zeros_like(b), M=M, rtol=rtol, atol=0.,
                               maxiter=maxiter, callback=callback)
        solve_seconds = time.perf_counter()-solve_start
        if not np.isfinite(candidate).all():
            raise ValueError('candidate returned nonfinite pressure')
        qref, qwref, rref, ref_metrics = _describe_solution(graph, Aref, d, b, reference)
        q, qw, r, candidate_metrics = _describe_solution(graph, Aref, d, b, candidate)
        comparison = _comparison(graph, d, reference, candidate, qref, qwref, q, qw)
    vectors = {'reference_drawdown': reference, 'candidate_drawdown': candidate,
               'reference_face_flux': qref, 'candidate_face_flux': q,
               'reference_well_outflow': qwref, 'candidate_well_outflow': qw,
               'reference_residual': rref, 'candidate_residual': r,
               'rhs': b.copy(), 'well_diagonal': d.copy()}
    metrics = {'reference_method': 'SciPy SuperLU sparse direct solve',
               'scipy_version': scipy.__version__,
               'reference_certified': False,
               'reference_certification_reason': 'no validated rigorous error bound computed',
               'candidate_method': 'component-preserving graph multilevel PCG',
               'damping': damping,
               'damping_note': 'rank_safe is an explicit adaptation from SPE10 legacy damping',
               'cg_status': int(status), 'cg_iterations': int(count[0]),
               'cg_rtol_requested': float(rtol), 'cg_maxiter': int(maxiter),
               'reference': ref_metrics, 'candidate': candidate_metrics,
               'comparison_to_uncertified_reference': comparison,
               'candidate_vs_reference_assembly_max_abs': delta_max,
               'independent_assembly_max_abs': maxdiff,
               'assembly': assembly_info, 'hierarchy': hierarchy,
               'timing_seconds': {'dual_assembly': assembly_seconds,
                                  'reference_direct': reference_seconds,
                                  'candidate_hierarchy_and_assembly': hierarchy_seconds,
                                  'candidate_pcg': solve_seconds},
               'threadpools': thread_info,
               'residual_note': 'math.fsum accumulates binary64 face terms; not interval certified',
               'acceptance': 'not decided: reference error uncertified and stage audit required'}
    return vectors, metrics
