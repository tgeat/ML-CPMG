"""Offline reference and verification for a full rectangular Neumann Darcy graph.

Node IDs follow C order for the supplied shape. Every interior face must occur
exactly once. The operator is the graph of the stored binary64 transmissibilities,
not a rounded assembled CSR. CSR and an optional inverse serve only as tools
for producing a reference; acceptance is determined from the face equations.
"""
from hashlib import sha256
from math import prod
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from . import interval_bounds as fc
from .. import double_double as gp


def _identity(shape, b, edge, hi, lo):
    digest = sha256()
    for value in (np.asarray(shape, dtype="<i8"), b, *edge, hi, lo):
        arr = np.ascontiguousarray(value)
        digest.update(str((arr.shape, arr.dtype.str)).encode("ascii"))
        digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def _grid(shape, b, edge):
    shape = tuple(int(n) for n in shape)
    ncell = prod(shape)
    if len(shape) != 3 or min(shape) < 1 or ncell < 2 or ncell != b.size:
        raise ValueError("Expected a 3D Cartesian shape with one RHS per cell")
    i, j, t = map(np.asarray, edge)
    if (any(v.ndim != 1 for v in (i, j, t)) or not i.shape == j.shape == t.shape
            or i.dtype.kind not in "iu" or j.dtype.kind not in "iu"
            or t.dtype != np.float64 or not np.isfinite(t).all() or np.any(t <= 0)
            or min(i.min(), j.min()) < 0 or max(i.max(), j.max()) >= ncell):
        raise ValueError("Invalid positive-weight nearest-neighbor graph")
    low = np.minimum(i, j)
    diff = np.maximum(i, j) - low
    sets = []
    gaps = []
    for axis, length in enumerate(shape):
        if length == 1:
            continue
        stride = prod(shape[axis+1:])
        idx = np.flatnonzero((diff == stride) & ((low // stride) % length < length-1))
        expected = (ncell // length) * (length-1)
        if idx.size != expected or np.max(np.bincount(low[idx])) != 1:
            raise ValueError(f"Edges on Cartesian axis {axis} are missing or duplicated")
        # Four times the minimum axis weight divided by axis length squared
        # bounds the first nonconstant path eigenvalue from below, since
        # sin(pi/(2*n)) >= 1/n. Tensor-product modes give the minimum axis bound.
        gaps.append(float(fc._down(4.0 * np.min(t[idx]) / (length * length))))
        sets.append((low[idx], low[idx] + stride, t[idx]))
    if sum(a.size for a, _, _ in sets) != t.size:
        raise ValueError("Unexpected non-neighbor, duplicate, or periodic edges")
    if not np.isfinite(min(gaps)) or min(gaps) <= 0:
        raise ValueError("No positive certified lower bound for the spectral gap")
    return sets, min(gaps)


def _residual_energy(sets, b, mean, hi, lo):
    """Accumulate the large flux words with TwoSum; enclose only their tails.

    Within each oriented axis the 'first' nodes are unique, and likewise the
    'second' nodes. This makes each indexed update a simultaneous scatter
    without dropped additions. Delaying high/low recombination until the end
    preserves cancellation between the source and six opposing face fluxes.
    """
    high = b.copy()
    lower = np.full_like(b, -mean[1])
    upper = np.full_like(b, -mean[0])
    energy_lo, energy_hi = [], []
    for first, second, weights in sets:
        diff, errors = fc._edge_dd_words(hi, lo, first, second)
        tails = fc._exact_sum_interval(errors)
        normlo, normhi = fc._norm_interval(fc._add((diff, diff), tails), weights)
        energy_lo.append(max(0.0, fc._down(normlo * normlo)))
        energy_hi.append(fc._up(normhi * normhi))
        product, error = fc.two_prod(weights, diff)
        corr = fc._add((error, error), fc._scale_positive(tails, weights))
        for cells, sign in ((first, -1), (second, 1)):
            summed, rounded = fc.two_sum(high[cells], sign * product)
            high[cells] = summed
            tail = corr if sign > 0 else (-corr[1], -corr[0])
            tail = fc._add(tail, (rounded, rounded))
            lower[cells], upper[cells] = fc._add((lower[cells], upper[cells]), tail)
    residual = fc._add((high, high), (lower, upper))
    norm = (float(max(0, fc._down(np.sqrt(fc.positive_sum_bounds(energy_lo)[0])))),
            float(fc._up(np.sqrt(fc.positive_sum_bounds(energy_hi)[1]))))
    return residual, norm


def make_reference(shape, b, edge, A, *, preconditioner=None, initial_pressure=None,
                   relative_target=5e-9, max_refinements=10, inner_maxiter=1200,
                   inner_rtol=1e-7, deadline=None):
    """Return reference (hi, lo, certificate) or raise on failure.

    initial_pressure may be an array or a (hi,lo) pair. If a solver's pilot
    supplies a starting point, disclose that fact. No timed solver may use
    this reference. 'deadline' is an optional absolute perf_counter value.
    The result is independently certified even when an existing hierarchy
    or an approximate solution is reused to reduce offline computation.
    """
    if not 0 < relative_target < 1 or not 0 < inner_rtol < 1:
        raise ValueError("Relative tolerances must lie between zero and one")
    fc._ieee_probe()
    b = np.asarray(b, dtype=float).ravel()
    sets, lower_gap = _grid(shape, b, edge)
    if not np.all(np.isfinite(b)) or A.shape != (b.size, b.size):
        raise ValueError("Invalid fine operator or RHS")
    mean = fc._source_projection(b)[1]
    if isinstance(initial_pressure, tuple):
        hi, lo = (np.asarray(v, dtype=float).ravel().copy() for v in initial_pressure)
    else:
        hi = (np.zeros_like(b) if initial_pressure is None
              else np.asarray(initial_pressure, dtype=float).ravel().copy())
        lo = np.zeros_like(b)
    if hi.shape != b.shape or lo.shape != b.shape:
        raise ValueError("Reference initial pressure has incorrect shape")
    hi, lo = gp.pair_project(hi, lo)
    M = preconditioner
    history = []
    def guard(*_):
        if deadline is not None and perf_counter() >= deadline:
            raise TimeoutError("Reference time budget exhausted")

    for iteration in range(max_refinements + 1):
        guard()
        residual, norm = _residual_energy(sets, b, mean, hi, lo)
        rnorm = fc._norm_interval(residual)[1]
        epsilon = float(fc._up(rnorm / fc._down(np.sqrt(lower_gap))))
        denom = fc._down(norm[0] - epsilon)
        ratio = float(fc._up(epsilon / denom)) if denom > 0 else None
        history.append({"outer": iteration, "residual_norm_upper": rnorm,
                        "relative_uncertainty_upper": ratio})
        if ratio is not None and ratio <= relative_target:
            return hi, lo, {
                "energy_uncertainty_upper": epsilon, "energy_norm_interval": norm,
                "relative_uncertainty_upper": ratio, "lambda2_lower": lower_gap,
                "method": "axiswise path Poincare + outward EFT face residual of actual DD pair",
                "identity_sha256": _identity(shape, b, edge, hi, lo),
                "initial_pressure_supplied": initial_pressure is not None,
                "external_inverse_supplied": preconditioner is not None, "history": history,
            }
        if iteration == max_refinements:
            break
        # These approximations produce corrections, never error bounds.
        target = residual[0] / 2 + residual[1] / 2
        target -= target.mean()
        if M is None:
            d_inv = 1 / np.asarray(A.diagonal()).ravel()
            M = LinearOperator(A.shape, matvec=lambda r: d_inv*r, dtype=float)
        delta, status = cg(A, target, M=M, rtol=inner_rtol, atol=0,
                           maxiter=inner_maxiter, callback=guard)
        history[-1]["inner_status"] = int(status)
        if status < 0 or not np.isfinite(delta).all():
            raise RuntimeError("Reference correction failed")
        hi, lo = gp.pair_project(*gp.pair_update(hi, lo, delta))
    raise RuntimeError(f"Uncertified reference after refinement: {history}")


def assess(shape, b, edge, p_hi, p_lo, reference_hi, reference_lo, certificate, q=None):
    """Check energy/flux 1e-6, actual flux and pressure rho/eta 1e-9.

    Supply the stored q when the timed solver exported it. When absent, q
    is reconstructed using the same binary64 graph-flux policy and labelled.
    """
    b = np.asarray(b, dtype=float).ravel()
    if certificate["identity_sha256"] != _identity(shape, b, edge, reference_hi, reference_lo):
        raise ValueError("Graph, source, or reference changed since certification")
    computed = q is None
    if computed:
        q = gp.graph_flux(p_hi, p_lo, edge)
    # The legacy certifier accepts a cell field exclusively for a positivity
    # and size check. Its physical operator is given entirely by edge[2];
    # the three SPE10 conductivity components have already determined it.
    positive_cell_marker = np.broadcast_to(np.float64(1.0), b.shape)
    result = fc.certify(positive_cell_marker, b, edge, p_hi, p_lo, q,
                        reference_hi, reference_lo, certificate["energy_uncertainty_upper"],
                        *certificate["energy_norm_interval"])
    result["flux_source"] = ("reconstructed from supplied pressure pair" if computed
                             else "supplied solver output")
    return result
