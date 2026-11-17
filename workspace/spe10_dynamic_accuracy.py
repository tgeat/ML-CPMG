"""Offline certificates for the well-grounded dynamic SPE10 pressure systems.

The pressure variable is drawdown x=p-p_BHP. The exact discrete operator is
the serialized face graph plus the nonnegative producer diagonal, L+D.
Neither the RHS nor pressure is projected. Approximate linear solvers only
generate reference candidates. Outward EFT residuals and a grounded-path
inequality determine the reference uncertainty independently.
"""
from hashlib import sha256
from math import prod
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

import fast_certificates as fc
import graph_precision as gp
from spe10_accuracy import _grid


def _identity(shape, b, edge, d, hi, lo):
    digest = sha256()
    for value in (np.asarray(shape, dtype='<i8'), b, *edge, d, hi, lo):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _validate(shape, b, edge, d):
    shape = tuple(int(n) for n in shape)
    b, d = (np.asarray(v, dtype=np.float64).ravel() for v in (b, d))
    if d.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(d).all():
        raise ValueError('RHS and producer diagonal must be finite cell vectors')
    if np.any(d < 0) or not np.any(d > 0) or not np.any(b):
        raise ValueError('A positive well anchor and a nonzero RHS are required')
    sets, _ = _grid(shape, b, edge)
    return shape, b, d, sets


def _grounded_gap(shape, sets, d):
    """Rigorous lower bound for the smallest eigenvalue of L+D.

    Choose an anchor a with d[a]>0 and a coordinate path to every cell i.
    Weighted Cauchy gives x[i]^2 <= R_i*x.T*(L+D)*x, where R_i is the
    anchor resistance plus the path resistance. Axis-minimum weights bound
    all such resistances. Summing R_i is an upper bound for ||(L+D)^-1||.
    """
    count = prod(shape)
    anchor = int(np.argmax(d))
    coordinates = np.unravel_index(anchor, shape)
    inverse_bound = float(fc._up(float(count) / d[anchor]))
    axis_bounds = []
    for axis, (_, _, weights) in zip((a for a in range(3) if shape[a] > 1), sets):
        length, position = shape[axis], int(coordinates[axis])
        distance = position*(position+1)//2 + (length-1-position)*(length-position)//2
        summed_distances = (count//length)*distance
        min_weight = float(weights.min())
        contribution = float(fc._up(float(summed_distances)/min_weight))
        inverse_bound = float(fc._up(inverse_bound+contribution))
        axis_bounds.append({'axis': axis, 'minimum_weight': min_weight,
                            'summed_anchor_distances': summed_distances})
    lower = float(fc._down(1.0/inverse_bound))
    if not np.isfinite(lower) or lower <= 0:
        raise ValueError('Grounded path bound is nonpositive or nonfinite')
    return lower, {'anchor_cell': anchor, 'anchor_diagonal': float(d[anchor]),
                   'inverse_eigenvalue_bound': inverse_bound, 'axis_bounds': axis_bounds}


def _combine_norms(norms):
    lower_squares = [max(0.0, float(fc._down(lo*lo))) for lo, _ in norms]
    upper_squares = [float(fc._up(hi*hi)) for _, hi in norms]
    lower = fc.positive_sum_bounds(lower_squares)[0]
    upper = fc.positive_sum_bounds(upper_squares)[1]
    return (float(max(0.0, fc._down(np.sqrt(lower)))), float(fc._up(np.sqrt(upper))))


def _well_words(hi, lo, d, cells):
    product, error = fc.two_prod(d[cells], hi[cells])
    tail, tail_error = fc.two_prod(d[cells], lo[cells])
    correction = fc._exact_sum_interval((error, tail, tail_error))
    return product, correction


def _residual_energy(sets, b, d, hi, lo):
    """Enclose b-L*x-D*x and the energy norm of the exact DD pair.

    The largest words cancel through error-free additions before the small
    intervals are recombined. In particular, well cancellation occurs before
    forming a binary64 cell residual, preserving useful DD refinement.
    """
    high, lower, upper = b.copy(), np.zeros_like(b), np.zeros_like(b)
    norms = []
    for first, second, weights in sets:
        difference, errors = fc._edge_dd_words(hi, lo, first, second)
        tails = fc._exact_sum_interval(errors)
        norms.append(fc._norm_interval(fc._add((difference, difference), tails), weights))
        product, error = fc.two_prod(weights, difference)
        correction = fc._add((error, error), fc._scale_positive(tails, weights))
        for cells, sign in ((first, -1), (second, 1)):
            high[cells], rounding = fc.two_sum(high[cells], sign*product)
            tail = correction if sign > 0 else (-correction[1], -correction[0])
            tail = fc._add(tail, (rounding, rounding))
            lower[cells], upper[cells] = fc._add((lower[cells], upper[cells]), tail)
    cells = np.flatnonzero(d)
    product, correction = _well_words(hi, lo, d, cells)
    high[cells], rounding = fc.two_sum(high[cells], -product)
    tail = fc._add((-correction[1], -correction[0]), (rounding, rounding))
    lower[cells], upper[cells] = fc._add((lower[cells], upper[cells]), tail)
    p, rounding = fc.two_sum(hi[cells], lo[cells])
    norms.append(fc._norm_interval(fc._exact_sum_interval((p, rounding)), d[cells]))
    return fc._add((high, high), (lower, upper)), _combine_norms(norms)


def _cell_difference(ph, pl, rh, rl, cells):
    high, error_high = fc.two_sum(ph[cells], -rh[cells])
    low, error_low = fc.two_sum(pl[cells], -rl[cells])
    main, error = fc.two_sum(high, low)
    return fc._add((main, main), fc._exact_sum_interval((error_high, error_low, error)))


def _well_flux_error(hi, lo, rates, d, cells):
    product, correction = _well_words(hi, lo, d, cells)
    difference, error = fc.two_sum(rates[cells], -product)
    return fc._sub(fc._exact_sum_interval((difference, error)), correction)


def _residual_report(residual, b):
    norm = fc._norm_interval(residual)
    bnorm_lower = fc._norm_interval((b, b))[0]
    bmax = float(np.max(np.abs(b)))
    max_abs = float(np.max(np.maximum(np.abs(residual[0]), np.abs(residual[1]))))
    return {'norm_lower': norm[0], 'norm_upper': norm[1], 'max_abs_upper': max_abs,
            'rho_upper': float(fc._up(norm[1]/bnorm_lower)),
            'eta_upper': float(fc._up(max_abs/bmax))}


def make_reference(shape, b, edge, d, A, *, preconditioner=None, initial_pressure=None,
                   relative_target=5e-9, max_refinements=10, inner_maxiter=1600,
                   inner_rtol=1e-8, deadline=None):
    """Return DD drawdown and an independent L+D energy uncertainty bound.

    A supplied pressure and preconditioner may accelerate reference creation.
    Their provenance is recorded, and neither contributes to the error bound.
    """
    if not 0 < relative_target < 1 or not 0 < inner_rtol < 1:
        raise ValueError('Reference tolerances must lie strictly between zero and one')
    fc._ieee_probe()
    shape, b, d, sets = _validate(shape, b, edge, d)
    if A.shape != (b.size, b.size):
        raise ValueError('Reference approximate operator has incorrect shape')
    lower_gap, gap_info = _grounded_gap(shape, sets, d)
    if isinstance(initial_pressure, tuple):
        hi, lo = (np.asarray(v, dtype=np.float64).ravel().copy() for v in initial_pressure)
    else:
        hi = np.zeros_like(b) if initial_pressure is None else np.asarray(initial_pressure, dtype=np.float64).ravel().copy()
        lo = np.zeros_like(b)
    if hi.shape != b.shape or lo.shape != b.shape:
        raise ValueError('Reference initial pressure has incorrect shape')
    hi, lo = gp.pair_normalize(hi, lo)
    M, history = preconditioner, []

    def guard(*_):
        if deadline is not None and perf_counter() >= deadline:
            raise TimeoutError('Reference time budget exhausted')

    with threadpool_limits(1), np.errstate(over='raise', invalid='raise', divide='raise'):
        for iteration in range(max_refinements+1):
            guard()
            residual, norm = _residual_energy(sets, b, d, hi, lo)
            rnorm = fc._norm_interval(residual)[1]
            epsilon = float(fc._up(rnorm/fc._down(np.sqrt(lower_gap))))
            denominator = float(fc._down(norm[0]-epsilon))
            relative = float(fc._up(epsilon/denominator)) if denominator > 0 else None
            history.append({'outer': iteration, 'residual_norm_upper': rnorm,
                            'relative_uncertainty_upper': relative})
            if relative is not None and relative <= relative_target:
                return hi, lo, {
                    'energy_uncertainty_upper': epsilon, 'energy_norm_interval': norm,
                    'relative_uncertainty_upper': relative, 'lambda_min_lower': lower_gap,
                    'spectral_bound': gap_info, 'history': history,
                    'identity_sha256': _identity(shape, b, edge, d, hi, lo),
                    'method': 'grounded coordinate-path bound and outward EFT face/well residual',
                    'pressure_variable': 'drawdown p-p_BHP; no constant projection',
                    'initial_pressure_supplied': initial_pressure is not None,
                    'external_inverse_supplied': preconditioner is not None,
                }
            if iteration == max_refinements:
                break
            target = residual[0]/2+residual[1]/2
            if M is None:
                inverse = 1/np.asarray(A.diagonal()).ravel()
                M = LinearOperator(A.shape, matvec=lambda value: inverse*value, dtype=np.float64)
            delta, status = cg(A, target, M=M, rtol=inner_rtol, atol=0,
                               maxiter=inner_maxiter, callback=guard)
            history[-1]['inner_status'] = int(status)
            if status < 0 or not np.isfinite(delta).all():
                raise RuntimeError('Reference correction failed')
            hi, lo = gp.pair_update(hi, lo, delta)
    raise RuntimeError(f'Uncertified reference after refinement: {history}')


def assess(shape, b, edge, d, p_hi, p_lo, reference_hi, reference_lo, certificate,
           q=None, well_rates=None):
    """Certify pressure energy, combined face/well rates and local balance.

    Stored rates are used when supplied. Reconstruction is explicitly labelled.
    The combined flux norm uses inverse face transmissibilities and inverse
    positive producer coefficients. It is dual to the L+D pressure energy.
    """
    started = perf_counter()
    try:
        with threadpool_limits(1), np.errstate(over='raise', invalid='raise', divide='raise'):
            fc._ieee_probe()
            shape, b, d, sets = _validate(shape, b, edge, d)
            ph, pl, rh, rl = (np.asarray(v, dtype=np.float64).ravel()
                              for v in (p_hi, p_lo, reference_hi, reference_lo))
            if any(v.shape != b.shape or not np.isfinite(v).all() for v in (ph, pl, rh, rl)):
                raise ValueError('Invalid pressure pair')
            if certificate['identity_sha256'] != _identity(shape, b, edge, d, rh, rl):
                raise ValueError('Graph, source, wells or reference changed since certification')
            face_source = 'supplied solver output' if q is not None else 'reconstructed from pressure'
            well_source = 'supplied solver output' if well_rates is not None else 'reconstructed from pressure'
            i, j, t = edge
            q = gp.graph_flux(ph, pl, edge) if q is None else np.asarray(q, dtype=np.float64).ravel()
            cells = np.flatnonzero(d)
            if well_rates is None:
                well_rates = np.zeros_like(b)
                main, correction = _well_words(ph, pl, d, cells)
                well_rates[cells] = main+(correction[0]/2+correction[1]/2)
            else:
                well_rates = np.asarray(well_rates, dtype=np.float64).ravel()
            if q.shape != t.shape or not np.isfinite(q).all():
                raise ValueError('Invalid stored interior flux')
            if well_rates.shape != b.shape or not np.isfinite(well_rates).all() or np.any(well_rates[d==0] != 0):
                raise ValueError('Stored producer rates must be a cell vector, zero off the wells')
            epsilon = float(certificate['energy_uncertainty_upper'])
            normlo, normhi = map(float, certificate['energy_norm_interval'])
            if not (np.isfinite(epsilon) and epsilon >= 0 and 0 <= normlo <= normhi < np.inf):
                raise ValueError('Invalid reference certificate bounds')
            denominator = float(fc._down(normlo-epsilon))
            if denominator <= 0:
                raise ValueError('Reference energy is not separated from zero')
            pressure_edge = fc._pressure_difference_edges(ph, pl, rh, rl, i, j)
            pressure_face_norm = fc._norm_interval(pressure_edge, t)
            del pressure_edge
            pressure_well_norm = fc._norm_interval(_cell_difference(ph, pl, rh, rl, cells), d[cells])
            pressure_norm = _combine_norms((pressure_face_norm, pressure_well_norm))
            _, face_error = fc._ideal_flux_and_actual_error(rh, rl, q, i, j, t)
            face_norm = fc._norm_interval(face_error, t, inverse_weights=True)
            del face_error
            well_error = _well_flux_error(rh, rl, well_rates, d, cells)
            well_norm = fc._norm_interval(well_error, d[cells], inverse_weights=True)
            flux_norm = _combine_norms((face_norm, well_norm))
            pressure_error = (max(0.0, float(fc._down(pressure_norm[0]-epsilon))),
                              float(fc._up(pressure_norm[1]+epsilon)))
            flux_error = (max(0.0, float(fc._down(flux_norm[0]-epsilon))),
                          float(fc._up(flux_norm[1]+epsilon)))
            relative_pressure = float(fc._up(pressure_error[1]/denominator))
            relative_flux = float(fc._up(flux_error[1]/denominator))
            true_residual, _ = _residual_energy(sets, b, d, ph, pl)
            true_report = _residual_report(true_residual, b)
            degree = np.bincount(i, minlength=b.size)+np.bincount(j, minlength=b.size)
            stored_divergence = fc._divergence_interval((q, q), i, j, b.size, degree)
            stored_balance = fc._sub((b, b), fc._add(stored_divergence, (well_rates, well_rates)))
            stored_report = _residual_report(stored_balance, b)
            passed = (relative_pressure <= 1e-6 and relative_flux <= 1e-6
                      and max(true_report['rho_upper'], true_report['eta_upper'],
                              stored_report['rho_upper'], stored_report['eta_upper']) <= 1e-9)
            return {
                'verified': True, 'accuracy_pass': bool(passed),
                'pressure_relative_upper': relative_pressure, 'flux_relative_upper': relative_flux,
                'pressure_energy_error_interval': pressure_error,
                'actual_weighted_flux_error_interval': flux_error,
                'internal_flux_relative_upper': float(fc._up(fc._up(face_norm[1]+epsilon)/denominator)),
                'well_flux_relative_upper': float(fc._up(fc._up(well_norm[1]+epsilon)/denominator)),
                'pressure_true_residual': true_report, 'stored_flux_true_imbalance': stored_report,
                'reference_energy_uncertainty_upper': epsilon,
                'reference_solution_energy_norm_interval': [denominator, float(fc._up(normhi+epsilon))],
                'pressure_metric': 'L+D drawdown energy; interior faces and producer well terms',
                'flux_metric': 'inverse-weighted interior faces and positive producer well rates',
                'source_convention': 'exact serialized injection vector; no projection',
                'flux_source': face_source, 'well_rate_source': well_source,
                'operator': 'exact serialized binary64 face and well coefficients',
                'method': 'independent EFT and per-operation outward binary64 intervals',
                'seconds': perf_counter()-started,
            }
    except (fc.Unverified, FloatingPointError, OverflowError, MemoryError, ValueError, KeyError) as error:
        return {'verified': False, 'accuracy_pass': False, 'reason': str(error),
                'seconds': perf_counter()-started}
