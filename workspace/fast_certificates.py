"""Independent binary64 interval certificates for an exact stored Darcy graph.

The input edge weights are exact serialized binary64 numbers.  A DD pressure
means the exact real sum hi+lo, without a nonoverlap assumption.  The reference
must have an external rigorous energy-error upper bound.  This module never
imports a solver or a reference-generation/graph-precision implementation.

Assumptions: IEEE-754 binary64, round-to-nearest basic NumPy operations,
gradual underflow, correctly rounded sqrt, no overflow.  EFT operands are
restricted to zero or [2**-400, 2**400] in magnitude: splitting stays below
2**428 and every nonzero partial-product quantum is at least 2**-904.
Thus the EFTs cannot overflow or underflow.  Interval arithmetic separately
handles subnormal results by rounding EVERY elementary operation outward.

Positive summation allows ANY addition tree with at most n-1 additions:
 |fl(sum)-sum| <= gamma_n*sum + delta_n,
 gamma_n=n*u/(1-n*u), delta_n=n*minsubnormal/(1-n*u).
Both constants are rounded upward; all subsequent bound arithmetic is also
directed.  There is no empirical tolerance or heuristic residual certificate.
"""

import time
import numpy as np

_U = 2.0**-53
_ETA = np.nextafter(np.float64(0), np.float64(1))
_SAFE_MIN, _SAFE_MAX = 2.0**-400, 2.0**400
_SPLITTER = 134217729.0
_SOURCE_MEAN_CACHE = {}


class Unverified(ValueError):
    pass


def _ieee_probe():
    """Reject common rounding-mode and FTZ/DAZ violations of the contract."""
    one = np.float64(1.0)
    tiny = np.float64(np.finfo(np.float64).tiny)
    if not (np.add(one, np.float64(_U)) == one
            and np.add(one, np.float64(3*_U)) == np.float64(1.0+4*_U)
            and np.multiply(tiny, np.float64(0.5)).view(np.uint64) == np.uint64(0x0008000000000000)
            and np.add(_ETA, _ETA).view(np.uint64) == np.uint64(2)):
        raise Unverified('Runtime binary64 nearest-even/gradual-underflow probe failed')


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _finite(x, name='interval'):
    if not np.all(np.isfinite(x)):
        raise Unverified(name + ' is nonfinite or overflows')


def _eft_operands(a, b):
    for value in (a, b):
        av = np.abs(value)
        if not np.all(np.isfinite(av)) or np.any((av != 0) & ((av < _SAFE_MIN) | (av > _SAFE_MAX))):
            raise Unverified('EFT operand outside proven normal/overflow-safe exponent window')


def two_sum(a, b):
    """Exact real a+b = s+e under the documented safe exponent window."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    _eft_operands(a, b)
    s = a+b
    bb = s-a
    aa = s-bb
    br = b-bb
    ar = a-aa
    e = ar+br
    return s, e


def two_prod(a, b):
    """Dekker exact product a*b=p+e; independent of platform FMA support."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    _eft_operands(a, b)
    ca, cb = _SPLITTER*a, _SPLITTER*b
    ahi, bhi = ca-(ca-a), cb-(cb-b)
    alo, blo = a-ahi, b-bhi
    p = a*b
    # Keep the standard Dekker sequence as separate NumPy operations.
    e1 = p-ahi*bhi
    e2 = e1-alo*bhi
    e3 = e2-ahi*blo
    e = alo*blo-e3
    return p, e


def _add(a, b):
    return _down(a[0]+b[0]), _up(a[1]+b[1])


def _sub(a, b):
    return _down(a[0]-b[1]), _up(a[1]-b[0])


def _scale_positive(a, scale):
    return _down(a[0]*scale), _up(a[1]*scale)


def _exact_sum_interval(words):
    lo = np.asarray(words[0], dtype=np.float64)
    hi = lo
    for word in words[1:]:
        lo, hi = _down(lo+word), _up(hi+word)
    return lo, hi


def _gamma_delta(n):
    if n < 0 or n >= 2**52:
        raise Unverified('Unsupported summation length')
    nu = float(n)*_U                 # exact for this range
    denominator = _down(1.0-nu)
    if denominator <= 0:
        raise Unverified('Summation bound denominator nonpositive')
    gamma = float(_up(nu/denominator))
    delta = float(_up(_up(float(n)*_ETA)/denominator))
    return gamma, delta


def positive_sum_bounds(values):
    """Rigorous bounds on the exact sum of nonnegative binary64 entries."""
    values = np.asarray(values, dtype=np.float64)
    _finite(values, 'nonnegative summands')
    if np.any(values < 0):
        raise Unverified('Expected nonnegative summands')
    if not values.size:
        return 0.0, 0.0
    s = float(np.sum(values, dtype=np.float64))
    _finite(s, 'positive sum')
    gamma, delta = _gamma_delta(values.size)
    lower = _down(_down(s-delta)/_up(1.0+gamma))
    upper = _up(_up(s+delta)/_down(1.0-gamma))
    return float(max(0.0, lower)), float(upper)


def _norm_interval(interval, weights=None, inverse_weights=False):
    lo, hi = interval
    if np.any(lo > hi):
        raise Unverified('Reversed interval')
    abslo = np.where(lo > 0, lo, np.where(hi < 0, -hi, 0.0))
    abshi = np.maximum(np.abs(lo), np.abs(hi))
    terms_lo = np.maximum(0.0, _down(abslo*abslo))
    terms_hi = _up(abshi*abshi)
    if weights is not None:
        if inverse_weights:
            terms_lo = np.maximum(0.0, _down(terms_lo/weights))
            terms_hi = _up(terms_hi/weights)
        else:
            terms_lo = np.maximum(0.0, _down(terms_lo*weights))
            terms_hi = _up(terms_hi*weights)
    sumlo = positive_sum_bounds(terms_lo)[0]
    sumhi = positive_sum_bounds(terms_hi)[1]
    return (float(max(0.0, _down(np.sqrt(sumlo)))),
            float(_up(np.sqrt(sumhi))))


def _edge_dd_words(hi, lo, i, j):
    sh, eh = two_sum(hi[i], -hi[j])
    sl, el = two_sum(lo[i], -lo[j])
    main, em = two_sum(sh, sl)
    return main, (eh, el, em)


def _ideal_flux_and_actual_error(hi, lo, q, i, j, t):
    main, errors = _edge_dd_words(hi, lo, i, j)
    product, product_error = two_prod(t, main)
    small = _exact_sum_interval(errors)
    small = _scale_positive(small, t)
    correction = _add((product_error, product_error), small)
    ideal_flux = _add((product, product), correction)
    # Crucial cancellation is done by exact TwoSum, not subtraction of two
    # wide flux intervals: actual-error bounds can be much smaller than ulp(q).
    d, e = two_sum(q, -product)
    actual_error = _sub(_exact_sum_interval((d, e)), correction)
    return ideal_flux, actual_error


def _pressure_difference_edges(hi, lo, refhi, reflo, i, j):
    sh, eh = two_sum(hi, -refhi)
    sl, el = two_sum(lo, -reflo)
    main, em = two_sum(sh, sl)
    # Exact cell difference = main + eh + el + em.  All three representation
    # errors are retained BEFORE taking neighboring-cell differences.
    remainder = _exact_sum_interval((eh, el, em))
    edge_main, edge_error = two_sum(main[i], -main[j])
    edge_remainder = _sub((remainder[0][i], remainder[1][i]),
                          (remainder[0][j], remainder[1][j]))
    edge_remainder = _add(edge_remainder, (edge_error, edge_error))
    return _add((edge_main, edge_main), edge_remainder)


def _divergence_interval(qinterval, i, j, N, degree):
    lo, hi = qinterval
    _finite(lo, 'flux interval lower')
    _finite(hi, 'flux interval upper')
    absq = np.maximum(np.abs(lo), np.abs(hi))
    # Each bincount cell sum has at most degree additions.  The final
    # subtraction adds one; degree+2 is a conservative operation count.
    gamma, delta = _gamma_delta(int(degree.max(initial=0))+2)
    abssum = (np.bincount(i, weights=absq, minlength=N)
              + np.bincount(j, weights=absq, minlength=N))
    absupper = _up(_up(abssum+delta)/_down(1.0-gamma))
    rounding = _up(_up(gamma*absupper)+delta)
    lower = (np.bincount(i, weights=lo, minlength=N)
             - np.bincount(j, weights=hi, minlength=N))
    upper = (np.bincount(i, weights=hi, minlength=N)
             - np.bincount(j, weights=lo, minlength=N))
    return _down(lower-rounding), _up(upper+rounding)


def _source_projection(b):
    """Enclose exact P b, including an exact rational mean computation.

    Repeated physical source values are counted first, so the balanced formal
    sources require just three integer terms.  Generic binary64 sources use
    the same exact path, with no summation tolerance or floating mean.
    """
    key = b.tobytes()
    mean_interval = _SOURCE_MEAN_CACHE.get(key)
    if mean_interval is None:
        values, counts = np.unique(b, return_counts=True)
        rational = [float(v).as_integer_ratio() for v in values]
        shifts = [den.bit_length()-1 for _, den in rational]
        common_shift = max(shifts, default=0)
        numerator = sum(int(count)*num*(1 << (common_shift-shift))
                        for (num, _), shift, count in zip(rational, shifts, counts))
        denominator = (1 << common_shift)*b.size
        mean = float(numerator/denominator)
        _finite(mean, 'source exact mean enclosure')
        num_float, den_float = mean.as_integer_ratio()
        comparison = num_float*denominator-numerator*den_float
        if comparison == 0:
            mean_interval = (mean, mean)
        elif comparison < 0:
            mean_interval = (mean, float(_up(mean)))
        else:
            mean_interval = (float(_down(mean)), mean)
        if len(_SOURCE_MEAN_CACHE) >= 4:
            _SOURCE_MEAN_CACHE.pop(next(iter(_SOURCE_MEAN_CACHE)))
        _SOURCE_MEAN_CACHE[key] = mean_interval
    if mean_interval == (0.0, 0.0):
        return (b, b), mean_interval
    return _sub((b, b), mean_interval), mean_interval


def _residual_report(flux_interval, source_interval, i, j, degree, bnorm_lower, bmax):
    div = _divergence_interval(flux_interval, i, j, source_interval[0].size, degree)
    residual = _sub(source_interval, div)
    norm = _norm_interval(residual)
    absupper = np.maximum(np.abs(residual[0]), np.abs(residual[1]))
    maximum = float(np.max(absupper, initial=0.0))
    rho = float(_up(norm[1]/bnorm_lower)) if bnorm_lower > 0 else (0.0 if norm[1] == 0 else np.inf)
    eta = float(_up(maximum/bmax)) if bmax > 0 else (0.0 if maximum == 0 else np.inf)
    return {'norm_lower': norm[0], 'norm_upper': norm[1], 'max_abs_upper': maximum,
            'rho_upper': rho, 'eta_upper': eta}


def certify(k, b, edge, pressure_hi, pressure_lo, q,
            reference_hi, reference_lo, reference_energy_uncertainty,
            reference_norm_lower, reference_norm_upper):
    """Return rigorous intervals, or verified=False with a concrete reason.

    reference_energy_uncertainty is an externally certified upper bound on
    ||reference_DD - p_exact||_A.  The supplied reference norm bounds enclose
    ||reference_DD||_A; they are enlarged by that uncertainty for denominators.
    Bounds already enclosing ||p_exact||_A are also safe (slightly enlarged).
    q is the actual stored face flux, not a recomputed pressure gradient.
    The singular Neumann equation uses exact-mean projected source P b; its
    exact rational mean is enclosed independently, including nonzero MMS means.

    This verifies mathematical intervals under the module's IEEE assumptions;
    accuracy_pass additionally requires both pressure true-residual gates,
    stored-flux imbalance gates, and both relative-error gates.
    """
    started = time.perf_counter()
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            _ieee_probe()
            k = np.asarray(k, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64).ravel()
            ph, pl, rh, rl = [np.asarray(v, dtype=np.float64).ravel()
                              for v in (pressure_hi, pressure_lo, reference_hi, reference_lo)]
            i, j = np.asarray(edge[0], dtype=np.int64), np.asarray(edge[1], dtype=np.int64)
            t, q = np.asarray(edge[2], dtype=np.float64), np.asarray(q, dtype=np.float64)
            N = b.size
            if k.size != N or any(v.size != N for v in (ph, pl, rh, rl)):
                raise Unverified('Cell array lengths disagree')
            if any(v.ndim != 1 for v in (i, j, t, q)) or not (len(i) == len(j) == len(t) == len(q)):
                raise Unverified('Edge array dimensions disagree')
            if N == 0 or np.any(i < 0) or np.any(j < 0) or np.any(i >= N) or np.any(j >= N):
                raise Unverified('Invalid graph endpoint')
            for name, value in [('k', k), ('b', b), ('pressure_hi', ph), ('pressure_lo', pl),
                                ('reference_hi', rh), ('reference_lo', rl), ('t', t), ('q', q)]:
                _finite(value, name)
            if np.any(k <= 0) or np.any(t <= 0):
                raise Unverified('Nonpositive conductivity or graph weight')
            degree = np.bincount(i, minlength=N)+np.bincount(j, minlength=N)
            if np.max(degree) > 6:
                raise Unverified('Expected Darcy graph with maximum degree six')
            epsilon = float(reference_energy_uncertainty)
            reflo, refhi = float(reference_norm_lower), float(reference_norm_upper)
            if not (np.isfinite(epsilon) and epsilon >= 0 and np.isfinite(reflo)
                    and np.isfinite(refhi) and 0 <= reflo <= refhi):
                raise Unverified('Invalid external reference certificate')
            denominator_lower = float(max(0.0, _down(reflo-epsilon)))
            denominator_upper = float(_up(refhi+epsilon))
            if denominator_lower <= 0:
                raise Unverified('Reference norm lower bound does not separate from zero')
            pressure_edge = _pressure_difference_edges(ph, pl, rh, rl, i, j)
            pressure_to_reference = _norm_interval(pressure_edge, weights=t)
            del pressure_edge
            _, flux_difference = _ideal_flux_and_actual_error(rh, rl, q, i, j, t)
            flux_to_reference = _norm_interval(flux_difference, weights=t, inverse_weights=True)
            del flux_difference
            pressure_error = (float(max(0.0, _down(pressure_to_reference[0]-epsilon))),
                              float(_up(pressure_to_reference[1]+epsilon)))
            flux_error = (float(max(0.0, _down(flux_to_reference[0]-epsilon))),
                          float(_up(flux_to_reference[1]+epsilon)))
            pressure_relative = (float(max(0.0, _down(pressure_error[0]/denominator_upper))),
                                 float(_up(pressure_error[1]/denominator_lower)))
            flux_relative = (float(max(0.0, _down(flux_error[0]/denominator_upper))),
                             float(_up(flux_error[1]/denominator_lower)))
            source, mean_interval = _source_projection(b)
            bnorm = _norm_interval(source)
            source_abs_lower = np.where(source[0] > 0, source[0],
                                       np.where(source[1] < 0, -source[1], 0.0))
            bmax = float(np.max(source_abs_lower))
            ideal_pressure_flux, _ = _ideal_flux_and_actual_error(ph, pl, q, i, j, t)
            pressure_residual = _residual_report(ideal_pressure_flux, source, i, j, degree, bnorm[0], bmax)
            del ideal_pressure_flux
            stored_flux_imbalance = _residual_report((q, q), source, i, j, degree, bnorm[0], bmax)
            for value in (pressure_error, flux_error, pressure_relative, flux_relative,
                          denominator_lower, denominator_upper):
                _finite(value, 'final certificate')
            passes = (pressure_relative[1] <= 1e-6 and flux_relative[1] <= 1e-6
                      and pressure_residual['rho_upper'] <= 1e-9
                      and pressure_residual['eta_upper'] <= 1e-9
                      and stored_flux_imbalance['rho_upper'] <= 1e-9
                      and stored_flux_imbalance['eta_upper'] <= 1e-9)
            return {'verified': True, 'accuracy_pass': bool(passes),
                    'pressure_energy_error_interval': pressure_error,
                    'actual_weighted_flux_error_interval': flux_error,
                    'pressure_relative_interval': pressure_relative,
                    'flux_relative_interval': flux_relative,
                    'pressure_relative_upper': pressure_relative[1],
                    'flux_relative_upper': flux_relative[1],
                    'reference_solution_energy_norm_interval': (denominator_lower, denominator_upper),
                    'reference_energy_uncertainty_upper': epsilon,
                    'pressure_true_residual': pressure_residual,
                    'stored_flux_true_imbalance': stored_flux_imbalance,
                    'source_exact_mean_interval': mean_interval,
                    'source_convention': 'exact-mean projected P b',
                    'pressure_metric': 'energy seminorm; common pressure gauge checked separately by runner',
                    'seconds': time.perf_counter()-started,
                    'operator': 'exact serialized binary64 edge weights',
                    'method': 'independent EFT + per-operation outward binary64 intervals',
                    'assumptions': 'IEEE binary64 nearest; gradual underflow; correctly rounded sqrt'}
    except (Unverified, FloatingPointError, OverflowError, MemoryError, ValueError) as exc:
        return {'verified': False, 'accuracy_pass': False, 'reason': str(exc),
                'seconds': time.perf_counter()-started}


certify_solution = certify
