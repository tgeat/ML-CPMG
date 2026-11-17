"""Double-double pressure arithmetic and binary64 Darcy graph fluxes.

A pressure is a pair of equally shaped binary64 arrays ``(hi, lo)``.  Its
meaning is the REAL sum of the two words, not ``np.asarray(hi + lo)``.  Keep
both words until after forming pressure differences.  Do not replace the pair
by its rounded binary64 sum before computing a flux or a certificate.

``two_sum`` is an error-free transformation (finite intermediates, IEEE-754
round-to-nearest, gradual underflow).  General DD addition and projection must
round to two words; their arithmetic errors are of order u**2, u=2**-53.  They
are not claimed to represent every sum of four binary64 numbers exactly.

All timed operations below are vectorized NumPy binary64 operations.  Decimal
is imported ONLY inside ``pair_to_decimal``, which is an offline conversion
helper and is not needed by a candidate solver.  No numerical state, factors,
reference pressures or geometry-dependent caches are kept in this module.
"""

import numpy as np


_SPLITTER = np.float64(134217729.0)  # 2**27 + 1, for IEEE binary64.


def two_sum(a, b):
    """Return s,e such that s=fl(a+b) and s+e is the exact sum.

    Broadcasting follows NumPy. Inputs and outputs are binary64. No overflow
    or NaN/Inf is permitted; errors below the binary64 subnormal range cannot
    be represented by any binary64 pair.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s = a + b
    b_virtual = s - a
    e = (a - (s - b_virtual)) + (b - b_virtual)
    return s, e


def pair_normalize(hi, lo):
    """Normalize the two words without first rounding their real sum."""
    return two_sum(hi, lo)


def pair_add(a_hi, a_lo, b_hi, b_lo):
    """Add normalized DD pairs, returning a normalized DD pair.

    This is the standard compensated DD addition with general TwoSum used in
    place of QuickTwoSum, so it does not require an operand-magnitude ordering.
    The two unavoidable tail additions are at the low-word scale.
    """
    s, e = two_sum(a_hi, b_hi)
    t, f = two_sum(a_lo, b_lo)
    e = e + t
    s, e = two_sum(s, e)
    e = e + f
    return two_sum(s, e)


def pair_sub(a_hi, a_lo, b_hi, b_lo):
    """Subtract normalized DD pairs without collapsing either input."""
    return pair_add(a_hi, a_lo, -np.asarray(b_hi, dtype=np.float64),
                    -np.asarray(b_lo, dtype=np.float64))


def pair_update(hi, lo, delta, delta_lo=None, *, inplace=False):
    """Accumulate a double or DD correction and return the new pressure pair.

    ``pair_update(hi, lo, correction)`` is the normal mixed-precision iterative
    refinement operation. With ``inplace=True``, hi and lo must be writable
    binary64 arrays; their contents are replaced after all inputs are read.
    """
    if delta_lo is None:
        delta_lo = np.float64(0.0)
    new_hi, new_lo = pair_add(hi, lo, delta, delta_lo)
    if inplace:
        hi[...] = new_hi
        lo[...] = new_lo
        return hi, lo
    return new_hi, new_lo


def pair_sum(hi, lo):
    """Sum all entries by a vectorized balanced DD reduction.

    Work and temporary storage are O(N); the reduction has O(log N) levels.
    A binary64 sum of the high words would lose the accuracy of the gauge.
    """
    h, l = pair_normalize(np.asarray(hi, dtype=np.float64).ravel(),
                          np.asarray(lo, dtype=np.float64).ravel())
    if h.size == 0:
        return np.float64(0.0), np.float64(0.0)
    while h.size > 1:
        pairs = h.size // 2
        next_h, next_l = pair_add(h[:2*pairs:2], l[:2*pairs:2],
                                  h[1:2*pairs:2], l[1:2*pairs:2])
        if h.size % 2:
            next_h = np.concatenate((next_h, h[-1:]))
            next_l = np.concatenate((next_l, l[-1:]))
        h, l = next_h, next_l
    return h[0], l[0]


def _two_product(a, b):
    """Dekker product and product error, for the physical Darcy value range.

    The splitter requires abs(a),abs(b) < max_float/(2**27+1), and the product
    and its error must not overflow/underflow. Darcy pressures, n<=64, and
    contrasts through 1e6 are far inside this range. These are the same usual
    nonexceptional-range assumptions as other error-free product algorithms.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    p = a*b
    a_split = _SPLITTER*a
    a_hi = a_split-(a_split-a)
    a_lo = a-a_hi
    b_split = _SPLITTER*b
    b_hi = b_split-(b_split-b)
    b_lo = b-b_hi
    error = ((a_hi*b_hi-p)+a_hi*b_lo+a_lo*b_hi)+a_lo*b_lo
    return p, error


def _pair_divide_double(hi, lo, divisor):
    """One compensated DD/double division, sufficient for a DD mean."""
    divisor = np.float64(divisor)
    first = np.float64(hi)/divisor
    product_hi, product_lo = _two_product(first, divisor)
    residual_hi, residual_lo = pair_sub(hi, lo, product_hi, product_lo)
    second = (residual_hi+residual_lo)/divisor
    return two_sum(first, second)


def pair_mean(hi, lo):
    """Return the ordinary cell-volume mean as a DD pair."""
    size = np.asarray(hi).size
    if not size:
        raise ValueError('the mean of an empty pressure vector is undefined')
    total_hi, total_lo = pair_sum(hi, lo)
    # On n=16/32/64 cubical grids size is a power of two, so the division
    # scales both words exactly. The compensated general path also handles N
    # that is not a power of two.
    if size & (size-1) == 0:
        exponent = -(size.bit_length()-1)
        return pair_normalize(np.ldexp(total_hi, exponent),
                              np.ldexp(total_lo, exponent))
    return _pair_divide_double(total_hi, total_lo, size)


def pair_project(hi, lo, *, inplace=False):
    """Subtract the DD mean, retaining both words of the gauge shift.

    The represented mean is zero to DD rounding accuracy; an exactly zero
    real mean is not in general representable by a fixed two-word vector.
    """
    mean_hi, mean_lo = pair_mean(hi, lo)
    return pair_update(hi, lo, -mean_hi, -mean_lo, inplace=inplace)


def graph_flux(hi, lo, edge):
    """Return ordinary binary64 integrated fluxes for the represented pressure.

    ``edge`` is precisely (tail, head, t) with the SAME binary64 conductances
    used to define the problem. Each pressure difference is formed in DD, its
    product with t is compensated, and only the final flux is rounded to one
    word. Thus the new pressure-storage error scales as O(u**2*t*|p|), rather
    than the O(u*t*|p|) floor caused by collapsing the pressure first. The
    unavoidable final flux rounding is O(u*|q|).
    """
    hi = np.asarray(hi, dtype=np.float64).ravel()
    lo = np.asarray(lo, dtype=np.float64).ravel()
    i, j, t = edge
    difference_hi, difference_lo = pair_sub(hi[i], lo[i], hi[j], lo[j])
    product_hi, product_lo = _two_product(t, difference_hi)
    correction = product_lo + np.asarray(t, dtype=np.float64)*difference_lo
    return product_hi+correction


def graph_residual(b, hi, lo, edge, *, return_flux=False):
    """Compute b-div(q) using the ordinary-double flux actually returned.

    The divergence uses the same oriented-edge convention as darcy_core. Its
    binary64 accumulation error remains; an independent Decimal certificate
    should sum the actual q entries independently, as well as checking their
    consistency with the exact real pressure hi+lo.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    i, j, _ = edge
    q = graph_flux(hi, lo, edge)
    divergence = (np.bincount(i, weights=q, minlength=b.size)
                  - np.bincount(j, weights=q, minlength=b.size))
    residual = b-divergence
    return (residual, q) if return_flux else residual


def pair_to_decimal(hi, lo, precision=None):
    """OFFLINE conversion to an object array of Decimal pressure values.

    By default the sum of the two finite binary64 words is EXACT: 1500 decimal
    significant digits suffice even for the span from max_float to the least
    subnormal. If precision is supplied, that context intentionally rounds
    the result and must not be mistaken for the exact represented pressure.
    This helper is for inspection/export; independent certification should
    implement its own Decimal.from_float conversion rather than trust a
    candidate arithmetic helper.
    """
    from decimal import Decimal, localcontext
    hi, lo = np.broadcast_arrays(np.asarray(hi, dtype=np.float64),
                                  np.asarray(lo, dtype=np.float64))
    out = np.empty(hi.shape, dtype=object)
    with localcontext() as ctx:
        ctx.prec = 1500 if precision is None else int(precision)
        for index in np.ndindex(hi.shape):
            a, b = float(hi[index]), float(lo[index])
            if not np.isfinite(a) or not np.isfinite(b):
                raise ValueError('DD pressure words must be finite')
            out[index] = Decimal.from_float(a)+Decimal.from_float(b)
    return out


# Short names for callers that use the request's add/update/project wording.
add = pair_add
update = pair_update
project = pair_project
