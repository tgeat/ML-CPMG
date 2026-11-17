"""Single-thread conservative explicit water transport for improved IMPES.

Pore volumes and rates must use consistent volume and volume/day units. Face
flux q is oriented tail -> head and is held fixed during at most M transport
substeps. The pressure solver supplies nonnegative pure-water injection and
total producer outflow vectors. Saturation is never clipped. Outward endpoint
rates bounded by floating-point summation roundoff may be nulled, with their
signed and absolute integrated water volumes recorded in an explicit ledger.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import math
import time

import numpy as np
from numba import njit


SWC = 0.2
SOR = 0.2
SWMAX = 1.0-SOR
MOBILE_RANGE = 1.0-SWC-SOR
MU_W = 0.3
MU_O = 3.0
# The unique maximum of Corey fractional-flow derivative in effective S.
# It solves 22*s**3 - 33*s**2 + 1 = 0. The upward bound 5 is conservative.
FW_DERIVATIVE_PEAK = 0.18598987556027885
FW_DERIVATIVE_UPPER = 5.0
UNIT_ROUNDOFF = 2.0**-53
ENDPOINT_RATE_ROUNDOFF_FACTOR = 128.0*UNIT_ROUNDOFF


@njit(cache=True)
def _properties(s):
    effective = (s-SWC)/MOBILE_RANGE
    water = effective*effective/MU_W
    oil = (1.0-effective)*(1.0-effective)/MU_O
    total = water+oil
    return total, water/total


def phase_properties(saturation):
    """Return total mobility [1/cP] and water fractional flow at each cell."""
    s = np.asarray(saturation, dtype=np.float64)
    if not np.isfinite(s).all() or np.any(s < SWC) or np.any(s > SWMAX):
        raise ValueError('Corey saturation must lie in [0.2, 0.8]')
    effective = (s-SWC)/MOBILE_RANGE
    water = effective*effective/MU_W
    oil = (1.0-effective)*(1.0-effective)/MU_O
    total = water+oil
    return total, water/total


@njit(cache=True)
def _derivative(effective):
    denominator = 11.0*effective*effective-2.0*effective+1.0
    return (20.0/MOBILE_RANGE)*effective*(1.0-effective)/(denominator*denominator)


@njit(cache=True)
def _local_derivative_bound(s, dsmax):
    """Bound f' over the saturation box allowed by the DSmax restriction."""
    low = max(0.0, (s-dsmax-SWC)/MOBILE_RANGE)
    high = min(1.0, (s+dsmax-SWC)/MOBILE_RANGE)
    if low <= FW_DERIVATIVE_PEAK <= high:
        return FW_DERIVATIVE_UPPER
    return np.nextafter(max(_derivative(low), _derivative(high)), math.inf)


@njit(cache=True)
def _sum_product(a, b):
    total = 0.0
    correction = 0.0
    for i in range(a.size):
        value = a[i]*b[i]-correction
        updated = total+value
        correction = (updated-total)-value
        total = updated
    return total


@njit(cache=True, inline='always')
def _add_compensated(values, tails, cell, term):
    """Neumaier addition for a cell's short signed sum of face/source terms."""
    previous = values[cell]
    summed = previous+term
    if abs(previous) >= abs(term):
        tails[cell] += (previous-summed)+term
    else:
        tails[cell] += (term-summed)+previous
    values[cell] = summed


@njit(cache=True)
def _flow_geometry(tail, head, flux, injection, production):
    outflow = production.copy()
    balance = injection-production
    for face in range(flux.size):
        i, j, q = tail[face], head[face], flux[face]
        balance[i] -= q
        balance[j] += q
        if q >= 0.0:
            outflow[i] += q
        else:
            outflow[j] -= q
    return outflow, balance


@njit(cache=True)
def _advance(s, tail, head, flux, pore_volume, injection, production,
             outflow, t_start, t_limit, steps, dsmax, cfl):
    fractional = np.empty_like(s)
    rate = np.empty_like(s)
    rate_tail = np.empty_like(s)
    absolute_terms = np.empty_like(s)
    clock = t_start
    injected_water = 0.0
    produced_water = 0.0
    injection_rate = np.sum(injection)
    dt_min, dt_max = math.inf, 0.0
    change_max = 0.0
    nstep = 0
    cfl_limited = 0
    ds_limited = 0
    bound_limited = 0
    stagnant_interval = False
    repair_count = 0
    repair_signed_volume = 0.0
    repair_absolute_volume = 0.0
    repair_max_rate = 0.0
    while nstep < steps and clock < t_limit:
        for cell in range(s.size):
            _, fractional[cell] = _properties(s[cell])
            rate[cell] = injection[cell]
            rate_tail[cell] = 0.0
            produced = fractional[cell]*production[cell]
            absolute_terms[cell] = injection[cell]+produced
            _add_compensated(rate, rate_tail, cell, -produced)
        water_production_rate = _sum_product(fractional, production)
        for face in range(flux.size):
            i, j, q = tail[face], head[face], flux[face]
            water_flux = q*(fractional[i] if q >= 0.0 else fractional[j])
            _add_compensated(rate, rate_tail, i, -water_flux)
            _add_compensated(rate, rate_tail, j, water_flux)
            absolute_terms[i] += abs(water_flux)
            absolute_terms[j] += abs(water_flux)
        max_rate = 0.0
        dt_cfl, dt_bound = math.inf, math.inf
        step_repair_signed_rate = 0.0
        step_repair_absolute_rate = 0.0
        for cell in range(s.size):
            volume_rate = rate[cell]+rate_tail[cell]
            at_outward_endpoint = ((s[cell] == SWMAX and volume_rate > 0.0)
                                   or (s[cell] == SWC and volume_rate < 0.0))
            if at_outward_endpoint:
                tolerance = ENDPOINT_RATE_ROUNDOFF_FACTOR*absolute_terms[cell]
                if abs(volume_rate) > tolerance:
                    raise ValueError('Substantive outward endpoint rate; check pressure flux balance')
                # Null only a numerically zero outward derivative. Its precise
                # change to the water update is added to the returned ledger.
                step_repair_signed_rate -= volume_rate
                step_repair_absolute_rate += abs(volume_rate)
                repair_max_rate = max(repair_max_rate, abs(volume_rate))
                repair_count += 1
                volume_rate = 0.0
            rate[cell] = volume_rate/pore_volume[cell]
            value = rate[cell]
            max_rate = max(max_rate, abs(value))
            if value > 0.0:
                margin = SWMAX-s[cell]
            elif value < 0.0:
                margin = s[cell]-SWC
            else:
                margin = math.inf
            if value != 0.0:
                if margin <= 0.0:
                    raise ValueError('Outward transport derivative at a saturation endpoint; check pressure flux balance')
                dt_bound = min(dt_bound, 0.99*margin/abs(value))
            if outflow[cell] > 0.0:
                derivative = _local_derivative_bound(s[cell], dsmax)
                if derivative > 0.0:
                    dt_cfl = min(dt_cfl, cfl*pore_volume[cell]/(outflow[cell]*derivative))
        remaining = t_limit-clock
        if max_rate == 0.0:
            # Exact fixed state. Rates still contribute to cumulative well totals.
            dt = remaining
            stagnant_interval = True
        else:
            dt_ds = np.nextafter(dsmax/max_rate, 0.0)
            dt = min(remaining, dt_ds, dt_cfl, dt_bound)
            if dt == dt_ds:
                ds_limited += 1
            elif dt == dt_cfl:
                cfl_limited += 1
            elif dt == dt_bound:
                bound_limited += 1
        if not math.isfinite(dt) or dt <= 0.0 or clock+dt <= clock:
            raise ValueError('No positive representable transport step; check saturation and pressure balance')
        actual_change = 0.0
        for cell in range(s.size):
            updated = s[cell]+dt*rate[cell]
            if not math.isfinite(updated) or updated < SWC or updated > SWMAX:
                raise ValueError('Explicit transport left physical saturation interval')
            actual_change = max(actual_change, abs(updated-s[cell]))
            s[cell] = updated
        if actual_change > dsmax+4.0*UNIT_ROUNDOFF*SWMAX:
            raise ValueError('Transport DSmax restriction exceeded')
        change_max = max(change_max, actual_change)
        injected_water += dt*injection_rate
        produced_water += dt*water_production_rate
        repair_signed_volume += dt*step_repair_signed_rate
        repair_absolute_volume += dt*step_repair_absolute_rate
        dt_min = min(dt_min, dt)
        dt_max = max(dt_max, dt)
        clock = t_limit if dt == remaining else clock+dt
        nstep += 1
    if nstep == 0:
        dt_min = 0.0
    return (clock, nstep, injected_water, produced_water, dt_min, dt_max,
            change_max, cfl_limited, ds_limited, bound_limited, stagnant_interval,
            repair_count, repair_signed_volume, repair_absolute_volume, repair_max_rate)


def transport(S, edge, q, pv, injection_vector, production_vector, *,
              t_start, t_limit, M=50, dsmax=0.001, cfl=0.95):
    """Advance at most M adaptive Euler substeps with a fixed total velocity.

    Time is absolute physical days. The final substep is shortened to t_limit.
    Per-cell water mass changes by inflow minus outflow and well source terms.
    Adaptive steps respect DSmax, the saturation interval, and a monotonicity
    restriction based on the maximum Corey slope over S +/- DSmax. Saturation
    is copied, so a failed call never partially changes the caller's state.
    At exactly .2 or .8, an outward volume rate is nulled only when its absolute
    value is at most 128*u times the sum of absolute water-rate terms for that
    cell, with u=2**-53. The repair is reported as an integrated water-volume
    ledger. Larger outward endpoint rates raise a pressure-balance error.

    Returned differences and water balances use floating-point summation.
    They are diagnostic values, not interval-certified error bounds.
    """
    started = time.perf_counter()
    original = np.asarray(S, dtype=np.float64)
    s = np.ascontiguousarray(original.ravel()).copy()
    tail, head, weights = (np.asarray(a) for a in edge)
    flux = np.ascontiguousarray(q, dtype=np.float64).ravel()
    pore_volume, injection, production = [np.ascontiguousarray(a, dtype=np.float64).ravel()
                                        for a in (pv, injection_vector, production_vector)]
    if (s.size == 0 or not np.isfinite(s).all() or np.any(s < SWC) or np.any(s > SWMAX)
            or any(a.shape != s.shape or not np.isfinite(a).all()
                   for a in (pore_volume, injection, production))
            or np.any(pore_volume <= 0.0) or np.any(injection < 0.0) or np.any(production < 0.0)):
        raise ValueError('Invalid saturation, pore volumes, or nonnegative well rates')
    if (tail.ndim != 1 or head.shape != tail.shape or weights.shape != tail.shape
            or flux.shape != tail.shape or tail.dtype.kind not in 'iu' or head.dtype.kind not in 'iu'
            or not np.isfinite(flux).all() or not np.isfinite(weights).all() or np.any(weights <= 0.0)
            or (tail.size and (min(tail.min(), head.min()) < 0 or max(tail.max(), head.max()) >= s.size))):
        raise ValueError('Invalid oriented interior face graph or total face flux')
    if (type(M) is not int or M < 1 or not np.isfinite(dsmax) or not 0.0 < dsmax <= MOBILE_RANGE
            or not np.isfinite(cfl) or not 0.0 < cfl < 1.0
            or not np.isfinite(t_start) or not np.isfinite(t_limit) or t_limit < t_start):
        raise ValueError('Invalid transport horizon, substep count, DSmax, or CFL')
    tail = np.ascontiguousarray(tail)
    head = np.ascontiguousarray(head)
    outflow, balance = _flow_geometry(tail, head, flux, injection, production)
    water_before = _sum_product(s, pore_volume)
    outcome = _advance(s, tail, head, flux, pore_volume, injection, production, outflow,
                       float(t_start), float(t_limit), M, float(dsmax), float(cfl))
    (clock, nstep, water_in, water_out, dt_min, dt_max, delta_max,
     cfl_limited, ds_limited, bound_limited, stagnant,
     repair_count, repair_signed_volume, repair_absolute_volume, repair_max_rate) = outcome
    water_after = _sum_product(s, pore_volume)
    mass_error = (water_after-water_before)-(water_in-water_out)
    ledger_mass_error = mass_error-repair_signed_volume
    mass_scale = max(abs(water_after-water_before), abs(water_in), abs(water_out),
                     np.finfo(np.float64).tiny)
    total_rate_scale = max(float(np.max(injection)), float(np.max(production)),
                           float(np.max(outflow)), np.finfo(np.float64).tiny)
    metrics = {
        'substeps': int(nstep), 'time_start_days': float(t_start), 'time_end_days': float(clock),
        'elapsed_days': float(clock-t_start), 'dt_min_days': float(dt_min), 'dt_max_days': float(dt_max),
        'max_saturation_change': float(delta_max), 'min_saturation': float(s.min()),
        'max_saturation': float(s.max()), 'water_before': float(water_before),
        'water_after': float(water_after), 'water_injected': float(water_in),
        'water_produced': float(water_out), 'water_balance_error': float(mass_error),
        'water_balance_relative': float(abs(mass_error)/mass_scale),
        'roundoff_repair_cell_substeps': int(repair_count),
        'roundoff_repair_signed_volume': float(repair_signed_volume),
        'roundoff_repair_abs_volume': float(repair_absolute_volume),
        'roundoff_repair_max_abs_rate': float(repair_max_rate),
        'roundoff_rate_threshold_factor': float(ENDPOINT_RATE_ROUNDOFF_FACTOR),
        'water_balance_error_after_roundoff_ledger': float(ledger_mass_error),
        'water_balance_relative_after_roundoff_ledger': float(abs(ledger_mass_error)/mass_scale),
        'total_balance_linf': float(np.max(np.abs(balance))),
        'total_balance_relative': float(np.max(np.abs(balance))/total_rate_scale),
        'cfl_limited_steps': int(cfl_limited), 'dsmax_limited_steps': int(ds_limited),
        'invariant_limited_steps': int(bound_limited), 'stationary_interval': bool(stagnant),
        'dsmax': float(dsmax), 'cfl_safety': float(cfl), 'seconds': time.perf_counter()-started,
        'dsmax_roundoff_allowance': float(4.0*UNIT_ROUNDOFF*SWMAX),
        'saturation_clipped': False, 'transport_threads': 1,
    }
    return {'S': s.reshape(original.shape), 'elapsed_days': float(clock-t_start),
            'time_days': float(clock), 'metrics': metrics}


def warmup():
    """Compile the same serial kernels on a two-cell example, for charged setup."""
    edge = (np.array([0], dtype=np.int32), np.array([1], dtype=np.int32), np.ones(1))
    transport(np.full(2, SWC), edge, np.ones(1), np.ones(2), np.array([1., 0.]),
              np.array([0., 1.]), t_start=0., t_limit=0.001, M=1)
