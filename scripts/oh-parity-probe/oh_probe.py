"""Small Oh Newton probes used only for numerical-parity investigation."""

import numpy as np


def probe_float32(theta, hh, vv, hv, iterations):
    """Evaluate the current NumPy-style float32 expression path."""
    theta, hh, vv, hv = _inputs(theta, hh, vv, hv, np.float32)

    with np.errstate(all="ignore"):
        a = 2.0 * theta / np.pi
        b = (hv / vv) / 0.23
        c = np.sqrt(hh / vv) - 1.0
        x = np.full_like(theta, 2.0)

        for _ in range(iterations):
            a_power = np.exp((x * x / 3.0) * np.log(a))
            numerator = a_power * (1.0 - b * x) + c
            denominator = ((2.0 * x / 3.0 * np.log(a) * (1.0 - b * x)) - b) * a_power
            x = x - numerator / denominator

    return np.column_stack((a, b, c, x)).astype(np.float32)


def probe_c_semantics(theta, hh, vv, hv, iterations):
    """Evaluate double libm expressions with C float assignment points."""
    theta, hh, vv, hv = _inputs(theta, hh, vv, hv, np.float32)

    with np.errstate(all="ignore"):
        hh_vv = (hh / vv).astype(np.float32)
        hv_vv = (hv / vv).astype(np.float32)
        a = (2.0 * theta.astype(np.float64) / np.pi).astype(np.float32)
        b = (hv_vv.astype(np.float64) / 0.23).astype(np.float32)
        c = (np.sqrt(hh_vv.astype(np.float64)) - 1.0).astype(np.float32)
        x = np.full_like(theta, 2.0)
        log_a = np.log(a.astype(np.float64))

        for _ in range(iterations):
            x_squared_third = (x * x / np.float32(3.0)).astype(np.float32)
            one_minus_bx = (np.float32(1.0) - b * x).astype(np.float32)
            a_power = np.exp(x_squared_third.astype(np.float64) * log_a)
            numerator = a_power * one_minus_bx.astype(np.float64) + c.astype(np.float64)
            two_x_third = (np.float32(2.0) * x / np.float32(3.0)).astype(np.float32)
            denominator = (
                two_x_third.astype(np.float64) * log_a * one_minus_bx.astype(np.float64)
                - b.astype(np.float64)
            ) * a_power
            x = (x.astype(np.float64) - numerator / denominator).astype(np.float32)

    return np.column_stack((a, b, c, x)).astype(np.float32)


def probe_float64(theta, hh, vv, hv, iterations):
    """Evaluate every operation and stored value in float64."""
    theta, hh, vv, hv = _inputs(theta, hh, vv, hv, np.float64)

    with np.errstate(all="ignore"):
        a = 2.0 * theta / np.pi
        b = (hv / vv) / 0.23
        c = np.sqrt(hh / vv) - 1.0
        x = np.full_like(theta, 2.0)

        for _ in range(iterations):
            a_power = np.exp((x * x / 3.0) * np.log(a))
            numerator = a_power * (1.0 - b * x) + c
            denominator = ((2.0 * x / 3.0 * np.log(a) * (1.0 - b * x)) - b) * a_power
            x = x - numerator / denominator

    return np.column_stack((a, b, c, x))


def _inputs(theta, hh, vv, hv, dtype):
    return tuple(np.asarray(value, dtype=dtype) for value in (theta, hh, vv, hv))
