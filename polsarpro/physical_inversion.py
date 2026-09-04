"""
This code is part of the Python PolSARpro software:

"A re-implementation of selected PolSARPro functions in Python,
following the scientific recommendations of PolInSAR 2021"

developed within an ESA funded project with SATIM.

Author: Olivier D'Hondt, 2026.
Scientific advisors: Armando Marino and Eric Pottier.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-----

# Description: module containing physical inversion functions

"""

import dask.array as da
import numpy as np
import xarray as xr

from polsarpro.auxil import validate_dataset
from polsarpro.util import C4_to_C3, S_to_C3, T3_to_C3, T4_to_C3


def dubois_surface_inversion(
    input_data: xr.Dataset,
    incidence_angle: xr.DataArray,
    freq_ghz: float,
    thresh1: float,
    thresh2: float,
    calibration_coeff: float | None = None,
) -> xr.Dataset:
    """Run the Dubois surface inversion on a PolSAR covariance dataset.

    The function converts the input dataset to C3 when it uses another
    supported polarimetric representation, then applies the Dubois empirical
    model using the supplied incidence-angle raster.

    Args:
        input_data (xr.Dataset): Input polarimetric dataset. Supported
            products are:

            - "S": Sinclair scattering matrix

            - "C3": Lexicographic covariance matrix (3x3)

            - "T3": Pauli coherency matrix (3x3)

            - "C4": 4x4 covariance matrix

            - "T4": 4x4 coherency matrix

            The dataset must share the same spatial grid as
            ``incidence_angle``.
        incidence_angle (xr.DataArray): Incidence angle raster in radians.
            Values must be numeric and in the range ``(0, pi/2)``.
        freq_ghz (float): Radar center frequency in GHz.
        thresh1 (float): Maximum allowed ``HV / VV`` ratio in dB for the
            Dubois validity mask.
        thresh2 (float): Maximum allowed ``HH / VV`` ratio in dB for the
            Dubois validity mask.
        calibration_coeff (float or None, optional): Optional multiplicative
            calibration coefficient. When provided, the HH, VV, and HV
            channels are scaled by ``sin(theta) / calibration_coeff`` before
            the inversion is applied. Defaults to None.

    Returns:
        xr.Dataset: Dataset containing the Dubois estimates and masks:
            ``dubois_ks`` (surface roughness parameter), ``dubois_er``
            (relative dielectric constant), ``dubois_mv`` (volumetric moisture
            estimate), ``dubois_mask_in`` (input validity mask),
            ``dubois_mask_out`` (model validity mask), and
            ``dubois_mask_valid_in_out`` (combined mask).

    Notes:
        The returned dataset preserves the input coordinates and uses the
        input spatial dimensions. Output variables are stored as ``float32``
        arrays.
    """
    if not isinstance(incidence_angle, xr.DataArray):
        raise TypeError("incidence_angle must be an xarray.DataArray.")
    if not np.issubdtype(incidence_angle.dtype, np.number):
        raise TypeError("incidence_angle must contain numeric values.")

    if not isinstance(freq_ghz, (int, float, np.number)):
        raise TypeError(f"freq_ghz must be a number, got {type(freq_ghz).__name__}.")
    if freq_ghz <= 0:
        raise ValueError(f"freg_ghz must be strictly positive, got {freq_ghz}.")

    if not isinstance(thresh1, (int, float, np.number)):
        raise TypeError(f"thresh1 must be a number, got {type(thresh1).__name__}.")
    if not isinstance(thresh2, (int, float, np.number)):
        raise TypeError(f"thresh2 must be a number, got {type(thresh2).__name__}.")

    if calibration_coeff is not None:
        if not isinstance(calibration_coeff, (int, float, np.number)):
            raise TypeError(
                "calibration_coeff must be a number or None, "
                f"got {type(calibration_coeff).__name__}."
            )
        if calibration_coeff <= 0:
            raise ValueError(
                f"calibration_coeff must be strictly positive, got {calibration_coeff}."
            )

    allowed_poltypes = ("S", "C3", "T3", "C4", "T4")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    converters = {
        "C3": lambda ds: ds,
        "T3": T3_to_C3,
        "C4": C4_to_C3,
        "T4": T4_to_C3,
        "S": S_to_C3,
    }
    C3 = converters[poltype](input_data)

    out = _apply_dubois_inversion(
        theta=incidence_angle,
        f0=freq_ghz,
        hh=C3.m11,
        vv=C3.m33,
        hv=C3.m22 / 2.0,
        calib=calibration_coeff,
        thresh1=thresh1,
        thresh2=thresh2,
    )
    original_non_nan_mask = input_data.to_array().notnull().all("variable")
    out["dubois_mask_valid_in_out"] = (
        out["dubois_mask_in"]
        * out["dubois_mask_out"]
        * original_non_nan_mask.astype(np.float32, copy=False)
    ).astype(np.float32, copy=False)

    return xr.Dataset(
        {k: (tuple(input_data.dims), v.data) for k, v in out.items()},
        attrs={
            "poltype": "dubois_surface_inversion",
            "description": "Results of the Dubois surface inversion.",
        },
        coords=input_data.coords,
    )


def oh_surface_inversion(
    input_data: xr.Dataset,
    incidence_angle: xr.DataArray,
    thresh1: float,
    thresh2: float,
    c_semantics: bool = False,
) -> xr.Dataset:
    """Run the legacy Oh surface inversion on a PolSAR covariance dataset.

    The function converts the input dataset to C3 when it uses another
    supported polarimetric representation, then applies the original Oh
    empirical model using the supplied incidence-angle raster.

    Args:
        input_data (xr.Dataset): Input polarimetric dataset. Supported
            products are:

            - "S": Sinclair scattering matrix

            - "C3": Lexicographic covariance matrix (3x3)

            - "T3": Pauli coherency matrix (3x3)

            - "C4": 4x4 covariance matrix

            - "T4": 4x4 coherency matrix

            The dataset must share the same spatial grid as
            ``incidence_angle``.
        incidence_angle (xr.DataArray): Incidence angle raster in radians.
            Values must be numeric and in the range ``(0, pi/2)``.
        thresh1 (float): Maximum allowed ``HV / VV`` ratio in dB for the Oh
            validity mask.
        thresh2 (float): Maximum allowed ``HH / VV`` ratio in dB for the Oh
            validity mask.
        c_semantics (bool, optional): Use C-style intermediate precision,
            float32 assignment points, and comparison-based validity masks
            for numerical parity experiments. Defaults to False.

    Returns:
        xr.Dataset: Dataset containing the Oh estimates and masks:
            ``oh_ks`` (normalized surface roughness), ``oh_er`` (relative
            dielectric constant), ``oh_mv`` (volumetric moisture estimate),
            ``oh_mask_in`` (input validity mask), ``oh_mask_out`` (model
            validity mask), and ``oh_mask_valid_in_out`` (combined mask).

    Notes:
        The PolSARpro C command-line help reverses the descriptions of
        ``thresh1`` and ``thresh2``. Its implementation applies ``thresh1``
        to ``HV / VV`` and ``thresh2`` to ``HH / VV``, which is the ordering
        used here and by :func:`dubois_surface_inversion`. For C/Python
        comparisons, pass the Python values directly as ``-th1`` and
        ``-th2``, respectively.

        The returned dataset preserves the input coordinates and uses the
        input spatial dimensions. Output variables are stored as ``float32``
        arrays.
    """
    if not isinstance(incidence_angle, xr.DataArray):
        raise TypeError("incidence_angle must be an xarray.DataArray.")
    if not np.issubdtype(incidence_angle.dtype, np.number):
        raise TypeError("incidence_angle must contain numeric values.")

    if not isinstance(thresh1, (int, float, np.number)):
        raise TypeError(f"thresh1 must be a number, got {type(thresh1).__name__}.")
    if not isinstance(thresh2, (int, float, np.number)):
        raise TypeError(f"thresh2 must be a number, got {type(thresh2).__name__}.")
    if not isinstance(c_semantics, bool):
        raise TypeError(
            f"c_semantics must be a boolean, got {type(c_semantics).__name__}."
        )

    allowed_poltypes = ("S", "C3", "T3", "C4", "T4")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    converters = {
        "C3": lambda ds: ds,
        "T3": T3_to_C3,
        "C4": C4_to_C3,
        "T4": T4_to_C3,
        "S": S_to_C3,
    }
    C3 = converters[poltype](input_data)

    inversion = _apply_oh_inversion_c if c_semantics else _apply_oh_inversion
    out = inversion(
        theta=incidence_angle,
        hh=C3.m11,
        vv=C3.m33,
        hv=C3.m22 / 2.0,
        thresh1=thresh1,
        thresh2=thresh2,
    )
    original_non_nan_mask = input_data.to_array().notnull().all("variable")
    out["oh_mask_valid_in_out"] = (
        out["oh_mask_in"]
        * out["oh_mask_out"]
        * original_non_nan_mask.astype(np.float32, copy=False)
    ).astype(np.float32, copy=False)

    return xr.Dataset(
        {k: (tuple(input_data.dims), v.data) for k, v in out.items()},
        attrs={
            "poltype": "oh_surface_inversion",
            "description": "Results of the Oh surface inversion.",
        },
        coords=input_data.coords,
    )


# helper function, do not use directly
def _apply_dubois_inversion(theta, f0, hh, vv, hv, calib, thresh1, thresh2):

    scale = np.sin(theta) / calib if calib is not None else 1.0
    hh = hh * scale
    vv = vv * scale
    hv = hv * scale

    lambd = 100 * 0.3 / f0
    eps = np.finfo(np.float32).eps

    theta_valid = np.isfinite(theta) & (theta > 0) & (theta < (np.pi / 2.0))
    hh_pos = hh > 0
    vv_pos = vv > 0
    base_valid = theta_valid & hh_pos & vv_pos

    vv_safe = xr.where(vv_pos, vv, 1.0)
    hh_safe = xr.where(hh_pos, hh, 1.0)
    theta_safe = xr.where(theta_valid, theta, 1.0)

    msk_valid = (
        base_valid
        & (hv / vv_safe < 10 ** (thresh1 / 10.0))
        & (hh / vv_safe < 10 ** (thresh2 / 10.0))
    )

    ks_inv = np.exp(
        1.36905 * np.log(hh_safe)
        - 0.83333 * np.log(vv_safe)
        + 0.446425 * np.log(np.cos(theta_safe))
        + 3.34525 * np.log(np.sin(theta_safe))
        - 0.375 * np.log(lambd)
        + 1.78989 * np.log(10)
    )
    msk_ks = base_valid & (ks_inv >= 0) & (ks_inv <= np.pi)
    ks_dub = xr.where(msk_valid & msk_ks, ks_inv, 0.0)

    ks_safe = xr.where(ks_dub > 0, ks_dub, np.nan)
    er_inv = (
        np.log10(hh_safe)
        + np.log10(vv_safe)
        + 5.12
        + np.log10(np.cos(theta_safe))
        - 2.5 * np.log10(ks_safe + eps)
        - 1.4 * np.log10(lambd)
    ) / (0.074 * np.tan(theta_safe) + eps)
    msk_er = base_valid & (er_inv > 0) & (er_inv <= 100)
    er_dub = xr.where(msk_valid & msk_ks & msk_er, er_inv, 0.0)

    er_safe = xr.where(er_dub > 0, er_dub, np.nan)
    mv_inv = (
        -5.3e-2 + 2.92e-2 * er_safe - 5.5e-4 * er_safe**2 + 4.3e-6 * er_safe**3
    ) * 100
    msk_mv = base_valid & (mv_inv > 0) & (mv_inv <= 100)
    mv_dub = xr.where(msk_valid & msk_ks & msk_er & msk_mv, mv_inv, 0.0)

    msk_out = (msk_valid & msk_ks & msk_er & msk_mv).astype(np.float32)

    return {
        "dubois_ks": ks_dub.astype(np.float32, copy=False),
        "dubois_er": er_dub.astype(np.float32, copy=False),
        "dubois_mv": mv_dub.astype(np.float32, copy=False),
        "dubois_mask_out": msk_out.astype(np.float32, copy=False),
        "dubois_mask_in": msk_valid.astype(np.float32, copy=False),
    }


def _apply_oh_inversion(theta, hh, vv, hv, thresh1, thresh2):

    theta_valid = np.isfinite(theta) & (theta > 0) & (theta < (np.pi / 2.0))
    hh_pos = np.isfinite(hh) & (hh > 0)
    vv_pos = np.isfinite(vv) & (vv > 0)
    base_valid = theta_valid & hh_pos & vv_pos & np.isfinite(hv)

    vv_safe = xr.where(vv_pos, vv, 1.0)
    hh_safe = xr.where(hh_pos, hh, 1.0)
    theta_safe = xr.where(theta_valid, theta, 1.0)

    hh_vv = hh_safe / vv_safe
    hv_vv = hv / vv_safe
    msk_valid = (
        base_valid & (hv_vv < 10 ** (thresh1 / 10.0)) & (hh_vv < 10 ** (thresh2 / 10.0))
    )

    a = 2.0 * theta_safe / np.pi
    b = (hv_vv / 0.23).astype(np.float64, copy=False)
    c = (np.sqrt(hh_vv) - 1.0).astype(np.float64, copy=False)
    a = a.astype(np.float64, copy=False)
    x = xr.apply_ufunc(
        _solve_oh_newton,
        a,
        b,
        c,
        msk_valid,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    active = msk_valid & np.isfinite(x)

    abs_x = np.abs(x)
    er_domain = active & (abs_x != 0) & (abs_x != 1)
    abs_x_safe = xr.where(er_domain, abs_x, 2.0)
    er_inv = np.abs((1.0 + 1.0 / abs_x_safe) / (1.0 - 1.0 / abs_x_safe)) ** 2
    msk_er = er_domain & np.isfinite(er_inv) & (er_inv >= 0) & (er_inv < 20)
    er_oh = xr.where(msk_valid & msk_er, er_inv, 0.0)

    er_calc = xr.where(msk_valid & msk_er & (er_oh > 0), er_oh, np.nan)
    mv_inv = (
        -5.3e-2
        + 2.92e-2 * er_calc
        - 5.5e-4 * np.exp(2.0 * np.log(er_calc))
        + 4.3e-6 * np.exp(3.0 * np.log(er_calc))
    ) * 100.0
    msk_mv = np.isfinite(mv_inv) & (mv_inv >= 0)
    mv_oh = xr.where(msk_valid & msk_mv, mv_inv, 0.0)

    c_safe = xr.where(active & np.isfinite(c) & (c != 0), c, np.nan)
    ks_argument = np.abs(np.power(a, np.power(x, 2.0) / 3.0) / c_safe)
    ks_log_argument = xr.where(
        np.isfinite(ks_argument) & (ks_argument > 0), ks_argument, np.nan
    )
    ks_inv = np.log(ks_log_argument)
    msk_ks = active & np.isfinite(ks_inv) & (ks_inv >= 0) & (ks_inv <= 3)
    ks_oh = xr.where(msk_valid & msk_ks, ks_inv, 0.0)

    msk_out = (msk_valid & msk_mv & msk_er & msk_ks).astype(np.float32)

    return {
        "oh_ks": ks_oh.astype(np.float32, copy=False),
        "oh_er": er_oh.astype(np.float32, copy=False),
        "oh_mv": mv_oh.astype(np.float32, copy=False),
        "oh_mask_out": msk_out.astype(np.float32, copy=False),
        "oh_mask_in": msk_valid.astype(np.float32, copy=False),
    }


def _solve_oh_newton(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Solve the Oh model's Newton iteration for one array block.

    Args:
        a: Incidence-angle term.
        b: Cross-polarization ratio term.
        c: Co-polarization ratio term.
        valid: Input-validity mask.

    Returns:
        The Newton solution, with invalid trajectories represented by NaN.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    a_valid = np.isfinite(a) & (a > 0)
    a_safe = np.where(a_valid, a, 1.0)
    log_a = np.log(a_safe)
    x = np.full_like(a, 2.0)
    active = valid & a_valid & np.isfinite(b) & np.isfinite(c)
    # A larger update overflowed the previous float32 iteration and could not
    # yield a valid result. Keep that trajectory invalid instead of recovering
    # a different root only because the guarded calculations use float64.
    float32_max = np.finfo(np.float32).max

    for _ in range(100):
        x_valid = active & np.isfinite(x) & (np.abs(x) <= float32_max)
        x_safe = np.where(x_valid, x, 2.0)
        b_safe = np.where(x_valid, b, 0.0)
        c_safe = np.where(x_valid, c, 0.0)
        log_a_safe = np.where(x_valid, log_a, 0.0)

        a_power = np.exp((x_safe**2 / 3.0) * log_a_safe)
        one_minus_bx = 1.0 - b_safe * x_safe
        numerator = a_power * one_minus_bx + c_safe
        denominator = (
            (2.0 * x_safe / 3.0 * log_a_safe * one_minus_bx) - b_safe
        ) * a_power

        can_divide = (
            x_valid
            & np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator != 0)
            & (np.abs(numerator) <= float32_max * np.abs(denominator))
        )
        numerator_safe = np.where(can_divide, numerator, 0.0)
        denominator_safe = np.where(can_divide, denominator, 1.0)
        x_update = x_safe - numerator_safe / denominator_safe
        active = can_divide & np.isfinite(x_update) & (np.abs(x_update) <= float32_max)
        x = np.where(active, x_update, np.nan)

    return x


def _solve_oh_newton_c(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Solve the Oh Newton iteration with the legacy C numeric semantics.

    Args:
        a: Incidence-angle term.
        b: Cross-polarization ratio term.
        c: Co-polarization ratio term.
        valid: Input-validity mask.

    Returns:
        The float32 Newton solution, including the C implementation's
        non-finite results.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)

    log_a = np.log(a.astype(np.float64))
    x = np.full_like(a, np.float32(2.0))

    for _ in range(100):
        x_squared_third = (x * x / np.float32(3.0)).astype(np.float32, copy=False)
        one_minus_bx = (np.float32(1.0) - b * x).astype(np.float32, copy=False)
        a_power = np.exp(x_squared_third.astype(np.float64) * log_a)
        numerator = a_power * one_minus_bx.astype(np.float64) + c.astype(np.float64)
        two_x_third = (np.float32(2.0) * x / np.float32(3.0)).astype(
            np.float32, copy=False
        )
        denominator = (
            two_x_third.astype(np.float64) * log_a * one_minus_bx.astype(np.float64)
            - b.astype(np.float64)
        ) * a_power
        x_update = (x.astype(np.float64) - numerator / denominator).astype(
            np.float32, copy=False
        )
        x = np.where(valid, x_update, np.float32(2.0)).astype(np.float32, copy=False)

    return x


def _apply_oh_inversion_c_block(theta, hh, vv, hv, thresh1, thresh2):
    # This suppression is deliberate: the reference C code lacks numerical
    # safeguards and silently propagates the resulting IEEE Inf and NaN values.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        theta = np.asarray(theta, dtype=np.float32)
        hh = np.asarray(hh, dtype=np.float32)
        vv = np.asarray(vv, dtype=np.float32)
        hv = np.asarray(hv, dtype=np.float32)

        hh_vv = (hh / vv).astype(np.float32, copy=False)
        hv_vv = (hv / vv).astype(np.float32, copy=False)
        thresh1 = np.float32(thresh1)
        thresh2 = np.float32(thresh2)
        hv_limit = np.power(10.0, np.float64(thresh1 / np.float32(10.0)))
        hh_limit = np.power(10.0, np.float64(thresh2 / np.float32(10.0)))
        msk_valid = (hv_vv < hv_limit) & (hh_vv < hh_limit)

        two_theta = (np.float32(2.0) * theta).astype(np.float32, copy=False)
        a = (two_theta.astype(np.float64) / np.pi).astype(np.float32, copy=False)
        b = (hv_vv.astype(np.float64) / 0.23).astype(np.float32, copy=False)
        c = (np.sqrt(hh_vv.astype(np.float64)) - 1.0).astype(np.float32, copy=False)
        x = _solve_oh_newton_c(a, b, c, msk_valid)

        abs_x = np.abs(x.astype(np.float64))
        er_inv = np.power(
            np.abs((1.0 + 1.0 / abs_x) / (1.0 - 1.0 / abs_x)), 2.0
        ).astype(np.float32, copy=False)
        msk_er = ~((er_inv >= 20) | (er_inv < 0))
        er_oh = np.where(
            msk_valid,
            er_inv * msk_er.astype(np.float32),
            np.float32(0.0),
        ).astype(np.float32, copy=False)

        er_calc = er_oh.astype(np.float64, copy=False)
        mv_inv = (
            -5.3e-2
            + 2.92e-2 * er_calc
            - 5.5e-4 * np.exp(2.0 * np.log(er_calc))
            + 4.3e-6 * np.exp(3.0 * np.log(er_calc))
        ) * 100.0
        mv_inv = mv_inv.astype(np.float32, copy=False)
        msk_mv = ~(mv_inv < 0)
        mv_oh = np.where(
            msk_valid,
            mv_inv * msk_mv.astype(np.float32),
            np.float32(0.0),
        ).astype(np.float32, copy=False)

        ks_inv = np.log(
            np.abs(
                np.power(
                    a.astype(np.float64),
                    np.power(x.astype(np.float64), 2.0) / 3.0,
                )
                / c.astype(np.float64)
            )
        ).astype(np.float32, copy=False)
        msk_ks = ~((ks_inv > 3) | (ks_inv < 0))
        ks_oh = np.where(
            msk_valid,
            ks_inv * msk_ks.astype(np.float32),
            np.float32(0.0),
        ).astype(np.float32, copy=False)

        msk_out = (msk_valid & msk_mv & msk_er & msk_ks).astype(np.float32)

    return ks_oh, er_oh, mv_oh, msk_out, msk_valid.astype(np.float32, copy=False)


def _apply_oh_inversion_c(theta, hh, vv, hv, thresh1, thresh2):
    outputs = xr.apply_ufunc(
        _apply_oh_inversion_c_block,
        theta,
        hh,
        vv,
        hv,
        kwargs={"thresh1": thresh1, "thresh2": thresh2},
        dask="parallelized",
        output_core_dims=[[], [], [], [], []],
        output_dtypes=[np.float32] * 5,
    )
    names = ("oh_ks", "oh_er", "oh_mv", "oh_mask_out", "oh_mask_in")
    return dict(zip(names, outputs))
