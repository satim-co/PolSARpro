import numpy as np
import xarray as xr
import dask.array as da
from polsarpro.util import S_to_C3, T3_to_C3, C4_to_C3, T4_to_C3
from polsarpro.auxil import validate_dataset


def dubois_surface_inversion(
    input_data: xr.Dataset,
    incidence_angle: xr.DataArray,
    f0: float, # In GHz!!!
    thresh1: float, # dB
    thresh2: float, # dB
    calibration_coeff: float | None = None, # sigma0? beta0?
) -> xr.Dataset:
    if not isinstance(incidence_angle, xr.DataArray):
        raise TypeError("incidence_angle must be an xarray.DataArray.")
    if not np.issubdtype(incidence_angle.dtype, np.number):
        raise TypeError("incidence_angle must contain numeric values.")

    if not isinstance(f0, (int, float, np.number)):
        raise TypeError(f"f0 must be a number, got {type(f0).__name__}.")
    if f0 <= 0:
        raise ValueError(f"f0 must be strictly positive, got {f0}.")

    if not isinstance(thresh1, (int, float, np.number)):
        raise TypeError(
            f"thresh1 must be a number, got {type(thresh1).__name__}."
        )
    if not isinstance(thresh2, (int, float, np.number)):
        raise TypeError(
            f"thresh2 must be a number, got {type(thresh2).__name__}."
        )

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
        f0=f0,
        hh=C3.m11,
        vv=C3.m33,
        hv=C3.m22 / 2.0,
        calib=calibration_coeff,
        thresh1=thresh1,
        thresh2=thresh2,
    )
    original_non_nan_mask = input_data.to_array().notnull().all("variable")
    out["dubois_mask_in_out_valid"] = (
        out["dubois_mask_in"]
        * out["dubois_mask_out"]
        * original_non_nan_mask.astype(np.float32, copy=False)
    ).astype(np.float32, copy=False)

    return xr.Dataset(
        {k: (tuple(input_data.dims), v.data) for k, v in out.items()},
        attrs={
            "poltype": "dubois",
            "description": "Results of the Dubois surface inversion.",
        },
        coords=input_data.coords,
    )


def _apply_dubois_inversion(theta, f0, hh, vv, hv, calib, thresh1, thresh2):
    scale = np.sin(theta) / calib if calib is not None else 1.0
    hh = hh * scale
    vv = vv * scale
    hv = hv * scale

    lambd = 100 * 0.3 / f0
    eps = np.finfo(np.float32).eps

    msk_valid = (hv / vv < 10 ** (thresh1 / 10.0)) & (hh / vv < 10 ** (thresh2 / 10.0))

    ks_inv = np.exp(
        1.36905 * np.log(hh)
        - 0.83333 * np.log(vv)
        + 0.446425 * np.log(np.cos(theta))
        + 3.34525 * np.log(np.sin(theta))
        - 0.375 * np.log(lambd)
        + 1.78989 * np.log(10)
    )
    msk_ks = (ks_inv >= 0) & (ks_inv <= np.pi)
    ks_dub = xr.where(msk_valid & msk_ks, ks_inv, 0.0)

    ks_safe = xr.where(ks_dub > 0, ks_dub, np.nan)
    er_inv = (
        np.log10(hh)
        + np.log10(vv)
        + 5.12
        + np.log10(np.cos(theta))
        - 2.5 * np.log10(ks_safe + eps)
        - 1.4 * np.log10(lambd)
    ) / (0.074 * np.tan(theta) + eps)
    msk_er = (er_inv > 0) & (er_inv <= 100)
    er_dub = xr.where(msk_valid & msk_ks & msk_er, er_inv, 0.0)

    er_safe = xr.where(er_dub > 0, er_dub, np.nan)
    mv_inv = (
        -5.3e-2
        + 2.92e-2 * er_safe
        - 5.5e-4 * (er_safe * er_safe)
        + 4.3e-6 * (er_safe * er_safe * er_safe)
    ) * 100
    msk_mv = (mv_inv > 0) & (mv_inv <= 100)
    mv_dub = xr.where(msk_valid & msk_ks & msk_er & msk_mv, mv_inv, 0.0)

    msk_out = (msk_valid & msk_ks & msk_er & msk_mv).astype(np.float32)

    return {
        "dubois_ks": ks_dub.astype(np.float32, copy=False),
        "dubois_er": er_dub.astype(np.float32, copy=False),
        "dubois_mv": mv_dub.astype(np.float32, copy=False),
        "dubois_mask_out": msk_out.astype(np.float32, copy=False),
        "dubois_mask_in": msk_valid.astype(np.float32, copy=False),
    }
