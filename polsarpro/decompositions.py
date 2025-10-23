"""
This code is part of the Python PolSARpro software:

"A re-implementation of selected PolSARPro functions in Python,
following the scientific recommendations of PolInSAR 2021"

developed within an ESA funded project with SATIM.

Author: Olivier D'Hondt, 2025.
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

# Description: module containing polarimetric decomposition functions

"""

import numpy as np
import xarray as xr
import dask.array as da
from polsarpro.util import boxcar, C3_to_T3, S_to_C3, S_to_T3, C4_to_T4, T3_to_C3
from polsarpro.auxil import validate_dataset


def freeman(
    input_data: xr.Dataset,
    boxcar_size: list[int, int] = [3, 3],
) -> xr.Dataset:
    """Applies the Freeman-Durden decomposition. This decomposition is based on physical modeling
      of the covariance matrix and returns 3 components Ps, Pd and Pv which are the powers of resp.
      surface, double bounce and volume backscattering.

    Args:
        input_data (xr.Dataset): Input image, may be a covariance (C3), coherency (T3) or Sinclair (S) matrix.
        boxcar_size (list[int, int], optional):  Boxcar dimensions along azimuth and range. Defaults to [3, 3].

    Returns:
        xr.Dataset: Ps, Pd and Pv components.
    """

    allowed_poltypes = ("S", "C3", "T3")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    # in_ = input_data.astype("complex64", copy=False)
    if poltype == "C3":
        in_ = input_data
    elif poltype == "T3":
        in_ = T3_to_C3(input_data)
    elif poltype == "S":
        in_ = S_to_C3(input_data)

    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = boxcar(in_, boxcar_size[0], boxcar_size[1])

    out = _compute_freeman_components(in_)
    return xr.Dataset(
        # add dimension names
        {k: (tuple(input_data.dims), v) for k, v in out.items()},
        attrs=dict(
            poltype="freeman",
            description="Results of the Freeman-Durden decomposition.",
        ),
        coords=input_data.coords,
    )


def h_a_alpha(
    input_data: xr.Dataset,
    boxcar_size: list[int, int] = [3, 3],
    flags: tuple[str] = ("entropy", "alpha", "anisotropy"),
) -> xr.Dataset:
    """Performs the H/A/Alpha polarimetric decomposition on full-pol SAR data.

    This function computes the H/A/Alpha decomposition from input polarimetric SAR data
    using eigenvalue analysis of the coherency matrix. The decomposition
    provides physical insight into scattering mechanisms through parameters such as
    entropy (H), anisotropy (A), and the alpha scattering angle (alpha). Additional
    eigenvalue-based parameters can also be computed by specifying corresponding flags.

    Args:
        input_data (xr.Dataset): Input polarimetric SAR dataset. Supported types are:

            - "S": Sinclair scattering matrix

            - "C3": Lexicographic covariance matrix

            - "T3": Pauli coherency matrix

            - "C4" and "T4": 4x4 versions of the above

            - "C2": Dual-polarimetric covariance

        boxcar_size (list[int, int], optional): Size of the spatial averaging window to be
            applied before decomposition (boxcar filter). Defaults to [3, 3].
        flags (tuple[str], optional): Parameters to compute and return from the decomposition.
            Possible values include:

            - "entropy": Scattering entropy (H)

            - "anisotropy": Scattering anisotropy (A)

            - "alpha": Mean alpha scattering angle (alpha)

            - "beta", "delta", "gamma", "lambda": Other angular or eigenvalue related parameters

            - "nhu", "epsilon" additional angles defined only for 4x4 matrices. Will be ignored if not processing 4x4 matrices.

            - "alphas", "betas", "deltas", "gammas", "lambdas": Per-eigenvalue versions of the above
            Defaults to ("entropy", "alpha", "anisotropy").

    Returns:
        xr.Dataset: An xarray.Dataset where data variable names correspond to the requested flags, and values are the corresponding 2D arrays (or 3D if the flag returns multiple values per pixel).

    Notes:
        For C2 inputs, only 'alpha(s)', 'delta(s)', 'anisotropy', 'entropy' and 'lambda(s)' can be computed. Other passed flags will be ignored.

        If the S matrix is given as an input, a 3x3 analysis will be assumed using the T3 matrix. For 4x4 and 2x2, use 'C4', 'T4' or 'C2' as an input.

    References:
        Cloude, S. R., & Pottier, E. (1997). An entropy based classification scheme for land
        applications of polarimetric SAR. *IEEE Transactions on Geoscience and Remote Sensing*,
        35(1), 68-78.
    """

    # check flags validity
    possible_flags = (
        "entropy",
        "anisotropy",
        "alpha",
        "beta",
        "delta",
        "epsilon",
        "gamma",
        "lambda",
        "nhu",
        "alphas",
        "betas",
        "deltas",
        "epsilons",
        "gammas",
        "lambdas",
        "nhus",
    )
    for flag in flags:
        if flag not in possible_flags:
            raise ValueError(
                f"Flag '{flag}' not recognized. Possible values are {possible_flags}."
            )
    allowed_poltypes = ("S", "C2", "C3", "C4", "T3", "T4")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    if poltype == "C2":
        in_ = input_data
    elif poltype == "C3":
        in_ = C3_to_T3(input_data)
    elif poltype == "T3":
        in_ = input_data
    elif poltype == "C4":
        in_ = C4_to_T4(input_data)
    elif poltype == "T4":
        in_ = input_data
    elif poltype == "S":
        in_ = S_to_T3(input_data)
    else:
        raise ValueError(f"Invalid polarimetric type: {input_data.poltype}")

    new_dims = tuple(input_data.dims)

    eps = 1e-30
    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = boxcar(in_, dim_az=boxcar_size[0], dim_rg=boxcar_size[1])

    # remove NaNs to avoid errors in eigh
    mask = in_.to_array().isnull().any("variable")
    in_ = in_.fillna(eps)

    M = _reconstruct_matrix_from_ds(in_)

    # eigendecomposition
    meta = (
        np.array([], dtype="float32").reshape((0, 0, 0)),
        np.array([], dtype="complex64").reshape((0, 0, 0, 0)),
    )
    l, v = da.apply_gufunc(np.linalg.eigh, "(i,j)->(i), (i,j)", M, meta=meta)

    l = l[..., ::-1]  # descending order
    v = v[..., ::-1]

    # returns a dict
    if in_.poltype in ("S", "T3", "C4"):
        out = _compute_h_a_alpha_parameters_T3(l, v, flags)
    elif in_.poltype in ("T4", "C4"):
        out = _compute_h_a_alpha_parameters_T4(l, v, flags)
    elif in_.poltype in ("C2"):
        out = _compute_h_a_alpha_parameters_C2(l, v, flags)

    return xr.Dataset(
        # add dimension names, account for 2D and 3D outputs
        {
            k: (new_dims, v) if v.ndim == 2 else (new_dims + ("i",), v)
            for k, v in out.items()
        },
        attrs=dict(
            poltype="h_a_alpha", description="Results of the H/A/Alpha decomposition."
        ),
        coords=input_data.coords,
    ).where(~mask)

def yamaguchi3(
    input_data: xr.Dataset,
    boxcar_size: list[int, int] = [3, 3],
) -> xr.Dataset:
    """Applies the Yamaguchi 3 component decomposition. This decomposition is based on physical modeling
      of the covariance matrix and returns 3 components Ps, Pd and Pv which are the powers of resp.
      surface, double bounce and volume backscattering.

    Args:
        input_data (xr.Dataset): Input image, may be a covariance (C3), coherency (T3) or Sinclair (S) matrix.
        boxcar_size (list[int, int], optional):  Boxcar dimensions along azimuth and range. Defaults to [3, 3].

    Returns:
        xr.Dataset: Ps, Pd and Pv components.
    """

    allowed_poltypes = ("S", "C3", "T3")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    # in_ = input_data.astype("complex64", copy=False)
    if poltype == "C3":
        in_ = input_data
    elif poltype == "T3":
        in_ = T3_to_C3(input_data)
    elif poltype == "S":
        in_ = S_to_C3(input_data)

    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = boxcar(in_, boxcar_size[0], boxcar_size[1])

    out = _compute_yamaguchi3_components(in_)
    return xr.Dataset(
        # add dimension names
        {k: (tuple(input_data.dims), v) for k, v in out.items()},
        attrs=dict(
            poltype="yamaguchi3",
            description="Results of the Yamaguchi 3 component decomposition.",
        ),
        coords=input_data.coords,
    )


# TODO: update docstrings
def yamaguchi4(
    input_data: xr.Dataset,
    boxcar_size: list[int, int] = [3, 3],
    mode: str = "y40",
) -> xr.Dataset:
    """Applies the Freeman-Durden decomposition. This decomposition is based on physical modeling
      of the covariance matrix and returns 3 components Ps, Pd and Pv which are the powers of resp.
      surface, double bounce and volume backscattering.

    Args:
        input_data (xr.Dataset): Input image, may be a covariance (C3), coherency (T3) or Sinclair (S) matrix.
        boxcar_size (list[int, int], optional):  Boxcar dimensions along azimuth and range. Defaults to [3, 3].

    Returns:
        xr.Dataset: Ps, Pd and Pv components.
    """

    allowed_poltypes = ("S", "C3", "T3")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    # in_ = input_data.astype("complex64", copy=False)
    if poltype == "C3":
        in_ = C3_to_T3(input_data)
    elif poltype == "T3":
        in_ = input_data
    elif poltype == "S":
        in_ = S_to_T3(input_data)

    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = boxcar(in_, boxcar_size[0], boxcar_size[1])

    out = _compute_yamaguchi4_components(in_, mode=mode)
    return xr.Dataset(
        # add dimension names
        {k: (tuple(input_data.dims), v) for k, v in out.items()},
        attrs=dict(
            poltype="yamaguchi4",
            description="Results of the Yamaguchi 4 component decomposition.",
        ),
        coords=input_data.coords,
    )


# below this line, functions are not meant to be called directly


def _compute_h_a_alpha_parameters_C2(l, v, flags):

    eps = 1e-30

    # Pseudo-probabilities (normalized eigenvalues)
    p = np.clip(l / (eps + l.sum(axis=2)[..., None]), eps, 1)

    outputs = {}
    if "entropy" in flags:
        H = np.sum(-p * np.log(p), axis=2) / np.float32(np.log(2))
        outputs["entropy"] = H

    if "anisotropy" in flags:
        A = (p[..., 0] - p[..., 1]) / (p[..., 0] + p[..., 1] + eps)
        outputs["anisotropy"] = A

    if "alpha" in flags or "alphas" in flags:
        # Alpha angles for each mechanism
        alphas = np.arccos(np.abs(v[:, :, 0, :]))
        # Convert to degrees
        alphas *= 180 / np.pi

    if "alpha" in flags:
        # Mean alpha
        alpha = np.sum(p * alphas, axis=2)
        outputs["alpha"] = alpha

    if "delta" in flags or "deltas" in flags:
        phases = np.atan2(v[:, :, 0, :].imag, eps + v[:, :, 0, :].real)
        deltas = np.atan2(v[:, :, 1, :].imag, eps + v[:, :, 1, :].real) - phases
        deltas = np.atan2(np.sin(deltas), eps + np.cos(deltas))
        deltas *= 180 / np.pi

    if "delta" in flags:
        delta = np.sum(p * deltas, axis=2)
        outputs["delta"] = delta

    # Average target eigenvalue
    if "lambda" in flags or "lambdas" in flags:
        # lambda is a python reserved keyword, using lambd instead
        lambd = np.sum(p * l, axis=2)
        outputs["lambda"] = lambd

    # extras outputs: non averaged parameters (ex: alpha1, alpha2, alpha3)
    if "alphas" in flags:
        outputs["alphas"] = alphas
    if "deltas" in flags:
        outputs["deltas"] = deltas
    if "lambdas" in flags:
        outputs["lambdas"] = l
    return outputs


def _compute_h_a_alpha_parameters_T3(l, v, flags):

    eps = 1e-30

    # Pseudo-probabilities (normalized eigenvalues)
    p = np.clip(l / (eps + l.sum(axis=2)[..., None]), eps, 1)

    outputs = {}
    if "entropy" in flags:
        H = np.sum(-p * np.log(p), axis=2) / np.float32(np.log(3))
        outputs["entropy"] = H

    if "anisotropy" in flags:
        A = (p[..., 1] - p[..., 2]) / (p[..., 1] + p[..., 2] + eps)
        outputs["anisotropy"] = A

    if "alpha" in flags or "alphas" in flags:
        # Alpha angles for each mechanism
        alphas = np.arccos(np.abs(v[:, :, 0, :]))
        # Convert to degrees
        alphas *= 180 / np.pi

    if "alpha" in flags:
        # Mean alpha
        alpha = np.sum(p * alphas, axis=2)
        outputs["alpha"] = alpha

    # Extra angles: beta, delta and gamma angles
    if "beta" in flags or "betas" in flags:
        betas = np.atan2(np.abs(v[:, :, 2, :]), eps + np.abs(v[:, :, 1, :]))
        betas *= 180 / np.pi

    if "beta" in flags:
        beta = np.sum(p * betas, axis=2)
        outputs["beta"] = beta

    if "delta" in flags or "gamma" in flags or "deltas" in flags or "gammas" in flags:
        phases = np.atan2(v[:, :, 0, :].imag, eps + v[:, :, 0, :].real)

    if "delta" in flags or "deltas" in flags:
        deltas = np.atan2(v[:, :, 1, :].imag, eps + v[:, :, 1, :].real) - phases
        deltas = np.atan2(np.sin(deltas), eps + np.cos(deltas))
        deltas *= 180 / np.pi

    if "delta" in flags:
        delta = np.sum(p * deltas, axis=2)
        outputs["delta"] = delta

    if "gamma" in flags or "gammas" in flags:
        gammas = np.atan2(v[:, :, 2, :].imag, eps + v[:, :, 2, :].real) - phases
        gammas = np.atan2(np.sin(gammas), eps + np.cos(gammas))
        gammas *= 180 / np.pi

    if "gamma" in flags:
        gamma = np.sum(p * gammas, axis=2)
        outputs["gamma"] = gamma

    # Average target eigenvalue
    if "lambda" in flags or "lambdas" in flags:
        # lambda is a python reserved keyword, using lambd instead
        lambd = np.sum(p * l, axis=2)
        outputs["lambda"] = lambd

    # extras outputs: non averaged parameters (ex: alpha1, alpha2, alpha3)
    if "alphas" in flags:
        outputs["alphas"] = alphas

    if "betas" in flags:
        outputs["betas"] = betas

    if "deltas" in flags:
        outputs["deltas"] = deltas

    if "gammas" in flags:
        outputs["gammas"] = gammas

    if "lambdas" in flags:
        outputs["lambdas"] = l
    return outputs


def _compute_h_a_alpha_parameters_T4(l, v, flags):

    eps = 1e-30

    # Pseudo-probabilities (normalized eigenvalues)
    p = np.clip(l / (eps + l.sum(axis=2)[..., None]), eps, 1)

    outputs = {}
    if "entropy" in flags:
        H = np.sum(-p * np.log(p), axis=2) / np.float32(np.log(4))
        outputs["entropy"] = H

    if "anisotropy" in flags:
        A = (p[..., 1] - p[..., 2]) / (p[..., 1] + p[..., 2] + eps)
        outputs["anisotropy"] = A

    if "alpha" in flags or "alphas" in flags:
        # Alpha angles for each mechanism
        alphas = np.arccos(np.abs(v[:, :, 0, :]))
        # Convert to degrees
        alphas *= 180 / np.pi

    if "alpha" in flags:
        # Mean alpha
        alpha = np.sum(p * alphas, axis=2)
        outputs["alpha"] = alpha

    # Extra angles: beta, delta and gamma angles
    if "beta" in flags or "betas" in flags:
        # betas = np.atan2(np.abs(v[:, :, 2, :]), eps + np.abs(v[:, :, 1, :]))
        beta_num = np.sqrt(
            np.real(
                v[:, :, 2, :] * v[:, :, 2, :].conj()
                + v[:, :, 3, :] * v[:, :, 3, :].conj()
            )
        )
        betas = np.atan2(beta_num, eps + np.abs(v[:, :, 1, :]))
        betas *= 180 / np.pi

    if "beta" in flags:
        beta = np.sum(p * betas, axis=2)
        outputs["beta"] = beta

    if "epsilon" in flags or "epsilons" in flags:
        epsilons = np.atan2(np.abs(v[:, :, 3, :]), eps + np.abs(v[:, :, 2, :]))
        epsilons *= 180 / np.pi

    if "epsilon" in flags:
        epsilon = np.sum(p * epsilons, axis=2)
        outputs["epsilon"] = epsilon

    if "delta" in flags or "gamma" in flags or "deltas" in flags or "gammas" in flags:
        phases = np.atan2(v[:, :, 0, :].imag, eps + v[:, :, 0, :].real)

    if "delta" in flags or "deltas" in flags:
        deltas = np.atan2(v[:, :, 1, :].imag, eps + v[:, :, 1, :].real) - phases
        deltas = np.atan2(np.sin(deltas), eps + np.cos(deltas))
        deltas *= 180 / np.pi

    if "delta" in flags:
        delta = np.sum(p * deltas, axis=2)
        outputs["delta"] = delta

    if "gamma" in flags or "gammas" in flags:
        gammas = np.atan2(v[:, :, 2, :].imag, eps + v[:, :, 2, :].real) - phases
        gammas = np.atan2(np.sin(gammas), eps + np.cos(gammas))
        gammas *= 180 / np.pi

    if "gamma" in flags:
        gamma = np.sum(p * gammas, axis=2)
        outputs["gamma"] = gamma

    if "nhu" in flags or "nhus" in flags:
        nhus = np.atan2(v[:, :, 3, :].imag, eps + v[:, :, 3, :].real) - phases
        nhus = np.atan2(np.sin(nhus), eps + np.cos(nhus))
        nhus *= 180 / np.pi

    if "nhu" in flags:
        nhu = np.sum(p * nhus, axis=2)
        outputs["nhu"] = nhu

    # Average target eigenvalue
    if "lambda" in flags or "lambdas" in flags:
        # lambda is a python reserved keyword, using lambd instead
        lambd = np.sum(p * l, axis=2)
        outputs["lambda"] = lambd

    # extras outputs: non averaged parameters (ex: alpha1, alpha2, alpha3)
    if "alphas" in flags:
        outputs["alphas"] = alphas

    if "betas" in flags:
        outputs["betas"] = betas

    if "epsilons" in flags:
        outputs["epsilons"] = epsilons

    if "deltas" in flags:
        outputs["deltas"] = deltas

    if "gammas" in flags:
        outputs["gammas"] = gammas

    if "nhus" in flags:
        outputs["nhus"] = nhus

    if "lambdas" in flags:
        outputs["lambdas"] = l
    return outputs


# this is a convenience function to give eigh the right data format
def _reconstruct_matrix_from_ds(ds):

    eps = 1e-30

    if {"y", "x"}.issubset(ds.dims):
        new_dims = ("y", "x")
    elif {"lat", "lon"}.issubset(ds.dims):
        new_dims = ("lat", "lon")
    else:
        ValueError(
            "Input data does not have valid dimension names. ('y', 'x') or ('lat', 'lon') allowed."
        )

    new_dims_array = new_dims + ("i", "j")
    if ds.poltype == "C2":
        # build each line of the T3 matrix
        C2_l1 = xr.concat((ds.m11, ds.m12), dim="j")
        C2_l2 = xr.concat((ds.m12.conj(), ds.m22), dim="j")

        # Concatenate all lines into a 2x2 matrix
        return (
            xr.concat((C2_l1, C2_l2), dim="i")
            .transpose(*new_dims_array)
            .chunk({new_dims[0]: "auto", new_dims[1]: "auto", "i": 2, "j": 2})
            + eps
        )
    if ds.poltype == "T3":
        # build each line of the T3 matrix
        T3_l1 = xr.concat((ds.m11, ds.m12, ds.m13), dim="j")
        T3_l2 = xr.concat((ds.m12.conj(), ds.m22, ds.m23), dim="j")
        T3_l3 = xr.concat((ds.m13.conj(), ds.m23.conj(), ds.m33), dim="j")

        # Concatenate all lines into a 3x3 matrix
        return (
            xr.concat((T3_l1, T3_l2, T3_l3), dim="i")
            .transpose(*new_dims_array)
            .chunk({new_dims[0]: "auto", new_dims[1]: "auto", "i": 3, "j": 3})
            + eps
        )
    elif ds.poltype == "T4":
        # build each line of the T4 matrix
        T4_l1 = xr.concat((ds.m11, ds.m12, ds.m13, ds.m14), dim="j")
        T4_l2 = xr.concat((ds.m12.conj(), ds.m22, ds.m23, ds.m24), dim="j")
        T4_l3 = xr.concat((ds.m13.conj(), ds.m23.conj(), ds.m33, ds.m34), dim="j")
        T4_l4 = xr.concat(
            (ds.m14.conj(), ds.m24.conj(), ds.m34.conj(), ds.m44), dim="j"
        )

        # concatenate all lines into a 4x4 matrix
        return (
            xr.concat((T4_l1, T4_l2, T4_l3, T4_l4), dim="i")
            .transpose(*new_dims_array)
            .chunk({new_dims[0]: "auto", new_dims[1]: "auto", "i": 4, "j": 4})
            + eps
        )
    else:
        raise NotImplementedError("Implemented only for C2, T3 and T4 poltypes.")


def _compute_freeman_components(C3):

    eps = 1e-30

    # Make copies to avoid modifying original data
    c11 = C3.m11.data.copy()
    c13r = C3.m13.data.real.copy()
    c13i = C3.m13.data.imag.copy()
    c22 = C3.m22.data.copy()
    c33 = C3.m33.data.copy()

    fv = 1.5 * c22
    c11 -= fv
    c33 -= fv
    c13r -= fv / 3

    # Volume scattering condition
    cnd1 = (c11 <= eps) | (c33 <= eps)
    fv = da.where(cnd1, 3 * (c11 + c22 + c33 + 2 * fv) / 8, fv)

    # Compute c13 power
    pow_c13 = c13r**2 + c13i**2

    # Data conditioning for non-realizable term
    cnd2 = ~cnd1 & (pow_c13 > c11 * c33)
    arg_sqrt = da.maximum(c11 * c33 / da.maximum(pow_c13, eps), 0)
    scale_factor = da.where(cnd2, da.sqrt(arg_sqrt), 1)
    c13r *= scale_factor
    c13i *= scale_factor

    # Recompute after conditioning
    pow_c13 = c13r**2 + c13i**2

    # Odd bounce dominates
    cnd3 = ~cnd1 & (c13r >= 0)
    alpha = da.where(cnd3, da.float32(-1), da.float32(eps))
    arg_div = c11 + c33 + 2 * c13r
    arg_div = da.where(arg_div == 0, eps, arg_div)
    fd = da.where(cnd3, (c11 * c33 - pow_c13) / arg_div, eps)
    fs = da.where(cnd3, c33 - fd, eps)
    arg_sqrt = da.maximum((fd + c13r) ** 2 + c13i**2, eps)
    arg_div = da.where(fs == 0, eps, fs)
    beta = da.where(cnd3, da.sqrt(arg_sqrt) / arg_div, eps)

    # Even bounce dominates
    cnd4 = ~cnd1 & (c13r < 0)
    beta = da.where(cnd4, 1, beta)
    arg_div = c11 + c33 - 2 * c13r
    arg_div = da.where(arg_div == 0, eps, arg_div)
    fs = da.where(cnd4, (c11 * c33 - pow_c13) / arg_div, fs)
    fd = da.where(cnd4, c33 - fs, fd)
    arg_sqrt = da.maximum((fs - c13r) ** 2 + c13i**2, eps)
    arg_div = da.where(fd == 0, eps, fd)
    alpha = da.where(cnd4, da.sqrt(arg_sqrt) / arg_div, alpha)

    # Compute Freeman components
    Ps = fs * (1 + beta**2)
    Pd = fd * (1 + alpha**2)
    Pv = 8 * fv / 3

    # Compute span on __original__ covariance, not modified one
    sp = C3.m11.data + C3.m22.data + C3.m33.data
    min_span, max_span = da.nanmin(sp), da.nanmax(sp)
    min_span = da.maximum(min_span, eps)

    Ps = da.where(Ps <= min_span, min_span, da.where(Ps > max_span, max_span, Ps))
    Pd = da.where(Pd <= min_span, min_span, da.where(Pd > max_span, max_span, Pd))
    Pv = da.where(Pv <= min_span, min_span, da.where(Pv > max_span, max_span, Pv))
    return {"odd": Ps, "double": Pd, "volume": Pv}


def _compute_yamaguchi3_components(C3):

    eps = 1e-30

    # Make copies to avoid modifying original data
    c11 = C3.m11.data.copy()
    c13r = C3.m13.data.real.copy()
    c13i = C3.m13.data.imag.copy()
    c22 = C3.m22.data.copy()
    c33 = C3.m33.data.copy()

    # Freeman-Yamaguchi algorithm
    ratio = 10.0 * da.log10(c33 / c11)
    msk_l = ratio <= -2
    msk_h = ratio > 2
    msk_m = (ratio > -2) & (ratio <= 2)

    fv = da.where(msk_m, 2 * c22, 15 * c22 / 8)
    c13r = da.where(msk_m, c13r - fv / 8, c13r - 2 * fv / 15)

    c11 = da.where(msk_l, c11 - 8 * fv / 15, c11)
    c33 = da.where(msk_l, c33 - 3 * fv / 15, c33)

    c11 = da.where(msk_h, c11 - 3 * fv / 15, c11)
    c33 = da.where(msk_h, c33 - 8 * fv / 15, c33)

    c11 = da.where(msk_m, c11 - 3 * fv / 8, c11)
    c33 = da.where(msk_m, c33 - 3 * fv / 8, c33)

    # Volume scattering condition
    cnd1 = (c11 <= eps) | (c33 <= eps)
    fv = da.where(cnd1 & msk_m, c11 + 3 * fv / 8 + c22 / 2 + c33 + 3 * fv / 8, fv)
    fv = da.where(cnd1 & msk_l, c11 + 8 * fv / 15 + c22 / 2 + c33 + 3 * fv / 15, fv)
    fv = da.where(cnd1 & msk_h, c11 + 3 * fv / 15 + c22 / 2 + c33 + 8 * fv / 15, fv)

    # Compute c13 power
    pow_c13 = c13r**2 + c13i**2

    # Data conditioning for non-realizable term
    cnd2 = ~cnd1 & (pow_c13 > c11 * c33)
    arg_sqrt = da.maximum(c11 * c33 / da.maximum(pow_c13, eps), 0)
    scale_factor = da.where(cnd2, da.sqrt(arg_sqrt), 1)
    c13r *= scale_factor
    c13i *= scale_factor

    # Recompute after conditioning
    pow_c13 = c13r**2 + c13i**2

    # Odd bounce dominates
    cnd3 = ~cnd1 & (c13r >= 0)
    alpha = da.where(
        cnd3, da.float32(-1) + 1j * da.float32(0), da.float32(eps) + 1j * da.float32(0)
    )
    arg_div = c11 + c33 + 2 * c13r
    arg_div = da.where(arg_div == 0, eps, arg_div)
    fd = da.where(cnd3, (c11 * c33 - pow_c13) / arg_div, eps)
    fs = da.where(cnd3, c33 - fd, eps)
    arg_div = da.where(fs == 0, eps, fs)
    beta = da.where(cnd3, (fd + c13r + 1j * c13i) / arg_div, eps)

    # Even bounce dominates
    cnd4 = ~cnd1 & (c13r < 0)
    beta = da.where(cnd4, 1 + 1j * 0, beta)
    arg_div = c11 + c33 - 2 * c13r
    arg_div = da.where(arg_div == 0, eps, arg_div)
    fs = da.where(cnd4, (c11 * c33 - pow_c13) / arg_div, fs)
    fd = da.where(cnd4, c33 - fs, fd)
    arg_div = da.where(fd == 0, eps, fd)
    alpha = da.where(cnd4, (c13r - fs + 1j*c13i) / arg_div, alpha)

    # Compute Freeman components
    Ps = fs * (1 + beta.real**2 + beta.imag**2)
    Pd = fd * (1 + alpha.real**2 + alpha.imag**2)
    Pv = fv

    # Compute span on __original__ covariance, not modified one
    sp = C3.m11.data + C3.m22.data + C3.m33.data
    min_span, max_span = da.nanmin(sp), da.nanmax(sp)
    min_span = da.maximum(min_span, eps)

    Ps = da.where(Ps <= min_span, min_span, da.where(Ps > max_span, max_span, Ps))
    Pd = da.where(Pd <= min_span, min_span, da.where(Pd > max_span, max_span, Pd))
    Pv = da.where(Pv <= min_span, min_span, da.where(Pv > max_span, max_span, Pv))
    return {"odd": Ps, "double": Pd, "volume": Pv}


def _compute_yamaguchi4_components(T3, mode="y4o"):

    eps = 1e-30

    span = T3.m11.data + T3.m22.data + T3.m33.data
    min_span = span.min()
    min_span = da.where(min_span > eps, min_span, eps)
    max_span = span.max()

    # Apply unitary rotation for corresponding modes
    if mode in ["y4r", "s4r"]:
        theta = 0.5 * da.atan(2 * T3.m23.data.real / (T3.m22.data - T3.m33.data))
        # T11, T22, T33, T12, T13, T23 = _unitary_rotation(T11, T22, T33, T12, T13, T23, theta)
        T3_new = _unitary_rotation(T3, theta)
    else:
        theta = da.zeros_like(T11)
        T3_new = T3.copy(deep=True)


    T11 = T3_new.m11.data
    T22 = T3_new.m22.data
    T33 = T3_new.m33.data

    T12 = T3_new.m12.data
    T13 = T3_new.m13.data
    T23 = T3_new.m23.data

    Pc = 2 * da.abs(T23.imag)

    # Surface scattering
    hv_type = da.ones_like(T11, dtype="uint8")

    if mode == "s4r":
        C1 = T11 - T22 + (7 / 8) * T33 + Pc / 16
        hv_type = da.where(C1 > 0, 1, 2)

    # Surface scattering
    ratio = 10(da.log10((T11 + T22 - 2 * T12.real) / (T11 + T22 + 2 * T12.real)))
    cnd = (hv_type == 1) & (ratio > -2) & (ratio <= 2)
    Pv = da.where(cnd, 2 * (2 * T33 - Pc), (15 / 8) * (2 * T33 - Pc))

    # Double bounce scattering
    Pv = da.where(hv_type == 2, (15 / 16) * (2 * T33 - Pc), Pv)

    TP = T11 + T22 + T33

    # 4 component algorithm

    C = T12 + T13
    amp_sq = (C * C.conj()).real  # |C|² (real-valued)

    # mask: True → volume scattering, False → double-bounce
    mask1 = hv_type == 1

    # -----------------------------------------------------------
    # Volume scattering (HV_type == 1)
    # -----------------------------------------------------------
    S1 = T11.real - Pv / 2.
    D1 = TP - Pv - Pc - S1

    # Adjust C according to ratio thresholds (real part only)
    C1 = C + (Pv / 6.) * ((ratio > 2.).astype(Pv.dtype) - (ratio <= -2.).astype(Pv.dtype))

    # (Pv + Pc) > TP 
    cond_pc = (Pv + Pc) > TP
    CO = 2. * T11.real + Pc - TP

    Ps1_else = da.where(CO > 0., S1 + amp_sq / S1, S1 - amp_sq / D1)
    Pd1_else = da.where(CO > 0., D1 - amp_sq / S1, D1 + amp_sq / D1)

    Ps1 = da.where(cond_pc, 0., Ps1_else)
    Pd1 = da.where(cond_pc, 0., Pd1_else)
    Pv1 = da.where(cond_pc, TP - Pc, Pv)

    # Handle Ps/Pd negativity
    cond_Ps_neg = Ps1 < 0
    cond_Pd_neg = Pd1 < 0

    Ps1 = da.where(
        cond_Ps_neg & cond_Pd_neg, 0.,
        da.where(cond_Ps_neg, 0.,
                da.where(cond_Pd_neg, TP - Pv1 - Pc, Ps1))
    )
    Pd1 = da.where(
        cond_Ps_neg & cond_Pd_neg, 0.,
        da.where(cond_Ps_neg, TP - Pv1 - Pc,
                da.where(cond_Pd_neg, 0., Pd1))
    )
    Pv1 = da.where(cond_Ps_neg & cond_Pd_neg, TP - Pc, Pv1)

    # -----------------------------------------------------------
    # Double-bounce scattering (HV_type == 2)
    # -----------------------------------------------------------
    S2 = T11.real
    D2 = TP - Pv - Pc - S2

    Pd2 = D2 + amp_sq / D2
    Ps2 = S2 - amp_sq / D2

    cond_Ps_neg = Ps2 < 0
    cond_Pd_neg = Pd2 < 0

    Ps2 = da.where(
        cond_Ps_neg & cond_Pd_neg, 0.,
        da.where(cond_Ps_neg, 0.,
                da.where(cond_Pd_neg, TP - Pv - Pc, Ps2))
    )
    Pd2 = da.where(
        cond_Ps_neg & cond_Pd_neg, 0.,
        da.where(cond_Ps_neg, TP - Pv - Pc,
                da.where(cond_Pd_neg, 0., Pd2))
    )
    Pv2 = da.where(cond_Ps_neg & cond_Pd_neg, TP - Pc, Pv)

    # -----------------------------------------------------------
    # Merge both cases (mask1 True → volume, False → double-bounce)
    # -----------------------------------------------------------
    Ps = da.where(mask1, Ps1, Ps2)
    Pd = da.where(mask1, Pd1, Pd2)
    Pv = da.where(mask1, Pv1, Pv2)
    Ps = da.where(Ps <= min_span, min_span, da.where(Ps > max_span, max_span, Ps))
    Pd = da.where(Pd <= min_span, min_span, da.where(Pd > max_span, max_span, Pd))
    Pv = da.where(Pv <= min_span, min_span, da.where(Pv > max_span, max_span, Pv))
    Pc = da.where(Pc <= min_span, min_span, da.where(Pc > max_span, max_span, Pc))

    # -----------------------------------------------------------
    # If Pv < 0 use three components and set Pc to 0
    # -----------------------------------------------------------
    out_3comp = _compute_yamaguchi3_components(T3_to_C3(T3_new))
    mask_v = Pv >= 0
    Ps = da.where(mask_v, Ps, out_3comp["odd"]) 
    Pd = da.where(mask_v, Pd, out_3comp["double"]) 
    Pv = da.where(mask_v, Pv, out_3comp["volume"]) 
    Pc = da.where(mask_v, Pc, 0) 
    # placeholder to remember names
    out = {"odd": Ps, "double": Pd, "volume": Pv, "helix": Pc}
    return out


# def _unitary_rotation(T11, T22, T33, T12, T13, T23, theta):
def _unitary_rotation(T3, theta):

    T3_out = {}

    c = 1 / np.sqrt(np.float32(2))

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    T11 = T3.m11.data.copy()
    T22 = T3.m22.data.copy()
    T33 = T3.m33.data.copy()
    # Decompose into real/imag
    T12_re, T12_im = da.real(T3.m12.data), da.imag(T3.m12.data)
    T13_re, T13_im = da.real(T3.m13.data), da.imag(T3.m13.data)
    T23_re, T23_im = da.real(T3.m23.data), da.imag(T3.m23.data)

    # --- Rotation ---
    T11_new = T11
    T12_re_new = T12_re * cos_t + T13_re * sin_t
    T12_im_new = T12_im * cos_t + T13_im * sin_t
    T13_re_new = -T12_re * sin_t + T13_re * cos_t
    T13_im_new = -T12_im * sin_t + T13_im * cos_t

    T22_new = T22 * cos_t**2 + 2.0 * T23_re * cos_t * sin_t + T33 * sin_t**2
    T23_re_new = (
        -T22 * cos_t * sin_t + T23_re * (cos_t**2 - sin_t**2) + T33 * cos_t * sin_t
    )
    T23_im_new = T23_im * (cos_t**2 + sin_t**2)
    T33_new = T22 * sin_t**2 + T33 * cos_t**2 - 2.0 * T23_re * cos_t * sin_t

    # Recombine
    T3_out["m11"] = T11_new 
    T3_out["m22"] = T22_new
    T3_out["m33"] = T33_new
    T3_out["m12"] = T12_re_new + 1j * T12_im_new
    T3_out["m13"] = T13_re_new + 1j * T13_im_new
    T3_out["m23"] = T23_re_new + 1j * T23_im_new

    attrs = {"poltype": "T3", "description": "Coherency matrix (3x3)"}
    return xr.Dataset(T3, attrs=attrs)
    # return T11, T22_new, T33_new, T12_new, T13_new, T23_new
