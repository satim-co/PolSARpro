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
"""

import numpy as np
import xarray as xr
import dask.array as da
from polsarpro.util import (
    boxcar_xarray,
    C3_to_T3_xarray,
    S_to_T3_xarray,
)


def h_a_alpha_xarray(
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
        input_data (xr.Dataset): The input polarimetric SAR data array. Expected to represent
            a 3x3 matrix (or 2x2 in the case of Sinclair) per pixel, typically with shape
            (..., 3, 3) or (..., 2, 2) depending on `input_poltype`.
        input_poltype (str, optional): The polarimetric basis of the input data. Supported types are:
            - "C3": Lexicographic covariance matrix
            - "T3": Pauli coherency matrix
            - "S": Sinclair scattering matrix
            Defaults to "C3".
        boxcar_size (list[int, int], optional): Size of the spatial averaging window to be
            applied before decomposition (boxcar filter). Defaults to [3, 3].
        flags (tuple[str], optional): Parameters to compute and return from the decomposition.
            Possible values include:
            - "entropy": Scattering entropy (H)
            - "anisotropy": Scattering anisotropy (A)
            - "alpha": Mean alpha scattering angle (alpha)
            - "beta", "delta", "gamma", "lambda": Other angular or eigenvalue related parameters
            - "alphas", "betas", "deltas", "gammas", "lambdas": Per-eigenvalue versions of the above
            Defaults to ("entropy", "alpha", "anisotropy").

    Returns:
        xr.Dataset: An xarray.Dataset where data variable names correspond to the requested flags,
        and values are the corresponding 2D arrays (or 3D if the flag returns multiple values per pixel).

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
        "gamma",
        "lambda",
        "alphas",
        "betas",
        "deltas",
        "gammas",
        "lambdas",
    )
    for flag in flags:
        if flag not in possible_flags:
            raise ValueError(
                f"Flag '{flag}' not recognized. Possible values are {possible_flags}."
            )

    if not isinstance(input_data, xr.Dataset):
        TypeError("Inputs must be of type xarray.Dataset")

    if not "poltype" in input_data.attrs:
        ValueError("Polarimetric type `poltype` not found in input attributes.")

    if input_data.poltype == "C3":
        in_ = C3_to_T3_xarray(input_data)
    elif input_data.poltype == "T3":
        in_ = input_data
    elif input_data.poltype == "S":
        in_ = S_to_T3_xarray(input_data)
    else:
        raise ValueError(f"Invalid polarimetric type: {input_data.poltype}")

    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = boxcar_xarray(in_, dim_az=boxcar_size[0], dim_rg=boxcar_size[1])

    # form a T3 matrix array that can be used in eigh
    # concatenate columns
    T3_l1 = xr.concat((in_.m11, in_.m12, in_.m13), dim="j")
    T3_l2 = xr.concat((in_.m12.conj(), in_.m22, in_.m23), dim="j")
    T3_l3 = xr.concat((in_.m13.conj(), in_.m23.conj(), in_.m33), dim="j")
    # concatenate lines
    T3 = (
        xr.concat((T3_l1, T3_l2, T3_l3), dim="i")
        .transpose("y", "x", "i", "j")
        .chunk(dict(y="auto", x="auto", i=3, j=3))
    )
    # Eigendecomposition
    meta = (
        np.array([], dtype="float32").reshape((0, 0, 0)),
        np.array([], dtype="complex64").reshape((0, 0, 0, 0)),
    )
    l, v = da.apply_gufunc(np.linalg.eigh, "(i,j)->(i), (i,j)", T3, meta=meta)

    l = l[..., ::-1]  # descending order
    v = v[..., ::-1]

    # returns a dict
    out = _compute_h_a_alpha_parameters(l, v, flags)
    return xr.Dataset(
        # add dimension names, account for 2D and 3D outputs
        {
            k: (["y", "x"], v) if v.ndim == 2 else (["y", "x", "i"], v)
            for k, v in out.items()
        },
        attrs=dict(
            poltype="h_a_alpha", description="Results of the H/A/Alpha decomposition."
        ),
    )

def _compute_h_a_alpha_parameters(l, v, flags):

    eps = 1e-30

    # Pseudo-probabilities (normalized eigenvalues)
    p = np.clip(l / (eps + l.sum(axis=2)[..., None]), eps, 1)

    outputs = {}
    if "entropy" in flags:
        H = np.sum(-p * np.log(p), axis=2) / np.float32(np.log(3))
        outputs["entropy"] = H

    if "anisotropy" in flags:
        A = (l[..., 1] - l[..., 2]) / (l[..., 1] + l[..., 2] + eps)
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
