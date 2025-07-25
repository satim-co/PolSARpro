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

import logging
import numpy as np
from scipy.ndimage import convolve
import dask.array as da
from bench import timeit
import xarray as xr
import xarray

log = logging.getLogger(__name__)


def boxcar(img: xarray.Dataset, dim_az: int, dim_rg: int) -> xarray.Dataset:
    """
    Apply a boxcar filter to an image.

    Args:
        img (complex or real array): Input image with arbitrary number of dimensions, shape (naz, nrg, ...).
        dim_az (int): Size in azimuth of the filter.
        dim_rg (int): Size in range of the filter.

    Returns:
        complex or real array: Filtered image, shape (naz, nrg, ...).

    Note:
        The filter is always applied along 2 dimensions (azimuth, range). Please ensure to provide a valid image.
    """
    if "x" in img.dims and "y" in img.dims:
        # pad the data with zeros
        res = img.pad(dict(x=dim_rg, y=dim_az), mode="constant", constant_values=0)
        # compute rolling mean
        res = res.rolling(x=dim_rg, y=dim_az, center=True).mean()
        # trim the padded borders
        res = res.isel(x=slice(dim_rg, -dim_rg), y=slice(dim_az, -dim_az))
        return res
        # return img.rolling(x=dim_rg, y=dim_az, center=True).mean()
    else:
        raise ValueError("'x' and 'y' must be in the dimensions of the input data.")


def S_to_C3(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the lexicographic covariance matrix C3.

    Args:
        S (xarray.Dataset): input image of scattering matrices with shape

    Returns:
        xarray.Dataset: C3 covariance matrix
    """

    if S.poltype != "S":
        raise ValueError("Input polarimetric type must be 'S'")

    # scattering vector, enforce type as in C version
    c = np.sqrt(np.float32(2))
    k1 = S.hh.astype("complex64", copy=False)
    k2 = ((S.hv + S.vh) / c).astype("complex64", copy=False)
    k3 = S.vv.astype("complex64", copy=False)

    # compute the Hermitian matrix elements
    C3_dict = {}

    # force real diagonal to save space
    C3_dict["m11"] = (k1 * k1.conj()).real
    C3_dict["m22"] = (k2 * k2.conj()).real
    C3_dict["m33"] = (k3 * k3.conj()).real

    # upper diagonal terms
    C3_dict["m12"] = k1 * k2.conj()
    C3_dict["m13"] = k1 * k3.conj()
    C3_dict["m23"] = k2 * k3.conj()

    attrs = {"poltype": "C3", "description": "Covariance matrix (3x3)"}
    return xr.Dataset(C3_dict, attrs=attrs)


def S_to_T3(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the Pauli coherency matrix T3.

    Args:
        S (xarray.Dataset): input image of scattering matrices with shape

    Returns:
        xarray.Dataset: T3 covariance matrix
    """

    if S.poltype != "S":
        raise ValueError("Input polarimetric type must be 'S'")

    # scattering vector
    c = np.sqrt(np.float32(2))
    k1 = ((S.hh + S.vv) / c).astype("complex64", copy=False)
    k2 = ((S.hh - S.vv) / c).astype("complex64", copy=False)
    k3 = ((S.hv + S.vh) / c).astype("complex64", copy=False)
    # k3 = (c * S.hv).astype("complex64", copy=False)

    # compute the Hermitian matrix elements
    T3_dict = {}

    # force real diagonal to save space
    T3_dict["m11"] = (k1 * k1.conj()).real
    T3_dict["m22"] = (k2 * k2.conj()).real
    T3_dict["m33"] = (k3 * k3.conj()).real

    # upper diagonal terms
    T3_dict["m12"] = k1 * k2.conj()
    T3_dict["m13"] = k1 * k3.conj()
    T3_dict["m23"] = k2 * k3.conj()

    attrs = {"poltype": "T3", "description": "Coherency matrix (3x3)"}
    return xr.Dataset(T3_dict, attrs=attrs)


def T3_to_C3(T3: xarray.Dataset) -> xarray.Dataset:
    """Converts the Pauli coherency matrix T3 to the lexicographic covariance matrix C3.

    Args:
        T3 (xarray.Dataset): input image of coherency matrices

    Returns:
        xarray.Dataset: C3 covariance matrix
    """

    if T3.poltype != "T3":
        raise ValueError("Input polarimetric type must be 'T3'")

    C3_dict = {}

    c = 1 / np.sqrt(np.float32(2))

    # force real diagonal to save space
    C3_dict["m11"] = 0.5 * (T3.m11 + T3.m22) + T3.m12.real
    C3_dict["m22"] = T3.m33
    C3_dict["m33"] = 0.5 * (T3.m11 + T3.m22) - T3.m12.real

    # upper diagonal terms
    C3_dict["m12"] = c * (T3.m13 + T3.m23)
    C3_dict["m13"] = 0.5 * (T3.m11.real - T3.m22.real) - 1j * T3.m12.imag
    C3_dict["m23"] = c * (T3.m13.conj() - T3.m23.conj())

    attrs = {"poltype": "C3", "description": "Covariance matrix (3x3)"}
    return xr.Dataset(C3_dict, attrs=attrs)


def C3_to_T3(C3: xarray.Dataset) -> xarray.Dataset:
    """Converts the lexicographic covariance matrix C3 to the Pauli coherency matrix T3.

    Args:
        C3 (xarray.Dataset): input image of covariance matrices

    Returns:
        xarray.Dataset: T3 coherency matrix
    """

    if C3.poltype != "C3":
        raise ValueError("Input polarimetric type must be 'T3'")

    T3_dict = {}

    c = 1 / np.sqrt(np.float32(2))

    # force real diagonal to save space
    T3_dict["m11"] = 0.5 * (C3.m11 + C3.m33) + C3.m13.real
    T3_dict["m22"] = 0.5 * (C3.m11 + C3.m33) - C3.m13.real
    T3_dict["m33"] = C3.m22
    # upper diagonal terms
    T3_dict["m12"] = 0.5 * (C3.m11 - C3.m33) - 1j * C3.m13.imag
    T3_dict["m13"] = c * (C3.m12 + C3.m23.conj())
    T3_dict["m23"] = c * (C3.m12 - C3.m23.conj())

    attrs = {"poltype": "T3", "description": "Coherency matrix (3x3)"}
    return xr.Dataset(T3_dict, attrs=attrs)


def vec_to_mat(vec: np.ndarray) -> np.ndarray:
    """Vector to matrix conversion. Input should have (naz, nrg, N) shape"""
    if vec.ndim != 3:
        raise ValueError("Vector valued image is expected (dimension 3)")
    return vec[:, :, None, :] * vec[:, :, :, None].conj()
