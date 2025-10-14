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
# from bench import timeit
import xarray as xr
import xarray

log = logging.getLogger(__name__)


# @timeit
def boxcar(img: np.ndarray, dim_az: int, dim_rg: int) -> np.ndarray:
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
    if type(dim_az) != int and type(dim_rg) != int:
        raise ValueError("dimaz and dimrg must be integers")
    if (dim_az < 1) or (dim_rg < 1):
        raise ValueError("dimaz and dimrg must be strictly positive")
    if img.ndim < 2:
        raise ValueError("Input must be at least of dimension 2")

    return _boxcar_core(img, dim_az, dim_rg)


# @timeit
def boxcar_dask(img: np.ndarray, dim_az: int, dim_rg: int) -> np.ndarray:
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
    if type(dim_az) != int and type(dim_rg) != int:
        raise ValueError("dimaz and dimrg must be integers")
    if (dim_az < 1) or (dim_rg < 1):
        raise ValueError("dimaz and dimrg must be strictly positive")
    if img.ndim < 2:
        raise ValueError("Input must be at least of dimension 2")

    process_args = dict(
        dim_az=dim_az,
        dim_rg=dim_rg,
        depth=(dim_az, dim_rg),
    )
    da_in = da.map_overlap(
        _boxcar_core,
        # da.from_array(img, chunks=(500, 500, 3, 3)),
        da.from_array(img, chunks="auto"),
        **process_args,
        dtype="complex64",
    )

    return np.asarray(da_in)


def boxcar_xarray(img: xarray.Dataset, dim_az: int, dim_rg: int) -> xarray.Dataset:
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


def _boxcar_core(img: np.ndarray, dim_az: int, dim_rg: int) -> np.ndarray:
    n_extra_dims = img.ndim - 2

    ker_dtype = img.dtype if not np.iscomplexobj(img) else img.real.dtype

    # this convolution mode reduces error between C and python implementations
    mode = "constant"
    if (dim_az > 1) or (dim_rg > 1):
        # avoid nan propagation
        msk = np.isnan(img)
        img_ = img.copy()
        img_[msk] = 0
        ker = np.ones((dim_az, dim_rg), dtype=ker_dtype) / (dim_az * dim_rg)
        ker = np.expand_dims(ker, axis=tuple(range(2, 2 + n_extra_dims)))
        if np.iscomplexobj(img_):
            imgout = convolve(img_.real, ker, mode=mode) + 1j * convolve(
                img_.imag, ker, mode=mode
            )
            imgout[msk] = np.nan + 1j * np.nan
        else:
            imgout = convolve(img_, ker, mode=mode)
            imgout[msk] = np.nan
        return imgout
    else:
        return img


# @timeit
def multilook(img: np.ndarray, dim_az: int, dim_rg: int) -> np.ndarray:
    """
    Computes the m by n presummed image.

    Args:
        img (array-like): Input image array with shape (naz, nrg,...).
        dim_az (int): Number of lines to sum. Must be an integer >= 1.
        dim_rg (int): Number of columns to sum. Must be an integer >= 1.

    Returns:
        array: Presummed image array with shape (M, N,...), where M and N are the largest multiples of dim_az and dim_rg that are less than or equal to img.shape[0] and img.shape[1], respectively.
    Note:
        Returns the input array if dim_az==1 and dim_rg==1.
    """
    # Check if m and n are integers >= 1
    if not isinstance(dim_az, int) or not isinstance(dim_rg, int):
        raise TypeError("Parameters m and n must be integers.")
    if dim_az < 1 or dim_rg < 1:
        raise ValueError(
            "Parameters m and n must be integers greater than or equal to 1."
        )

    # Check if m and n are valid in relation to the image dimensions
    if dim_az > img.shape[0] or dim_rg > img.shape[1]:
        raise ValueError(
            "Cannot presum with these parameters; m or n is too large for the image dimensions."
        )

    # skip if m = n = 1, avoids conditionals in calls
    if (dim_az > 1) or (dim_rg > 1):
        M = (img.shape[0] // dim_az) * dim_az
        N = (img.shape[1] // dim_rg) * dim_rg

        img_trimmed = img[:M, :N]

        s = img_trimmed[::dim_az].copy()  # Make a copy once for efficiency
        for i in range(1, dim_az):
            s += img_trimmed[i::dim_az]

        t = s[:, ::dim_rg].copy()
        for j in range(1, dim_rg):
            t += s[:, j::dim_rg]

        return t / float(dim_az * dim_rg)
    else:
        return img


def S_to_C3(S: np.ndarray) -> np.ndarray:
    """Converts the Sinclair scattering matrix S to the lexicographic covariance matrix C3.

    Args:
        S (np.ndarray): input image of scattering matrices with shape (naz, nrg, 2, 2)

    Returns:
        np.ndarray: C3 covariance matrix
    """
    if np.isrealobj(S):
        raise ValueError("Inputs must be complex-valued.")
    if S.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if S.shape[-2:] != (2, 2):
        raise ValueError("S must have a shape like (naz, nrg, 2, 2)")
    return _S_to_C3_core(S)


def S_to_C3_dask(S: np.ndarray) -> np.ndarray:
    """Converts the Sinclair scattering matrix S to the lexicographic covariance matrix C3.

    Args:
        S (np.ndarray): input image of scattering matrices with shape (naz, nrg, 2, 2)

    Returns:
        np.ndarray: C3 covariance matrix
    """
    if np.isrealobj(S):
        raise ValueError("Inputs must be complex-valued.")
    if S.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if S.shape[-2:] != (2, 2):
        raise ValueError("S must have a shape like (naz, nrg, 2, 2)")

    da_in = da.map_blocks(
        _S_to_C3_core,
        da.from_array(S, chunks="auto"),
        # da.from_array(S, chunks=(500, 500, -1, -1)),
        dtype="complex64",
    )

    return np.asarray(da_in)


def S_to_C3_xarray(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the lexicographic covariance matrix C3.

    Args:
        S (xarray.Dataset): input image of scattering matrices with shape

    Returns:
        xarray.Dataset: C3 covariance matrix
    """

    if S.poltype == "S":
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
    else:
        raise ValueError("Input polarimetric type must be 'S'")


def _S_to_C3_core(S: np.ndarray) -> np.ndarray:

    k = np.dstack(
        (S[..., 0, 0], (1.0 / np.sqrt(2)) * (S[..., 0, 1] + S[..., 1, 0]), S[..., 1, 1])
    )

    return vec_to_mat(k).astype("complex64")


def S_to_T3(S: np.ndarray) -> np.ndarray:
    """Converts the Sinclair scattering matrix S to the Pauli coherency matrix T3.

    Args:
        S (np.ndarray): input image of scattering matrices with shape (naz, nrg, 2, 2)

    Returns:
        np.ndarray: T3 coherency matrix
    """
    if S.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if S.shape[-2:] != (2, 2):
        raise ValueError("S must have a shape like (naz, nrg, 2, 2)")
    return _S_to_T3_core(S)


def S_to_T3_dask(S: np.ndarray) -> np.ndarray:
    """Converts the Sinclair scattering matrix S to the Pauli coherency matrix T3.

    Args:
        S (np.ndarray): input image of scattering matrices with shape (naz, nrg, 2, 2)

    Returns:
        np.ndarray: T3 coherency matrix
    """
    if S.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if S.shape[-2:] != (2, 2):
        raise ValueError("S must have a shape like (naz, nrg, 2, 2)")

    da_in = da.map_blocks(
        _S_to_T3_core,
        da.from_array(S, chunks="auto"),
        # da.from_array(S, chunks=(500, 500, -1, -1)),
        dtype="complex64",
    )

    return np.asarray(da_in)


def S_to_T3_xarray(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the Pauli coherency matrix T3.

    Args:
        S (xarray.Dataset): input image of scattering matrices with shape

    Returns:
        xarray.Dataset: T3 covariance matrix
    """

    if S.poltype == "S":
        # scattering vector
        c = np.sqrt(np.float32(2))
        k1 = ((S.hh + S.vv) / c).astype("complex64", copy=False)
        k2 = ((S.hh - S.vv) / c).astype("complex64", copy=False)
        k3 = (c * S.hv).astype("complex64", copy=False)

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
    else:
        raise ValueError("Input polarimetric type must be 'S'")


def _S_to_T3_core(S: np.ndarray) -> np.ndarray:

    k = np.dstack(
        (S[..., 0, 0] + S[..., 1, 1], S[..., 0, 0] - S[..., 1, 1], 2 * S[..., 0, 1])
    ) / np.sqrt(2)

    return vec_to_mat(k).astype("complex64")


# @timeit
def T3_to_C3(T3: np.ndarray) -> np.ndarray:
    """Converts the Pauli coherency matrix T3 to the lexicographic covariance matrix C3.

    Args:
        T3 (np.ndarray): input image of coherency matrices with shape (naz, nrg, 3, 3)

    Returns:
        np.ndarray: C3 covariance matrix
    """
    if np.isrealobj(T3):
        raise ValueError("Inputs must be complex-valued.")
    if T3.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if T3.shape[-2:] != (3, 3):
        raise ValueError("T3 must have a shape like (naz, nrg, 3, 3)")

    return _T3_to_C3_core(T3=T3)


# @timeit
def T3_to_C3_dask(T3: np.ndarray) -> np.ndarray:
    """Converts the Pauli coherency matrix T3 to the lexicographic covariance matrix C3.

    Args:
        T3 (np.ndarray): input image of coherency matrices with shape (naz, nrg, 3, 3)

    Returns:
        np.ndarray: C3 covariance matrix
    """
    if np.isrealobj(T3):
        raise ValueError("Inputs must be complex-valued.")
    if T3.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if T3.shape[-2:] != (3, 3):
        raise ValueError("T3 must have a shape like (naz, nrg, 3, 3)")

    da_in = da.map_blocks(
        _T3_to_C3_core,
        # da.from_array(T3, chunks=(500, 500, -1, -1)),
        da.from_array(T3, chunks="auto"),
        dtype="complex64",
    )

    return np.asarray(da_in)


def T3_to_C3_xarray(T3: xarray.Dataset) -> xarray.Dataset:
    """Converts the Pauli coherency matrix T3 to the lexicographic covariance matrix C3.

    Args:
        T3 (xarray.Dataset): input image of coherency matrices

    Returns:
        xarray.Dataset: C3 covariance matrix
    """

    if T3.poltype == "T3":

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

        # TODO: remove once the function has been tested
        # T3.m13.real - 1j * T3.m13.imag - T3.m23.real + 1j * T3.m23.imag
        # C3[..., 0, 0] = (T3[..., 0, 0] + 2 * T3[..., 0, 1].real + T3[..., 1, 1]) / 2
        # C3[..., 0, 1] = (T3[..., 0, 2] + T3[..., 1, 2]) / np.sqrt(2)
        # C3[..., 0, 2] = (T3[..., 0, 0].real - T3[..., 1, 1].real) / 2
        # C3[..., 0, 2] += 1j * -T3[..., 0, 1].imag
        # C3[..., 1, 1] = T3.m33
        # C3[..., 1, 2] = (T3[..., 0, 2].real - T3[..., 1, 2].real) / np.sqrt(2)
        # C3[..., 1, 2] += 1j * (-T3[..., 0, 2].imag + T3[..., 1, 2].imag) / np.sqrt(2)
        # C3[..., 2, 2] = (T3[..., 0, 0] - 2 * T3[..., 0, 1].real + T3[..., 1, 1]) / 2

    else:
        raise ValueError("Input polarimetric type must be 'T3'")


def _T3_to_C3_core(T3: np.ndarray) -> np.ndarray:

    C3 = np.zeros_like(T3, dtype="complex64")

    # Reproject T3 matrix in the lexicographic basis
    C3[..., 0, 0] = (T3[..., 0, 0] + 2 * T3[..., 0, 1].real + T3[..., 1, 1]) / 2
    C3[..., 0, 1] = (T3[..., 0, 2] + T3[..., 1, 2]) / np.sqrt(2)
    C3[..., 0, 2] = (T3[..., 0, 0].real - T3[..., 1, 1].real) / 2
    C3[..., 0, 2] += 1j * -T3[..., 0, 1].imag
    C3[..., 1, 1] = T3[..., 2, 2]
    C3[..., 1, 2] = (T3[..., 0, 2].real - T3[..., 1, 2].real) / np.sqrt(2)
    C3[..., 1, 2] += 1j * (-T3[..., 0, 2].imag + T3[..., 1, 2].imag) / np.sqrt(2)
    C3[..., 2, 2] = (T3[..., 0, 0] - 2 * T3[..., 0, 1].real + T3[..., 1, 1]) / 2

    C3[..., 1, 0] = np.conj(C3[..., 0, 1])
    C3[..., 2, 0] = np.conj(C3[..., 0, 2])
    C3[..., 2, 1] = np.conj(C3[..., 1, 2])

    return C3


def C3_to_T3(C3: np.ndarray) -> np.ndarray:
    """Converts the lexicographic covariance matrix T3 to the Pauli coherency matrix T3.

    Args:
        C3 (np.ndarray): input image of coherency matrices with shape (naz, nrg, 3, 3)

    Returns:
        np.ndarray: T3 coherency matrix
    """
    if np.isrealobj(C3):
        raise ValueError("Inputs must be complex-valued.")
    if C3.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if C3.shape[-2:] != (3, 3):
        raise ValueError("C3 must have a shape like (naz, nrg, 3, 3)")

    return _C3_to_T3_core(C3=C3)


def C3_to_T3_dask(C3: np.ndarray) -> np.ndarray:
    """Converts the lexicographic covariance matrix C3 to the Pauli coherency matrix T3.

    Args:
        C3 (np.ndarray): input image of coherency matrices with shape (naz, nrg, 3,

    Returns:
        np.ndarray: T3 coherency matrix
    """
    if np.isrealobj(C3):
        raise ValueError("Inputs must be complex-valued.")
    if C3.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if C3.shape[-2:] != (3, 3):
        raise ValueError("C3 must have a shape like (naz, nrg, 3, 3)")

    da_in = da.map_blocks(
        _C3_to_T3_core,
        da.from_array(C3, chunks="auto"),
        # da.from_array(C3, chunks=(500, 500, -1, -1)),
        dtype="complex64",
    )

    return np.asarray(da_in)


def C3_to_T3_xarray(C3: xarray.Dataset) -> xarray.Dataset:
    """Converts the lexicographic covariance matrix C3 to the Pauli coherency matrix T3.

    Args:
        C3 (xarray.Dataset): input image of covariance matrices

    Returns:
        xarray.Dataset: T3 coherency matrix
    """

    if C3.poltype == "C3":

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

        # TODO: remove once the function has been tested
        # T3[..., 0, 0] = 0.5 * (C3[..., 0, 0] + 2 * C3[..., 0, 2].real + C3[..., 2, 2])
        # T3[..., 0, 1] = 0.5 * (C3[..., 0, 0] - C3[..., 2, 2]) - 1j * C3[..., 0, 2].imag
        # T3[..., 0, 2] = (C3[..., 0, 1] + np.conj(C3[..., 1, 2])) / np.sqrt(2)
        # T3[..., 1, 1] = 0.5 * (C3[..., 0, 0] - 2 * C3[..., 0, 2].real + C3[..., 2, 2])
        # T3[..., 1, 2] = (C3[..., 0, 1] - np.conj(C3[..., 1, 2])) / np.sqrt(2)
        # T3[..., 2, 2] = C3[..., 1, 1]

    else:
        raise ValueError("Input polarimetric type must be 'T3'")


def _C3_to_T3_core(C3: np.ndarray) -> np.ndarray:
    T3 = np.zeros_like(C3, dtype="complex64")

    # Reproject T3 matrix in the lexicographic basis
    T3[..., 0, 0] = 0.5 * (C3[..., 0, 0] + 2 * C3[..., 0, 2].real + C3[..., 2, 2])
    T3[..., 0, 1] = 0.5 * (C3[..., 0, 0] - C3[..., 2, 2]) - 1j * C3[..., 0, 2].imag
    T3[..., 0, 2] = (C3[..., 0, 1] + np.conj(C3[..., 1, 2])) / np.sqrt(2)
    T3[..., 1, 1] = 0.5 * (C3[..., 0, 0] - 2 * C3[..., 0, 2].real + C3[..., 2, 2])
    T3[..., 1, 2] = (C3[..., 0, 1] - np.conj(C3[..., 1, 2])) / np.sqrt(2)
    T3[..., 2, 2] = C3[..., 1, 1]

    T3[..., 1, 0] = np.conj(T3[..., 0, 1])
    T3[..., 2, 0] = np.conj(T3[..., 0, 2])
    T3[..., 2, 1] = np.conj(T3[..., 1, 2])

    return T3


def vec_to_mat(vec: np.ndarray) -> np.ndarray:
    """Vector to matrix conversion. Input should have (naz, nrg, N) shape"""
    if vec.ndim != 3:
        raise ValueError("Vector valued image is expected (dimension 3)")
    return vec[:, :, None, :] * vec[:, :, :, None].conj()


# @timeit
def span(M: np.ndarray) -> np.ndarray:
    """Span computation for a image of matrices. Input should have (naz, nrg, N, N) shape"""
    if M.ndim != 4:
        raise ValueError("Matrix valued image is expected (dimension 4)")
    if M.shape[2] != M.shape[3]:
        raise ValueError("Input shape (naz, nrg, N, N) expected")

    return np.diagonal(M, axis1=2, axis2=3).real.sum(axis=-1)
