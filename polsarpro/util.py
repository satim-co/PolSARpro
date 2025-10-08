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

# Description: module containing various utility functions

"""

import logging
import numpy as np
from scipy.ndimage import convolve
import dask.array as da
import xarray as xr
import xarray
import matplotlib.pyplot as plt
from polsarpro.auxil import validate_dataset

log = logging.getLogger(__name__)


def S_to_C2(S: xarray.Dataset, p1: str = "hh", p2: str = "hv") -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the lexicographic dual polarization covariance matrix C2.

    Args:
        S (xarray.Dataset): input image of scattering matrices
        p1 (str): first polarization.
        p2 (str): second polarization.

    Returns:
        xarray.Dataset: C2 covariance matrix
    Note:
        p1 and p2 must be different and belong to {'hh', 'hv', 'vv', 'vh'}
    """

    validate_dataset(S, allowed_poltypes="S")
    # if S.poltype != "S":
        # raise ValueError("Input polarimetric type must be 'S'")

    pols = {"hh", "hv", "vv", "vh"}
    if not {p1, p2}.issubset(pols):
        raise ValueError(f"Polarizations need to be in {pols}")

    if p1 == p2:
        raise ValueError(f"Polarizations must be different.")

    # scattering vector, enforce type as in C version
    k1 = S[p1].astype("complex64", copy=False)
    k2 = S[p2].astype("complex64", copy=False)

    # compute the Hermitian matrix elements
    C2 = {}

    # force real diagonal to save space
    C2["m11"] = (k1 * k1.conj()).real
    C2["m22"] = (k2 * k2.conj()).real

    # upper diagonal terms
    C2["m12"] = k1 * k2.conj()

    attrs = {
        "poltype": "C2",
        "description": f"Covariance matrix (2x2), with polarizations {p1} and {p2}",
    }
    return xr.Dataset(C2, attrs=attrs)


def S_to_C3(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the lexicographic covariance matrix C3.

    Args:
        S (xarray.Dataset): input image of scattering matrices

    Returns:
        xarray.Dataset: C3 covariance matrix
    """
    validate_dataset(S, allowed_poltypes="S")
    # if S.poltype != "S":
    #     raise ValueError("Input polarimetric type must be 'S'")

    # scattering vector, enforce type as in C version
    c = np.sqrt(np.float32(2))
    k1 = S.hh.astype("complex64", copy=False)
    k2 = ((S.hv + S.vh) / c).astype("complex64", copy=False)
    k3 = S.vv.astype("complex64", copy=False)

    # compute the Hermitian matrix elements
    C3 = {}

    # force real diagonal to save space
    C3["m11"] = (k1 * k1.conj()).real
    C3["m22"] = (k2 * k2.conj()).real
    C3["m33"] = (k3 * k3.conj()).real

    # upper diagonal terms
    C3["m12"] = k1 * k2.conj()
    C3["m13"] = k1 * k3.conj()
    C3["m23"] = k2 * k3.conj()

    attrs = {"poltype": "C3", "description": "Covariance matrix (3x3)"}
    return xr.Dataset(C3, attrs=attrs)


def S_to_C4(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the lexicographic covariance matrix C4.

    Args:
        S (xarray.Dataset): input image of scattering matrices

    Returns:
        xarray.Dataset: C4 covariance matrix
    """
    validate_dataset(S, allowed_poltypes="S")
    # if S.poltype != "S":
    #     raise ValueError("Input polarimetric type must be 'S'")

    # scattering vector, enforce type as in C version
    k1 = S.hh.astype("complex64", copy=False)
    k2 = S.hv.astype("complex64", copy=False)
    k3 = S.vh.astype("complex64", copy=False)
    k4 = S.vv.astype("complex64", copy=False)

    # compute the Hermitian matrix elements
    C4 = {}
    # force real diagonal to save space
    C4["m11"] = (k1 * k1.conj()).real
    C4["m22"] = (k2 * k2.conj()).real
    C4["m33"] = (k3 * k3.conj()).real
    C4["m44"] = (k4 * k4.conj()).real

    # upper diagonal terms
    C4["m12"] = k1 * k2.conj()
    C4["m13"] = k1 * k3.conj()
    C4["m14"] = k1 * k4.conj()
    C4["m23"] = k2 * k3.conj()
    C4["m24"] = k2 * k4.conj()
    C4["m34"] = k3 * k4.conj()

    attrs = {"poltype": "C4", "description": "Covariance matrix (4x4)"}
    return xr.Dataset(C4, attrs=attrs)


def S_to_T3(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the Pauli coherency matrix T3.

    Args:
        S (xarray.Dataset): input image of scattering matrices

    Returns:
        xarray.Dataset: T3 covariance matrix
    """
    validate_dataset(S, allowed_poltypes="S")
    # if S.poltype != "S":
    #     raise ValueError("Input polarimetric type must be 'S'")

    # scattering vector
    c = np.sqrt(np.float32(2))
    k1 = ((S.hh + S.vv) / c).astype("complex64", copy=False)
    k2 = ((S.hh - S.vv) / c).astype("complex64", copy=False)
    k3 = ((S.hv + S.vh) / c).astype("complex64", copy=False)

    # compute the Hermitian matrix elements
    T3 = {}

    # force real diagonal to save space
    T3["m11"] = (k1 * k1.conj()).real
    T3["m22"] = (k2 * k2.conj()).real
    T3["m33"] = (k3 * k3.conj()).real

    # upper diagonal terms
    T3["m12"] = k1 * k2.conj()
    T3["m13"] = k1 * k3.conj()
    T3["m23"] = k2 * k3.conj()

    attrs = {"poltype": "T3", "description": "Coherency matrix (3x3)"}
    return xr.Dataset(T3, attrs=attrs)


def S_to_T4(S: xarray.Dataset) -> xarray.Dataset:
    """Converts the Sinclair scattering matrix S to the Pauli coherency matrix T4.

    Args:
        S (xarray.Dataset): input image of scattering matrices

    Returns:
        xarray.Dataset: T4 coherency matrix
    """
    validate_dataset(S, allowed_poltypes="S")
    # if S.poltype != "S":
    #     raise ValueError("Input polarimetric type must be 'S'")

    # scattering vector
    c = np.sqrt(np.float32(2))
    k1 = ((S.hh + S.vv) / c).astype("complex64", copy=False)
    k2 = ((S.hh - S.vv) / c).astype("complex64", copy=False)
    k3 = ((S.hv + S.vh) / c).astype("complex64", copy=False)
    # NOTE: it might work only with ALOS-1 (hv-vh is inverted, according to C-PSP)
    k4 = (1j * (S.vh - S.hv) / c).astype("complex64", copy=False)
    # alternate version
    # k4 = (1j * (S.hv - S.vh) / c).astype("complex64", copy=False)

    # compute the Hermitian matrix elements
    T4 = {}

    # force real diagonal to save space
    T4["m11"] = (k1 * k1.conj()).real
    T4["m22"] = (k2 * k2.conj()).real
    T4["m33"] = (k3 * k3.conj()).real
    T4["m44"] = (k4 * k4.conj()).real

    # upper diagonal terms
    T4["m12"] = k1 * k2.conj()
    T4["m13"] = k1 * k3.conj()
    T4["m14"] = k1 * k4.conj()
    T4["m23"] = k2 * k3.conj()
    T4["m24"] = k2 * k4.conj()
    T4["m34"] = k3 * k4.conj()

    attrs = {"poltype": "T4", "description": "Coherency matrix (4x4)"}
    return xr.Dataset(T4, attrs=attrs)


def T3_to_C3(T3: xarray.Dataset) -> xarray.Dataset:
    """Converts the Pauli coherency matrix T3 to the lexicographic covariance matrix C3.

    Args:
        T3 (xarray.Dataset): input image of coherency matrices

    Returns:
        xarray.Dataset: C3 covariance matrix
    """

    validate_dataset(T3, allowed_poltypes="T3")
    # if T3.poltype != "T3":
        # raise ValueError("Input polarimetric type must be 'T3'")

    C3 = {}

    c = 1 / np.sqrt(np.float32(2))

    # force real diagonal to save space
    C3["m11"] = 0.5 * (T3.m11 + T3.m22) + T3.m12.real
    C3["m22"] = T3.m33
    C3["m33"] = 0.5 * (T3.m11 + T3.m22) - T3.m12.real

    # upper diagonal terms
    C3["m12"] = c * (T3.m13 + T3.m23)
    C3["m13"] = 0.5 * (T3.m11.real - T3.m22.real) - 1j * T3.m12.imag
    C3["m23"] = c * (T3.m13.conj() - T3.m23.conj())

    attrs = {"poltype": "C3", "description": "Covariance matrix (3x3)"}
    return xr.Dataset(C3, attrs=attrs)


def T4_to_C4(T4: xarray.Dataset) -> xarray.Dataset:
    """Converts the Pauli coherency matrix T4 to the lexicographic covariance matrix C4.

    Args:
        T4 (xarray.Dataset): input image of coherency matrices

    Returns:
        xarray.Dataset: C4 covariance matrix
    """
    validate_dataset(T4, allowed_poltypes="T4")
    # if T4.poltype != "T4":
    #     raise ValueError("Input polarimetric type must be 'T4'")

    C4 = {}

    # force real diagonal to save space
    C4["m11"] = 0.5 * (T4.m11 + 2 * T4.m12.real + T4.m22)
    C4["m22"] = 0.5 * (T4.m33 - 2 * T4.m34.imag + T4.m44)
    C4["m33"] = 0.5 * (T4.m33 + T4.m44 + 2 * T4.m34.imag)
    C4["m44"] = 0.5 * (T4.m11 - 2 * T4.m12.real + T4.m22)

    # upper diagonal terms
    C4["m12"] = 0.5 * (
        T4.m13.real + T4.m23.real - T4.m14.imag - T4.m24.imag
    ) + 0.5 * 1j * (T4.m13.imag + T4.m23.imag + T4.m14.real + T4.m24.real)

    C4["m13"] = 0.5 * (
        T4.m13.real + T4.m23.real + T4.m14.imag + T4.m24.imag
    ) + 0.5 * 1j * (T4.m13.imag + T4.m23.imag - T4.m14.real - T4.m24.real)

    C4["m14"] = 0.5 * (T4.m11 - T4.m22) - 1j * T4.m12.imag

    C4["m23"] = 0.5 * (T4.m33 - T4.m44) - 1j * T4.m34.real

    C4["m24"] = 0.5 * (
        T4.m13.real - T4.m23.real - T4.m14.imag + T4.m24.imag
    ) + 0.5 * 1j * (-T4.m13.imag + T4.m23.imag - T4.m14.real + T4.m24.real)

    C4["m34"] = 0.5 * (
        T4.m13.real - T4.m23.real + T4.m14.imag - T4.m24.imag
    ) + 0.5 * 1j * (-T4.m13.imag + T4.m23.imag + T4.m14.real - T4.m24.real)

    attrs = {"poltype": "C4", "description": "Covariance matrix (4x4)"}
    return xr.Dataset(C4, attrs=attrs)


def C3_to_T3(C3: xarray.Dataset) -> xarray.Dataset:
    """Converts the lexicographic covariance matrix C3 to the Pauli coherency matrix T3.

    Args:
        C3 (xarray.Dataset): input image of covariance matrices

    Returns:
        xarray.Dataset: T3 coherency matrix
    """
    validate_dataset(C3, allowed_poltypes="C3")
    if C3.poltype != "C3":
        raise ValueError("Input polarimetric type must be 'C3'")

    T3 = {}

    c = 1 / np.sqrt(np.float32(2))

    # force real diagonal to save space
    T3["m11"] = 0.5 * (C3.m11 + C3.m33) + C3.m13.real
    T3["m22"] = 0.5 * (C3.m11 + C3.m33) - C3.m13.real
    T3["m33"] = C3.m22
    # upper diagonal terms
    T3["m12"] = 0.5 * (C3.m11 - C3.m33) - 1j * C3.m13.imag
    T3["m13"] = c * (C3.m12 + C3.m23.conj())
    T3["m23"] = c * (C3.m12 - C3.m23.conj())

    attrs = {"poltype": "T3", "description": "Coherency matrix (3x3)"}
    return xr.Dataset(T3, attrs=attrs)


def C4_to_T4(C4: xarray.Dataset) -> xarray.Dataset:
    """Converts the lexicographic covariance matrix C4 to the Pauli coherency matrix T4.

    Args:
        C4 (xarray.Dataset): input image of covariance matrices

    Returns:
        xarray.Dataset: T4 coherency matrix
    """
    validate_dataset(C4, allowed_poltypes="C4")
    # if C4.poltype != "C4":
    #     raise ValueError("Input polarimetric type must be 'C4'")

    T4 = {}

    # force real diagonal to save space
    # diagonal terms
    T4["m11"] = 0.5 * (C4.m11 + 2 * C4.m14.real + C4.m44)
    T4["m22"] = 0.5 * (C4.m11 - 2 * C4.m14.real + C4.m44)
    T4["m33"] = 0.5 * (C4.m22 + C4.m33 + 2 * C4.m23.real)
    T4["m44"] = 0.5 * (C4.m22 + C4.m33 - 2 * C4.m23.real)

    # m12
    T4["m12"] = 0.5 * (C4.m11 - C4.m44) - 1j * C4.m14.imag

    # m13
    T4["m13"] = 0.5 * (C4.m12 + C4.m13 + C4.m24.conj() + C4.m34.conj())

    # m14
    T4["m14"] = 0.5 * (
        C4.m12.imag - C4.m13.imag - C4.m24.imag + C4.m34.imag
    ) + 0.5 * 1j * (-C4.m12.real + C4.m13.real - C4.m24.real + C4.m34.real)

    # m23
    T4["m23"] = 0.5 * (C4.m12 + C4.m13 - C4.m24.conj() - C4.m34.conj())

    # m24
    T4["m24"] = 0.5 * (
        C4.m12.imag - C4.m13.imag + C4.m24.imag - C4.m34.imag
    ) + 0.5 * 1j * (-C4.m12.real + C4.m13.real + C4.m24.real - C4.m34.real)

    # m34
    T4["m34"] = -C4.m23.imag + 0.5j * (-C4.m22 + C4.m33)

    attrs = {"poltype": "T4", "description": "Coherency matrix (4x4)"}
    return xr.Dataset(T4, attrs=attrs)


# this function is used only in unit tests
def vec_to_mat(vec: np.ndarray) -> np.ndarray:
    if vec.ndim != 3:
        raise ValueError("Vector valued image is expected (dimension 3)")
    return vec[:, :, None, :] * vec[:, :, :, None].conj()


def boxcar(img: xarray.Dataset, dim_az: int, dim_rg: int) -> xarray.Dataset:
    """
    Apply a boxcar filter to an image.

    Args:
        img (xarray.Dataset): Input image with variables of shape (naz, nrg, ...).
        dim_az (int): Size in azimuth of the filter.
        dim_rg (int): Size in range of the filter.

    Returns:
        xarray.Dataset: Filtered image, shape (naz, nrg, ...).

    Note:
        The filter is always applied along 2 dimensions (azimuth, range). Please ensure to provide a valid image.
    """
    validate_dataset(img)

    # if not isinstance(img, xarray.Dataset):
        # raise TypeError("Function only valid for xarray / PolSARpro datasets.")

    if type(dim_az) != int and type(dim_rg) != int:
        raise ValueError("dimaz and dimrg must be integers")
    if (dim_az < 1) or (dim_rg < 1):
        raise ValueError("dimaz and dimrg must be strictly positive")

    process_args = dict(
        dim_az=dim_az,
        dim_rg=dim_rg,
        depth=(dim_az, dim_rg),
    )
    data_out = {}
    for var in img.data_vars:
        # if data is not chunked, apply directly
        if isinstance(img[var].data, np.ndarray):
            da_in = _boxcar_core(img[var].data, dim_az=dim_az, dim_rg=dim_rg)
        # for dask chunked arrays, apply blockwise parallel processing
        else:
            da_in = da.map_overlap(
                _boxcar_core,
                img[var].data,
                **process_args,
                dtype="complex64",
            )
        data_out[var] = (img[var].dims, da_in)

    return xr.Dataset(data_out, coords=img.coords, attrs=img.attrs)


# this should not be used directly
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


# This is an alternative version that uses xarray's rolling mean
# performance is not clear so I just keep it here for now
# def boxcar(img: xarray.Dataset, dim_az: int, dim_rg: int) -> xarray.Dataset:
#     """
#     Apply a boxcar filter to an image.

#     Args:
#         img (complex or real array): Input image with arbitrary number of dimensions, shape (naz, nrg, ...).
#         dim_az (int): Size in azimuth (or latitude) of the filter.
#         dim_rg (int): Size in range (or longitude) of the filter.

#     Returns:
#         complex or real array: Filtered image, shape (naz, nrg, ...).

#     Note:
#         The filter is always applied along 2 dimensions (azimuth, range).
#         If the input is a geocoded image, azimuth and range become latitude and longitude.
#         Please ensure to provide a valid image.
#     """

#     if not isinstance(img, xarray.Dataset):
#         raise TypeError("Input must be a valid PolSARPro Dataset.")

#     if not all((isinstance(dim_az, int), isinstance(dim_rg, int))):
#         raise TypeError("Parameters dim_az and dim_rg must be integers.")

#     if dim_az <= 0 or dim_rg <= 0:
#         raise ValueError("Parameters dim_az and dim_rg must strictly positive.")

#     dims_sar = {"y", "x"}
#     dims_geo = {"lat", "lon"}

#     if dims_sar.issubset(img.dims):
#         dict_filter = dict(x=dim_rg, y=dim_az)
#         dict_slice = dict(x=slice(dim_rg, -dim_rg), y=slice(dim_az, -dim_az))
#     elif dims_geo.issubset(img.dims):
#         dict_filter = dict(lon=dim_rg, lat=dim_az)
#         dict_slice = dict(lon=slice(dim_rg, -dim_rg), lat=slice(dim_az, -dim_az))
#     else:
#         raise ValueError("Input data must have dimensions ('y','x') or ('lat', 'lon').")

#     # pad the data with zeros
#     res = img.pad(dict_filter, mode="constant", constant_values=0)

#     # compute rolling mean
#     res = res.rolling(**dict_filter, center=True).mean()

#     # trim the padded borders
#     res = res.isel(**dict_slice)
#     return res


def plot_h_alpha_plane(ds, bins=500, min_pts=5):
    """Plot H-Alpha 2D histogram.

    Args:
        ds (xr.Dataset): Dataset containing 'entropy' and 'alpha' variables.
        bins (int): Number of bins along each dimension.
        min_pts (int): If the number of points in one bin is less than this value, display is omitted.
    """

    if not ds.poltype == "h_a_alpha":
        raise ValueError("Input must be a valid PolSARpro H/A/Alpha result.")
    
    if not {"entropy", "alpha"}.issubset(ds.data_vars):
        raise ValueError("Entropy or Alpha is missing from the input data.")

    def curve1(n_points):

        m = np.linspace(1e-30, 1, n_points)
        l1 = np.ones(n_points)
        l2 = m
        l3 = m
        l = np.vstack([l1, l2, l3])

        p = l / l.sum(0)
        H = np.sum(-p * np.log(p), 0) / np.float32(np.log(3))
        alpha = p[1] + p[2]

        return H, alpha

    def curve2(n_points):
        m = np.linspace(1e-30, 1, n_points)
        l1 = np.zeros_like(m)
        l2 = np.ones_like(m)
        l3 = np.zeros_like(m)
        # l1[m < 0.5] = 0
        l1[m >= 0.5] = 2 * m[m >= 0.5] - 1

        # l3[m < 0.5] = 2 * m[m<0.5]
        l3[m >= 0.5] = 1

        l = np.vstack([l1, l2, l3])

        p = l / l.sum(0)
        H = np.sum(-p * np.log(p + 1e-30), 0) / np.float32(np.log(3))
        alpha = p[1] + p[2]

        return H[m >= 0.5], alpha[m >= 0.5]

    x1, y1 = curve1(bins)
    x2, y2 = curve2(bins)

    # Flatten and remove NaNs
    entropy = ds.entropy.values.ravel()
    alpha = ds.alpha.values.ravel()
    mask = ~np.isnan(entropy) & ~np.isnan(alpha)
    entropy = entropy[mask].clip(0, 1)
    alpha = alpha[mask].clip(0, 90)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 2D histogram
    h = ax.hist2d(entropy, alpha, bins=bins, range=((0, 1), (0, 90)), cmap="jet", cmin=min_pts)
    plt.colorbar(h[3], ax=ax, label="Number of points")

    # Overlay curves
    ax.plot(x1, 90*y1, color="black", linewidth=1)
    ax.plot(x2, 90*y2, color="black", linewidth=1)

    # Labels and title
    ax.set_xlabel("Entropy (H)")
    ax.set_ylabel("Alpha (°)")
    ax.set_title("H-Alpha plane")

    return ax, fig
