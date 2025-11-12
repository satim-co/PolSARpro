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

# Description: module containing speckle filtering functions

"""

import numpy as np
import xarray as xr
import dask.array as da
from polsarpro.util import boxcar, S_to_C3
from polsarpro.auxil import validate_dataset
from scipy.ndimage import convolve
from polsarpro.util import _boxcar_core


def PWF(
    input_data: xr.Dataset,
    train_window_size: list[int, int] = [9, 9],
    test_window_size: list[int, int] = [1, 1],
) -> xr.Dataset:
    """
    Speckle reduction using Polarimetric Whitening Filter.

    Args:
        input_data (xr.Dataset): Input polarimetric SAR dataset. Supported types are:

            - "S": Sinclair scattering matrix

            - "C3": Lexicographic covariance matrix

            - "T3": Pauli coherency matrix

            - "C4" and "T4": 4x4 versions of the above

            - "C2": Dual-polarimetric covariance
        train_window_size (list[int, int], optional): Size of the spatial averaging window to be applied to compute the inverse covariance (boxcar filter). Defaults to [9, 9].
        test_window_size (list[int, int], optional): Size of the spatial averaging window to be applied on the input covariance (boxcar filter). Defaults to [1, 1].

    Returns:
        xr.Dataset: Result of the filtering.

    Notes:
        If the S matrix is given as an input, it will be converted to C3.
    """

    allowed_poltypes = ("S", "C2", "C3", "C4", "T3", "T4")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    if test_window_size[0] >= train_window_size[0] or test_window_size[1] >= train_window_size[1]:
        raise ValueError("Testing window size must be smaller than training window size.") 

    if poltype in ["C2", "C3", "T3", "C4", "T4"]:
        in_ = input_data
    elif poltype == "S":
        in_ = S_to_C3(input_data)
    else:
        raise ValueError(f"Invalid polarimetric type: {input_data.poltype}")

    new_dims = tuple(input_data.dims)

    eps = 1e-30
    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    box_train = boxcar(in_, dim_az=train_window_size[0], dim_rg=train_window_size[1])
    box_test = boxcar(in_, dim_az=test_window_size[0], dim_rg=test_window_size[1])

    # remove NaNs to avoid errors in eigh
    mask = box_train.to_array().isnull().any("variable")
    box_train = box_train.fillna(eps)

    M_box = _reconstruct_matrix_from_ds(box_train).data
    # M = _reconstruct_matrix_from_ds(in_).data
    M = _reconstruct_matrix_from_ds(box_test).data
    # eigendecomposition
    meta = (
        np.array([], dtype="complex64").reshape((0, 0, 0, 0)),
    )
    M_inv = da.apply_gufunc(np.linalg.inv, "(i,j)->(i,j)", M_box, meta=meta)
    prod = np.matmul(M_inv, M)
    trace = np.trace(prod, axis1=2, axis2=3).real

    out = {"pwf": trace}
    # return trace

    return xr.Dataset(
        # add dimension names, account for 2D and 3D outputs
        {
            k: (new_dims, v) if v.ndim == 2 else (new_dims + ("i",), v)
            for k, v in out.items()
        },
        attrs=dict(
            poltype="pwf",
            description="Results of the PWF (Polarimetric Whitening Filter) speckle reduction method.",
        ),
        coords=input_data.coords,
    ).where(~mask)


# below this line, functions are not meant to be called directly


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
        M2_l1 = xr.concat((ds.m11, ds.m12), dim="j")
        M2_l2 = xr.concat((ds.m12.conj(), ds.m22), dim="j")

        # Concatenate all lines into a 2x2 matrix
        return (
            xr.concat((M2_l1, M2_l2), dim="i")
            .transpose(*new_dims_array)
            .chunk({new_dims[0]: "auto", new_dims[1]: "auto", "i": 2, "j": 2})
            + eps
        )
    if ds.poltype in ["T3", "C3"]:
        # build each line of the T3 matrix
        M3_l1 = xr.concat((ds.m11, ds.m12, ds.m13), dim="j")
        M3_l2 = xr.concat((ds.m12.conj(), ds.m22, ds.m23), dim="j")
        M3_l3 = xr.concat((ds.m13.conj(), ds.m23.conj(), ds.m33), dim="j")

        # Concatenate all lines into a 3x3 matrix
        return (
            xr.concat((M3_l1, M3_l2, M3_l3), dim="i")
            .transpose(*new_dims_array)
            .chunk({new_dims[0]: "auto", new_dims[1]: "auto", "i": 3, "j": 3})
            + eps
        )
    elif ds.poltype in ["T4", "C4"]:
        # build each line of the T4 matrix
        M4_l1 = xr.concat((ds.m11, ds.m12, ds.m13, ds.m14), dim="j")
        M4_l2 = xr.concat((ds.m12.conj(), ds.m22, ds.m23, ds.m24), dim="j")
        M4_l3 = xr.concat((ds.m13.conj(), ds.m23.conj(), ds.m33, ds.m34), dim="j")
        M4_l4 = xr.concat(
            (ds.m14.conj(), ds.m24.conj(), ds.m34.conj(), ds.m44), dim="j"
        )

        # concatenate all lines into a 4x4 matrix
        return (
            xr.concat((M4_l1, M4_l2, M4_l3, M4_l4), dim="i")
            .transpose(*new_dims_array)
            .chunk({new_dims[0]: "auto", new_dims[1]: "auto", "i": 4, "j": 4})
            + eps
        )
    else:
        raise NotImplementedError(
            "Implemented only for C2, C3, T3 and C4 and T4 poltypes."
        )




def refined_lee(
    input_data: xr.Dataset, window_size: int = 7, num_looks: int = 1
) -> xr.Dataset:
    """
    Apply the Refined Lee Speckle Filter to a PolSAR dataset.

    Args:
        input_data (xr.Dataset): Input PolSAR dataset containing covariance or coherency matrices. Works for 2x2, 3x3 and 4x4 matrices.
        window_size (int, optional): Size of the filtering window (default is 7).
        num_looks (int or float, optional): Number of looks of the input data (default is 1).

    Returns:
        xr.Dataset: Speckle filtered PolSAR dataset.
    
    References:
        Lee, J.-S., Grunes, M. R., & Kwok, R. (1999). "Polarimetric SAR Speckle Filtering and Its Implications for Classification." IEEE Transactions on Geoscience and Remote Sensing, 37(5), 2363-2373. doi:10.1109/36.789467
    """
    # Validate input dataset
    allowed_poltypes = ("C2", "C3", "C4", "T3", "T4")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    # test that num_looks is a number > 0
    if num_looks <= 0 or not isinstance(num_looks, (int, float)):
        raise ValueError("num_looks must be a positive number")

    span = _compute_span(input_data)

    # use a different window size for gradient computation
    # also return an offset for efficient dilated convolutions
    nwg, off = _get_window_params(window_size)

    # smooth power image prior to gradient computation
    process_args = dict(dim_az=nwg, dim_rg=nwg, depth=(nwg, nwg))
    span_smooth = da.map_overlap(  # convolutions require overlapping chunks
        _boxcar_core, # used instead of boxcar() to avoid extra conversions
        span,
        **process_args,
        dtype="float32",
    )

    # directional gradient-based mask selection
    process_args = dict(off=off, depth=(off, off))
    mask_index = da.map_overlap(  # operation requires overlapping chunks
        _compute_mask_index,
        span_smooth,
        **process_args,
        dtype="int64",
    )

    # filter coefficient computation
    process_args = dict(
        window_size=window_size, num_looks=num_looks, depth=(window_size, window_size)
    )
    coeff = da.map_overlap(  # convolutions require overlapping chunks
        _compute_reflee_coefficients,
        span,
        mask_index,
        **process_args,
        dtype="float32",
    )

    # apply filter to input matrix elements
    process_args = dict(window_size=window_size, depth=(window_size, window_size))
    out = {}
    for var in input_data.data_vars:
        img = input_data[var].data
        filtered_img = da.map_overlap(  # convolutions require overlapping chunks
            _apply_reflee_filter,
            img,
            coeff,
            mask_index,
            **process_args,
            dtype=img.dtype,
        )
        out[var] = filtered_img

    return xr.Dataset(
        # add dimension names
        {k: (tuple(input_data.dims), v) for k, v in out.items()},
        attrs=dict(
            poltype=poltype,
            description=input_data.description,
        ),
        coords=input_data.coords,
    )


# functions for internal computations -- do not use directly


def _compute_mask_index(span: np.array, off: int) -> np.array:
    mask_index = np.zeros_like(span, dtype=da.uint8)

    # use a short name for more compact expressions
    I = np.pad(span, off, mode="reflect")

    # directional gradients on dilated 3x3 windows
    d0 = (
        -I[: -2 * off, : -2 * off]
        + I[: -2 * off, 2 * off :]
        - I[off:-off, : -2 * off]
        + I[off:-off, 2 * off :]
        - I[2 * off :, : -2 * off]
        + I[2 * off :, 2 * off :]
    )

    d1 = (
        I[: -2 * off, off:-off]
        + I[: -2 * off, 2 * off :]
        - I[off:-off, : -2 * off]
        + I[off:-off, 2 * off :]
        - I[2 * off :, : -2 * off]
        - I[2 * off :, off:-off]
    )

    d2 = (
        I[: -2 * off, : -2 * off]
        + I[: -2 * off, off:-off]
        + I[: -2 * off, 2 * off :]
        - I[2 * off :, : -2 * off]
        - I[2 * off :, off:-off]
        - I[2 * off :, 2 * off :]
    )

    d3 = (
        I[: -2 * off, : -2 * off]
        + I[: -2 * off, off:-off]
        + I[off:-off, : -2 * off]
        - I[off:-off, 2 * off :]
        - I[2 * off :, off:-off]
        - I[2 * off :, 2 * off :]
    )

    # find index of maximum gradient
    dist = np.stack([d0, d1, d2, d3], axis=0)
    mask_index = np.argmax(np.abs(dist), axis=0)

    # determine the sign of the gradient for proper mask selection
    sign = _extract_value_at_indices(dist, mask_index) < 0
    mask_index = mask_index + 4 * sign

    return mask_index


def _compute_reflee_coefficients(
    span: np.ndarray,
    mask_index: np.ndarray,
    window_size: int,
    num_looks: int,
) -> np.ndarray:
    eps = 1e-30
    # compute local statistics
    mean_local = _convolve_and_select(span, mask_index, window_size)
    mean_local_sq = _convolve_and_select(span**2, mask_index, window_size)
    var_local = mean_local_sq - mean_local**2
    sigma2 = 1 / num_looks
    cv_span2 = abs(var_local) / (eps + mean_local**2)
    coeff = (cv_span2 - sigma2) / (cv_span2 * (1 + sigma2) + eps)
    coeff = np.where(coeff < 0, 0, coeff)

    return coeff


def _apply_reflee_filter(
    img: np.array, coeff: da.Array, mask_index: da.Array, window_size: int
) -> dict:
    mean_local = _convolve_and_select(img, mask_index, window_size)
    filtered_img = mean_local + coeff * (img - mean_local)
    return filtered_img


# convolves with all masks and selects according to index
def _convolve_and_select(img, mask_index, window_size):
    # adds small overhead and avoids chunk alignment issues
    masks = _make_masks(window_size)
    mode = "constant"
    imgout = np.zeros((masks.shape[0],) + img.shape, dtype=img.dtype)
    msk = np.isnan(img)
    img_ = np.where(msk, 0, img)
    for i in np.arange(masks.shape[0]):
        ker = masks[i]
        if np.iscomplexobj(img_):
            imgout[i] = convolve(img_.real, ker, mode=mode) + 1j * convolve(
                img_.imag, ker, mode=mode
            )
            imgout[i][msk] = np.nan + 1j * np.nan
        else:
            imgout[i] = convolve(img_, ker, mode=mode)
            imgout[i][msk] = np.nan

    return _extract_value_at_indices(imgout, mask_index)


# dask doesn't support ND fancy indexing
def _extract_value_at_indices(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    val = np.zeros_like(indices, dtype=arr.dtype)
    for i in np.arange(arr.shape[0]):
        val = np.where(indices == i, arr[i], val)
    return val


def _compute_span(ds_in: xr.Dataset) -> da.Array:
    # directly return a dask array to avoid unnecessary conversions
    span = da.zeros_like(ds_in.m11.data)
    poltype = ds_in.poltype
    if poltype in ("C2", "T2"):
        span = ds_in.m11.data + ds_in.m22.data
    elif poltype in ("C3", "T3"):
        span = ds_in.m11.data + ds_in.m22.data + ds_in.m33.data
    elif poltype in ("C4", "T4"):
        span = ds_in.m11.data + ds_in.m22.data + ds_in.m33.data + ds_in.m44.data
    return span


def _get_window_params(nw: int) -> tuple[int, int]:

    # compute for any allowed window sizes the corresponding
    # gradient window size and the subsampling step
    params = {
        3: (1, 1),
        5: (3, 1),
        7: (3, 2),
        9: (5, 2),
        11: (5, 3),
        13: (5, 4),
        15: (7, 4),
        17: (7, 5),
        19: (7, 6),
        21: (9, 6),
        23: (9, 7),
        25: (9, 8),
        27: (11, 8),
        29: (11, 9),
        31: (11, 10),
    }

    try:
        window_size_grad, sub = params[nw]
    except KeyError:
        raise ValueError("Window size must be one of {3, 5, 7, …, 31}")

    return window_size_grad, sub


def _make_masks(window_size: int) -> np.array:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    h = window_size // 2
    grid_i, grid_j = np.meshgrid(
        np.arange(-h, h + 1, dtype=np.int32),
        np.arange(-h, h + 1, dtype=np.int32),
        indexing="ij",
    )

    # Masks (same order as original C Nmax indices)
    w0 = (grid_j >= 0).astype("float32")  # right half
    w1 = (grid_j >= grid_i).astype("float32")  # upper-right triangle
    w2 = (grid_i <= 0).astype("float32")  # top half
    w3 = (grid_j <= grid_i)[::-1].astype("float32")  # upper-left triangle
    w4 = (grid_j <= 0).astype("float32")  # left half
    w5 = (grid_j <= grid_i).astype("float32")  # lower-left triangle
    w6 = (grid_i >= 0).astype("float32")  # bottom half
    w7 = (grid_j >= grid_i)[::-1].astype("float32")  # lower-right triangle

    w0 /= np.sum(w0)
    w1 /= np.sum(w1)
    w2 /= np.sum(w2)
    w3 /= np.sum(w3)
    w4 /= np.sum(w4)
    w5 /= np.sum(w5)
    w6 /= np.sum(w6)
    w7 /= np.sum(w7)
    masks = np.stack([w0, w1, w2, w3, w4, w5, w6, w7], axis=0).astype("float32")

    return masks
