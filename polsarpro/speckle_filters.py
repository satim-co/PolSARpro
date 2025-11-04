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
from scipy.ndimage import convolve

from polsarpro.auxil import validate_dataset
from polsarpro.util import boxcar, _boxcar_core


def refined_lee(
    input_data: xr.Dataset, window_size: int = 7, num_looks: int = 4
) -> xr.Dataset:
    """
    Apply the Refined Lee Speckle Filter to a PolSAR dataset.

    Parameters
    ----------
    input_data : xr.Dataset
        Input PolSAR dataset containing covariance or coherency matrices.
    window_size : int, optional
        Size of the filtering window (default is 7).
    num_looks : int, optional
        Number of looks of the input data (default is 4).

    Returns
    -------
    xr.Dataset
        Speckle filtered PolSAR dataset.
    """
    # Validate input dataset
    allowed_poltypes = ("C2", "C3", "C4", "T3", "T4")
    poltype = validate_dataset(input_data, allowed_poltypes=allowed_poltypes)

    # TODO: add checks on window_size and num_looks

    span = _compute_span(input_data)
    # directional masks creation
    masks = _make_masks(window_size)

    # use a different window size for gradient computation
    # also return an offset for efficient dilated convolutions
    nwg, _ = _get_window_params(window_size)

    # smooth power image prior to gradient computation
    process_args = dict(dim_az=nwg, dim_rg=nwg, depth=(window_size, window_size))
    span_smooth = da.map_overlap(  # convolutions require overlapping chunks
        _boxcar_core,
        span,
        **process_args,
        dtype="float32",
    )

    # directional gradient-based mask selection
    process_args = dict(window_size=window_size, depth=(window_size, window_size))
    mask_index = da.map_overlap(  # convolutions require overlapping chunks
        _compute_mask_index,
        span_smooth,
        **process_args,
        dtype="int64",
    )

    # filter coefficient computation
    process_args = dict(
        span=span, mask_index=mask_index, masks=masks, num_looks=num_looks, depth=(window_size, window_size)
    )
    coeff = da.map_overlap(  # convolutions require overlapping chunks
        _compute_reflee_coefficients,
        **process_args,
        dtype="float32",
    )

    # apply filter to input matrix elements
    # out = _apply_reflee_filter(input_data, coeff, mask_index, masks)

    # outputs
    out = {}
    out["mask_index"] = mask_index.compute()  # .compute()
    out["coeff"] = coeff.compute()
    return out

    # return xr.Dataset(
    #     # add dimension names
    #     {k: (tuple(input_data.dims), v) for k, v in out.items()},
    #     attrs=dict(
    #         poltype=poltype,
    #         description=input_data.description,
    #     ),
    #     coords=input_data.coords,
    # )


# functions for internal computations -- do not use directly


def _compute_mask_index(span: da.Array, window_size: int) -> da.Array:
    mask_index = np.zeros_like(span, dtype=da.uint8)

    # remove if not needed and change window_size to off
    _, off = _get_window_params(window_size)
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
    mask_index = np.argmax(da.abs(dist), axis=0)

    # determine the sign of the gradient for proper mask selection
    sign = _extract_value_at_indices(dist, mask_index) > 0
    mask_index = mask_index + 4 * sign

    return mask_index


def _compute_reflee_coefficients(
    span: da.Array, mask_index: da.Array, masks: da.Array, num_looks: int
) -> da.Array:
    # compute local statistics

    mean_local = _convolve_and_select(span, mask_index, masks)
    mean_local_sq = _convolve_and_select(span**2, mask_index, masks)
    var_local = mean_local_sq - mean_local**2

    sigma2 = 1 / num_looks

    coeff = var_local - mean_local_sq * sigma2
    coeff /= var_local * (1 + sigma2)
    coeff = np.clip(coeff, 0, 1)
    return coeff


def _apply_reflee_filter(
    input_data: xr.Dataset, coeff: da.Array, mask_index: da.Array, masks: da.Array
) -> dict:
    # apply filter to each element of the covariance/coherency matrix
    filtered_elements = {}
    for var in input_data.data_vars:
        img = input_data[var].data
        mean_local = _convolve_and_select(img, mask_index, masks)
        filtered_img = mean_local + coeff * (img - mean_local)
        filtered_elements[var] = filtered_img
    return filtered_elements


# TODO: use numba to avoid all convolutions
# convolves with all masks and selects according to index
def _convolve_and_select(img, mask_index, masks):
    mode = "constant"
    imgout = np.zeros((mask_index.shape[0],) + img.shape, dtype=img.dtype)
    # TODO update kernel normalization in NaN areas

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
def _extract_value_at_indices(arr: da.Array, indices: da.Array) -> da.Array:
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


def _make_masks(window_size: int) -> da.Array:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    h = window_size // 2
    grid_i, grid_j = da.meshgrid(
        da.arange(-h, h + 1, dtype=da.int32),
        da.arange(-h, h + 1, dtype=da.int32),
        indexing="ij",
    )

    # Masks (same order as original C Nmax indices)
    w0 = (grid_j >= 0).astype("float32")  # right half
    w1 = (grid_j >= grid_i).astype("float32")  # upper-right triangle
    w2 = (grid_i <= 0).astype("float32")  # top half
    w3 = (grid_i >= grid_j).astype("float32")  # upper-left triangle
    w4 = (grid_j <= 0).astype("float32")  # left half
    w5 = (grid_j <= grid_i).astype("float32")  # lower-left triangle
    w6 = (grid_i >= 0).astype("float32")  # bottom half
    w7 = (grid_i <= grid_j).astype("float32")  # lower-right triangle

    w0 /= da.sum(w0)
    w1 /= da.sum(w1)
    w2 /= da.sum(w2)
    w3 /= da.sum(w3)
    w4 /= da.sum(w4)
    w5 /= da.sum(w5)
    w6 /= da.sum(w6)
    w7 /= da.sum(w7)
    masks = da.stack([w0, w1, w2, w3, w4, w5, w6, w7], axis=0).astype("float32")

    return masks
