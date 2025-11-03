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
from polsarpro.util import boxcar


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

    # TODO: adapt to chunked data (use map_overlap?)
    # directional gradient-based mask selection
    mask_index = _compute_mask_index(span, window_size)

    # filter coefficient computation
    coeff = _compute_reflee_coefficients(span, mask_index, masks, num_looks)

    # apply filter to input matrix elements
    out = _apply_reflee_filter(input_data, coeff, mask_index, masks)
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


def _compute_mask_index(span: xr.Dataset, window_size: int) -> da.Array:
    mask_index = da.zeros_like(span, dtype=da.uint8)

    # use a different window size for gradient computation
    # also return an offset for efficient dilated convolutions
    nwg, off = _get_window_params(window_size)

    # smooth power image prior to gradient computation
    span_smooth = boxcar(span, dim_az=nwg, dim_rg=nwg)

    # use a short name for more compact expressions
    I = da.pad(span_smooth, off, mode="reflect")

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
    dist = da.stack([d0, d1, d2, d3], axis=0)
    mask_index = da.argmax(da.abs(dist), axis=0)

    # determine the sign of the gradient for proper mask selection
    shp_in = span.m11.data.shape
    sign = dist[mask_index, da.arange(shp_in[0])[:, None], da.arange(shp_in[1])] > 0
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
    coeff = da.clip(coeff, 0, 1)
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
    imgout = da.zeros(img.shape + (1,), dtype=img.dtype)
    # TODO update kernel normalization in NaN areas
    for i in da.range(masks.shape[0]):
        msk = np.isnan(img)
        img_ = img.copy()
        img_[msk] = 0
        ker = masks[i]
        if np.iscomplexobj(img_):
            imgout = convolve(img_.real, ker, mode=mode) + 1j * convolve(
                img_.imag, ker, mode=mode
            )
            imgout[msk] = np.nan + 1j * np.nan
        else:
            imgout = convolve(img_, ker, mode=mode)
            imgout[msk] = np.nan

    res = imgout[mask_index, da.arange(img.shape[0])[:, None], da.arange(img.shape[1])]
    return res


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
