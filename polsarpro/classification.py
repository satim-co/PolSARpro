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

# Description: module containing polarimetric classification functions

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
import dask.array as da
from dask.distributed import print
from polsarpro.util import boxcar, C3_to_T3, S_to_C3, S_to_T3, C4_to_T4, T3_to_C3
from polsarpro.auxil import validate_dataset
from polsarpro.decompositions import h_a_alpha


def wishart_h_a_alpha(
    input_data: xr.Dataset,
    boxcar_size: list[int] = [5, 5],
    h_a_alpha_result: Optional[xr.Dataset] = None,
    max_iter: int = 10,
    tol_pct: float = None,
) -> xr.Dataset:
    """
    Performs the Wishart H/A/Alpha classification on polarimetric SAR data.

    This function implements an iterative unsupervised classification scheme based on
    the H/A/Alpha decomposition and the Wishart distance measure. The algorithm:
    1. Computes initial H/A/Alpha decomposition parameters
    2. Performs initial classification using the H-Alpha decision boundaries
    3. Iteratively refines the classification using Wishart distance to class centers

    Args:
        input_data (xr.Dataset): Input polarimetric SAR dataset. Supported types are:
            - "S": Sinclair scattering matrix
            - "C3": Lexicographic covariance matrix (3x3)
            - "T3": Pauli coherency matrix (3x3)
            - "C4": 4x4 covariance matrix
            - "T4": 4x4 coherency matrix
        boxcar_size (list[int, int], optional): Size of the spatial averaging window
            (boxcar filter) applied before decomposition. Defaults to [5, 5].
        h_a_alpha_result (xr.Dataset, optional): Pre-computed H/A/Alpha decomposition
            results. If provided, the function will use these results for the initial
            H-Alpha classification instead of computing them from input_data. The dataset
            must contain at least 'entropy' and 'alpha' variables. If None (default),
            the H/A/Alpha decomposition is computed from input_data.
        max_iter (int, optional): Maximum number of iterations for the classification
            refinement loop. Must be a positive integer. Defaults to 10.
        tol_pct (float or None, optional): Threshold for early stopping based on the
            percentage of pixels changing class between iterations. This parameter is kept only for compatibility with the C version (see notes below). When the percentage
            of pixels that switch class is less than this value, the algorithm stops.
            If set, must be in the range [0.0, 100.0]. Defaults to None.

    Returns:
        xr.Dataset: Dataset containing the classification map with variable 'label'
            containing integer labels (1-8) corresponding to the 8 H-Alpha zones.

    References:
        Cloude, S. R., & Pottier, E. (1997). An entropy based classification scheme for land
        applications of polarimetric SAR. *IEEE Transactions on Geoscience and Remote Sensing*,
        35(1), 68-78.

    Notes:
        It is recommended to leave the tol_pct parameter to its default None state to take advantage of Dask lazy processing. Setting this parameter to a value might result in a performance loss. It has been observed that without early termination the python implementation is faster than the legacy version.

    """
    # Validate max_iter parameter
    if not isinstance(max_iter, int):
        raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}.")

    # Validate stop_threshold parameter
    if not isinstance(tol_pct, (int, float)) and tol_pct is not None:
        raise TypeError(
            f"tol_pct must be a number or None, got {type(tol_pct).__name__}."
        )
    if tol_pct is not None and (tol_pct < 0.0 or tol_pct > 100.0):
        raise ValueError(
            f"tol_pct must be in the range [0.0, 100.0], got {tol_pct}."
        )
    poltype = validate_dataset(
        input_data, allowed_poltypes=("C3", "T3", "C4", "T4", "S")
    )

    # Handle optional pre-computed h_a_alpha result
    if h_a_alpha_result is not None:
        # Validate h_a_alpha_result
        if not isinstance(h_a_alpha_result, xr.Dataset):
            raise TypeError("h_a_alpha_result must be an xarray.Dataset.")
        if h_a_alpha_result.attrs.get("poltype") != "h_a_alpha":
            raise ValueError("h_a_alpha_result must have poltype='h_a_alpha'.")
        required_vars = {"entropy", "alpha"}
        if not required_vars.issubset(set(h_a_alpha_result.data_vars)):
            raise ValueError(
                f"h_a_alpha_result must contain variables: {required_vars}"
            )
        ds_ha = h_a_alpha_result
    else:
        # Compute H/A/Alpha decomposition from input_data
        ds_ha = None

    # Convert input to T3/T4 format for Wishart distance computation
    if poltype == "S":
        in_ = S_to_T3(input_data)
    elif poltype == "C3":
        in_ = C3_to_T3(input_data)
    elif poltype == "T3":
        in_ = input_data
    elif poltype == "C4":
        in_ = C4_to_T4(input_data)
    elif poltype == "T4":
        in_ = input_data

    # Apply boxcar filtering to the coherency matrix
    in_ = boxcar(in_, dim_az=boxcar_size[0], dim_rg=boxcar_size[1])

    # Compute H/A/Alpha decomposition if not provided
    if ds_ha is None:
        ds_ha = h_a_alpha(
            input_data=in_,
            boxcar_size=[1, 1],  # Already filtered
            flags=("entropy", "anisotropy", "alpha"),
        )

    # Initial classification using H-Alpha decision boundaries
    class_map_init = _h_alpha_classifier(ds_ha)

    # Iterative Wishart classification
    nclass = 8
    if tol_pct is None:
        # lazy computation for a faster execution
        class_map, percent_changed = _wishart_classifier_without_early_stop(
            in_, class_map_init, nclass, max_iter
        )
    else:
        # in this case results are computed on the fly
        class_map, percent_changed = _wishart_classifier_with_early_stop(
            in_, class_map_init, nclass, max_iter, tol_pct
        )

    # Initial classification using H-A-Alpha decision boundaries
    class_map_init_16 = da.where(ds_ha.anisotropy.data <= 0.5, class_map, class_map + 8)

    nclass = 16
    if tol_pct is None:
        # lazy computation for a faster execution
        class_map_16, percent_changed_16 = _wishart_classifier_without_early_stop(
            in_, class_map_init_16, nclass, max_iter
        )
    else:
        # in this case results are computed on the fly
        class_map_16, percent_changed_16 = _wishart_classifier_with_early_stop(
            in_, class_map_init_16, nclass, max_iter, tol_pct
        )

    # Build output dataset
    result = xr.Dataset(
        {
            "wishart_h_alpha_class": (in_.dims, class_map),
            # "h_alpha_class": (in_.dims, class_map_init),
            "wishart_h_alpha_percent_change": percent_changed,
            "wishart_h_a_alpha_class": (in_.dims, class_map_16),
            # "h_a_alpha_class": (in_.dims, class_map_init_16),
            "wishart_h_a_alpha_percent_change": percent_changed_16,
        },
        coords=in_.coords,
        attrs=dict(
            poltype="wishart_h_a_alpha",
            description="Wishart H/A/Alpha classification result",
        ),
    )

    return result


def _reconstruct_matrix_from_ds(ds):

    # eps = 1e-30

    new_dims_array = ("c", "i", "j")

    if ds.poltype in ["T3", "C3"]:
        # build each line of the T3 matrix
        M3_l1 = xr.concat((ds.m11, ds.m12, ds.m13), dim="j")
        M3_l2 = xr.concat((ds.m12.conj(), ds.m22, ds.m23), dim="j")
        M3_l3 = xr.concat((ds.m13.conj(), ds.m23.conj(), ds.m33), dim="j")

        # Concatenate all lines into a 3x3 matrix
        return (
            xr.concat((M3_l1, M3_l2, M3_l3), dim="i")
            .transpose(*new_dims_array)
            .chunk({"i": 3, "j": 3})
            # + eps
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
            .chunk({"i": 4, "j": 4})
            # + eps
        )
    else:
        raise NotImplementedError(
            "Implemented only for C2, C3, T3 and C4 and T4 poltypes."
        )


def _trace_product_3(M_inv: da.Array, cov_ds: xr.Dataset) -> da.Array:
    """
    Compute Tr(M_inv @ C3) for 3x3 covariance/coherency matrices.

    Uses direct element-wise computation for efficiency:
    Tr(M_inv @ C3) = sum_ij M_inv[i,j] * C3[j,i]

    Args:
        M_inv: Inverse matrix array of shape (3, 3)
        cov_ds: Dataset containing covariance/coherency matrix elements
            (m11, m12, m22, m13, m23, m33)

    Returns:
        Trace of the product, same shape as cov_ds variables
    """
    m11 = cov_ds.m11.data
    m12 = cov_ds.m12.data
    m22 = cov_ds.m22.data
    m13 = cov_ds.m13.data
    m23 = cov_ds.m23.data
    m33 = cov_ds.m33.data

    M_inv_broad = M_inv[None, None]
    trace = (
        M_inv_broad[..., 0, 0] * m11[..., None]
        + M_inv_broad[..., 0, 1] * da.conj(m12)[..., None]
        + M_inv_broad[..., 0, 2] * da.conj(m13)[..., None]
        + M_inv_broad[..., 1, 0] * m12[..., None]
        + M_inv_broad[..., 1, 1] * m22[..., None]
        + M_inv_broad[..., 1, 2] * da.conj(m23)[..., None]
        + M_inv_broad[..., 2, 0] * m13[..., None]
        + M_inv_broad[..., 2, 1] * m23[..., None]
        + M_inv_broad[..., 2, 2] * m33[..., None]
    )

    return trace


def _trace_product_4(M_inv: da.Array, cov_ds: xr.Dataset) -> da.Array:
    """
    Compute Tr(M_inv @ C4) for 4x4 covariance/coherency matrices.

    Uses direct element-wise computation for efficiency:
    Tr(M_inv @ C4) = sum_ij M_inv[i,j] * C4[j,i]

    Args:
        M_inv: Inverse matrix array of shape (4, 4)
        cov_ds: Dataset containing covariance/coherency matrix elements
            (m11, m12, m22, m13, m23, m33, m14, m24, m34, m44)

    Returns:
        Trace of the product, same shape as cov_ds variables
    """
    m11 = cov_ds.m11.data
    m12 = cov_ds.m12.data
    m22 = cov_ds.m22.data
    m13 = cov_ds.m13.data
    m23 = cov_ds.m23.data
    m33 = cov_ds.m33.data
    m14 = cov_ds.m14.data
    m24 = cov_ds.m24.data
    m34 = cov_ds.m34.data
    m44 = cov_ds.m44.data
    M_inv_broad = M_inv[None, None]
    trace = (
        M_inv_broad[..., 0, 0] * m11[..., None]
        + M_inv_broad[..., 0, 1] * da.conj(m12)[..., None]
        + M_inv_broad[..., 0, 2] * da.conj(m13)[..., None]
        + M_inv_broad[..., 0, 3] * da.conj(m14)[..., None]
        + M_inv_broad[..., 1, 0] * m12[..., None]
        + M_inv_broad[..., 1, 1] * m22[..., None]
        + M_inv_broad[..., 1, 2] * da.conj(m23)[..., None]
        + M_inv_broad[..., 1, 3] * da.conj(m24)[..., None]
        + M_inv_broad[..., 2, 0] * m13[..., None]
        + M_inv_broad[..., 2, 1] * m23[..., None]
        + M_inv_broad[..., 2, 2] * m33[..., None]
        + M_inv_broad[..., 2, 3] * da.conj(m34)[..., None]
        + M_inv_broad[..., 3, 0] * m14[..., None]
        + M_inv_broad[..., 3, 1] * m24[..., None]
        + M_inv_broad[..., 3, 2] * m34[..., None]
        + M_inv_broad[..., 3, 3] * m44[..., None]
    )

    return trace


def _h_alpha_classifier(ds_ha: xr.Dataset) -> da.Array:
    """
    Classify pixels based on H-Alpha decomposition using decision boundaries.

    Parameters
    ----------
    ds_ha : xarray.Dataset
        Dataset containing 'entropy' and 'alpha' variables.

    Returns
    -------
    dask.array.Array
        Classification map with integer class labels (1-9).
    """
    alpha = ds_ha.alpha.data
    H = ds_ha.entropy.data

    # Decision boundaries
    thresh_al1 = 55.0
    thresh_al2 = 50.0
    thresh_al3 = 48.0
    thresh_al4 = 42.0
    thresh_al5 = 40.0
    thresh_H1 = 0.9
    thresh_H2 = 0.5

    # Apply thresholds
    # In Python/dask: True=1, False=0, so we use direct boolean operations
    a1 = alpha <= thresh_al1
    a2 = alpha <= thresh_al2
    a3 = alpha <= thresh_al3
    a4 = alpha <= thresh_al4
    a5 = alpha <= thresh_al5

    h1 = H <= thresh_H1
    h2 = H <= thresh_H2

    # 2-D regions
    # In Python/dask: ~ for NOT, & for AND, | for OR
    r1 = (~a3) & h2
    r2 = a3 & (~a4) & h2
    r3 = a4 & h2
    r4 = (~a2) & h1 & (~h2)
    r5 = a2 & (~a5) & h1 & (~h2)
    r6 = a5 & h1 & (~h2)
    r7 = (~a1) & (~h1)
    r8 = a1 & (~a5) & (~h1)
    r9 = a5 & (~h1)  # Non feasible region

    # Compute class labels (1-9)
    # Each region contributes its class number where the region is True
    class_map = (
        1 * r1 + 2 * r2 + 3 * r3 + 4 * r4 + 5 * r5 + 6 * r6 + 7 * r7 + 8 * r8 + 9 * r9
    )

    return class_map


def _update_wishart_class_centers(input_data, class_map, nclass):

    # Compute class center -- broadcast arrays to avoid looping
    mask = class_map[..., None] == da.arange(1, nclass + 1)[None, None]
    npts = mask.sum((0, 1))

    # Matrix dims
    n = 3 if input_data.poltype in ("C3", "T3") else 4

    # Class centers
    center = (mask * input_data.expand_dims(dim="c", axis=2)).sum(("y", "x")) / npts

    # Reconstruct matrix and regularize
    M_center = _reconstruct_matrix_from_ds(center) # + 1e-30 * da.eye(n)
    return M_center


def _update_wishart_class_map(in_, M_center):
    # Compute inverse and determinant
    meta = (np.array([], dtype="complex64").reshape((0, 0, 0)),)
    M_inv = da.apply_gufunc(np.linalg.inv, "(i,j)->(i,j)", M_center, meta=meta)
    M_det = da.apply_gufunc(np.linalg.det, "(i,j)->()", M_center, meta=meta)

    # Compute Wishart distance
    if M_center.shape[1] == 3:
        dist = da.log(da.abs(M_det)) + _trace_product_3(M_inv, in_)
    else:
        dist = da.log(da.abs(M_det)) + _trace_product_4(M_inv, in_)

    # As in C version, let's clip the determinant
    eps = 1e-30
    M_det = M_det.real.clip(eps) + 1j* M_det.imag.clip(eps)

    # Assign class numbers
    # +1 to convert from 0-based to 1-based class labels
    class_map = da.argmin(dist, axis=-1) + 1
    return class_map


def _wishart_classifier_with_early_stop(in_, class_map, nclass, max_iter, tol_pct):
    total_pixels = class_map.shape[0] * class_map.shape[1]
    for iteration in range(max_iter):
        print(f"Iteration #{iteration+1}")
        # Store previous classification for convergence check
        class_map_prev = class_map

        # Returns class centers as a dask array with shape (nclass, n, n)
        M_center = _update_wishart_class_centers(in_, class_map, nclass).persist()

        # Assign new classes, use persist to avoid recomputing
        class_map = _update_wishart_class_map(in_, M_center).persist()

        # Check for convergence based on percentage of pixels changing class
        changed_pixels = (class_map != class_map_prev).sum()
        percent_changed = (changed_pixels / total_pixels) * 100.0
        percent_changed = percent_changed.compute()
        if percent_changed < tol_pct:
            break
    return class_map, percent_changed


def _wishart_classifier_without_early_stop(in_, class_map, nclass, max_iter):
    total_pixels = class_map.shape[0] * class_map.shape[1]
    for iteration in range(max_iter):
        # Store previous classification for convergence check
        class_map_prev = class_map

        # Returns class centers as a dask array with shape (nclass, n, n)
        M_center = _update_wishart_class_centers(in_, class_map, nclass)

        # Assign new classes
        class_map = _update_wishart_class_map(in_, M_center)

        # Check for convergence based on percentage of pixels changing class
        changed_pixels = (class_map != class_map_prev).sum()
        percent_changed = (changed_pixels / total_pixels) * 100.0

    return class_map, percent_changed
