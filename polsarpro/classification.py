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

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import label

from polsarpro.auxil import validate_dataset
from polsarpro.decompositions import h_a_alpha
from polsarpro.util import (C3_to_T3, C4_to_T4, S_to_C3, S_to_T3, T3_to_C3,
                            boxcar)


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
    3. Iteratively refines the classification using Wishart distance to 8 class centers
    4. Split the 8 classes into 16 based on a 0.5 threshold on anisotropy
    5. Apply Wishart iterations on these classes

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
        xr.Dataset: Dataset containing the 8 and 18 class maps.
            Pixels where any input polarimetric element is NaN are excluded from
            class-center estimation and iterative class assignment, and are encoded
            as class ``0`` in the output maps.

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
        raise ValueError(f"tol_pct must be in the range [0.0, 100.0], got {tol_pct}.")
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

    # Track invalid pixels and replace NaNs for numerical stability.
    eps = 1e-30
    valid_mask = (~in_.to_array().isnull().any("variable")).data
    in_ = in_.fillna(eps)

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
    class_map_init = da.where(valid_mask, _h_alpha_classifier(ds_ha), 0).astype("uint8")

    # Iterative Wishart classification
    nclass = 8
    if tol_pct is None:
        # lazy computation for a faster execution
        class_map, percent_changed = _wishart_classifier_without_early_stop(
            in_, class_map_init, nclass, max_iter, valid_mask=valid_mask
        )
    else:
        # in this case results are computed on the fly
        class_map, percent_changed = _wishart_classifier_with_early_stop(
            in_, class_map_init, nclass, max_iter, tol_pct, valid_mask=valid_mask
        )

    # Divide classes according to anisotropy
    class_map_init_16 = da.where(
        valid_mask,
        da.where(ds_ha.anisotropy.data <= 0.5, class_map, class_map + 8),
        0,
    ).astype("uint8")

    nclass = 16
    if tol_pct is None:
        # lazy computation for a faster execution
        class_map_16, percent_changed_16 = _wishart_classifier_without_early_stop(
            in_, class_map_init_16, nclass, max_iter, valid_mask=valid_mask
        )
    else:
        # in this case results are computed on the fly
        class_map_16, percent_changed_16 = _wishart_classifier_with_early_stop(
            in_, class_map_init_16, nclass, max_iter, tol_pct, valid_mask=valid_mask
        )

    # Build output dataset
    result = xr.Dataset(
        {
            "wishart_h_alpha_class": (in_.dims, class_map),
            "wishart_h_alpha_percent_change": percent_changed,
            "wishart_h_a_alpha_class": (in_.dims, class_map_16),
            "wishart_h_a_alpha_percent_change": percent_changed_16,
        },
        coords=in_.coords,
        attrs=dict(
            poltype="wishart_h_a_alpha",
            description="Wishart H/A/Alpha classification result",
        ),
    )

    return result

def wishart_supervised(
    input_data: xr.Dataset,
    training_labels: xr.DataArray,
    boxcar_size: list[int] = [5, 5],
) -> xr.Dataset:
    """
    Performs Wishart supervised classification based on user inputs.
    Args:
        input_data (xr.Dataset): Input polarimetric SAR dataset. Supported types are:
            - "S": Sinclair scattering matrix
            - "C3": Lexicographic covariance matrix (3x3)
            - "T3": Pauli coherency matrix (3x3)
            - "C4": 4x4 covariance matrix
            - "T4": 4x4 coherency matrix
        training_labels: (xr.DataArray): 2D array of the same spatial dimensions
            as input_data, containing integer class labels for training pixels.
            Each connected region for a class is used as a separate training
            cluster. Pixels with label 0 are considered unlabeled and are not
            used in training.
        boxcar_size (list[int, int], optional): Size of the spatial averaging window
            (boxcar filter) applied before decomposition. Defaults to [5, 5].

    Returns:
        xr.Dataset: Dataset containing the output classes.
            Pixels where any input polarimetric element is NaN are excluded from
            training and class assignment, and are encoded as class ``0`` in the
            output map.
    """

    poltype = validate_dataset(
        input_data, allowed_poltypes=("C3", "T3", "C4", "T4", "S")
    )


    if {"y", "x"}.issubset(input_data.dims):
        dims = ("y", "x")
    elif {"lat", "lon"}.issubset(input_data.dims):
        dims = ("lat", "lon")

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


    if training_labels.ndim != 2:
        raise ValueError("training_labels must be a 2D DataArray.")
    if training_labels.shape != (input_data.sizes[dims[0]], input_data.sizes[dims[1]]):
        raise ValueError(
            "training_labels must have the same spatial shape as input_data."
        )

    # Exclude invalid raster pixels (NaN in any matrix element).
    eps = 1e-30
    valid_mask = (~in_.to_array().isnull().any("variable")).data
    in_ = in_.fillna(eps)

    training_labels_data = training_labels.data
    if isinstance(training_labels_data, da.Array):
        training_labels_data = training_labels_data.compute()
    training_labels_data = np.asarray(training_labels_data)
    if isinstance(valid_mask, da.Array):
        valid_mask_np = np.asarray(valid_mask.compute())
    else:
        valid_mask_np = np.asarray(valid_mask)
    training_labels_data = training_labels_data.copy()
    training_labels_data[~valid_mask_np] = 0

    lab, cluster_classes = _label_training_clusters(training_labels_data)
    nclass = len(cluster_classes) - 1

    # Apply boxcar filtering to the coherency matrix
    in_ = boxcar(in_, dim_az=boxcar_size[0], dim_rg=boxcar_size[1])

    centers = _update_wishart_class_centers(
        in_, lab, nclass=nclass, valid_mask=valid_mask
    )
    cluster_map = _update_wishart_class_map(in_=in_, M_center=centers, valid_mask=valid_mask)

    # Remap training clusters back to their semantic class labels
    class_map = cluster_map.map_blocks(
        lambda block: cluster_classes[block],
        dtype=cluster_classes.dtype,
    )

    # Build output dataset
    result = xr.Dataset(
        {
            "wishart_supervised_class": (in_.dims, class_map),
        },
        coords=in_.coords,
        attrs=dict(
            poltype="wishart_supervised",
            description="Wishart supervised classification result",
        ),
    )

    return result


def _label_training_clusters(
    training_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Label connected training regions and keep their semantic class labels."""
    if not np.issubdtype(training_labels.dtype, np.integer):
        raise TypeError("training_labels must contain integer class labels.")
    if np.any(training_labels < 0):
        raise ValueError("training_labels must contain non-negative class labels.")

    lab = np.zeros(training_labels.shape, dtype=np.int32)
    cluster_classes = [0]
    cluster_id = 0

    for class_value in np.unique(training_labels):
        if class_value == 0:
            continue

        # labeled connected regions for current class
        class_regions, nregions = label(training_labels == class_value)
        if nregions == 0:
            continue
        
        # update cluster -> class mapping list
        region_mask = class_regions > 0
        lab[region_mask] = class_regions[region_mask] + cluster_id
        cluster_classes.extend([class_value] * nregions)
        cluster_id += nregions

    if cluster_id == 0:
        raise ValueError("training_labels must contain at least one labeled pixel.")

    return lab, np.asarray(cluster_classes, dtype=training_labels.dtype)


def _reconstruct_matrix_from_ds(ds):

    eps = 1e-30

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
            .chunk({"i": 4, "j": 4})
            + eps
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

    return da.real(trace)


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

    return da.real(trace)


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

    return class_map.astype("uint8")


def _update_wishart_class_centers(input_data, class_map, nclass, valid_mask=None):

    if {"y", "x"}.issubset(input_data.dims):
        dims = ("y", "x")
    elif {"lat", "lon"}.issubset(input_data.dims):
        dims = ("lat", "lon")

    # Compute class center -- broadcast arrays to avoid looping
    mask = class_map[..., None] == da.arange(1, nclass + 1)[None, None]
    if valid_mask is not None:
        mask = mask & valid_mask[..., None]
    npts = mask.sum((0, 1))
    # Some classes can be temporarily empty: guard denominator to avoid 0/0.
    npts_safe = da.maximum(npts, 1)

    # Matrix dims
    n = 3 if input_data.poltype in ("C3", "T3") else 4

    # Class centers
    center = (mask * input_data.expand_dims(dim="c", axis=2)).sum((dims[0], dims[1])) / npts_safe

    # Reconstruct matrix and regularize
    M_center = _reconstruct_matrix_from_ds(center)  + 1e-30 * da.eye(n)

    # otherwise M_center type is complex128
    return M_center.astype("complex64")


def _update_wishart_class_map(in_, M_center, valid_mask=None):

    # Compute inverse and determinant
    meta = (np.array([], dtype="complex64").reshape((0, 0, 0)),)
    M_inv = da.apply_gufunc(np.linalg.inv, "(i,j)->(i,j)", M_center.data, meta=meta)
    M_det = da.apply_gufunc(np.linalg.det, "(i,j)->()", M_center.data, meta=meta)

    # use custom C-like functions instead (Debug only, use with 3x3)
    # M_inv = inverse_hermitian_3x3(M_center.data)
    # M_det = det_hermitian_3x3(M_center.data)

    # As in C version, let's clip the determinant
    eps = 1e-30
    M_det = M_det.real.clip(eps) + 1j * M_det.imag.clip(eps)

    # Compute Wishart distance
    if M_center.shape[1] == 3:
        dist = da.log(da.abs(M_det)) + _trace_product_3(M_inv, in_)
    else:
        dist = da.log(da.abs(M_det)) + _trace_product_4(M_inv, in_)

    # Assign class numbers
    # +1 to convert from 0-based to 1-based class labels
    class_map = da.argmin(dist, axis=-1) + 1
    if valid_mask is not None:
        class_map = da.where(valid_mask, class_map, 0)
    return class_map


def _wishart_classifier_with_early_stop(
    in_, class_map, nclass, max_iter, tol_pct, valid_mask=None
):
    if valid_mask is None:
        total_pixels = class_map.shape[0] * class_map.shape[1]
    else:
        total_pixels = valid_mask.sum()

    for iteration in range(max_iter):
        print(f"Iteration #{iteration+1}")
        # Store previous classification for convergence check
        class_map_prev = class_map

        # Returns class centers as a dask array with shape (nclass, n, n)
        M_center = _update_wishart_class_centers(
            in_, class_map, nclass, valid_mask=valid_mask
        ).persist()

        # Assign new classes, use persist to avoid recomputing
        class_map = _update_wishart_class_map(
            in_, M_center, valid_mask=valid_mask
        ).persist()

        # Check for convergence based on percentage of pixels changing class
        changed_mask = class_map != class_map_prev
        if valid_mask is not None:
            changed_mask = changed_mask & valid_mask
        changed_pixels = changed_mask.sum()
        percent_changed = (changed_pixels / total_pixels) * 100.0
        percent_changed = percent_changed.compute()
        if percent_changed < tol_pct:
            break
    return class_map, percent_changed


def _wishart_classifier_without_early_stop(
    in_, class_map, nclass, max_iter, valid_mask=None
):
    if valid_mask is None:
        total_pixels = class_map.shape[0] * class_map.shape[1]
    else:
        total_pixels = valid_mask.sum()
    for iteration in range(max_iter):
        # Store previous classification for convergence check
        class_map_prev = class_map

        # Returns class centers as a dask array with shape (nclass, n, n)
        M_center = _update_wishart_class_centers(
            in_, class_map, nclass, valid_mask=valid_mask
        )

        # Assign new classes
        class_map = _update_wishart_class_map(in_, M_center, valid_mask=valid_mask)

        # Check for convergence based on percentage of pixels changing class
        changed_mask = class_map != class_map_prev
        if valid_mask is not None:
            changed_mask = changed_mask & valid_mask
        changed_pixels = changed_mask.sum()
        percent_changed = (changed_pixels / total_pixels) * 100.0

    return class_map, percent_changed


def inverse_hermitian_3x3(M: da.Array) -> da.Array:
    """
    Invert a batch of 3x3 Hermitian matrices using explicit cofactor expansion.

    Args:
        M: Complex Dask array with shape (..., 3, 3)

    Returns:
        Complex Dask array with shape (..., 3, 3)
    """

    # --- Extract elements ---
    m11 = M[..., 0, 0]
    m12 = M[..., 0, 1]
    m13 = M[..., 0, 2]
    m21 = M[..., 1, 0]
    m22 = M[..., 1, 1]
    m23 = M[..., 1, 2]
    m31 = M[..., 2, 0]
    m32 = M[..., 2, 1]
    m33 = M[..., 2, 2]

    # --- Cofactors (adjugate before transpose, but already aligned) ---
    c00 = m22 * m33 - m23 * m32
    c01 = -(m12 * m33 - m13 * m32)
    c02 = m12 * m23 - m22 * m13

    c10 = -(m21 * m33 - m31 * m23)
    c11 = m11 * m33 - m13 * m31
    c12 = -(m11 * m23 - m13 * m21)

    c20 = m21 * m32 - m22 * m31
    c21 = -(m11 * m32 - m12 * m31)
    c22 = m11 * m22 - m12 * m21

    # --- Determinant ---
    det = m11 * c00 + m21 * c01 + m31 * c02
    eps = 1e-30
    det_real = da.maximum(det.real, eps)
    det_imag = da.maximum(det.imag, eps)
    det = det_real + 1j * det_imag

    # Optional safety (uncomment if needed)
    # det = da.where(det == 0, da.nan, det)

    inv_det = 1.0 / det

    # --- Apply normalization ---
    i00 = c00 * inv_det
    i01 = c01 * inv_det
    i02 = c02 * inv_det

    i10 = c10 * inv_det
    i11 = c11 * inv_det
    i12 = c12 * inv_det

    i20 = c20 * inv_det
    i21 = c21 * inv_det
    i22 = c22 * inv_det

    # --- Stack back to matrix form ---
    row0 = da.stack([i00, i01, i02], axis=-1)
    row1 = da.stack([i10, i11, i12], axis=-1)
    row2 = da.stack([i20, i21, i22], axis=-1)

    M_inv = da.stack([row0, row1, row2], axis=-2)

    return M_inv


def det_hermitian_3x3(M: da.Array) -> da.Array:
    """
    Compute determinant of 3x3 Hermitian matrices using the same
    method as the provided C implementation.

    Args:
        M: Complex Dask array with shape (..., 3, 3)

    Returns:
        Complex Dask array with shape (...)
    """

    # --- Extract elements ---
    m11 = M[..., 0, 0]
    m12 = M[..., 0, 1]
    m13 = M[..., 0, 2]
    m21 = M[..., 1, 0]
    m22 = M[..., 1, 1]
    m23 = M[..., 1, 2]
    m31 = M[..., 2, 0]
    m32 = M[..., 2, 1]
    m33 = M[..., 2, 2]

    # --- First row of cofactors (same as C: IHM[0][*]) ---
    c00 = m22 * m33 - m23 * m32
    c01 = -(m12 * m33 - m13 * m32)
    c02 = m12 * m23 - m22 * m13

    # --- Determinant (complex) ---
    # C version does this in real/imag parts explicitly
    det = m11 * c00 + m21 * c01 + m31 * c02

    # --- Optional epsilon handling (mimics C behaviour) ---
    # if eps is not None:
    eps = 1e-30
    det_real = da.maximum(det.real, eps)
    det_imag = da.maximum(det.imag, eps)
    det = det_real + 1j * det_imag

    return det
