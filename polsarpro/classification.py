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

import numpy as np
import xarray as xr
import dask.array as da
from polsarpro.util import boxcar, C3_to_T3, S_to_C3, S_to_T3, C4_to_T4, T3_to_C3
from polsarpro.auxil import validate_dataset
from polsarpro.decompositions import h_a_alpha

def wishart_h_a_alpha(input_data, boxcar_size=[5, 5], h_a_alpha_result=None):
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

    Returns:
        xr.Dataset: Dataset containing the classification map with variable 'class'
            containing integer labels (1-9) corresponding to the 9 H-Alpha zones.

    References:
        Cloude, S. R., & Pottier, E. (1997). An entropy based classification scheme for land
        applications of polarimetric SAR. *IEEE Transactions on Geoscience and Remote Sensing*,
        35(1), 68-78.
    """
    poltype = validate_dataset(input_data, allowed_poltypes=("C3", "T3", "C4", "T4", "S"))

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
    class_map = _h_alpha_classifier(ds_ha)

    # Reconstruct matrix from dataset for Wishart iteration
    M = _reconstruct_matrix_from_ds(in_)

    # Iterative Wishart classification
    # Maximum iterations
    max_iter = 3

    for iteration in range(max_iter):
        # Compute class centers (coherency matrix averages per class)
        class_centers = _compute_class_centers(M, class_map)

        # Compute distance to all classes and reclassify
        class_map = _wishart_classify(M, class_centers)

    # Build output dataset
    result = xr.Dataset(
        {"class": (in_.dims, class_map)},
        coords=in_.coords,
        attrs=dict(
            poltype="wishart_h_a_alpha",
            description="Wishart H/A/Alpha classification result",
        ),
    )

    return result


def _reconstruct_matrix_from_ds(ds):
    """
    Reconstruct the coherency matrix from a T3/T4 dataset.

    Args:
        ds: Dataset with m11, m12, m22, m13, m23, m33 (and m14, m24, m34, m44 for T4)

    Returns:
        Dask array of shape (..., n, n) where n is 3 or 4
    """
    if ds.poltype == "T3":
        # Build 3x3 Hermitian matrix
        m11 = ds.m11.data
        m12 = ds.m12.data
        m22 = ds.m22.data
        m13 = ds.m13.data
        m23 = ds.m23.data
        m33 = ds.m33.data

        M = da.stack([
            da.stack([m11, m12, m13], axis=-1),
            da.stack([da.conj(m12), m22, m23], axis=-1),
            da.stack([da.conj(m13), da.conj(m23), m33], axis=-1),
        ], axis=-2)

    elif ds.poltype == "T4":
        # Build 4x4 Hermitian matrix
        m11 = ds.m11.data
        m12 = ds.m12.data
        m13 = ds.m13.data
        m14 = ds.m14.data
        m22 = ds.m22.data
        m23 = ds.m23.data
        m24 = ds.m24.data
        m33 = ds.m33.data
        m34 = ds.m34.data
        m44 = ds.m44.data

        M = da.stack([
            da.stack([m11, m12, m13, m14], axis=-1),
            da.stack([da.conj(m12), m22, m23, m24], axis=-1),
            da.stack([da.conj(m13), da.conj(m23), m33, m34], axis=-1),
            da.stack([da.conj(m14), da.conj(m24), da.conj(m34), m44], axis=-1),
        ], axis=-2)
    else:
        raise ValueError(f"Unsupported poltype: {ds.poltype}. Expected T3 or T4.")

    return M


def _compute_class_centers(M, class_map, num_classes=9):
    """
    Compute the average coherency matrix for each class.

    Args:
        M: Coherency matrix array of shape (y, x, n, n)
        class_map: Classification map of shape (y, x)
        num_classes: Number of classes (default 9)

    Returns:
        Dictionary mapping class labels to their center matrices
    """
    centers = {}
    n = M.shape[-1]

    for k in range(1, num_classes + 1):
        # Create mask for class k
        mask = (class_map == k)

        # Compute mean of matrices in this class
        # Sum all matrices in the class and divide by count
        mask_expanded = mask[..., None, None]
        masked_sum = da.sum(M * mask_expanded, axis=(0, 1))
        count = da.sum(mask)

        # Avoid division by zero - use a small regularization
        count = da.maximum(count, 1)
        center = masked_sum / count

        # Regularize the center matrix to ensure it's non-singular
        # Add a small value to the diagonal
        reg = 1e-6
        if n == 3:
            center = center + reg * da.eye(3, dtype=center.dtype)
        else:
            center = center + reg * da.eye(4, dtype=center.dtype)

        centers[k] = center

    return centers


def _trace3_hm1hm2(Hm1, Hm2):
    """
    Compute Tr(Hm1 * Hm2) for 3x3 matrices.

    Args:
        Hm1: First matrix array of shape (..., 3, 3)
        Hm2: Second matrix array of shape (..., 3, 3)

    Returns:
        Trace of the product, shape (...)
    """
    # Matrix multiplication followed by trace
    product = da.matmul(Hm1, Hm2)
    trace = product[..., 0, 0] + product[..., 1, 1] + product[..., 2, 2]
    return trace


def _trace4_hm1hm2(Hm1, Hm2):
    """
    Compute Tr(Hm1 * Hm2) for 4x4 matrices.

    Args:
        Hm1: First matrix array of shape (..., 4, 4)
        Hm2: Second matrix array of shape (..., 4, 4)

    Returns:
        Trace of the product, shape (...)
    """
    # Matrix multiplication followed by trace
    product = da.matmul(Hm1, Hm2)
    trace = (product[..., 0, 0] + product[..., 1, 1] +
             product[..., 2, 2] + product[..., 3, 3])
    return trace


def _wishart_distance(M, center, center_inv, logdet_center, n=3):
    """
    Compute the Wishart distance between a coherency matrix and a class center.

    The Wishart distance is defined as:
    d = ln(det(center)) + Tr(center^{-1} * M)

    Args:
        M: Coherency matrix array of shape (..., n, n)
        center: Class center matrix of shape (n, n)
        center_inv: Pre-computed inverse of center
        logdet_center: Pre-computed log determinant of center
        n: Matrix dimension (3 or 4)

    Returns:
        Distance array of shape (...)
    """
    # Compute trace term: Tr(center^{-1} * M)
    if n == 3:
        trace_term = _trace3_hm1hm2(M, center_inv)
    else:  # n == 4
        trace_term = _trace4_hm1hm2(M, center_inv)

    # Wishart distance
    distance = logdet_center + trace_term

    return distance


def _compute_logdet(center):
    """
    Compute log determinant of a small matrix using numpy via map_blocks.

    Args:
        center: Matrix array of shape (n, n)

    Returns:
        Log determinant value (scalar dask array)
    """
    # Use map_blocks to apply numpy slogdet
    def _slogdet_block(x):
        sign, logdet = np.linalg.slogdet(x)
        return np.array(logdet, dtype=np.float64)

    # Compute logdet - result is a scalar
    logdet_da = da.map_blocks(_slogdet_block, center, drop_axis=[0, 1], dtype=np.float64)
    return logdet_da


def _wishart_classify(M, class_centers):
    """
    Classify each pixel based on minimum Wishart distance to class centers.

    Args:
        M: Coherency matrix array of shape (y, x, n, n)
        class_centers: Dictionary mapping class labels to center matrices

    Returns:
        Classification map of shape (y, x)
    """
    n = M.shape[-1]

    # Initialize with the first class
    classes = list(class_centers.keys())
    k0 = classes[0]
    center0 = class_centers[k0]

    # Compute inverse and log determinant for each center
    center0_inv = da.linalg.inv(center0)
    center0_logdet = _compute_logdet(center0)

    # Compute distance to first class
    if n == 3:
        min_distance = _trace3_hm1hm2(M, center0_inv)
    else:
        min_distance = _trace4_hm1hm2(M, center0_inv)
    min_distance = center0_logdet + min_distance

    class_map = da.full(M.shape[:-2], k0, dtype=np.int32)

    # Compare with other classes
    for k in classes[1:]:
        center = class_centers[k]
        center_inv = da.linalg.inv(center)
        center_logdet = _compute_logdet(center)
        distance = _wishart_distance(M, center, center_inv, center_logdet, n=n)

        # Update class map where this distance is smaller
        mask = distance < min_distance
        class_map = da.where(mask, k, class_map)
        min_distance = da.where(mask, distance, min_distance)

    return class_map


def _h_alpha_classifier(ds_ha):
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
    a1 = (alpha <= thresh_al1)
    a2 = (alpha <= thresh_al2)
    a3 = (alpha <= thresh_al3)
    a4 = (alpha <= thresh_al4)
    a5 = (alpha <= thresh_al5)
    
    h1 = (H <= thresh_H1)
    h2 = (H <= thresh_H2)
    
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
    class_map = (1 * r1 + 2 * r2 + 3 * r3 + 4 * r4 + 
                 5 * r5 + 6 * r6 + 7 * r7 + 8 * r8 + 9 * r9)
    
    return class_map
