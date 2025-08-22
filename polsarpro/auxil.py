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

Description: module containing auxiliary helper functions
"""

import xarray as xr
import numpy as np
from typing import Sequence

# Standardized required variables for each poltype
POLTYPES = {
    "S": {
        "description": "Scattering matrix",
        "vars": {
            "hh": {"dtype": "complex64"},
            "hv": {"dtype": "complex64"},
            "vh": {"dtype": "complex64"},
            "vv": {"dtype": "complex64"},
        },
        # for some data, (e.g. H/A/Alpha) variables may not be required by default
        "optional_vars": None,
    },
    "C3": {
        "description": "Covariance matrix (3x3)",
        "vars": {
            "m11": {"dtype": "float32"},
            "m12": {"dtype": "complex64"},
            "m22": {"dtype": "float32"},
            "m13": {"dtype": "complex64"},
            "m23": {"dtype": "complex64"},
            "m33": {"dtype": "float32"},
        },
        "optional_vars": None,
    },
    "C4": {
        "description": "Covariance matrix (4x4)",
        "vars": {
            "m11": {"dtype": "float32"},
            "m12": {"dtype": "complex64"},
            "m13": {"dtype": "complex64"},
            "m14": {"dtype": "complex64"},
            "m22": {"dtype": "float32"},
            "m23": {"dtype": "complex64"},
            "m24": {"dtype": "complex64"},
            "m33": {"dtype": "float32"},
            "m34": {"dtype": "complex64"},
            "m44": {"dtype": "float32"},
        },
        "optional_vars": None,
    },
    "T3": {
        "description": "Coherency matrix (3x3)",
        "vars": {
            "m11": {"dtype": "float32"},
            "m12": {"dtype": "complex64"},
            "m22": {"dtype": "float32"},
            "m13": {"dtype": "complex64"},
            "m23": {"dtype": "complex64"},
            "m33": {"dtype": "float32"},
        },
        "optional_vars": None,
    },
    "T4": {
        "description": "Coherency matrix (4x4)",
        "vars": {
            "m11": {"dtype": "float32"},
            "m12": {"dtype": "complex64"},
            "m13": {"dtype": "complex64"},
            "m14": {"dtype": "complex64"},
            "m22": {"dtype": "float32"},
            "m23": {"dtype": "complex64"},
            "m24": {"dtype": "complex64"},
            "m33": {"dtype": "float32"},
            "m34": {"dtype": "complex64"},
            "m44": {"dtype": "float32"},
        },
        "optional_vars": None,
    },
}


# Allowed 2D dimension conventions
ALLOWED_DIMS: tuple[tuple[str, str], ...] = (
    ("y", "x"),
    ("lat", "lon"),
    # some H/A/Alpha outputs
    ("y", "x", "i"),
    ("lat", "lon", "i"),
)


def validate_dataset(
    ds: xr.Dataset,
    allowed_poltypes: str | Sequence[str] | None = None,
    check_dims: bool = True,
    check_vars: bool = True,
    check_dtypes: bool = True,
) -> str:
    """
    Validate a PolSAR dataset against standard conventions.
    Used in processing functions to ensure the validity of inputs.

    Args:
        ds: Dataset to validate.
        allowed_poltypes: Single poltype (e.g. "S") or sequence of poltypes
            (e.g. ("S", "T3")). If None, accept any known poltype.
        check_dims: If True, ensure that dims match one of ALLOWED_DIMS.
        check_vars: If True, ensure that expected variables are present.
        check_dtypes: If True, ensure that variable dtypes are correct.

    Returns:
        The detected poltype string.

    Raises:
        ValueError: If any check fails.
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")

    # Check poltype is in attrs
    if "poltype" not in ds.attrs:
        raise ValueError("Missing required 'poltype' in dataset attributes.")

    # Check poltype is a valid name
    poltype = str(ds.attrs["poltype"])  # .upper()
    if poltype not in POLTYPES:
        raise ValueError(
            f"Unsupported poltype '{poltype}'. Must be one of {list(POLTYPES)}."
        )

    # Check poltype is in the input list
    if allowed_poltypes is not None:
        if isinstance(allowed_poltypes, str):
            allowed = (allowed_poltypes,)
        else:
            allowed = tuple(p for p in allowed_poltypes)
        if poltype not in allowed:
            raise ValueError(
                f"poltype='{poltype}' not allowed. Must be one of {allowed}."
            )

    # Check dimensions are valid (2D or 3D coordinates)
    if check_dims:
        ds_dims = tuple(ds.dims)
        if ds_dims not in ALLOWED_DIMS:
            raise ValueError(
                f"Dataset dims {list(ds.dims)} do not match any allowed convention "
                f"{[list(d) for d in ALLOWED_DIMS]}."
            )

    # Check variable names and shapes
    specs = POLTYPES[poltype]
    if check_vars:
        if not ds.data_vars:
            raise ValueError("No variable found in Dataset.")

        allowed_vars = specs["vars"]
        # Is this variable allowed?
        for v in ds.data_vars:
            if v not in allowed_vars:
                raise ValueError(f"Unexpected variable '{v}' for poltype '{poltype}'.")

        # Is this variable missing?
        for v in allowed_vars:
            optional_vars = specs.get("optional_vars") or []
            if (v not in ds.data_vars) and (v not in optional_vars):
                raise ValueError(f"Dataset is missing required variable: '{v}'")

        shapes = [ds[v].shape for v in allowed_vars if v in ds.data_vars]
        for shape in shapes:
            if len(shape) < 2:
                raise ValueError(f"Expected >= 2D variables, but found shape {shape}")

    if check_dtypes:
        for v in ds.data_vars:
            expected_dtype = np.dtype(specs["vars"][v]["dtype"])
            if ds[v].dtype != expected_dtype:
                raise ValueError(
                    f"Variable '{v}': expected {expected_dtype} dtype, but found {ds[v].dtype}."
                )
