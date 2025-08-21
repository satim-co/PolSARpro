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

Description: module containing auxiliary helper functions
"""

import xarray as xr
from typing import Sequence

# Standardized required variables for each poltype
POLTYPE_VARS: dict[str, tuple[str, ...]] = {
    "S": ("hh", "hv", "vh", "vv"),
    "C2": ("m11", "m12", "m22"),
    "C3": ("m11", "m12", "m13", "m22", "m23", "m33"),
    "C4": ("m11", "m12", "m13", "m14", "m22", "m23", "m24",
           "m33", "m34", "m44"),
    "T3": ("m11", "m12", "m13", "m22", "m23", "m33"),
    "T4": ("m11", "m12", "m13", "m14", "m22", "m23", "m24",
           "m33", "m34", "m44"),
}

# Allowed 2D dimension conventions
ALLOWED_DIMS: tuple[tuple[str, str], ...] = (
    ("y", "x"),
    ("lat", "lon"),
)


def validate_dataset(
    ds: xr.Dataset,
    allowed_poltypes: str | Sequence[str] | None = None,
    check_dims: bool = True,
    check_vars: bool = True,
) -> str:
    """
    Validate a PolSAR dataset against standard conventions.

    Args:
        ds: Dataset to validate.
        allowed_poltypes: Single poltype (e.g. "S") or sequence of poltypes
            (e.g. ("S", "T3")). If None, accept any known poltype.
        check_dims: If True, ensure that dims match one of ALLOWED_DIMS.
        check_vars: If True, ensure that expected variables are present.

    Returns:
        The detected poltype string.

    Raises:
        ValueError: If any check fails.
    """
    # --- Check poltype ---
    if "poltype" not in ds.attrs:
        raise ValueError("Missing required 'poltype' in dataset attributes.")

    poltype = str(ds.attrs["poltype"]).upper()
    if poltype not in POLTYPE_VARS:
        raise ValueError(
            f"Unsupported poltype '{poltype}'. Must be one of {list(POLTYPE_VARS)}."
        )

    if allowed_poltypes is not None:
        if isinstance(allowed_poltypes, str):
            allowed = (allowed_poltypes.upper(),)
        else:
            allowed = tuple(p.upper() for p in allowed_poltypes)
        if poltype not in allowed:
            raise ValueError(f"poltype='{poltype}' not allowed. Must be one of {allowed}.")

    # --- Check dims ---
    if check_dims:
        ds_dims = tuple(ds.dims)
        if ds_dims not in ALLOWED_DIMS:
            raise ValueError(
                f"Dataset dims {list(ds.dims)} do not match any allowed convention "
                f"{[list(d) for d in ALLOWED_DIMS]}."
            )

    # --- Check vars ---
    if check_vars:
        required_vars = POLTYPE_VARS[poltype]
        missing = [v for v in required_vars if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"Dataset poltype={poltype} is missing required variables: {missing}"
            )

        # Ensure all variables have the same 2D shape
        shapes = {ds[v].shape for v in required_vars}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent variable shapes found: {shapes}")

        shape = next(iter(shapes))
        if len(shape) != 2:
            raise ValueError(f"Expected 2D variables, but found shape {shape}")

    return poltype

