"""
This code is part of the Python PolSARpro software:

"A re-implementation of selected PolSARPro functions in Python,
following the scientific recommendations of PolInSAR 2021"

developed within an ESA funded project with SATIM.

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

# Description: polarisation-state synthesis functions

"""

from numbers import Real
from typing import Literal

import numpy as np
import xarray as xr

from polsarpro.auxil import validate_dataset
from polsarpro.util import C3_to_T3, S_to_T3


def polarisation_synthesis(
    input_data: xr.Dataset,
    phi: float = 0.0,
    tau: float = 0.0,
    *,
    basis: Literal["pauli", "sinclair"] = "pauli",
) -> xr.DataArray:
    """Synthesize power channels for a rotated polarisation state.

    The input is converted to the Pauli coherency basis before applying a
    real rotation by ``phi`` and an elliptical rotation by ``tau``. The
    returned values are raw powers; no display scaling or clipping is applied.

    Args:
        input_data: Input polarimetric dataset with ``poltype`` ``"S"``,
            ``"C3"``, or ``"T3"``.
        phi: Polarisation orientation angle in degrees.
        tau: Polarisation ellipticity angle in degrees.
        basis: Output channel convention. ``"pauli"`` returns the diagonal
            powers of the rotated coherency matrix. ``"sinclair"`` returns
            synthesized copolar and cross-polar powers using the legacy
            C-PolSARpro red, green, and blue channel assignment.

    Returns:
        DataArray containing the raw power channels as ``float32`` values along
        a ``band`` dimension labeled ``red``, ``green``, and ``blue``.
        Coordinates and Dask laziness are preserved from the input.

    Notes:
        This function follows the angle signs and channel normalization of
        C-PolSARpro's ``polar_synt`` routine. At zero rotation, the Sinclair
        blue and red channels are HH and VV power, respectively, while green
        is the reciprocal cross-polar power. The result can be displayed as an
        RGB image with ``result.plot.imshow(rgb="band", robust=True)``.
    """
    for name, angle in (("phi", phi), ("tau", tau)):
        if not isinstance(angle, Real):
            raise TypeError(f"{name} must be a real number.")
        if not np.isfinite(angle):
            raise ValueError(f"{name} must be finite, got {angle}.")

    if basis not in ("pauli", "sinclair"):
        raise ValueError("basis must be either 'pauli' or 'sinclair'.")

    poltype = validate_dataset(input_data, allowed_poltypes=("S", "C3", "T3"))
    converters = {
        "S": S_to_T3,
        "C3": C3_to_T3,
        "T3": lambda data: data,
    }
    T3 = converters[poltype](input_data)

    phi_rad = np.deg2rad(float(phi))
    tau_rad = np.deg2rad(float(tau))
    cos_phi = np.cos(2.0 * phi_rad)
    sin_phi = np.sin(2.0 * phi_rad)
    sin_4phi = np.sin(4.0 * phi_rad)
    cos_tau = np.cos(2.0 * tau_rad)
    sin_tau = np.sin(2.0 * tau_rad)
    sin_4tau = np.sin(4.0 * tau_rad)

    t11_phi = T3.m11
    t12_re_phi = T3.m12.real * cos_phi + T3.m13.real * sin_phi
    t13_im_phi = -T3.m12.imag * sin_phi + T3.m13.imag * cos_phi
    t22_phi = T3.m22 * cos_phi**2 + T3.m23.real * sin_4phi + T3.m33 * sin_phi**2
    t23_im_phi = T3.m23.imag
    t33_phi = T3.m22 * sin_phi**2 - T3.m23.real * sin_4phi + T3.m33 * cos_phi**2

    new_t11 = t11_phi * cos_tau**2 + t13_im_phi * sin_4tau + t33_phi * sin_tau**2
    new_t12_re = t12_re_phi * cos_tau + t23_im_phi * sin_tau
    new_t22 = t22_phi
    new_t33 = t11_phi * sin_tau**2 - t13_im_phi * sin_4tau + t33_phi * cos_tau**2

    if basis == "pauli":
        blue = new_t11
        red = new_t22
        green = new_t33
    else:
        blue = 0.5 * (new_t11 + new_t22) + new_t12_re
        red = 0.5 * (new_t11 + new_t22) - new_t12_re
        green = 0.5 * new_t33

    attrs = {
        "poltype": "polarisation_synthesis",
        "description": "Polarisation synthesis power channels.",
        "basis": basis,
        "phi": float(phi),
        "tau": float(tau),
    }
    band = xr.IndexVariable("band", ["red", "green", "blue"])
    return (
        xr.concat((red, green, blue), dim=band)
        .astype(np.float32, copy=False)
        .rename("Polarisation synthesis")
        .assign_attrs(attrs)
    )
