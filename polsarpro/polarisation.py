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

from numbers import Integral, Real
from typing import Literal

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from polsarpro.auxil import validate_dataset
from polsarpro.util import C3_to_T3, S_to_T3, T3_to_C3, boxcar


def estimate_orientation(
    input_data: xr.Dataset,
    boxcar_size: tuple[int, int] = (3, 3),
) -> xr.Dataset:
    """Estimate the local polarimetric orientation angle.

    The input is converted to the Pauli coherency basis and spatially averaged
    before estimating the angle from the real T23 coupling. The returned angle
    follows the quotient-based arctangent convention of C-PolSARpro.

    Args:
        input_data: Polarimetric dataset with ``poltype`` ``"S"``, ``"C3"``,
            or ``"T3"``.
        boxcar_size: Averaging-window size along the two spatial dimensions.

    Returns:
        Dataset containing the float32 ``orientation_angle`` in degrees.
        Coordinates and Dask laziness are preserved from the input.
    """
    poltype = validate_dataset(input_data, allowed_poltypes=("S", "C3", "T3"))
    if len(boxcar_size) != 2:
        raise ValueError("boxcar_size must contain two values.")

    converters = {
        "S": S_to_T3,
        "C3": C3_to_T3,
        "T3": lambda data: data,
    }
    T3 = boxcar(converters[poltype](input_data), *boxcar_size)
    angle = 0.25 * np.arctan(2.0 * T3.m23.real / (T3.m22 - T3.m33))
    return xr.Dataset(
        {"orientation_angle": np.rad2deg(angle).astype(np.float32, copy=False)},
        attrs={
            "poltype": "orientation_estimation",
            "description": "Polarimetric orientation angle estimation.",
        },
    )


def correct_orientation(input_data: xr.Dataset, orientation: xr.Dataset) -> xr.Dataset:
    """Correct a polarimetric dataset for a spatially varying orientation.

    Args:
        input_data: Polarimetric dataset with ``poltype`` ``"S"``, ``"C3"``,
            or ``"T3"``.
        orientation: Dataset returned by :func:`estimate_orientation`.

    Returns:
        Orientation-corrected dataset in the same polarimetric representation
        as ``input_data``. Coordinates and Dask laziness are preserved.
    """
    poltype = validate_dataset(input_data, allowed_poltypes=("S", "C3", "T3"))
    validate_dataset(orientation, allowed_poltypes="orientation_estimation")
    _, angle = xr.align(input_data, orientation.orientation_angle, join="exact")

    if poltype == "S":
        return _correct_orientation_S(input_data, angle)

    T3 = C3_to_T3(input_data) if poltype == "C3" else input_data
    corrected = _correct_orientation_T3(T3, angle)
    if poltype == "C3":
        corrected = T3_to_C3(corrected)
        corrected.attrs["description"] = "Orientation-compensated covariance matrix."
    return corrected


def orientation_compensation(
    input_data: xr.Dataset,
    boxcar_size: tuple[int, int] = (3, 3),
) -> tuple[xr.Dataset, xr.Dataset]:
    """Estimate and compensate the local polarimetric orientation.

    Args:
        input_data: Polarimetric dataset with ``poltype`` ``"S"``, ``"C3"``,
            or ``"T3"``.
        boxcar_size: Estimation-window size along the two spatial dimensions.

    Returns:
        Orientation-corrected data and the estimated orientation-angle Dataset.
    """
    orientation = estimate_orientation(input_data, boxcar_size=boxcar_size)
    return correct_orientation(input_data, orientation), orientation


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

    new_t11, new_t12_re, new_t22, new_t33 = _rotated_powers(T3, phi, tau)

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


def polarimetric_signature(
    input_data: xr.Dataset,
    row: int,
    col: int,
    *,
    n_phi: int = 181,
    n_tau: int = 91,
) -> xr.Dataset:
    """Compute the co-polar and cross-polar signatures of one pixel.

    The polarisation orientation angle ``phi`` spans -90 to 90 degrees, and
    the polarisation ellipticity angle ``tau`` spans -45 to 45 degrees. Both
    endpoints are included. Results are raw powers without display
    normalization or dB scaling.

    Args:
        input_data: Polarimetric dataset with ``poltype`` ``"S"``, ``"C3"``,
            or ``"T3"`` and spatial dimensions ``(y, x)`` or ``(lat, lon)``.
        row: Zero-based position along the first spatial dimension.
        col: Zero-based position along the second spatial dimension.
        n_phi: Number of evenly spaced polarisation orientation angles, at
            least two.
        n_tau: Number of evenly spaced polarisation ellipticity angles, at
            least two.

    Returns:
        Dataset with ``copol`` and ``xpol`` float32 powers on ``(phi, tau)``.
        Source pixel positions are recorded in the dataset attributes.

    Raises:
        ValueError: If the selected pixel contains a NaN or infinite matrix
            value.

    Notes:
        The default 181 phi and 91 tau values include both endpoints at exact
        1-degree steps. C-PolSARpro uses 180 and 90 values over the same ranges,
        giving steps of 180/179 and 90/89 degrees. Use ``n_phi=180`` and
        ``n_tau=90`` to reproduce the C angle grid. The C writer also normalizes
        each surface for display; this function returns raw powers.
    """
    poltype = validate_dataset(input_data, allowed_poltypes=("S", "C3", "T3"))
    if len(input_data.dims) != 2:
        raise ValueError("Signature input must have two spatial dimensions.")
    row_dim, col_dim = input_data.dims
    for name, index, dim in (("row", row, row_dim), ("col", col, col_dim)):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError(f"{name} must be an integer pixel position.")
        if not 0 <= index < input_data.sizes[dim]:
            raise IndexError(f"{name}={index} is outside the {dim} dimension.")
    for name, count in (("n_phi", n_phi), ("n_tau", n_tau)):
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(f"{name} must be an integer.")
        if count < 2:
            raise ValueError(f"{name} must be at least 2.")

    pixel = input_data.isel(
        {row_dim: slice(row, row + 1), col_dim: slice(col, col + 1)}
    )
    converters = {"S": S_to_T3, "C3": C3_to_T3, "T3": lambda data: data}
    T3 = converters[poltype](pixel).isel({row_dim: 0, col_dim: 0}, drop=True).compute()
    if any(not np.isfinite(value.item()) for value in T3.data_vars.values()):
        raise ValueError(
            f"Selected pixel at row={row}, col={col} contains non-finite values."
        )

    phi_values = np.linspace(-90.0, 90.0, n_phi)
    tau_values = np.linspace(-45.0, 45.0, n_tau)
    phi = xr.DataArray(phi_values, dims="phi", coords={"phi": phi_values})
    tau = xr.DataArray(tau_values, dims="tau", coords={"tau": tau_values})
    t11, t12_re, t22, t33 = _rotated_powers(T3, phi, tau)
    copol = (0.5 * (t11 + t22) + t12_re).transpose("phi", "tau")
    xpol = (0.5 * t33).transpose("phi", "tau")

    return xr.Dataset(
        {"copol": copol.astype(np.float32), "xpol": xpol.astype(np.float32)},
        attrs={
            "poltype": "polarimetric_signature",
            "description": "Single-pixel co-polar and cross-polar power signatures.",
            "row": int(row),
            "col": int(col),
        },
    )


def plot_polarimetric_signature(
    signature: xr.Dataset,
    *,
    azimuth_angle: float = -60.0,
    elevation_angle: float = 30.0,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot co-polar and cross-polar signature surfaces in three dimensions.

    Args:
        signature: Dataset returned by :func:`polarimetric_signature`.
        azimuth_angle: Camera azimuth angle in degrees. This controls the plot
            view and is unrelated to the SAR image azimuth coordinate.
        elevation_angle: Camera elevation angle in degrees.

    Returns:
        Matplotlib figure and the co-polar and cross-polar axes. The returned
        handles can be edited or passed to ``Figure.savefig``.
    """
    if not isinstance(signature, xr.Dataset):
        raise TypeError("signature must be an xarray.Dataset.")
    if signature.attrs.get("poltype") != "polarimetric_signature":
        raise ValueError("Input must be a polarimetric signature dataset.")
    if set(signature.data_vars) != {"copol", "xpol"}:
        raise ValueError("Signature dataset must contain copol and xpol variables.")
    for name in ("copol", "xpol"):
        if signature[name].dims != ("phi", "tau"):
            raise ValueError(f"{name} must have dimensions ('phi', 'tau').")
    for name, angle in (
        ("azimuth_angle", azimuth_angle),
        ("elevation_angle", elevation_angle),
    ):
        if not isinstance(angle, Real):
            raise TypeError(f"{name} must be a real number.")
        if not np.isfinite(angle):
            raise ValueError(f"{name} must be finite, got {angle}.")

    tau, phi = np.meshgrid(signature.tau.values, signature.phi.values)
    figure, axes_array = plt.subplots(
        1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"}
    )
    axes = tuple(axes_array)
    titles = {"copol": "Co-polar signature", "xpol": "Cross-polar signature"}
    for axis, name in zip(axes, ("copol", "xpol"), strict=True):
        axis.plot_surface(tau, phi, signature[name].values, cmap="viridis")
        axis.set_xlabel("Ellipticity angle, tau (degrees)")
        axis.set_ylabel("Orientation angle, phi (degrees)")
        axis.set_zlabel("Power")
        axis.set_title(titles[name])
        axis.view_init(elev=elevation_angle, azim=azimuth_angle)

    figure.tight_layout()
    return figure, axes


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


def _correct_orientation_S(S: xr.Dataset, angle: xr.DataArray) -> xr.Dataset:
    """Apply the inverse real rotation to a full Sinclair matrix."""
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    cos_sq = cos_angle**2
    sin_sq = sin_angle**2
    sin_cos = sin_angle * cos_angle

    return xr.Dataset(
        {
            "hh": cos_sq * S.hh + sin_cos * (S.hv + S.vh) + sin_sq * S.vv,
            "hv": -sin_cos * S.hh + cos_sq * S.hv - sin_sq * S.vh + sin_cos * S.vv,
            "vh": -sin_cos * S.hh - sin_sq * S.hv + cos_sq * S.vh + sin_cos * S.vv,
            "vv": sin_sq * S.hh - sin_cos * (S.hv + S.vh) + cos_sq * S.vv,
        },
        attrs={
            "poltype": "S",
            "description": "Orientation-compensated scattering matrix.",
        },
    ).astype(np.complex64)


def _correct_orientation_T3(T3: xr.Dataset, angle: xr.DataArray) -> xr.Dataset:
    """Apply the inverse real rotation to a Pauli coherency matrix."""
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(2.0 * angle_rad)
    sin_angle = -np.sin(2.0 * angle_rad)
    cos_sq = cos_angle**2
    sin_sq = sin_angle**2
    sin_cos = sin_angle * cos_angle

    return xr.Dataset(
        {
            "m11": T3.m11,
            "m12": cos_angle * T3.m12 + sin_angle * T3.m13,
            "m13": -sin_angle * T3.m12 + cos_angle * T3.m13,
            "m22": cos_sq * T3.m22 + 2.0 * sin_cos * T3.m23.real + sin_sq * T3.m33,
            "m23": sin_cos * (T3.m33 - T3.m22)
            + cos_sq * T3.m23
            - sin_sq * T3.m23.conj(),
            "m33": sin_sq * T3.m22 - 2.0 * sin_cos * T3.m23.real + cos_sq * T3.m33,
        },
        attrs={
            "poltype": "T3",
            "description": "Orientation-compensated coherency matrix.",
        },
    ).astype(
        {
            "m11": np.float32,
            "m12": np.complex64,
            "m13": np.complex64,
            "m22": np.float32,
            "m23": np.complex64,
            "m33": np.float32,
        }
    )


def _rotated_powers(
    T3: xr.Dataset, phi: float | xr.DataArray, tau: float | xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return the T11, real T12, T22, and T33 powers after rotation.

    Angles are in degrees. Scalar angles retain the input's spatial chunks;
    DataArray angles broadcast by their named dimensions.
    """
    phi_rad = np.deg2rad(phi)
    tau_rad = np.deg2rad(tau)
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
    return new_t11, new_t12_re, new_t22, new_t33
