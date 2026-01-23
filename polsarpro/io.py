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

# Description: module containing read / write functions

"""

import logging
from pathlib import Path
import numpy as np
import xarray as xr
import xarray
from polsarpro.auxil import validate_dataset


log = logging.getLogger(__name__)

SLC_NO_TAG = ""
ALLOWED_SLC_TAGS = ("S1", "S2", "S3")

def open_netcdf_beam(file_path: str | Path) -> xarray.Dataset:
    """Open data in the NetCDF-BEAM format exported by SNAP and create a valid python PolSARpro Dataset. Also works for complex matrix datasets written with polmat_to_netcdf.

    Args:
        file_path (str|Path): path of the input file.

    Returns:
        xarray.Dataset: output dataset with python PolSARpro specific metadata.

    Note:
        Only polarimetric data is allowed.
        Supported polarimetric types are 'S' scattering matrix, 'C3' and 'C4' covariance matrices, 'T3' and 'T4' coherency matrices as well as 'C2' dual-polarimetric covariance matrix.
    """

    # use chunks to create dask instead of numpy arrays
    ds = xr.open_dataset(file_path, chunks={})
    var_names = set(ds.data_vars)

    # file comes from SNAP, else assume python PolSARpro dataset
    if "poltype" not in ds.attrs:
        meta = ds.metadata.attrs
        # check if image is in the SAR geometry or in geographic coordinates
        is_geocoded = bool(meta["Abstracted_Metadata:is_terrain_corrected"])

    T3_vars = {
        "T11",
        "T22",
        "T33",
        "T12_real",
        "T12_imag",
        "T13_real",
        "T13_imag",
        "T23_real",
        "T23_imag",
    }
    C2_vars = {
        "C11",
        "C22",
        "C12_real",
        "C12_imag",
    }
    C3_vars = {
        "C11",
        "C22",
        "C33",
        "C12_real",
        "C12_imag",
        "C13_real",
        "C13_imag",
        "C23_real",
        "C23_imag",
    }
    C4_vars = C3_vars.union(
        {
            "C14_real",
            "C14_imag",
            "C24_real",
            "C24_imag",
            "C34_real",
            "C34_imag",
            "C44",
        }
    )
    T4_vars = T3_vars.union(
        {
            "T14_real",
            "T14_imag",
            "T24_real",
            "T24_imag",
            "T34_real",
            "T34_imag",
            "T44",
        }
    )

    # S matrix - SLC is a special case as band names may vary
    tag = _parse_slc_bands(var_names)

    # infers polarimetric type from dataset variable names
    data = {}

    if C4_vars.issubset(var_names):
        poltype = "C4"
        description = "Covariance matrix (4x4)"
        data["m11"] = ds.C11
        data["m22"] = ds.C22
        data["m33"] = ds.C33
        data["m44"] = ds.C44
        data["m12"] = ds.C12_real + 1j * ds.C12_imag
        data["m13"] = ds.C13_real + 1j * ds.C13_imag
        data["m14"] = ds.C14_real + 1j * ds.C14_imag
        data["m23"] = ds.C23_real + 1j * ds.C23_imag
        data["m24"] = ds.C24_real + 1j * ds.C24_imag
        data["m34"] = ds.C34_real + 1j * ds.C34_imag
    elif C3_vars.issubset(var_names):
        poltype = "C3"
        description = "Covariance matrix (3x3)"
        data["m11"] = ds.C11
        data["m22"] = ds.C22
        data["m33"] = ds.C33
        data["m12"] = ds.C12_real + 1j * ds.C12_imag
        data["m13"] = ds.C13_real + 1j * ds.C13_imag
        data["m23"] = ds.C23_real + 1j * ds.C23_imag
    elif C2_vars.issubset(var_names):
        poltype = "C2"
        description = "Covariance matrix (2x2)"
        data["m11"] = ds.C11
        data["m22"] = ds.C22
        data["m12"] = ds.C12_real + 1j * ds.C12_imag
    elif T4_vars.issubset(var_names):
        poltype = "T4"
        description = "Coherency matrix (4x4)"
        data["m11"] = ds.T11
        data["m22"] = ds.T22
        data["m33"] = ds.T33
        data["m44"] = ds.T44
        data["m12"] = ds.T12_real + 1j * ds.T12_imag
        data["m13"] = ds.T13_real + 1j * ds.T13_imag
        data["m14"] = ds.T14_real + 1j * ds.T14_imag
        data["m23"] = ds.T23_real + 1j * ds.T23_imag
        data["m24"] = ds.T24_real + 1j * ds.T24_imag
        data["m34"] = ds.T34_real + 1j * ds.T34_imag
    elif T3_vars.issubset(var_names):
        poltype = "T3"
        description = "Coherency matrix (3x3)"
        data["m11"] = ds.T11
        data["m22"] = ds.T22
        data["m33"] = ds.T33
        data["m12"] = ds.T12_real + 1j * ds.T12_imag
        data["m13"] = ds.T13_real + 1j * ds.T13_imag
        data["m23"] = ds.T23_real + 1j * ds.T23_imag
    elif tag is not None:
        poltype = "S"
        description = "Scattering matrix"
        if tag == SLC_NO_TAG: # Usual SNAP band names
            data["hh"] = ds.i_HH + 1j * ds.q_HH
            data["hv"] = ds.i_HV + 1j * ds.q_HV
            data["vh"] = ds.i_VH + 1j * ds.q_VH
            data["vv"] = ds.i_VV + 1j * ds.q_VV
        else: # BIOMASS data
            data["hh"] = ds[f"i_{tag}_HH"] + 1j * ds[f"q_{tag}_HH"]
            data["hv"] = ds[f"i_{tag}_HV"] + 1j * ds[f"q_{tag}_HV"]
            data["vh"] = ds[f"i_{tag}_VH"] + 1j * ds[f"q_{tag}_VH"]
            data["vv"] = ds[f"i_{tag}_VV"] + 1j * ds[f"q_{tag}_VV"]
    else:
        raise ValueError(
            "Polarimetric type not recognized. Possible types are 'S', 'C2', 'C3', 'T3', 'C4', 'T4' matrices."
        )

    # make a new dataset with PolSARpro metadata
    if "poltype" not in ds.attrs:
        ds_out = xr.Dataset(
            data, attrs={"poltype": poltype, "description": description}
        )

        # coordinates: "y" & "x" for SAR geometry, "lat" & "lon" for geocoded
        if {"y", "x"}.issubset(ds.dims) and not is_geocoded:
            # make sure coordinates are only pixel indices
            return ds_out.assign_coords(
                {"y": np.arange(ds.sizes["y"]), "x": np.arange(ds.sizes["x"])}
            ).drop_vars(("lon", "lat"), errors="ignore")
        else:
            return ds_out
    # input data was already a python PolSARpro dataset
    else:
        ds_out = xr.Dataset(
            data,
            attrs={"poltype": poltype, "description": description},
            coords=ds.coords,
        )
        return ds_out


def polmat_to_netcdf(ds: xarray.Dataset, file_path: str | Path):
    """Writes complex polarimetric matrices to a nectdf file.
    Due to the lack of complex number support in netcdf files, real and imaginary parts are stored separately.
    Variable names follow the netcdf-beam convention used by SNAP.

    Args:
        ds (xarray.Dataset): input dataset with polarimetric matrix elements.
        file_path (str|Path): output file path.
    """

    poltype = validate_dataset(ds, allowed_poltypes=["S", "C2", "C3", "T3", "C4", "T4"])

    data_out = {}
    if poltype == "S":
        data_out["i_HH"] = ds.hh.real
        data_out["q_HH"] = ds.hh.imag
        data_out["i_HV"] = ds.hv.real
        data_out["q_HV"] = ds.hv.imag
        data_out["i_VH"] = ds.vh.real
        data_out["q_VH"] = ds.vh.imag
        data_out["i_VV"] = ds.vv.real
        data_out["q_VV"] = ds.vv.imag
    else:
        # automatically extract data for T and C matrices
        n = int(poltype[-1])
        name = poltype[0]
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                arr = ds[f"m{i}{j}"]
                if i == j:
                    data_out[f"{name}{i}{j}"] = arr
                else:
                    data_out[f"{name}{i}{j}_real"] = arr.real
                    data_out[f"{name}{i}{j}_imag"] = arr.imag

    # make a new dataset with PolSARpro metadata
    # Preserve chunking when writing
    encoding = {var: {"chunksizes": data_out[var].data.chunksize, "zlib": True} for var in data_out}
    ds_out = xr.Dataset(
        # extract dask arrays and dims from xarray data
        {k: (ds.dims, v.data) for k, v in data_out.items()},
        attrs={"poltype": poltype, "description": ds.description},
        coords=ds.coords,
    )
    ds_out.to_netcdf(file_path, encoding=encoding)

# --- helper functions not to be called outside of the module

def _parse_slc_bands(var_names: set[str]) -> str | None:
    pol_list = ("HH", "HV", "VH", "VV")
    tags = ("S1", "S2", "S3")

    # Legacy SNAP-style SLC (no tag)
    base_set = {f"{x}_{p}" for p in pol_list for x in ("i", "q")}
    if base_set.issubset(var_names):
        return SLC_NO_TAG

    # Tagged SLC variants
    found_tags = []
    for tag in tags:
        tagged_set = {f"{x}_{tag}_{p}" for p in pol_list for x in ("i", "q")}
        if tagged_set.issubset(var_names):
            found_tags.append(tag)

    if len(found_tags) > 1:
        raise ValueError(
            f"Multiple SLC tags found: {found_tags}"
        )

    if found_tags:
        return found_tags[0]

    # Not an SLC dataset
    return None