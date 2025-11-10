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


def read_T3_psp(input_dir: str):
    """Reads a T3 matrix in the PolSARPro format.

    Args:
        input_dir (str): Input directory containing the elements of the T3 matrix and the configuration file config.txt.
    """

    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Directory {input_dir} not found.")

    pth_cfg = input_path / "config.txt"
    if pth_cfg.is_file():
        # read config and store in a dict
        dict_cfg = {}
        with open(pth_cfg, "r") as f:
            lines = f.readlines()
            lines = [line.replace("\n", "") for line in lines if "---" not in line]
            for i in range(0, len(lines), 2):
                dict_cfg[lines[i]] = lines[i + 1]
    else:
        raise FileNotFoundError("Configuration file config.txt does not exist.")

    valid_mask_path = input_path / "mask_valid_pixels.bin"
    is_valid_mask = True
    if valid_mask_path.is_file():
        valid_mask = read_psp_bin(valid_mask_path).astype(bool)
    else:
        is_valid_mask = False
        log.info("Valid pixel mask not found, skipping.")

    # image dimensions (azimuth, range)
    naz = int(dict_cfg["Nrow"])
    nrg = int(dict_cfg["Ncol"])

    T3 = {}

    for c in range(3):
        for r in range(c + 1):
            elt = f"T{r+1}{c+1}"
            if r == c:
                data = read_psp_bin(input_path / f"{elt}.bin", dtype="float32")

                if is_valid_mask:
                    data = np.where(valid_mask, data, np.nan)
                T3[f"m{r+1}{c+1}"] = (("y", "x"), data)
            else:
                data_real = read_psp_bin(
                    input_path / f"{elt}_real.bin", dtype="float32"
                )
                data_imag = read_psp_bin(
                    input_path / f"{elt}_imag.bin", dtype="float32"
                )
                if is_valid_mask:
                    data = np.where(valid_mask, data, np.nan + 1j * np.nan)
                T3[f"m{r+1}{c+1}"] = (("y", "x"), data_real + 1j * data_imag)

    attrs = {"poltype": "T3", "description": "Coherency matrix (3x3)"}
    return xr.Dataset(T3, attrs=attrs).chunk({"x": 512, "y": 512})


def read_C3_psp(input_dir: str):
    """Reads a C3 matrix in the PolSARpro format.

    Args:
        input_dir (str): Input directory containing the elements of the C3 matrix and the configuration file config.txt.
    """

    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Directory {input_dir} not found.")

    pth_cfg = input_path / "config.txt"
    if pth_cfg.is_file():
        # read config and store in a dict
        dict_cfg = {}
        with open(pth_cfg, "r") as f:
            lines = f.readlines()
            lines = [line.replace("\n", "") for line in lines if "---" not in line]
            for i in range(0, len(lines), 2):
                dict_cfg[lines[i]] = lines[i + 1]
    else:
        raise FileNotFoundError("Configuration file config.txt does not exist.")

    valid_mask_path = input_path / "mask_valid_pixels.bin"
    is_valid_mask = True
    if valid_mask_path.is_file():
        valid_mask = read_psp_bin(valid_mask_path).astype(bool)
    else:
        is_valid_mask = False
        log.info("Valid pixel mask not found, skipping.")

    # image dimensions (azimuth, range)
    naz = int(dict_cfg["Nrow"])
    nrg = int(dict_cfg["Ncol"])

    C3 = {}

    for c in range(3):
        for r in range(c + 1):
            elt = f"C{r+1}{c+1}"
            if r == c:
                data = read_psp_bin(input_path / f"{elt}.bin", dtype="float32")

                if is_valid_mask:
                    data = np.where(valid_mask, data, np.nan)
                C3[f"m{r+1}{c+1}"] = (("y", "x"), data)
            else:
                data_real = read_psp_bin(
                    input_path / f"{elt}_real.bin", dtype="float32"
                )
                data_imag = read_psp_bin(
                    input_path / f"{elt}_imag.bin", dtype="float32"
                )
                if is_valid_mask:
                    data = np.where(valid_mask, data, np.nan + 1j * np.nan)
                C3[f"m{r+1}{c+1}"] = (("y", "x"), data_real + 1j * data_imag)

    attrs = {"poltype": "C3", "description": "Covariance matrix (3x3)"}
    return xr.Dataset(C3, attrs=attrs).chunk("auto").chunk({"x": 512, "y": 512})


def read_psp_bin(file_name: str, dtype: str = "float32"):
    """Reads a raster bin file in the PolSARPro format.

    Args:
        file_name (str): Input file.
    Note:
        The parent directory must contain a config.txt file.
    """
    file_path = Path(file_name)
    input_dir = file_path.parent

    pth_cfg = input_dir / "config.txt"
    if pth_cfg.exists():
        # read config and store in a dict
        cfg = {}
        with open(pth_cfg, "r") as f:
            lines = f.readlines()
            lines = [line.replace("\n", "") for line in lines if "---" not in line]
            for i in range(0, len(lines), 2):
                cfg[lines[i]] = lines[i + 1]
    else:
        raise FileNotFoundError("Configuration file config.txt does not exist.")

    # image dimensions (azimuth, range)
    naz = int(cfg["Nrow"])
    nrg = int(cfg["Ncol"])

    return np.fromfile(file_path, dtype=dtype, count=naz * nrg).reshape((naz, nrg))


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

    # Scattering matrix
    pol_list = ("H", "V")
    # construct the set of all i_HH, q_HH, i_HV...
    S_vars = {f"{x}_{p1}{p2}" for p1 in pol_list for p2 in pol_list for x in ("i", "q")}
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

    # infers polarimetric type from dataset variable names
    data = {}
    if S_vars.issubset(var_names):
        poltype = "S"
        description = "Scattering matrix"
        data["hh"] = ds.i_HH + 1j * ds.q_HH
        data["hv"] = ds.i_HV + 1j * ds.q_HV
        data["vv"] = ds.i_VV + 1j * ds.q_VV
        data["vh"] = ds.i_VH + 1j * ds.q_VH
    elif C4_vars.issubset(var_names):
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

    validate_dataset(ds, allowed_poltypes=["S", "C2", "C3", "T3", "C4", "T4"])
    poltype = ds.poltype

    data_out = {}
    if poltype == "S":
        data_out["i_HH"] = ds.hh.real
        data_out["q_HH"] = ds.hh.imag
        data_out["i_HV"] = ds.hv.real
        data_out["q_HV"] = ds.hv.imag
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
    encoding = {var: {"chunksizes": data_out[var].data.chunksize} for var in data_out}
    ds_out = xr.Dataset(
        # extract dask arrays and dims from xarray data
        {k: (ds.dims, v.data) for k, v in data_out.items()},
        attrs={"poltype": poltype, "description": ds.description},
        coords=ds.coords,
    )
    ds_out.to_netcdf(file_path, encoding=encoding)

# Reader below does not work due to a bug in gdal PALSAR driver (adds a line of zeros at the top)
# def read_demo_data(data_dir: str) -> xarray.Dataset:
#     """Reads ALOS-1 PALSAR San Francisco demo dataset.

#     Args:
#         data_dir (str): Input directory

#     Returns:
#         xarray.Dataset: Returns the elements of the Sinclair scattering matrix in a dataset.
#     Note:
#         The dataset must be downloaded at https://ietr-lab.univ-rennes1.fr/polsarpro-bio/sample_datasets/dataset/SAN_FRANCISCO_ALOS1.zip and unzipped in the directory of your choice.
#     """

#     # read metadata to extract the radiometric calibaration factor
#     def extract_calibration_factor(data_path):
#         from math import sqrt, pow

#         file_path = data_path / "ceos_leader.txt"
#         with open(file_path, "r") as file:
#             for line in file:
#                 if "Calibration Factor:" in line:
#                     parts = line.strip().split()
#                     # Assume the last element is the calibration value
#                     try:
#                         value = float(parts[-1])
#                         # dB to linear
#                         value = sqrt(pow(10.0, (value - 32.0) / 10.0))
#                         return value
#                     except ValueError("Invalid metadata."):
#                         continue
#         return None

#     data_path = Path(data_dir)
#     calfac = extract_calibration_factor(data_path=data_path)
#     # silence warnings when dataset is in the SAR geometry
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
#         # open with chunks to use dask arrays
#         S = riox.open_rasterio(
#             data_path / "VOL-ALPSRP202350750-P1.1__A",
#             chunks="auto",
#             # chunks=(-1, "auto", "auto"),
#             band_as_variable=True,
#         ).rename(band_1="hh", band_2="hv", band_3="vh", band_4="vv")
#     # convert digital number to RCS
#     S *= calfac

#     # S['hv'] = 0.5*(S.hv + S.vh)
#     # S['vh'] = S.hv
#     # set polarimetric data type
#     S.attrs = {"poltype": "S", "description": "Scattering matrix"}
#     return S
