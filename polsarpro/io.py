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
"""

import logging
from pathlib import Path
import numpy as np
import rioxarray as riox
import warnings
from rasterio.errors import NotGeoreferencedWarning
import xarray


log = logging.getLogger(__name__)


def read_T3(input_dir: str):
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
        valid_mask = read_PSP_bin(valid_mask_path).astype(bool)
    else:
        is_valid_mask = False
        log.info("Valid pixel mask not found, skipping.")

    # image dimensions (azimuth, range)
    naz = int(dict_cfg["Nrow"])
    nrg = int(dict_cfg["Ncol"])

    T3 = np.zeros((naz, nrg, 3, 3), dtype="complex64")

    for c in range(3):
        for r in range(c + 1):
            elt = f"T{r+1}{c+1}"
            if r == c:
                T3[..., r, c].real = np.fromfile(
                    input_path / f"{elt}.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))
            else:
                T3[..., r, c].real = np.fromfile(
                    input_path / f"{elt}_real.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))
                T3[..., r, c].imag = np.fromfile(
                    input_path / f"{elt}_imag.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))

    T3[..., 1, 0] = T3[..., 0, 1].conj()
    T3[..., 2, 0] = T3[..., 0, 2].conj()
    T3[..., 2, 1] = T3[..., 1, 2].conj()

    if is_valid_mask:
        T3[~valid_mask] = np.nan + 1j * np.nan

    return T3


def read_C3(input_dir: str):
    """Reads a C3 matrix in the PolSARPro format.

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
        valid_mask = read_PSP_bin(valid_mask_path).astype(bool)
    else:
        is_valid_mask = False
        log.info("Valid pixel mask not found, skipping.")

    # image dimensions (azimuth, range)
    naz = int(dict_cfg["Nrow"])
    nrg = int(dict_cfg["Ncol"])

    C3 = np.zeros((naz, nrg, 3, 3), dtype="complex64")

    for c in range(3):
        for r in range(c + 1):
            elt = f"C{r+1}{c+1}"
            if r == c:
                C3[..., r, c].real = np.fromfile(
                    input_path / f"{elt}.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))
            else:
                C3[..., r, c].real = np.fromfile(
                    input_path / f"{elt}_real.bin",
                    dtype="float32",
                    count=naz * nrg,
                ).reshape((naz, nrg))
                C3[..., r, c].imag = np.fromfile(
                    input_path / f"{elt}_imag.bin",
                    dtype="float32",
                    count=naz * nrg,
                ).reshape((naz, nrg))

    C3[..., 1, 0] = C3[..., 0, 1].conj()
    C3[..., 2, 0] = C3[..., 0, 2].conj()
    C3[..., 2, 1] = C3[..., 1, 2].conj()

    if is_valid_mask:
        C3[~valid_mask] = np.nan + 1j * np.nan

    return C3


def read_PSP_bin(file_name: str, dtype: str = "float32"):
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
        dict_cfg = {}
        with open(pth_cfg, "r") as f:
            lines = f.readlines()
            lines = [line.replace("\n", "") for line in lines if "---" not in line]
            for i in range(0, len(lines), 2):
                dict_cfg[lines[i]] = lines[i + 1]
    else:
        raise FileNotFoundError("Configuration file config.txt does not exist.")

    # image dimensions (azimuth, range)
    naz = int(dict_cfg["Nrow"])
    nrg = int(dict_cfg["Ncol"])

    return np.fromfile(file_path, dtype=dtype, count=naz * nrg).reshape((naz, nrg))


def read_demo_data(data_dir: str) -> xarray.Dataset:
    """Reads ALOS-1 PALSAR San Francisco demo dataset.

    Args:
        data_dir (str): Input directory

    Returns:
        xarray.Dataset: Returns the elements of the Sinclair scattering matrix in a dataset.
    Note:
        The dataset must be downloaded at https://ietr-lab.univ-rennes1.fr/polsarpro-bio/sample_datasets/dataset/SAN_FRANCISCO_ALOS1.zip and unzipped in the directory of your choice.
    """

    # read metadata to extract the radiometric calibaration factor
    def extract_calibration_factor(data_path):
        from math import sqrt, pow

        file_path = data_path / "ceos_leader.txt"
        with open(file_path, "r") as file:
            for line in file:
                if "Calibration Factor:" in line:
                    parts = line.strip().split()
                    # Assume the last element is the calibration value
                    try:
                        value = float(parts[-1])
                        # dB to linear
                        value = sqrt(pow(10.0, (value - 32.0) / 10.0))
                        return value
                    except ValueError("Invalid metadata."):
                        continue
        return None

    data_path = Path(data_dir)
    calfac = extract_calibration_factor(data_path=data_path)
    # silence warnings when dataset is in the SAR geometry
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        # open with chunks to use dask arrays
        S = riox.open_rasterio(
            data_path / "VOL-ALPSRP202350750-P1.1__A",
            chunks="auto",
            # chunks=(-1, "auto", "auto"),
            band_as_variable=True,
        ).rename(band_1="hh", band_2="hv", band_3="vh", band_4="vv")
    # convert digital number to RCS
    S *= calfac
    # set polarimetric data type
    S.attrs = {"poltype": "S", "description": "Scattering matrix"}
    return S
