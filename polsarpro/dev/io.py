import logging
import numpy as np
import xarray as xr
from pathlib import Path

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
