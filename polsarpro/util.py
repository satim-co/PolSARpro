from pathlib import Path
import logging
import numpy as np
from scipy.ndimage import convolve
import dask.array as da
from bench import timeit

log = logging.getLogger(__name__)


@timeit
def boxcar(img: np.ndarray, dim_az: int, dim_rg: int):
    """
    Apply a boxcar filter to an image.

    Args:
        img (complex or real array): Input image with arbitrary number of dimensions, shape (naz, nrg, ...).
        dim_az (int): Size in azimuth of the filter.
        dim_rg (int): Size in range of the filter.

    Returns:
        complex or real array: Filtered image, shape (naz, nrg, ...).

    Note:
        The filter is always applied along 2 dimensions (azimuth, range). Please ensure to provide a valid image.
    """
    if type(dim_az) != int and type(dim_rg) != int:
        raise ValueError("dimaz and dimrg must be integers")
    if (dim_az < 1) or (dim_rg < 1):
        raise ValueError("dimaz and dimrg must be strictly positive")
    if img.ndim < 2:
        raise ValueError("Input must be at least of dimension 2")

    return _boxcar_core(img, dim_az, dim_rg)


@timeit
def boxcar_dask(img: np.ndarray, dim_az: int, dim_rg: int):
    """
    Apply a boxcar filter to an image.

    Args:
        img (complex or real array): Input image with arbitrary number of dimensions, shape (naz, nrg, ...).
        dim_az (int): Size in azimuth of the filter.
        dim_rg (int): Size in range of the filter.

    Returns:
        complex or real array: Filtered image, shape (naz, nrg, ...).

    Note:
        The filter is always applied along 2 dimensions (azimuth, range). Please ensure to provide a valid image.
    """
    if type(dim_az) != int and type(dim_rg) != int:
        raise ValueError("dimaz and dimrg must be integers")
    if (dim_az < 1) or (dim_rg < 1):
        raise ValueError("dimaz and dimrg must be strictly positive")
    if img.ndim < 2:
        raise ValueError("Input must be at least of dimension 2")

    process_args = dict(
        dim_az=dim_az,
        dim_rg=dim_rg,
        depth=(dim_az, dim_rg),
    )
    da_in = da.map_overlap(
        _boxcar_core,
        da.from_array(img, chunks=(500, 500, 3, 3)),
        **process_args,
        dtype="complex64",
    )

    return np.asarray(da_in)


def _boxcar_core(img, dim_az, dim_rg):
    n_extra_dims = img.ndim - 2

    ker_dtype = img.dtype if not np.iscomplexobj(img) else img.real.dtype
    
    # this convolution mode reduces error between C and python implementations
    mode = "constant"
    if (dim_az > 1) or (dim_rg > 1):
        # avoid nan propagation
        msk = np.isnan(img)
        img_ = img.copy()
        img_[msk] = 0
        ker = np.ones((dim_az, dim_rg), dtype=ker_dtype) / (dim_az * dim_rg)
        ker = np.expand_dims(ker, axis=tuple(range(2, 2 + n_extra_dims)))
        if np.iscomplexobj(img_):
            imgout = convolve(img_.real, ker, mode=mode) + 1j * convolve(
                img_.imag, ker, mode=mode
            )
            imgout[msk] = np.nan + 1j * np.nan
        else:
            imgout = convolve(img_, ker, mode=mode)
            imgout[msk] = np.nan
        return imgout
    else:
        return img


@timeit
def multilook(img: np.ndarray, dim_az: int, dim_rg: int):
    """
    Computes the m by n presummed image.

    Args:
        img (array-like): Input image array with shape (naz, nrg,...).
        dim_az (int): Number of lines to sum. Must be an integer >= 1.
        dim_rg (int): Number of columns to sum. Must be an integer >= 1.

    Returns:
        array: Presummed image array with shape (M, N,...), where M and N are the largest multiples of dim_az and dim_rg that are less than or equal to img.shape[0] and img.shape[1], respectively.
    Note:
        Returns the input array if dim_az==1 and dim_rg==1.
    """
    # Check if m and n are integers >= 1
    if not isinstance(dim_az, int) or not isinstance(dim_rg, int):
        raise TypeError("Parameters m and n must be integers.")
    if dim_az < 1 or dim_rg < 1:
        raise ValueError(
            "Parameters m and n must be integers greater than or equal to 1."
        )

    # Check if m and n are valid in relation to the image dimensions
    if dim_az > img.shape[0] or dim_rg > img.shape[1]:
        raise ValueError(
            "Cannot presum with these parameters; m or n is too large for the image dimensions."
        )

    # skip if m = n = 1, avoids conditionals in calls
    if (dim_az > 1) or (dim_rg > 1):
        M = (img.shape[0] // dim_az) * dim_az
        N = (img.shape[1] // dim_rg) * dim_rg

        img_trimmed = img[:M, :N]

        s = img_trimmed[::dim_az].copy()  # Make a copy once for efficiency
        for i in range(1, dim_az):
            s += img_trimmed[i::dim_az]

        t = s[:, ::dim_rg].copy()
        for j in range(1, dim_rg):
            t += s[:, j::dim_rg]

        return t / float(dim_az * dim_rg)
    else:
        return img


def read_T3(input_dir: str):
    """Reads a T3 matrix in the PolSARPro format.

    Args:
        input_dir (str): Input directory containing the elements of the T3 matrix and the configuration file config.txt.
    """
    pth_cfg = input_dir / "config.txt"

    # TODO add exceptions

    # read config and store in a dict
    dict_cfg = {}
    with open(pth_cfg, "r") as f:
        lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines if "---" not in line]
        for i in range(0, len(lines), 2):
            dict_cfg[lines[i]] = lines[i + 1]
    
    valid_mask_path = input_dir / "mask_valid_pixels.bin"
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
                    input_dir / f"{elt}.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))
            else:
                T3[..., r, c].real = np.fromfile(
                    input_dir / f"{elt}_real.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))
                T3[..., r, c].imag = np.fromfile(
                    input_dir / f"{elt}_imag.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))

    T3[..., 1, 0] = T3[..., 0, 1].conj()
    T3[..., 2, 0] = T3[..., 0, 2].conj()
    T3[..., 2, 1] = T3[..., 1, 2].conj()

    if is_valid_mask:
        T3[~valid_mask] = np.nan + 1j*np.nan

    return T3


def read_C3(input_dir: str):
    """Reads a C3 matrix in the PolSARPro format.

    Args:
        input_dir (str): Input directory containing the elements of the C3 matrix and the configuration file config.txt.
    """
    pth_cfg = input_dir / "config.txt"

    # TODO add exceptions

    # read config and store in a dict
    dict_cfg = {}
    with open(pth_cfg, "r") as f:
        lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines if "---" not in line]
        for i in range(0, len(lines), 2):
            dict_cfg[lines[i]] = lines[i + 1]

    valid_mask_path = input_dir / "mask_valid_pixels.bin"
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
                    input_dir / f"{elt}.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))
            else:
                C3[..., r, c].real = np.fromfile(
                    input_dir / f"{elt}_real.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))
                C3[..., r, c].imag = np.fromfile(
                    input_dir / f"{elt}_imag.bin", dtype="float32", count=naz * nrg
                ).reshape((naz, nrg))

    C3[..., 1, 0] = C3[..., 0, 1].conj()
    C3[..., 2, 0] = C3[..., 0, 2].conj()
    C3[..., 2, 1] = C3[..., 1, 2].conj()

    if is_valid_mask:
        C3[~valid_mask] = np.nan + 1j*np.nan

    return C3


def read_PSP_bin(file_name: str, dtype: str = "float32"):

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


def S_to_C3(S):
    pass


@timeit
def T3_to_C3(T3: np.ndarray) -> np.ndarray:
    """Converts the Pauli coherency matrix T3 to the lexicographic covariance matrix.

    Args:
        T3 (np.ndarray): input image of coherency matrices with shape (naz, nrg, 3, 3)

    Returns:
        np.ndarray: C3 covariance matrix
    """
    if T3.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if T3.shape[-2:] != (3, 3):
        raise ValueError("T3 must have a shape like (naz, nrg, 3, 3)")

    return _T3_to_C3_core(T3=T3)


@timeit
def T3_to_C3_dask(T3: np.ndarray) -> np.ndarray:
    """Converts the Pauli coherency matrix T3 to the lexicographic covariance matrix.

    Args:
        T3 (np.ndarray): input image of coherency matrices with shape (naz, nrg, 3, 3)

    Returns:
        np.ndarray: C3 covariance matrix
    """
    if T3.ndim != 4:
        raise ValueError("A matrix-valued image is expected (dimension 4)")
    if T3.shape[-2:] != (3, 3):
        raise ValueError("T3 must have a shape like (naz, nrg, 3, 3)")

    da_in = da.map_blocks(
        _T3_to_C3_core,
        da.from_array(T3, chunks=(500, 500, -1, -1)),
        dtype="complex64",
    )

    return np.asarray(da_in)


def _T3_to_C3_core(T3: np.ndarray) -> np.ndarray:

    C3 = np.zeros_like(T3, dtype="complex64")

    # Reproject T3 matrix in the lexicographic basis
    C3[..., 0, 0] = (T3[..., 0, 0] + 2 * T3[..., 0, 1].real + T3[..., 1, 1]) / 2
    C3[..., 0, 1] = (T3[..., 0, 2] + T3[..., 1, 2]) / np.sqrt(2)
    C3[..., 0, 2] = (T3[..., 0, 0].real - T3[..., 1, 1].real) / 2
    C3[..., 0, 2] += 1j * -T3[..., 0, 1].imag
    C3[..., 1, 1] = T3[..., 2, 2]
    C3[..., 1, 2] = (T3[..., 0, 2].real - T3[..., 1, 2].real) / np.sqrt(2)
    C3[..., 1, 2] += 1j * (-T3[..., 0, 2].imag + T3[..., 1, 2].imag) / np.sqrt(2)
    C3[..., 2, 2] = (T3[..., 0, 0] - 2 * T3[..., 0, 1].real + T3[..., 1, 1]) / 2

    C3[..., 1, 0] = np.conj(C3[..., 0, 1])
    C3[..., 2, 0] = np.conj(C3[..., 0, 2])
    C3[..., 2, 1] = np.conj(C3[..., 1, 2])

    return C3


def vec_to_mat(vec):
    return vec[None, :] * vec[:, None].conj()


@timeit
def span(M):
    if M.ndim != 4:
        raise ValueError("Matrix valued image is expected (dimension 4)")
    if M.shape[2] != M.shape[3]:
        raise ValueError("Input shape [naz, nrg, N, N] expected")

    return M.diagonal(axis1=2, axis2=3).real.sum(axis=-1)
