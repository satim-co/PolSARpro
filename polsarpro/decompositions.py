import numpy as np
import warnings
from polsarpro.util import boxcar, T3_to_C3, C3_to_T3, S_to_T3, S_to_C3
from polsarpro.util import _boxcar_core, _T3_to_C3_core, _S_to_T3_core, _S_to_C3_core
from polsarpro.util import span
from bench import timeit
import dask.array as da
from dask.diagnostics import ProgressBar


# TODO: better poltype handling
# idea1: read functions return {"data": array, "poltype": "T3"}
# idea2: in the long run, use xarray metadata
# Note: metadata is going to be added to numpy 2.3

# TODO: think about block processing options:
# idea1: block process by creating dask / xarray data_arrays on top of np.ndarray
# idea2: in the long run we should directly pass a data array (and keep numpy compatible?)


# H-A-Alpha decomposition
# TODO: discuss useful flags.
# Are combinations like CombH1mA relevant in python, since they are straightforward, e.g. H*(1-alpha)?
def h_a_alpha(
    input_data: np.ndarray,
    input_poltype: str = "C3",
    boxcar_size: list[int, int] = [3, 3],
    # TODO: add flags
) -> np.ndarray:
    if input_poltype == "C3":
        in_ = C3_to_T3(input_data)
    elif input_poltype == "T3":
        in_ = input_data
    elif input_poltype == "S":
        in_ = S_to_T3(input_data)
    else:
        raise ValueError("Invalid polarimetric type")

    eps = 1e-30

    # TODO: Notes to remove
    # eigh puts eig vals in ascending order
    # The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i]. Will return a matrix object if a is a matrix object.

    # Eigendecomposition
    l, v = np.linalg.eigh(in_)
    l = l[..., ::-1]  # put in descending order
    v = v[..., ::-1]

    # Alpha angle for each mechanism
    arg_sqrt = (v[:, :, 0, :] * v[:, :, 0, :].conj()).real
    alpha_i = np.arccos(np.sqrt(arg_sqrt))
    alpha_i *= 180 / np.pi

    # Pseudo-probabilities (normalized eigenvalues)
    p = np.clip(l / (eps + l.sum(axis=2)[..., None]), eps, 1)

    # Mean alpha
    alpha = np.sum(p * alpha_i, axis=2)

    # Entropy
    H = np.sum(-p * np.log(p), axis=2) / np.float32(np.log(3))

    # Anisotropy
    A = (p[..., 0] - p[..., 1]) / (p[..., 0] + p[..., 1] + eps)

    return H, A, alpha


def h_a_alpha_dask(
    input_data: np.ndarray,
    input_poltype: str = "C3",
    boxcar_size: list[int, int] = [3, 3],
    # TODO: add flags
) -> np.ndarray:
    if input_poltype == "C3":
        in_ = C3_to_T3(input_data)
    elif input_poltype == "T3":
        in_ = input_data
    elif input_poltype == "S":
        in_ = S_to_T3(input_data)
    else:
        raise ValueError("Invalid polarimetric type")


# @timeit
def freeman(
    input_data: np.ndarray,
    input_poltype: str = "C3",
    boxcar_size: list[int, int] = [3, 3],
) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """Applies the Freeman-Durden decomposition. This decomposition is based on physical modeling
      of the covariance matrix and returns 3 components Ps, Pd and Pv which are the powers of resp.
      surface, double bounce and volume backscattering.

    Args:
        input_data (np.ndarray): Input array, may be a covariance, coherency or Sinclair matrix.
        input_poltype (str, optional): Polarimetric input type (covariance "C3", coherency "T3" or Sinclair "S"). Defaults to "C3".
        boxcar_size (list[int, int], optional):  Boxcar dimensions along azimuth and range. Defaults to [3, 3].

    Returns:
        list[np.ndarray, np.ndarray, np.ndarray]: Ps, Pd and Pv components.
    """
    if input_poltype == "C3":
        in_ = input_data
    elif input_poltype == "T3":
        in_ = T3_to_C3(input_data)
    elif input_poltype == "S":
        in_ = S_to_C3(input_data)
    else:
        raise ValueError("Invalid polarimetric type")

    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = boxcar(in_, boxcar_size[0], boxcar_size[1])

    return _compute_freeman_components(in_)


# @timeit
def freeman_dask(
    input_data: np.ndarray,
    input_poltype: str = "C3",
    boxcar_size: list[int, int] = [3, 3],
) -> list[np.ndarray, np.ndarray, np.ndarray]:
    """Applies the Freeman-Durden decomposition. This decomposition is based on physical modeling
      of the covariance matrix and returns 3 components Ps, Pd and Pv which are the powers of resp.
      surface, double bounce and volume backscattering.

    Args:
        input_data (np.ndarray): Input array, may be a covariance, coherency or Sinclair matrix.
        input_poltype (str, optional): Polarimetric input type (covariance "C3", coherency "T3" or Sinclair "S"). Defaults to "C3".
        multilook_size (list[int, int], optional): Multilook dimensions along azimuth and range. Defaults to [1, 1].
        boxcar_size (list[int, int], optional):  Boxcar dimensions along azimuth and range. Defaults to [3, 3].

    Returns:
        list[np.ndarray, np.ndarray, np.ndarray]: Ps, Pd and Pv components.
    """

    in_ = da.from_array(input_data, chunks="auto")
    if input_poltype == "T3":
        in_ = da.map_blocks(
            _T3_to_C3_core,
            in_,
            dtype="complex64",
        )
    elif input_poltype == "S":
        in_ = da.map_blocks(
            _S_to_C3_core,
            in_,
            dtype="complex64",
            chunks=in_.chunksize[:2] + (3, 3),
        )
    else:
        raise ValueError("Invalid polarimetric type")

    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = da.map_overlap(
        _boxcar_core,
        in_,
        dim_az=boxcar_size[0],
        dim_rg=boxcar_size[1],
        depth=(boxcar_size[0], boxcar_size[1]),
        dtype="complex64",
        boundary=1e-30,
    )

    out = _compute_freeman_components_dask(in_)
    with ProgressBar():
        return da.compute(out)[0]


# convenience function not to be called by users
# @timeit
def _compute_freeman_components(
    C3: np.ndarray,
) -> list[np.ndarray, np.ndarray, np.ndarray]:

    eps = 1e-30
    c11 = C3[..., 0, 0].real.copy()
    c13r = C3[..., 0, 2].real.copy()
    c13i = C3[..., 0, 2].imag.copy()
    c22 = C3[..., 1, 1].real.copy()
    c33 = C3[..., 2, 2].real.copy()

    fv = 1.5 * c22
    c11 -= fv
    c33 -= fv
    c13r -= fv / 3

    # intermediate parameters
    fd = np.zeros(c11.shape[:2], dtype="float32")
    fs = np.zeros(c11.shape[:2], dtype="float32")
    alpha = np.zeros(c11.shape[:2], dtype="float32")
    beta = np.zeros(c11.shape[:2], dtype="float32")

    # volume scattering
    cnd1 = (c11 <= eps) | (c33 <= eps)
    # in this case, fs = 0 and fd = 0
    fv[cnd1] = 3 * (c11[cnd1] + c22[cnd1] + c33[cnd1] + 2 * fv[cnd1]) / 8

    # pre-computing c13 power
    pow_c13 = c13r**2 + c13i**2

    # data conditioning for non realizable S_{hh}S_{vv}^* term

    cnd2 = ~cnd1 & (pow_c13 > c11 * c33)
    c13r[cnd2] *= np.sqrt(c11[cnd2] * c33[cnd2] / pow_c13[cnd2])
    c13i[cnd2] *= np.sqrt(c11[cnd2] * c33[cnd2] / pow_c13[cnd2])

    # recompute after conditioning
    pow_c13 = c13r**2 + c13i**2

    # odd bounce dominates
    cnd3 = ~cnd1 & (c13r >= 0)
    alpha[cnd3] = -1
    fd[cnd3] = (c11[cnd3] * c33[cnd3] - pow_c13[cnd3]) / (
        c11[cnd3] + c33[cnd3] + 2 * c13r[cnd3]
    )
    fs[cnd3] = c33[cnd3] - fd[cnd3]
    beta[cnd3] = np.sqrt((fd[cnd3] + c13r[cnd3]) ** 2 + c13i[cnd3] ** 2) / fs[cnd3]

    # even bounce dominates
    cnd4 = ~cnd1 & (c13r < 0)
    beta[cnd4] = 1
    fs[cnd4] = (c11[cnd4] * c33[cnd4] - pow_c13[cnd4]) / (
        c11[cnd4] + c33[cnd4] - 2 * c13r[cnd4]
    )
    fd[cnd4] = c33[cnd4] - fs[cnd4]
    alpha[cnd4] = np.sqrt((fs[cnd4] - c13r[cnd4]) ** 2 + c13i[cnd4] ** 2) / fd[cnd4]

    Ps = fs * (1 + beta**2)
    Pd = fd * (1 + alpha**2)
    Pv = 8 * fv / 3
    sp = span(C3)
    min_span, max_span = np.nanmin(sp), np.nanmax(sp).max()
    min_span = min_span if min_span >= eps else eps
    Ps = Ps.clip(min_span, max_span)
    Pd = Pd.clip(min_span, max_span)
    Pv = Pv.clip(min_span, max_span)

    return Ps, Pd, Pv


def _compute_freeman_components_dask(
    C3: da.Array,
) -> tuple[da.Array, da.Array, da.Array]:
    """Compute Freeman decomposition components.

    Args:
        C3 (da.Array): A dask.Array 3x3 covariance matrix.

    Returns:
        tuple[da.Array, da.Array, da.Array]: Freeman decomposition components (Ps, Pd, Pv).
    """

    eps = 1e-30

    # Extract real and imaginary parts
    c11 = C3[..., 0, 0].real.copy()
    c13r = C3[..., 0, 2].real.copy()
    c13i = C3[..., 0, 2].imag.copy()
    c22 = C3[..., 1, 1].real.copy()
    c33 = C3[..., 2, 2].real.copy()

    fv = 1.5 * c22
    c11 -= fv
    c33 -= fv
    c13r -= fv / 3

    # Volume scattering condition
    cnd1 = (c11 <= eps) | (c33 <= eps)
    fv = da.where(cnd1, 3 * (c11 + c22 + c33 + 2 * fv) / 8, fv)

    # Compute c13 power
    pow_c13 = c13r**2 + c13i**2

    # Data conditioning for non-realizable term
    cnd2 = ~cnd1 & (pow_c13 > c11 * c33)
    arg_sqrt = da.maximum(c11 * c33 / da.maximum(pow_c13, eps), 0)
    scale_factor = da.where(cnd2, da.sqrt(arg_sqrt), 1)
    c13r *= scale_factor
    c13i *= scale_factor

    # Recompute after conditioning
    pow_c13 = c13r**2 + c13i**2

    # Odd bounce dominates
    cnd3 = ~cnd1 & (c13r >= 0)
    alpha = da.where(cnd3, da.float32(-1), da.float32(eps))
    arg_div = c11 + c33 + 2 * c13r
    arg_div = np.where(arg_div == 0, eps, arg_div)
    fd = da.where(cnd3, (c11 * c33 - pow_c13) / arg_div, eps)
    fs = da.where(cnd3, c33 - fd, eps)
    arg_sqrt = da.maximum((fd + c13r) ** 2 + c13i**2, eps)
    arg_div = np.where(fs == 0, eps, fs)
    beta = da.where(cnd3, da.sqrt(arg_sqrt) / arg_div, eps)

    # Even bounce dominates
    cnd4 = ~cnd1 & (c13r < 0)
    beta = da.where(cnd4, 1, beta)
    arg_div = c11 + c33 - 2 * c13r
    arg_div = np.where(arg_div == 0, eps, arg_div)
    fs = da.where(cnd4, (c11 * c33 - pow_c13) / arg_div, fs)
    fd = da.where(cnd4, c33 - fs, fd)
    arg_sqrt = da.maximum((fs - c13r) ** 2 + c13i**2, eps)
    arg_div = np.where(fd == 0, eps, fd)
    alpha = da.where(cnd4, da.sqrt(arg_sqrt) / arg_div, alpha)

    # Compute Freeman components
    Ps = fs * (1 + beta**2)
    Pd = fd * (1 + alpha**2)
    Pv = 8 * fv / 3

    sp = span(C3)
    min_span, max_span = da.nanmin(sp), da.nanmax(sp)
    min_span = max(min_span, eps)

    Ps = da.where(Ps <= min_span, min_span, da.where(Ps > max_span, max_span, Ps))
    Pd = da.where(Pd <= min_span, min_span, da.where(Pd > max_span, max_span, Pd))
    Pv = da.where(Pv <= min_span, min_span, da.where(Pv > max_span, max_span, Pv))

    return Ps, Pd, Pv
