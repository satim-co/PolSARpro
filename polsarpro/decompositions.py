import numpy as np
from polsarpro.util import T3_to_C3, T3_to_C3_dask 
from polsarpro.util import S_to_C3, S_to_C3_dask 
from polsarpro.util import span, boxcar, boxcar_dask
from bench import timeit
import dask.array as da

# TODO: better poltype handling
# idea1: read functions return {"data": array, "poltype": "T3"}
# idea2: in the long run, use xarray metadata
# Note: metadata is going to be added to numpy 2.3

# TODO: think about block processing options:
# idea1: block process by creating dask / xarray data_arrays on top of np.ndarray
# idea2: in the long run we should directly pass a data array (and keep numpy compatible?)


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


@timeit
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
    if input_poltype == "C3":
        in_ = input_data
    elif input_poltype == "T3":
        in_ = T3_to_C3_dask(input_data)
    elif input_poltype == "S":
        in_ = S_to_C3_dask(input_data)
    else:
        raise ValueError("Invalid polarimetric type")

    # pre-processing step, it is recommended to filter the matrices to mitigate speckle effects
    in_ = boxcar_dask(in_, boxcar_size[0], boxcar_size[1])

    da_in = da.from_array(in_, chunks=(500, 500, 3, 3))
    out = _compute_freeman_components_dask(da_in)
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
        C3 (da.Array): A NumPy 3x3 covariance matrix.

    Returns:
        tuple[da.Array, da.Array, da.Array]: Freeman decomposition components (Ps, Pd, Pv).
    """

    eps = 1e-12

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

    # Intermediate parameters
    fd = da.zeros(c11.shape[:2], dtype="float32")
    fs = da.zeros(c11.shape[:2], dtype="float32")
    alpha = da.zeros(c11.shape[:2], dtype="float32")
    beta = da.zeros(c11.shape[:2], dtype="float32")

    # Volume scattering condition
    cnd1 = (c11 <= eps) | (c33 <= eps)
    fv = da.where(cnd1, 3 * (c11 + c22 + c33 + 2 * fv) / 8, fv)

    # Compute c13 power
    pow_c13 = c13r**2 + c13i**2

    # Data conditioning for non-realizable term
    cnd2 = ~cnd1 & (pow_c13 > c11 * c33)
    scale_factor = da.where(cnd2, da.sqrt(c11 * c33 / (pow_c13 + eps)), 1)
    c13r *= scale_factor
    c13i *= scale_factor

    # Recompute after conditioning
    pow_c13 = c13r**2 + c13i**2

    # Odd bounce dominates
    cnd3 = ~cnd1 & (c13r >= 0)
    alpha = da.where(cnd3, -1, alpha)
    fd = da.where(cnd3, (c11 * c33 - pow_c13) / (c11 + c33 + 2 * c13r), fd)
    fs = da.where(cnd3, c33 - fd, fs)
    beta = da.where(cnd3, da.sqrt((fd + c13r) ** 2 + c13i**2) / (fs + eps), beta)

    # Even bounce dominates
    cnd4 = ~cnd1 & (c13r < 0)
    beta = da.where(cnd4, 1, beta)
    fs = da.where(cnd4, (c11 * c33 - pow_c13) / (c11 + c33 - 2 * c13r), fs)
    fd = da.where(cnd4, c33 - fs, fd)
    alpha = da.where(cnd4, da.sqrt((fs - c13r) ** 2 + c13i**2) / (fd + eps), alpha)

    # Compute Freeman components
    Ps = fs * (1 + beta**2)
    Pd = fd * (1 + alpha**2)
    Pv = 8 * fv / 3

    # Ensure values are within reasonable bounds
    sp = span(C3)
    min_span, max_span = da.nanmin(sp), da.nanmax(sp)
    min_span = max(min_span, eps)  # Ensure min_span is at least eps

    Ps = da.where(Ps < min_span, min_span, da.where(Ps > max_span, max_span, Ps))
    Pd = da.where(Pd < min_span, min_span, da.where(Pd > max_span, max_span, Pd))
    Pv = da.where(Pv < min_span, min_span, da.where(Pv > max_span, max_span, Pv))

    return Ps, Pd, Pv
