import numpy as np
import warnings
from polsarpro.util import boxcar, T3_to_C3, C3_to_T3, S_to_T3, S_to_C3
from polsarpro.util import (
    _boxcar_core,
    _T3_to_C3_core,
    _S_to_T3_core,
    _S_to_C3_core,
    _C3_to_T3_core,
)
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
def h_a_alpha(
    input_data: np.ndarray,
    input_poltype: str = "C3",
    boxcar_size: list[int, int] = [3, 3],
    flags: tuple[str] = ("entropy", "alpha", "anisotropy"),
) -> np.ndarray:

    """Performs the H/A/Alpha polarimetric decomposition on full-pol SAR data.

    This function computes the H/A/Alpha decomposition from input polarimetric SAR data 
    using eigenvalue analysis of the coherency matrix. The decomposition 
    provides physical insight into scattering mechanisms through parameters such as 
    entropy (H), anisotropy (A), and the alpha scattering angle (alpha). Additional 
    eigenvalue-based parameters can also be computed by specifying corresponding flags.

    Args:
        input_data (np.ndarray): The input polarimetric SAR data array. Expected to represent
            a 3x3 matrix (or 2x2 in the case of Sinclair) per pixel, typically with shape 
            (..., 3, 3) or (..., 2, 2) depending on `input_poltype`.
        input_poltype (str, optional): The polarimetric basis of the input data. Supported types are:
            - "C3": Lexicographic covariance matrix
            - "T3": Pauli coherency matrix
            - "S": Sinclair scattering matrix
            Defaults to "C3".
        boxcar_size (list[int, int], optional): Size of the spatial averaging window to be
            applied before decomposition (boxcar filter). Defaults to [3, 3].
        flags (tuple[str], optional): Parameters to compute and return from the decomposition.
            Possible values include:
            - "entropy": Scattering entropy (H)
            - "anisotropy": Scattering anisotropy (A)
            - "alpha": Mean alpha scattering angle (alpha)
            - "beta", "delta", "gamma", "lambda": Other angular or eigenvalue related parameters
            - "alphas", "betas", "deltas", "gammas", "lambdas": Per-eigenvector versions of the above
            Defaults to ("entropy", "alpha", "anisotropy").

    Returns:
        dict[str, np.ndarray]: A dictionary where keys correspond to the requested flags, 
        and values are the corresponding 2D arrays (or 3D if the flag returns multiple values per pixel).

    Raises:
        ValueError: If `input_poltype` is not one of the supported types, or if any
        requested flag is unrecognized.

    References:
        Cloude, S. R., & Pottier, E. (1997). An entropy based classification scheme for land
        applications of polarimetric SAR. *IEEE Transactions on Geoscience and Remote Sensing*,
        35(1), 68-78.
    """

    # check flags validity
    possible_flags = (
        "entropy",
        "anisotropy",
        "alpha",
        "beta",
        "delta",
        "gamma",
        "lambda",
        "alphas",
        "betas",
        "deltas",
        "gammas",
        "lambdas",
    )
    for flag in flags:
        if flag not in possible_flags:
            raise ValueError(
                f"Flag '{flag}' not recognized. Possible values are {possible_flags}."
            )

    if np.isrealobj(input_data):
        raise ValueError("Inputs must be complex-valued.")

    in_ = input_data.astype("complex64", copy=False)

    if input_poltype == "C3":
        in_ = C3_to_T3(input_data)
    elif input_poltype == "T3":
        pass
    elif input_poltype == "S":
        in_ = S_to_T3(input_data)
    else:
        raise ValueError("Invalid polarimetric type")

    # pre-processing step, filtering is required for full rank matrices
    in_ = boxcar(in_, boxcar_size[0], boxcar_size[1])

    # # Eigendecomposition
    l, v = np.linalg.eigh(in_)
    l = l[..., ::-1]  # put in descending order
    v = v[..., ::-1]

    outputs = _compute_h_a_alpha_parameters(l, v, flags)
    return outputs


def h_a_alpha_dask(
    input_data: np.ndarray,
    input_poltype: str = "C3",
    boxcar_size: list[int, int] = [3, 3],
    flags: tuple[str] = ("entropy", "alpha", "anisotropy"),
) -> np.ndarray:
    """Performs the H/A/Alpha polarimetric decomposition on full-pol SAR data.

    This function computes the H/A/Alpha decomposition from input polarimetric SAR data 
    using eigenvalue analysis of the coherency matrix. The decomposition 
    provides physical insight into scattering mechanisms through parameters such as 
    entropy (H), anisotropy (A), and the alpha scattering angle (alpha). Additional 
    eigenvalue-based parameters can also be computed by specifying corresponding flags.

    Args:
        input_data (np.ndarray): The input polarimetric SAR data array. Expected to represent
            a 3x3 matrix (or 2x2 in the case of Sinclair) per pixel, typically with shape 
            (..., 3, 3) or (..., 2, 2) depending on `input_poltype`.
        input_poltype (str, optional): The polarimetric basis of the input data. Supported types are:
            - "C3": Lexicographic covariance matrix
            - "T3": Pauli coherency matrix
            - "S": Sinclair scattering matrix
            Defaults to "C3".
        boxcar_size (list[int, int], optional): Size of the spatial averaging window to be
            applied before decomposition (boxcar filter). Defaults to [3, 3].
        flags (tuple[str], optional): Parameters to compute and return from the decomposition.
            Possible values include:
            - "entropy": Scattering entropy (H)
            - "anisotropy": Scattering anisotropy (A)
            - "alpha": Mean alpha scattering angle (alpha)
            - "beta", "delta", "gamma", "lambda": Other angular or eigenvalue related parameters
            - "alphas", "betas", "deltas", "gammas", "lambdas": Per-eigenvector versions of the above
            Defaults to ("entropy", "alpha", "anisotropy").

    Returns:
        dict[str, np.ndarray]: A dictionary where keys correspond to the requested flags, 
        and values are the corresponding 2D arrays (or 3D if the flag returns multiple values per pixel).

    Raises:
        ValueError: If `input_poltype` is not one of the supported types, or if any
        requested flag is unrecognized.

    References:
        Cloude, S. R., & Pottier, E. (1997). An entropy based classification scheme for land
        applications of polarimetric SAR. *IEEE Transactions on Geoscience and Remote Sensing*,
        35(1), 68-78.
    """

    # check flags validity
    possible_flags = (
        "entropy",
        "anisotropy",
        "alpha",
        "beta",
        "delta",
        "gamma",
        "lambda",
        "alphas",
        "betas",
        "deltas",
        "gammas",
        "lambdas",
    )
    for flag in flags:
        if flag not in possible_flags:
            raise ValueError(
                f"Flag '{flag}' not recognized. Possible values are {possible_flags}."
            )

    if np.isrealobj(input_data):
        raise ValueError("Inputs must be complex-valued.")

    in_ = da.from_array(input_data.astype("complex64", copy=False), chunks="auto")
    if input_poltype == "C3":
        in_ = da.map_blocks(
            _C3_to_T3_core,
            in_,
            dtype="complex64",
        )
    if input_poltype == "T3":
        pass
    elif input_poltype == "S":
        in_ = da.map_blocks(
            _S_to_T3_core,
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

    # Eigendecomposition
    meta = (
        np.array([], dtype="float32").reshape((0, 0, 0)),
        np.array([], dtype="complex64").reshape((0, 0, 0, 0)),
    )
    l, v = da.apply_gufunc(np.linalg.eigh, "(i,j)->(i), (i,j)", in_, meta=meta)

    l = l[..., ::-1]  # put in descending order
    v = v[..., ::-1]

    out = _compute_h_a_alpha_parameters(l, v, flags)
    with ProgressBar():
        return da.compute(out)[0]


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

    if np.isrealobj(input_data):
        raise ValueError("Inputs must be complex-valued.")

    in_ = input_data.astype("complex64", copy=False)
    if input_poltype == "C3":
        pass
    elif input_poltype == "T3":
        in_ = T3_to_C3(in_)
    elif input_poltype == "S":
        in_ = S_to_C3(in_)
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

    if np.isrealobj(input_data):
        raise ValueError("Inputs must be complex-valued.")

    in_ = da.from_array(input_data.astype("complex64", copy=False), chunks="auto")
    if input_poltype == "C3":
        pass
    elif input_poltype == "T3":
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


def _compute_h_a_alpha_parameters(l, v, flags):
    eps = 1e-30

    # Pseudo-probabilities (normalized eigenvalues)
    p = np.clip(l / (eps + l.sum(axis=2)[..., None]), eps, 1)

    outputs = {}
    if "entropy" in flags:
        H = np.sum(-p * np.log(p), axis=2) / np.float32(np.log(3))
        outputs["entropy"] = H

    if "anisotropy" in flags:
        A = (l[..., 1] - l[..., 2]) / (l[..., 1] + l[..., 2] + eps)
        outputs["anisotropy"] = A

    if "alpha" in flags or "alphas" in flags:
        # Alpha angles for each mechanism
        alphas = np.arccos(np.abs(v[:, :, 0, :]))
        # Convert to degrees
        alphas *= 180 / np.pi
        # Mean alpha
        alpha = np.sum(p * alphas, axis=2)
        outputs["alpha"] = alpha

    # Extra angles: beta, delta and gamma angles
    if "beta" in flags or "betas" in flags:
        betas = np.atan2(np.abs(v[:, :, 2, :]), eps + np.abs(v[:, :, 1, :]))
        betas *= 180 / np.pi
        beta = np.sum(p * betas, axis=2)
        outputs["beta"] = beta

    if "delta" in flags or "gamma" in flags or "deltas" in flags or "gammas" in flags:
        phases = np.atan2(v[:, :, 0, :].imag, eps + v[:, :, 0, :].real)

    if "delta" in flags or "deltas" in flags:
        deltas = np.atan2(v[:, :, 1, :].imag, eps + v[:, :, 1, :].real) - phases
        deltas = np.atan2(np.sin(deltas), eps + np.cos(deltas))
        deltas *= 180 / np.pi
        delta = np.sum(p * deltas, axis=2)
        outputs["delta"] = delta

    if "gamma" in flags or "gammas" in flags:
        gammas = np.atan2(v[:, :, 2, :].imag, eps + v[:, :, 2, :].real) - phases
        gammas = np.atan2(np.sin(gammas), eps + np.cos(gammas))
        gammas *= 180 / np.pi
        gamma = np.sum(p * gammas, axis=2)
        outputs["gamma"] = gamma

    # Average target eigenvalue
    if "lambda" in flags or "lambdas" in flags:
        # lambda is a python reserved keyword, using lambd instead
        lambd = np.sum(p * l, axis=2)
        outputs["lambda"] = lambd

    # extras outputs: non averaged parameters (ex: alpha2, alpha2, alpha3)
    if "alphas" in flags:
        outputs["alphas"] = alphas

    if "betas" in flags:
        outputs["betas"] = betas

    if "deltas" in flags:
        outputs["deltas"] = deltas

    if "gammas" in flags:
        outputs["gammas"] = gammas

    if "lambdas" in flags:
        outputs["lambdas"] = l
    return outputs
