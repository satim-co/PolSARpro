import pytest
import numpy as np
import xarray as xr
from polsarpro.util import vec_to_mat


@pytest.fixture(scope="function")
def synthetic_poldata(request):
    """Generate only the requested synthetic polarimetric datasets.

    Args:
        request.param: one or more of {"S", "C3", "T3"}.
                       If None, all are generated.

    Returns:
        - xr.Dataset if one type requested (e.g., "T3")
        - dict[str, xr.Dataset] if multiple types requested
    """
    N = 128
    D = 3
    dims = ("y", "x")

    # Parse requested types
    requested = getattr(request, "param", None)
    if requested is None:
        requested = ["S", "C3", "T3"]
    elif isinstance(requested, str):
        requested = [requested]

    result = {}

    # Generate only what's needed
    if "S" in requested:
        S = (np.random.randn(N, N, 2, 2) + 1j * np.random.randn(N, N, 2, 2)).astype(
            "complex64"
        )
        S_dict = dict(
            hh=xr.DataArray(S[..., 0, 0], dims=dims),
            hv=xr.DataArray(S[..., 0, 1], dims=dims),
            vh=xr.DataArray(S[..., 1, 0], dims=dims),
            vv=xr.DataArray(S[..., 1, 1], dims=dims),
        )
        result["S"] = xr.Dataset(
            S_dict,
            attrs=dict(poltype="S"),
            coords={"y": np.arange(N), "x": np.arange(N)},
        )

    if any(t in requested for t in ("C3", "T3")):
        # Only generate v/vp when needed
        v = np.random.randn(N, N, D) + 1j * np.random.randn(N, N, D)
        vp = np.random.randn(N, N, D) + 1j * np.random.randn(N, N, D)
        if "C2" in requested:
            C2 = vec_to_mat(v[..., :2]).astype("complex64")
            C2_dict = dict(
                m11=xr.DataArray(C2[..., 0, 0].real, dims=dims),
                m22=xr.DataArray(C2[..., 1, 1].real, dims=dims),
                m12=xr.DataArray(C2[..., 0, 1], dims=dims),
            )
            result["C2"] = xr.Dataset(
                C2_dict,
                attrs=dict(poltype="C2", description="..."),
                coords={"y": np.arange(N), "x": np.arange(N)},
            )
        if "C3" in requested:
            C3 = vec_to_mat(v).astype("complex64")
            C3_dict = dict(
                m11=xr.DataArray(C3[..., 0, 0].real, dims=dims),
                m22=xr.DataArray(C3[..., 1, 1].real, dims=dims),
                m33=xr.DataArray(C3[..., 2, 2].real, dims=dims),
                m12=xr.DataArray(C3[..., 0, 1], dims=dims),
                m13=xr.DataArray(C3[..., 0, 2], dims=dims),
                m23=xr.DataArray(C3[..., 1, 2], dims=dims),
            )
            result["C3"] = xr.Dataset(
                C3_dict,
                attrs=dict(poltype="C3", description="..."),
                coords={"y": np.arange(N), "x": np.arange(N)},
            )
        if "C4" in requested:
            v = np.random.randn(N, N, 4) + 1j * np.random.randn(N, N, 4)
            C4 = vec_to_mat(v).astype("complex64")
            C4_dict = dict(
                m11=xr.DataArray(C4[..., 0, 0].real, dims=dims),
                m22=xr.DataArray(C4[..., 1, 1].real, dims=dims),
                m33=xr.DataArray(C4[..., 2, 2].real, dims=dims),
                m44=xr.DataArray(C4[..., 3, 3].real, dims=dims),
                m12=xr.DataArray(C4[..., 0, 1], dims=dims),
                m13=xr.DataArray(C4[..., 0, 2], dims=dims),
                m14=xr.DataArray(C4[..., 0, 3], dims=dims),
                m23=xr.DataArray(C4[..., 1, 2], dims=dims),
                m24=xr.DataArray(C4[..., 1, 3], dims=dims),
                m34=xr.DataArray(C4[..., 2, 3], dims=dims),
            )
            result["C4"] = xr.Dataset(
                C4_dict,
                attrs=dict(poltype="C4", description="..."),
                coords={"y": np.arange(N), "x": np.arange(N)},
            )
        if "T3" in requested:
            T3 = vec_to_mat(vp).astype("complex64")
            T3_dict = dict(
                m11=xr.DataArray(T3[..., 0, 0].real, dims=dims),
                m22=xr.DataArray(T3[..., 1, 1].real, dims=dims),
                m33=xr.DataArray(T3[..., 2, 2].real, dims=dims),
                m12=xr.DataArray(T3[..., 0, 1], dims=dims),
                m13=xr.DataArray(T3[..., 0, 2], dims=dims),
                m23=xr.DataArray(T3[..., 1, 2], dims=dims),
            )
            result["T3"] = xr.Dataset(
                T3_dict,
                attrs=dict(poltype="T3", description="..."),
                coords={"y": np.arange(N), "x": np.arange(N)},
            )
        if "T4" in requested:
            v = np.random.randn(N, N, 4) + 1j * np.random.randn(N, N, 4)
            T4 = vec_to_mat(v).astype("complex64")
            T4_dict = dict(
                m11=xr.DataArray(T4[..., 0, 0].real, dims=dims),
                m22=xr.DataArray(T4[..., 1, 1].real, dims=dims),
                m33=xr.DataArray(T4[..., 2, 2].real, dims=dims),
                m44=xr.DataArray(T4[..., 3, 3].real, dims=dims),
                m12=xr.DataArray(T4[..., 0, 1], dims=dims),
                m13=xr.DataArray(T4[..., 0, 2], dims=dims),
                m14=xr.DataArray(T4[..., 0, 3], dims=dims),
                m23=xr.DataArray(T4[..., 1, 2], dims=dims),
                m24=xr.DataArray(T4[..., 1, 3], dims=dims),
                m34=xr.DataArray(T4[..., 2, 3], dims=dims),
            )
            result["T4"] = xr.Dataset(
                T4_dict,
                attrs=dict(poltype="T4", description="..."),
                coords={"y": np.arange(N), "x": np.arange(N)},
            )

    return result
