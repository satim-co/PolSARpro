import numpy as np
import xarray as xr
import pytest
from polsarpro.decompositions import h_a_alpha
from polsarpro.util import vec_to_mat


@pytest.mark.filterwarnings("ignore:invalid")
def test_h_a_alpha():
    N = 128
    D = 3
    v = np.random.randn(N, N, D) + 1j * np.random.randn(N, N, D)
    vp = np.random.randn(N, N, D) + 1j * np.random.randn(N, N, D)
    S = (np.random.randn(N, N, 2, 2) + 1j * np.random.randn(N, N, 2, 2)).astype(
        "complex64"
    )

    # fake input matrices
    C3 = vec_to_mat(v).astype("complex64")
    T3 = vec_to_mat(vp).astype("complex64")

    dims = ("y", "x")
    S_dict = dict(
        hh=xr.DataArray(S[..., 0, 0], dims=dims),
        hv=xr.DataArray(S[..., 0, 1], dims=dims),
        vh=xr.DataArray(S[..., 1, 0], dims=dims),
        vv=xr.DataArray(S[..., 1, 1], dims=dims),
    )
    C3_dict = dict(
        m11=xr.DataArray(C3[..., 0, 0].real, dims=dims),
        m22=xr.DataArray(C3[..., 1, 1].real, dims=dims),
        m33=xr.DataArray(C3[..., 2, 2].real, dims=dims),
        m12=xr.DataArray(C3[..., 0, 1], dims=dims),
        m13=xr.DataArray(C3[..., 0, 2], dims=dims),
        m23=xr.DataArray(C3[..., 1, 2], dims=dims),
    )
    T3_dict = dict(
        m11=xr.DataArray(T3[..., 0, 0].real, dims=dims),
        m22=xr.DataArray(T3[..., 1, 1].real, dims=dims),
        m33=xr.DataArray(T3[..., 2, 2].real, dims=dims),
        m12=xr.DataArray(T3[..., 0, 1], dims=dims),
        m13=xr.DataArray(T3[..., 0, 2], dims=dims),
        m23=xr.DataArray(T3[..., 1, 2], dims=dims),
    )
    for in_dict, poltype in zip([S_dict, C3_dict, T3_dict], ["S", "C3", "T3"]):
        input_data = xr.Dataset(in_dict, attrs=dict(poltype=poltype))
        res = h_a_alpha(
            input_data=input_data,
            boxcar_size=[5, 5],
            flags=("alpha", "entropy", "anisotropy", "betas"),
        )
        var = "hh" if "hh" in input_data.data_vars else "m11"
        shp = input_data[var].shape
        assert all((res[it].shape == shp for it in ["entropy", "alpha", "anisotropy"]))
        assert res.betas.shape == shp + (3,)
        assert all(
            (
                res[it].dtype == "float32"
                for it in ["entropy", "alpha", "anisotropy", "betas"]
            )
        )
