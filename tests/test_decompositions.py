import numpy as np
import xarray as xr
import pytest
from polsarpro.decompositions import h_a_alpha, freeman, yamaguchi3, yamaguchi4, tsvm
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
        result["S"] = xr.Dataset(S_dict, attrs=dict(poltype="S"))

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
                C2_dict, attrs=dict(poltype="C2", description="...")
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
            result["C3"] = xr.Dataset(C3_dict, attrs=dict(poltype="C3"))
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
                C4_dict, attrs=dict(poltype="C4", description="...")
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
            result["T3"] = xr.Dataset(T3_dict, attrs=dict(poltype="T3"))

    return result


@pytest.mark.parametrize(
    "synthetic_poldata", ["S", "C2", "C3", "C4", "T3"], indirect=True
)
def test_h_a_alpha(synthetic_poldata):
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        res = h_a_alpha(
            input_data=ds,
            boxcar_size=[5, 5],
            flags=("alpha", "entropy", "anisotropy", "betas"),
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert all((res[it].shape == shp for it in ["entropy", "alpha", "anisotropy"]))
        assert res.betas.shape == shp + (3,)
        assert all(
            (
                res[it].dtype == "float32"
                for it in ["entropy", "alpha", "anisotropy", "betas"]
            )
        )


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
@pytest.mark.filterwarnings("ignore:invalid")
def test_freeman(synthetic_poldata):

    input_data = synthetic_poldata

    for _, ds in input_data.items():
        # uses dask specific functions and cannot be run on numpy arrays
        ds = ds.chunk("auto")
        res = freeman(
            input_data=ds,
            boxcar_size=[5, 5],
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert all((res[it].shape == shp for it in ["odd", "double", "volume"]))
        assert all((res[it].dtype == "float32" for it in ["odd", "double", "volume"]))


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
@pytest.mark.filterwarnings("ignore:invalid")
def test_yamaguchi3(synthetic_poldata):

    input_data = synthetic_poldata

    for _, ds in input_data.items():
        # uses dask specific functions and cannot be run on numpy arrays
        ds = ds.chunk("auto")
        res = yamaguchi3(
            input_data=ds,
            boxcar_size=[5, 5],
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert all((res[it].shape == shp for it in ["odd", "double", "volume"]))
        assert all((res[it].dtype == "float32" for it in ["odd", "double", "volume"]))


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
@pytest.mark.filterwarnings("ignore:invalid")
def test_yamaguchi4(synthetic_poldata):

    input_data = synthetic_poldata

    for _, ds in input_data.items():
        for mode in ["y4o", "y4r", "s4r"]:
            # uses dask specific functions and cannot be run on numpy arrays
            ds = ds.chunk("auto")
            res = yamaguchi4(input_data=ds, boxcar_size=[5, 5], mode=mode)
            var = "hh" if "hh" in ds.data_vars else "m11"
            shp = ds[var].shape
            assert all(
                (res[it].shape == shp for it in ["odd", "double", "volume", "helix"])
            )
            assert all(
                (
                    res[it].dtype == "float32"
                    for it in ["odd", "double", "volume", "helix"]
                )
            )


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
def test_tsvm(synthetic_poldata):
    input_data = synthetic_poldata

    # input flags and corresponding output names
    in_out = {
        "alpha_phi_tau_psi": ["alpha_s", "phi_s", "tau_m", "psi"],
        "alpha": ["alpha_s1", "alpha_s2", "alpha_s3"],
        "phi": ["phi_s1", "phi_s2", "phi_s3"],
        "tau": ["tau_m1", "tau_m2", "tau_m3"],
        "psi": ["psi1", "psi2", "psi3"],
    }

    for _, ds in input_data.items():
        res = tsvm(
            input_data=ds,
            boxcar_size=[5, 5],
            flags=in_out.keys(),
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        for flag in in_out:
            for name in in_out[flag]:
                assert res[name].dtype == "float32"
                assert res[name].shape == shp
