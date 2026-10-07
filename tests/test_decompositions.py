import numpy as np
import pytest
import xarray as xr
from polsarpro.decompositions import (
    h_a_alpha,
    freeman,
    freeman2,
    yamaguchi3,
    yamaguchi4,
    tsvm,
    vanzyl,
    cameron,
)


@pytest.mark.parametrize(
    "synthetic_poldata", ["S", "C2", "C3", "C4", "T3", "T4"], indirect=True
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
@pytest.mark.parametrize("condition_threshold", [None, 0.01])
def test_freeman2(synthetic_poldata, condition_threshold):
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        res = freeman2(
            input_data=ds,
            boxcar_size=[5, 5],
            condition_threshold=condition_threshold,
        ).compute()
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert all((res[it].shape == shp for it in ["ground", "volume"]))
        assert all((res[it].dtype == "float32" for it in ["ground", "volume"]))


@pytest.mark.parametrize(
    "powers, correlation, excluded",
    [
        ((4, 1, 1), 0.5 + 0.5j, False),
        ((1, 0.2, 1), 0.2 + 0.1j, True),
        ((4, 1, 1), 0.5 + 1e-7j, False),
        ((4, 1, 1), 3 + 0.001j, False),
        ((4, 1, 1), 1.5 + 1j, True),
        ((4, 0, 1), 2j, False),
        ((0, 0, 0), 0j, True),
    ],
    ids=["normal", "HH-VV", "imaginary", "correlation", "ground", "volume", "zero"],
)
@pytest.mark.parametrize("scale", [1, 100])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_freeman2_condition(powers, correlation, excluded, scale, chunked):
    ds = xr.Dataset(
        {
            **{
                name: (("y", "x"), np.full((8, 8), value * scale, dtype="float32"))
                for name, value in zip(("m11", "m22", "m33"), powers)
            },
            **{
                name: (("y", "x"), np.full((8, 8), value * scale, dtype="complex64"))
                for name, value in (("m12", 0j), ("m13", correlation), ("m23", 0j))
            },
        },
        attrs={"poltype": "C3"},
    )
    if chunked:
        ds = ds.chunk({"y": 4, "x": 4})
    # Both default safeguards and optional exclusion must be warning-free.
    baseline = freeman2(ds, (1, 1)).compute()
    result = freeman2(ds, (1, 1), condition_threshold=0.01).compute()
    assert bool(result.to_array().isnull().all()) == excluded
    if not excluded:
        assert bool(np.isfinite(result.to_array()).all())
        xr.testing.assert_equal(result, baseline)


@pytest.mark.parametrize("threshold", [0, -0.01, 1, np.nan, np.inf])
@pytest.mark.parametrize("synthetic_poldata", ["C3"], indirect=True)
def test_freeman2_threshold(synthetic_poldata, threshold):
    with pytest.raises(ValueError, match="condition_threshold"):
        freeman2(synthetic_poldata["C3"], condition_threshold=threshold)


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
        "alpha_phi_tau_psi": ("alpha_s", "phi_s", "tau_m", "psi"),
        "alpha": ("alpha_s1", "alpha_s2", "alpha_s3"),
        "phi": ("phi_s1", "phi_s2", "phi_s3"),
        "tau": ("tau_m1", "tau_m2", "tau_m3"),
        "psi": ("psi1", "psi2", "psi3"),
    }

    for _, ds in input_data.items():
        res = tsvm(
            input_data=ds,
            boxcar_size=[5, 5],
            flags=tuple(in_out.keys()),
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        for flag in in_out:
            for name in in_out[flag]:
                assert res[name].dtype == "float32"
                assert res[name].shape == shp


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
@pytest.mark.filterwarnings("ignore:invalid")
def test_vanzyl(synthetic_poldata):

    input_data = synthetic_poldata

    for _, ds in input_data.items():
        # uses dask specific functions and cannot be run on numpy arrays
        ds = ds.chunk("auto")
        res = vanzyl(
            input_data=ds,
            boxcar_size=[5, 5],
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert all((res[it].shape == shp for it in ["odd", "double", "volume"]))
        assert all((res[it].dtype == "float32" for it in ["odd", "double", "volume"]))


@pytest.mark.parametrize("synthetic_poldata", ["S"], indirect=True)
def test_cameron(synthetic_poldata):
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        input_data = ds.chunk(x=64, y=64)
        res = cameron(input_data=input_data)
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert res["cameron"].shape == shp
        assert res["cameron"].dtype == "int32"
