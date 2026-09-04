import numpy as np
import pytest
import xarray as xr

from polsarpro.physical_inversion import (
    _solve_oh_newton_c,
    dubois_surface_inversion,
    oh_surface_inversion,
)


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["S", "C3", "T3", "C4", "T4"], "size": 16, "chunk_size": 4}],
    indirect=True,
)
def test_dubois_surface_inversion(synthetic_poldata):
    for _, ds in synthetic_poldata.items():
        ds = ds.copy()
        first_var = next(iter(ds.data_vars))
        ds[first_var] = ds[first_var].where(~((ds.y == 0) & (ds.x == 0)))

        incidence_angle = ds[first_var].real.astype("float32") * 0 + 35.0

        res = dubois_surface_inversion(
            input_data=ds,
            incidence_angle=incidence_angle,
            freq_ghz=5.3,
            thresh1=3.0,
            thresh2=3.0,
            calibration_coeff=1.0,
        )

        expected_vars = {
            "dubois_ks",
            "dubois_er",
            "dubois_mv",
            "dubois_mask_out",
            "dubois_mask_in",
            "dubois_mask_valid_in_out",
        }
        assert expected_vars.issubset(set(res.data_vars))

        shp = ds[first_var].shape
        for name in expected_vars:
            assert res[name].shape == shp
            assert res[name].dtype == "float32"

        original_non_nan_mask = ds.to_array().notnull().all("variable")
        expected_mask = (
            res["dubois_mask_in"]
            * res["dubois_mask_out"]
            * original_non_nan_mask.astype("float32")
        ).astype("float32")
        xr.testing.assert_allclose(res["dubois_mask_valid_in_out"], expected_mask)


@pytest.mark.parametrize(
    "kwargs, exc_type",
    [
        (dict(freq_ghz=0.0), ValueError),
        (dict(freq_ghz="bad"), TypeError),
        (dict(thresh1="bad"), TypeError),
        (dict(thresh2="bad"), TypeError),
        (dict(calibration_coeff="bad"), TypeError),
        (dict(calibration_coeff=0.0), ValueError),
    ],
)
@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["C3"], "size": 8, "chunk_size": 4}],
    indirect=True,
)
def test_dubois_surface_inversion_invalid_scalars(synthetic_poldata, kwargs, exc_type):
    ds = synthetic_poldata["C3"]
    first_var = next(iter(ds.data_vars))
    incidence_angle = ds[first_var].real.astype("float32") * 0 + 35.0

    base_kwargs = dict(
        input_data=ds,
        incidence_angle=incidence_angle,
        freq_ghz=5.3,
        thresh1=3.0,
        thresh2=3.0,
        calibration_coeff=1.0,
    )
    base_kwargs.update(kwargs)

    with pytest.raises(exc_type):
        dubois_surface_inversion(**base_kwargs)


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["C3"], "size": 8, "chunk_size": 4}],
    indirect=True,
)
def test_dubois_surface_inversion_invalid_incidence_angle_type(synthetic_poldata):
    ds = synthetic_poldata["C3"]

    with pytest.raises(TypeError):
        dubois_surface_inversion(
            input_data=ds,
            incidence_angle=ds.m11.values,
            freq_ghz=5.3,
            thresh1=3.0,
            thresh2=3.0,
        )


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["S", "C3", "T3", "C4", "T4"], "size": 8, "chunk_size": 4}],
    indirect=True,
)
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_oh_surface_inversion(synthetic_poldata):
    for _, ds in synthetic_poldata.items():
        ds = ds.copy()
        first_var = next(iter(ds.data_vars))
        ds[first_var] = ds[first_var].where(~((ds.y == 0) & (ds.x == 0)))

        incidence_angle = ds[first_var].real.astype("float32") * 0 + 0.6

        res = oh_surface_inversion(
            input_data=ds,
            incidence_angle=incidence_angle,
            thresh1=3.0,
            thresh2=3.0,
        )

        expected_vars = {
            "oh_ks",
            "oh_er",
            "oh_mv",
            "oh_mask_out",
            "oh_mask_in",
            "oh_mask_valid_in_out",
        }
        assert expected_vars.issubset(set(res.data_vars))
        assert res.attrs["poltype"] == "oh_surface_inversion"

        shape = ds[first_var].shape
        for name in expected_vars:
            assert res[name].shape == shape
            assert res[name].dtype == "float32"

        original_non_nan_mask = ds.to_array().notnull().all("variable")
        expected_mask = (
            res["oh_mask_in"]
            * res["oh_mask_out"]
            * original_non_nan_mask.astype("float32")
        ).astype("float32")
        xr.testing.assert_allclose(res["oh_mask_valid_in_out"], expected_mask)


@pytest.mark.parametrize("name", ["thresh1", "thresh2"])
@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["C3"], "size": 8, "chunk_size": 4}],
    indirect=True,
)
def test_oh_surface_inversion_invalid_threshold(synthetic_poldata, name):
    ds = synthetic_poldata["C3"]
    incidence_angle = ds.m11.astype("float32") * 0 + 0.6
    kwargs = {"thresh1": 3.0, "thresh2": 3.0, name: "bad"}

    with pytest.raises(TypeError):
        oh_surface_inversion(
            input_data=ds,
            incidence_angle=incidence_angle,
            **kwargs,
        )


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["C3"], "size": 4, "chunk_size": 2}],
    indirect=True,
)
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_oh_surface_inversion_c_semantics(synthetic_poldata):
    ds = synthetic_poldata["C3"]
    incidence_angle = ds.m11.astype("float32") * 0 + 0.6

    result = oh_surface_inversion(
        input_data=ds,
        incidence_angle=incidence_angle,
        thresh1=3.0,
        thresh2=3.0,
        c_semantics=True,
    ).compute()

    assert result.attrs["poltype"] == "oh_surface_inversion"
    assert all(value.dtype == "float32" for value in result.values())


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["C3"], "size": 4, "chunk_size": 2}],
    indirect=True,
)
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_oh_surface_inversion_c_semantics_accepts_nan_results(synthetic_poldata):
    ds = synthetic_poldata["C3"].copy()
    ds["m11"] = xr.full_like(ds.m11, -1.0)
    ds["m22"] = xr.full_like(ds.m22, 0.2)
    ds["m33"] = xr.full_like(ds.m33, 1.0)
    incidence_angle = xr.full_like(ds.m11.real, 0.6)

    result = oh_surface_inversion(
        input_data=ds,
        incidence_angle=incidence_angle,
        thresh1=3.0,
        thresh2=3.0,
        c_semantics=True,
    ).compute()

    assert result[["oh_ks", "oh_er", "oh_mv"]].to_array().isnull().all()
    assert (result["oh_mask_in"] == 1).all()
    assert (result["oh_mask_out"] == 1).all()


def test_oh_newton_c_semantics_float32_regression():
    a = np.array([0.3, 0.5, 0.7, 0.6], dtype=np.float32)
    b = np.array([0.2, 0.5, 0.8, 0.3], dtype=np.float32)
    c = np.array([-0.1, 0.2, 0.5, -0.4], dtype=np.float32)
    valid = np.array([True, True, True, False])

    with np.errstate(all="ignore"):
        result = _solve_oh_newton_c(a, b, c, valid)

    expected_bits = np.array(
        [0x40060595, 0xFFC00000, 0xFFC00000, 0x40000000], dtype=np.uint32
    )
    np.testing.assert_array_equal(result.view(np.uint32), expected_bits)


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["C3"], "size": 4, "chunk_size": 2}],
    indirect=True,
)
def test_oh_surface_inversion_invalid_c_semantics(synthetic_poldata):
    ds = synthetic_poldata["C3"]
    incidence_angle = ds.m11.astype("float32") * 0 + 0.6

    with pytest.raises(TypeError):
        oh_surface_inversion(
            input_data=ds,
            incidence_angle=incidence_angle,
            thresh1=3.0,
            thresh2=3.0,
            c_semantics="yes",
        )


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": ["C3"], "size": 8, "chunk_size": 4}],
    indirect=True,
)
def test_oh_surface_inversion_invalid_incidence_angle_type(synthetic_poldata):
    ds = synthetic_poldata["C3"]

    with pytest.raises(TypeError):
        oh_surface_inversion(
            input_data=ds,
            incidence_angle=ds.m11.values,
            thresh1=3.0,
            thresh2=3.0,
        )
