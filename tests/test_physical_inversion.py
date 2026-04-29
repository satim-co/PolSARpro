import pytest
import xarray as xr

from polsarpro.physical_inversion import dubois_surface_inversion


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
            f0=5.3,
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
            "dubois_mask_in_out_valid",
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
        xr.testing.assert_allclose(res["dubois_mask_in_out_valid"], expected_mask)


@pytest.mark.parametrize(
    "kwargs, exc_type",
    [
        (dict(f0=0.0), ValueError),
        (dict(f0="bad"), TypeError),
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
        f0=5.3,
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
            f0=5.3,
            thresh1=3.0,
            thresh2=3.0,
        )
