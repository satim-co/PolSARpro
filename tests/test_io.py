import pytest
import numpy as np
import xarray as xr
from pathlib import Path

from polsarpro.io import open_netcdf_beam


@pytest.fixture
def tmp_netcdf(tmp_path: Path):
    """Helper to write a temporary NetCDF file."""

    def _create_ds(variables: dict, geocoded=False):
        y = np.arange(2)
        x = np.arange(3)
        dims = ("lat", "lon") if geocoded else ("y", "x")
        coords = {"y": y, "x": x} if dims == ("y", "x") else {"lat": y, "lon": x}
        ds = xr.Dataset(
            {name: (dims, np.ones((len(y), len(x)))) for name in variables},
            coords=coords,
        )
        # add metadata attribute
        ds["metadata"] = xr.DataArray(
            attrs={"Abstracted_Metadata:is_terrain_corrected": int(geocoded)}
        )
        file_path = tmp_path / "test.nc"
        ds.to_netcdf(file_path)
        return file_path

    return _create_ds


def test_s_matrix(tmp_netcdf):
    # Create S matrix dataset
    S_vars = [
        f"{x}_{p1}{p2}" for p1 in ("H", "V") for p2 in ("H", "V") for x in ("i", "q")
    ]
    file_path = tmp_netcdf(S_vars, geocoded=False)

    ds_out = open_netcdf_beam(file_path)

    assert set(ds_out.data_vars) == {"hh", "hv", "vh", "vv"}
    assert ds_out.attrs["poltype"] == "S"
    assert "description" in ds_out.attrs
    assert "y" in ds_out.coords and "x" in ds_out.coords


def test_c3_matrix(tmp_netcdf):
    C3_vars = [
        "C11",
        "C22",
        "C33",
        "C12_real",
        "C12_imag",
        "C13_real",
        "C13_imag",
        "C23_real",
        "C23_imag",
    ]
    file_path = tmp_netcdf(C3_vars, geocoded=True)

    ds_out = open_netcdf_beam(file_path)

    assert set(ds_out.data_vars) == {"m11", "m22", "m33", "m12", "m13", "m23"}
    assert ds_out.attrs["poltype"] == "C3"
    assert "description" in ds_out.attrs
    # Should not add y/x for geocoded datasets
    assert "y" not in ds_out.coords


def test_t3_matrix(tmp_netcdf):
    T3_vars = [
        "T11",
        "T22",
        "T33",
        "T12_real",
        "T12_imag",
        "T13_real",
        "T13_imag",
        "T23_real",
        "T23_imag",
    ]
    file_path = tmp_netcdf(T3_vars, geocoded=True)

    ds_out = open_netcdf_beam(file_path)

    assert set(ds_out.data_vars) == {"m11", "m22", "m33", "m12", "m13", "m23"}
    assert ds_out.attrs["poltype"] == "T3"
    assert "description" in ds_out.attrs


def test_invalid_vars(tmp_netcdf):
    file_path = tmp_netcdf(["random_var"], geocoded=False)

    with pytest.raises(ValueError, match="Polarimetric type not recognized"):
        open_netcdf_beam(file_path)
