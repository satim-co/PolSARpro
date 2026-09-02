import pytest
import numpy as np
import rasterio
import xarray as xr
from pathlib import Path

from polsarpro.io import (
    _parse_slc_bands,
    _validate_biomass_l1a_scs_name,
    open_biomass_l1a,
    open_netcdf_beam,
    polmat_to_netcdf,
)

VALID_BIOMASS_NAME = (
    "BIO_S2_SCS__1S_20251216T034800_20251216T034815_"
    "T_G01_M01_C02_T017_F289_01_DJQGAN"
)
VALID_BIOMASS_STEM = (
    "bio_s2_scs__1s_20251216t034800_20251216t034815_"
    "t_g01_m01_c02_t017_f289"
)


def _write_biomass_raster(
    path,
    *,
    shape=(4, 2, 3),
    dtype="float32",
    polarizations=("HH", "HV", "VH", "VV"),
    data=None,
    nodata=None,
):
    if data is None:
        data = np.ones(shape, dtype=dtype)
    else:
        data = np.asarray(data, dtype=dtype)
        shape = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=shape[0],
        height=shape[1],
        width=shape[2],
        dtype=dtype,
        nodata=nodata,
    ) as raster:
        raster.write(data)
        raster.update_tags(PolarisationsSequence=" ".join(polarizations))


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

def test_s_matrix_biomass(tmp_netcdf):
    # Create S matrix dataset
    tag = "S1"
    S_vars = [
        f"{x}_{tag}_{p1}{p2}" for p1 in ("H", "V") for p2 in ("H", "V") for x in ("i", "q")
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


@pytest.mark.parametrize(
    "synthetic_poldata",
    ["S", "C2", "C3", "C4", "T3", "T4"],
    indirect=True,
)
def test_polmat_to_netcdf(synthetic_poldata, tmp_path):

    input_data = synthetic_poldata
    for _, ds in input_data.items():
        poltype = ds.poltype
        out_file = tmp_path / f"test_{poltype}.nc"
        polmat_to_netcdf(ds, out_file)

        ds_out = xr.open_dataset(out_file)

        if poltype == "S":
            expected_vars = {
                "i_HH",
                "q_HH",
                "i_HV",
                "q_HV",
                "i_VH",
                "q_VH",
                "i_VV",
                "q_VV",
            }
        elif poltype == "C2":
            expected_vars = {"C11", "C12_real", "C12_imag", "C22"}
        elif poltype == "T3":
            expected_vars = {
                "T11",
                "T12_real",
                "T12_imag",
                "T13_real",
                "T13_imag",
                "T22",
                "T23_real",
                "T23_imag",
                "T33",
            }
        elif poltype == "C3":
            expected_vars = {
                "C11",
                "C12_real",
                "C12_imag",
                "C13_real",
                "C13_imag",
                "C22",
                "C23_real",
                "C23_imag",
                "C33",
            }
        elif poltype == "T4":
            expected_vars = {
                "T11",
                "T12_real",
                "T12_imag",
                "T13_real",
                "T13_imag",
                "T14_real",
                "T14_imag",
                "T22",
                "T23_real",
                "T23_imag",
                "T24_real",
                "T24_imag",
                "T33",
                "T34_real",
                "T34_imag",
                "T44",
            }
        elif poltype == "C4":
            expected_vars = {
                "C11",
                "C12_real",
                "C12_imag",
                "C13_real",
                "C13_imag",
                "C14_real",
                "C14_imag",
                "C22",
                "C23_real",
                "C23_imag",
                "C24_real",
                "C24_imag",
                "C33",
                "C34_real",
                "C34_imag",
                "C44",
            }
        assert set(ds_out.data_vars.keys()) == expected_vars
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        for var in expected_vars:
            assert ds_out[var].shape == shp
            assert not np.isnan(ds_out[var].values).any()


def test_parse_slc_bands():
    tag = "S3"
    var_names = {
        f"i_{tag}_HH",
        f"q_{tag}_HH",
        f"i_{tag}_HV",
        f"q_{tag}_HV",
        f"i_{tag}_VH",
        f"q_{tag}_VH",
        f"i_{tag}_VV",
        f"q_{tag}_VV",
    }
    pol_list = ("H", "V")
    var_names_no_tags = {
        f"{x}_{p1}{p2}" for p1 in pol_list for p2 in pol_list for x in ("i", "q")
    }
    var_names_miss = {
        f"i_{tag}_HH",
        f"q_{tag}_HH",
        f"i_{tag}_HV",
        f"q_{tag}_VH",
        f"i_{tag}_VV",
    }
    var_names_bad_prefix = {
        f"i_{tag}_HH",
        f"q_{tag}_HH",
        f"{tag}_HV",
        f"q_{tag}_HV",
        f"i_{tag}_VH",
        f"q_{tag}_VH",
        f"i_{tag}_VV",
        f"q_{tag}_VV",
    }
    var_names_bad_pol = {
        f"i_{tag}_HH",
        f"q_{tag}_HH",
        f"i_{tag}_HV",
        f"q_{tag}_HV",
        f"i_{tag}_GG",
        f"q_{tag}_VH",
        f"i_{tag}_VV",
        f"q_{tag}_VV",
    }
    tagx = "Sx"
    var_names_bad_tag = {
        f"i_{tagx}_HH",
        f"q_{tagx}_HH",
        f"i_{tagx}_HV",
        f"q_{tagx}_HV",
        f"i_{tagx}_VH",
        f"q_{tagx}_VH",
        f"i_{tagx}_VV",
        f"q_{tagx}_VV",
    }
    assert _parse_slc_bands(var_names=var_names) == tag
    assert _parse_slc_bands(var_names=var_names_no_tags) == ""
    assert _parse_slc_bands(var_names=var_names_miss) is None
    assert _parse_slc_bands(var_names=var_names_bad_prefix) is None
    assert _parse_slc_bands(var_names=var_names_bad_pol) is None
    assert _parse_slc_bands(var_names=var_names_bad_tag) is None


@pytest.mark.parametrize("sensor", ["S1", "S2", "S3"])
def test_biomass_name_valid(sensor):
    """Accept standard L1a SCS product names for every BIOMASS sensor mode."""
    name = VALID_BIOMASS_NAME.replace("S2_SCS", f"{sensor}_SCS")

    assert _validate_biomass_l1a_scs_name(Path("products") / name) == name


@pytest.mark.parametrize(
    "name",
    [
        VALID_BIOMASS_NAME.replace("BIO_", "BAD_", 1),
        VALID_BIOMASS_NAME.replace("S2_SCS", "S4_SCS"),
        VALID_BIOMASS_NAME.replace("SCS", "DGM"),
        VALID_BIOMASS_NAME.replace("1S_", "1M_", 1),
        VALID_BIOMASS_NAME.replace("SCS__", "SCS1_"),
        VALID_BIOMASS_NAME.replace("G01_", "G1_"),
    ],
    ids=[
        "satellite",
        "sensor",
        "dgm",
        "monitoring",
        "calibration",
        "field-width",
    ],
)
def test_biomass_name_invalid(name):
    """Reject unsupported product families and malformed directory fields."""
    with pytest.raises(ValueError, match="Unsupported BIOMASS product"):
        _validate_biomass_l1a_scs_name(name)


@pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
def test_biomass_files_valid(tmp_path):
    """Read valid rasters lazily and reconstruct the expected PSP dataset."""
    product_path = tmp_path / VALID_BIOMASS_NAME
    measurement_path = product_path / "measurement"
    measurement_path.mkdir(parents=True)
    raster_shape = (4, 2, 3)
    amplitude = np.broadcast_to(
        np.arange(1, 5, dtype="float32")[:, None, None], raster_shape
    )
    phase_values = np.array((0, np.pi / 2, np.pi, -np.pi / 2), dtype="float32")
    phase = np.broadcast_to(phase_values[:, None, None], raster_shape)
    _write_biomass_raster(
        measurement_path / f"{VALID_BIOMASS_STEM}_i_abs.tiff",
        data=amplitude,
    )
    _write_biomass_raster(
        measurement_path / f"{VALID_BIOMASS_STEM}_i_phase.tiff",
        data=phase,
    )

    result = open_biomass_l1a(product_path, chunks={"y": 1, "x": 2})

    assert set(result.data_vars) == {"hh", "hv", "vh", "vv"}
    assert result.attrs == {
        "poltype": "S",
        "description": "Scattering matrix",
    }
    assert tuple(result.dims) == ("y", "x")
    np.testing.assert_array_equal(result.y, np.arange(2))
    np.testing.assert_array_equal(result.x, np.arange(3))
    assert all(
        result[channel].dtype == np.dtype("complex64") for channel in result
    )
    assert result.hh.chunks == ((1, 1), (2, 1))
    np.testing.assert_allclose(result.hh, 1 + 0j, atol=1e-6)
    np.testing.assert_allclose(result.hv, 0 + 2j, atol=1e-6)
    np.testing.assert_allclose(result.vh, -3 + 0j, atol=1e-6)
    np.testing.assert_allclose(result.vv, 0 - 4j, atol=1e-6)


@pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
def test_biomass_nodata(tmp_path):
    """Propagate amplitude and phase nodata values to complex NaNs."""
    product_path = tmp_path / VALID_BIOMASS_NAME
    measurement_path = product_path / "measurement"
    measurement_path.mkdir(parents=True)
    amplitude = np.ones((4, 2, 3), dtype="float32")
    phase = np.zeros((4, 2, 3), dtype="float32")
    amplitude[0, 0, 0] = -9999
    phase[1, 0, 1] = -9999

    _write_biomass_raster(
        measurement_path / f"{VALID_BIOMASS_STEM}_i_abs.tiff",
        data=amplitude,
        nodata=-9999,
    )
    _write_biomass_raster(
        measurement_path / f"{VALID_BIOMASS_STEM}_i_phase.tiff",
        data=phase,
        nodata=-9999,
    )

    result = open_biomass_l1a(product_path, chunks=None)

    assert np.isnan(result.hh[0, 0])
    assert np.isnan(result.hv[0, 1])
    assert result.vv[0, 0] == 1 + 0j


@pytest.mark.parametrize("missing_suffix", ["i_abs.tiff", "i_phase.tiff"])
def test_biomass_file_missing(tmp_path, missing_suffix):
    """Report which required measurement raster is absent."""
    product_path = tmp_path / VALID_BIOMASS_NAME
    measurement_path = product_path / "measurement"
    measurement_path.mkdir(parents=True)

    for suffix in ("i_abs.tiff", "i_phase.tiff"):
        if suffix != missing_suffix:
            (measurement_path / f"{VALID_BIOMASS_STEM}_{suffix}").touch()

    with pytest.raises(FileNotFoundError, match=missing_suffix):
        open_biomass_l1a(product_path)


@pytest.mark.parametrize(
    "raster_name, options, error, message",
    [
        ("amplitude", {"shape": (3, 2, 3)}, ValueError, "four bands"),
        ("phase", {"dtype": "uint16"}, TypeError, "dtype float32"),
        ("phase", {"shape": (4, 3, 3)}, ValueError, "shapes do not match"),
        (
            "amplitude",
            {"polarizations": ("HH", "HV", "VV", "VH")},
            ValueError,
            "must contain polarizations",
        ),
    ],
    ids=["bands", "dtype", "shape", "polarizations"],
)
@pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
def test_biomass_raster_invalid(tmp_path, raster_name, options, error, message):
    """Reject invalid band counts, dtypes, shapes, and polarization order."""
    product_path = tmp_path / VALID_BIOMASS_NAME
    measurement_path = product_path / "measurement"
    measurement_path.mkdir(parents=True)

    for name, suffix in (("amplitude", "i_abs.tiff"), ("phase", "i_phase.tiff")):
        raster_options = options if name == raster_name else {}
        _write_biomass_raster(
            measurement_path / f"{VALID_BIOMASS_STEM}_{suffix}",
            **raster_options,
        )

    with pytest.raises(error, match=message):
        open_biomass_l1a(product_path, chunks=None)
