import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from polsarpro.polarisation import (
    _rotated_powers,
    plot_polarimetric_signature,
    polarimetric_signature,
    polarisation_synthesis,
)


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
def test_pol_synth_output(synthetic_poldata):
    """Check the channel schema for each supported input and output basis."""
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        for basis in ("pauli", "sinclair"):
            res = polarisation_synthesis(
                input_data=ds,
                phi=17.0,
                tau=-11.0,
                basis=basis,
            )
            var = "hh" if "hh" in ds.data_vars else "m11"
            assert res.shape == (3,) + ds[var].shape
            assert res.dims == ("band",) + tuple(ds.dims)
            assert res.band.values.tolist() == ["red", "green", "blue"]
            assert res.dtype == "float32"


@pytest.mark.parametrize(
    "basis, expected",
    [
        ("pauli", [4.0, 6.0, 2.0]),
        ("sinclair", [2.0, 3.0, 4.0]),
    ],
)
def test_pol_synth_chunks(basis, expected):
    """Check zero-angle powers and preservation of spatial Dask chunks."""
    data = xr.Dataset(
        {
            "m11": (("y", "x"), np.full((5, 7), 2.0, dtype=np.float32)),
            "m22": (("y", "x"), np.full((5, 7), 4.0, dtype=np.float32)),
            "m33": (("y", "x"), np.full((5, 7), 6.0, dtype=np.float32)),
            "m12": (("y", "x"), np.full((5, 7), 1.0 + 2.0j, dtype=np.complex64)),
            "m13": (("y", "x"), np.zeros((5, 7), dtype=np.complex64)),
            "m23": (("y", "x"), np.zeros((5, 7), dtype=np.complex64)),
        },
        attrs={"poltype": "T3"},
    ).chunk({"y": 2, "x": 3})

    result = polarisation_synthesis(data, phi=0.0, tau=0.0, basis=basis)

    assert result.chunks == ((1, 1, 1), (2, 2, 1), (3, 3, 1))
    np.testing.assert_array_equal(result.compute().values[:, 0, 0], expected)


def test_pol_synth_broadcast():
    """Check that the shared rotation accepts separate phi and tau axes."""
    data = xr.Dataset(
        {
            "m11": xr.DataArray(2.0),
            "m22": xr.DataArray(4.0),
            "m33": xr.DataArray(6.0),
            "m12": xr.DataArray(1.0 + 2.0j),
            "m13": xr.DataArray(0.0j),
            "m23": xr.DataArray(0.0j),
        }
    )
    phi = xr.DataArray([0.0, 45.0], dims="phi", coords={"phi": [0.0, 45.0]})
    tau = xr.DataArray([0.0, 45.0], dims="tau", coords={"tau": [0.0, 45.0]})

    t11, t12_re, t22, t33 = _rotated_powers(data, phi, tau)

    for term in (t11, t12_re, t33):
        assert set(term.dims) == {"phi", "tau"}
        assert term.shape == (2, 2)
    assert t22.dims == ("phi",)
    assert (
        t11.sel(phi=0.0, tau=0.0).item(),
        t12_re.sel(phi=0.0, tau=0.0).item(),
        t22.sel(phi=0.0).item(),
        t33.sel(phi=0.0, tau=0.0).item(),
    ) == (2.0, 1.0, 4.0, 6.0)


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
@pytest.mark.parametrize("geocoded", [False, True])
def test_signature_pixel(synthetic_poldata, geocoded):
    """Check positional selection and zero-angle powers for each input format."""
    data = next(iter(synthetic_poldata.values()))
    if geocoded:
        data = data.rename({"y": "lat", "x": "lon"})

    result = polarimetric_signature(data, row=3, col=5, n_phi=3, n_tau=3)
    synthesis = polarisation_synthesis(data, basis="sinclair").isel(
        {list(data.dims)[0]: 3, list(data.dims)[1]: 5}
    )

    assert set(result.data_vars) == {"copol", "xpol"}
    assert result.copol.dims == result.xpol.dims == ("phi", "tau")
    assert result.copol.shape == result.xpol.shape == (3, 3)
    assert result.copol.dtype == result.xpol.dtype == np.float32
    assert isinstance(result.copol.data, np.ndarray)
    np.testing.assert_array_equal(result.phi, [-90.0, 0.0, 90.0])
    np.testing.assert_array_equal(result.tau, [-45.0, 0.0, 45.0])
    assert not result.phi.attrs
    assert not result.tau.attrs
    np.testing.assert_allclose(
        result.copol.sel(phi=0, tau=0), synthesis.sel(band="blue")
    )
    np.testing.assert_allclose(
        result.xpol.sel(phi=0, tau=0), synthesis.sel(band="green")
    )
    assert result.attrs["row"] == 3
    assert result.attrs["col"] == 5


@pytest.mark.parametrize("synthetic_poldata", ["T3"], indirect=True)
def test_signature_grid(synthetic_poldata):
    """Check that the default endpoints have exact one-degree spacing."""
    data = next(iter(synthetic_poldata.values()))

    result = polarimetric_signature(data, row=0, col=0)

    assert result.copol.shape == result.xpol.shape == (181, 91)
    np.testing.assert_allclose(np.diff(result.phi), 1.0)
    np.testing.assert_allclose(np.diff(result.tau), 1.0)


@pytest.mark.parametrize(
    "name, value, error",
    [
        ("row", -1, IndexError),
        ("col", 128, IndexError),
        ("row", 1.5, TypeError),
        ("n_phi", 1, ValueError),
        ("n_tau", 2.5, TypeError),
    ],
)
@pytest.mark.parametrize("synthetic_poldata", ["T3"], indirect=True)
def test_signature_invalid(synthetic_poldata, name, value, error):
    data = next(iter(synthetic_poldata.values()))
    kwargs = {"row": 0, "col": 0, name: value}

    with pytest.raises(error):
        polarimetric_signature(data, **kwargs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("synthetic_poldata", ["T3"], indirect=True)
def test_signature_nonfinite(synthetic_poldata, value):
    """Reject a selected pixel containing a non-finite matrix element."""
    data = next(iter(synthetic_poldata.values())).compute()
    data["m11"].values[3, 5] = value

    with pytest.raises(ValueError, match="row=3, col=5 contains non-finite values"):
        polarimetric_signature(data, row=3, col=5)


@pytest.mark.parametrize("synthetic_poldata", ["T3"], indirect=True)
def test_signature_plot(synthetic_poldata):
    """Check labeled 3D axes, camera angles, and editable handles."""
    data = next(iter(synthetic_poldata.values()))
    signature = polarimetric_signature(data, row=0, col=0, n_phi=5, n_tau=3)

    figure, axes = plot_polarimetric_signature(
        signature, azimuth_angle=25.0, elevation_angle=40.0
    )

    assert len(axes) == 2
    assert all(axis.name == "3d" for axis in axes)
    assert [axis.get_title() for axis in axes] == [
        "Co-polar signature",
        "Cross-polar signature",
    ]
    assert all(axis.get_xlabel() == "Ellipticity angle, tau (degrees)" for axis in axes)
    assert all(axis.get_ylabel() == "Orientation angle, phi (degrees)" for axis in axes)
    assert all(axis.get_zlabel() == "Power" for axis in axes)
    assert all(axis.azim == 25.0 and axis.elev == 40.0 for axis in axes)
    assert all(len(axis.collections) == 1 for axis in axes)
    plt.close(figure)
