import numpy as np
import pytest
import xarray as xr

from polsarpro.polarisation import _rotated_powers, polarisation_synthesis


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
