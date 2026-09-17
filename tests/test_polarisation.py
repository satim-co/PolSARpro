import pytest

from polsarpro.polarisation import polarisation_synthesis


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
def test_polarisation_synthesis(synthetic_poldata):
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
