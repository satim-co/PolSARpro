import pytest
from polsarpro.speckle_filters import refined_lee, pwf


@pytest.mark.parametrize(
    "synthetic_poldata", ["C2", "C3", "C4", "T3", "T4"], indirect=True
)
def test_refined_lee(synthetic_poldata):
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        input_data = ds.chunk(x=64, y=64)
        res = refined_lee(
            input_data=input_data,
            window_size=7,
            num_looks=4,
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert res.data_vars.dtypes == ds.data_vars.dtypes
        assert all((res[it].shape == shp for it in ds.data_vars))


@pytest.mark.parametrize(
    "synthetic_poldata", ["S", "C2", "C3", "C4", "T3", "T4"], indirect=True
)
def test_pwf(synthetic_poldata):
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        input_data = ds.chunk(x=64, y=64)
        res = pwf(
            input_data=input_data,
            train_window_size=[7, 7],
            test_window_size=[3, 3],
        )
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert res["pwf"].dtype == "float32"
        assert res["pwf"].shape == shp
