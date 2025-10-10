import pytest
import numpy as np
import xarray as xr

from polsarpro.auxil import validate_dataset, POLTYPES


def make_dataset(poltype: str, wrong_dtype: bool = False, desc: str | None = None):
    """Utility to build a dataset for a given poltype, with optional mistakes."""
    if poltype not in POLTYPES:
        # use S variables by default if poltype is not correct
        specs = POLTYPES["S"]
    else:
        specs = POLTYPES[poltype]
    shape = (5, 5)

    data_vars = {}
    for v, props in specs["vars"].items():
        dtype = props["dtype"]
        if wrong_dtype:
            # flip float32 -> complex64, complex64 -> float32
            dtype = "complex64" if dtype == "float32" else "float32"
        arr = np.zeros(shape, dtype=dtype)
        data_vars[v] = (("y", "x"), arr)

    ds = xr.Dataset(data_vars)
    ds.attrs["poltype"] = poltype
    ds.attrs["description"] = desc if desc is not None else specs["description"]
    return ds


@pytest.mark.parametrize("poltype", list(POLTYPES.keys()))
def test_valid_datasets_pass(poltype):
    ds = make_dataset(poltype)
    # Should not raise
    validate_dataset(ds, allowed_poltypes=poltype)


def test_invalid_poltype_raises():
    ds = make_dataset("INVALID")
    with pytest.raises(ValueError, match=f"Unsupported poltype"):
        validate_dataset(ds, allowed_poltypes=("S", "T3", "C4"))


def test_missing_variable_raises():
    ds = make_dataset("S")
    ds = ds.drop_vars("hh")
    with pytest.raises(ValueError, match=f"Dataset is missing required variable: 'hh'"):
        validate_dataset(ds, allowed_poltypes="S")


def test_wrong_dtype_detected():
    ds = make_dataset("T3", wrong_dtype=True)
    with pytest.raises(ValueError, match="dtype"):
        validate_dataset(ds, allowed_poltypes="T3", check_dtypes=True)


def test_skip_checks_allows_pass():
    # Wrong dtype, but skip dtype check
    ds = make_dataset("T3", wrong_dtype=True)
    # Should not raise
    validate_dataset(ds, allowed_poltypes="T3", check_dtypes=False)
