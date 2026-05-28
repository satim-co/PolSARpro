import pytest
import numpy as np
from polsarpro.auxil import POLTYPES
from polsarpro.util import vec_to_mat
from polsarpro.util import S_to_C3, S_to_C4, S_to_C2
from polsarpro.util import S_to_T3, S_to_T4
from polsarpro.util import T3_to_C3, T4_to_C4, T4_to_T3
from polsarpro.util import C3_to_T3, C4_to_T4
from polsarpro.util import boxcar
from polsarpro.util import multilook
from polsarpro.util import pauli_rgb


def _assert_polmatrix(res, poltype, shape):
    expected_vars = POLTYPES[poltype]["vars"]

    assert res.poltype == poltype
    assert set(res.data_vars) == set(expected_vars)
    assert all(res[var].shape == shape for var in expected_vars)
    assert all(res[var].dtype == spec["dtype"] for var, spec in expected_vars.items())


def test_vec_to_mat():

    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    M = vec_to_mat(v)
    assert M.shape == (N, N, D, D)
    # M has to be Hermitian
    assert np.allclose(M.transpose((0, 1, 3, 2)), M.conj())
    assert np.allclose(M.diagonal(axis1=2, axis2=3).imag, 0)


@pytest.mark.parametrize("synthetic_poldata", ["S"], indirect=True)
@pytest.mark.parametrize(
    "converter, poltype",
    [
        (S_to_C2, "C2"),
        (S_to_C3, "C3"),
        (S_to_C4, "C4"),
        (S_to_T3, "T3"),
        (S_to_T4, "T4"),
    ],
)
def test_S_conversions(synthetic_poldata, converter, poltype):
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        res = converter(ds)
        _assert_polmatrix(res, poltype, ds.hh.shape)


@pytest.mark.parametrize(
    "synthetic_poldata, input_poltype, converter, poltype",
    [
        ("T3", "T3", T3_to_C3, "C3"),
        ({"poltypes": ["T3", "T4"]}, "T4", T4_to_C4, "C4"),
        ({"poltypes": ["T3", "T4"]}, "T4", T4_to_T3, "T3"),
        ("C3", "C3", C3_to_T3, "T3"),
        ({"poltypes": ["C3", "C4"]}, "C4", C4_to_T4, "T4"),
    ],
    indirect=["synthetic_poldata"],
)
def test_matrix_conversions(synthetic_poldata, input_poltype, converter, poltype):
    input_data = synthetic_poldata
    ds = input_data[input_poltype]
    res = converter(ds)

    _assert_polmatrix(res, poltype, ds.m11.shape)
    if converter is T4_to_T3:
        assert all(
            np.allclose(res[var].compute().values, ds[var].compute().values)
            for var in res.data_vars
        )


@pytest.mark.parametrize("synthetic_poldata", ["C3"], indirect=True)
def test_boxcar(synthetic_poldata):
    input_data = synthetic_poldata

    for _, ds in input_data.items():
        res = boxcar(img=ds, dim_az=5, dim_rg=3)
        _assert_polmatrix(res, ds.poltype, ds.m11.shape)


@pytest.mark.parametrize(
    "synthetic_poldata, input_poltype",
    [
        ({"poltypes": ["C2", "C3"]}, "C2"),
        ("C3", "C3"),
        ({"poltypes": ["C3", "C4"]}, "C4"),
        ("T3", "T3"),
        ({"poltypes": ["T3", "T4"]}, "T4"),
    ],
    indirect=["synthetic_poldata"],
)
def test_multilook(synthetic_poldata, input_poltype):
    ds = synthetic_poldata[input_poltype]
    dim_az = 5
    dim_rg = 2

    input_data = ds.chunk(x=64, y=64)
    res = multilook(input_data=input_data, dim_az=dim_az, dim_rg=dim_rg)
    for var in input_data.data_vars:
        shp = ds[var].shape
        naz_out = shp[0] // dim_az
        nrg_out = shp[1] // dim_rg
        assert res[var].shape == (naz_out, nrg_out)
        assert res[var].dtype == input_data[var].dtype


@pytest.mark.parametrize("synthetic_poldata", ["S", "C3", "T3"], indirect=True)
def test_pauli_rgb(synthetic_poldata):
    input_data = synthetic_poldata
    for _, ds in input_data.items():
        input_data = ds.chunk(x=64, y=64)
        res = pauli_rgb(input_data=input_data)
        var = "hh" if "hh" in ds.data_vars else "m11"
        shp = ds[var].shape
        assert res.shape == (3,) + shp
        assert res.dtype == "float32"
