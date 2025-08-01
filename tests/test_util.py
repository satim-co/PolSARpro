import numpy as np
import xarray as xr
from polsarpro.util import vec_to_mat
from polsarpro.util import S_to_C3, S_to_C4
from polsarpro.util import S_to_T3, S_to_T4
from polsarpro.util import T3_to_C3, T4_to_C4
from polsarpro.util import C3_to_T3, C4_to_T4
from polsarpro.util import boxcar


def test_vec_to_mat():

    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    M = vec_to_mat(v)
    assert M.shape == (N, N, D, D)
    # M has to be Hermitian
    assert np.allclose(M.transpose((0, 1, 3, 2)), M.conj())
    assert np.allclose(M.diagonal(axis1=2, axis2=3).imag, 0)


def test_S_to_C3():

    N = 128
    S = np.random.rand(N, N, 2, 2) + 1j * np.random.rand(N, N, 2, 2)

    dims = ("y", "x")
    S_dict = dict(
        hh=xr.DataArray(S[..., 0, 0], dims=dims),
        hv=xr.DataArray(S[..., 0, 1], dims=dims),
        vh=xr.DataArray(S[..., 1, 0], dims=dims),
        vv=xr.DataArray(S[..., 1, 1], dims=dims),
    )
    Sx = xr.Dataset(S_dict, attrs=dict(poltype="S"))
    C3x = S_to_C3(Sx)

    # test ouput shapes and types
    assert C3x.poltype == "C3"
    assert all(C3x[var].shape == (N, N) for var in C3x.data_vars)
    assert all(C3x[var].dtype == "float32" for var in ["m11", "m22", "m33"])
    assert all(C3x[var].dtype == "complex64" for var in ["m12", "m13", "m23"])

def test_S_to_C4():

    N = 128
    S = np.random.rand(N, N, 2, 2) + 1j * np.random.rand(N, N, 2, 2)

    dims = ("y", "x")
    S_dict = dict(
        hh=xr.DataArray(S[..., 0, 0], dims=dims),
        hv=xr.DataArray(S[..., 0, 1], dims=dims),
        vh=xr.DataArray(S[..., 1, 0], dims=dims),
        vv=xr.DataArray(S[..., 1, 1], dims=dims),
    )
    Sx = xr.Dataset(S_dict, attrs=dict(poltype="S"))
    C4x = S_to_C4(Sx)

    # test ouput shapes and types
    assert C4x.poltype == "C4"
    assert all(C4x[var].shape == (N, N) for var in C4x.data_vars)
    assert all(C4x[var].dtype == "float32" for var in ["m11", "m22", "m33"])
    assert all(C4x[var].dtype == "complex64" for var in ["m12", "m13", "m23"])


def test_S_to_T3():

    N = 128
    S = np.random.rand(N, N, 2, 2) + 1j * np.random.rand(N, N, 2, 2)

    # Xarray version
    dims = ("y", "x")
    S_dict = dict(
        hh=xr.DataArray(S[..., 0, 0], dims=dims),
        hv=xr.DataArray(S[..., 0, 1], dims=dims),
        vh=xr.DataArray(S[..., 1, 0], dims=dims),
        vv=xr.DataArray(S[..., 1, 1], dims=dims),
    )
    Sx = xr.Dataset(S_dict, attrs=dict(poltype="S"))
    T3x = S_to_T3(Sx)

    # test ouput shapes and types
    assert T3x.poltype == "T3"
    assert all(T3x[var].shape == (N, N) for var in T3x.data_vars)
    assert all(T3x[var].dtype == "float32" for var in ["m11", "m22", "m33"])
    assert all(T3x[var].dtype == "complex64" for var in ["m12", "m13", "m23"])

def test_S_to_T4():

    N = 128
    S = np.random.rand(N, N, 2, 2) + 1j * np.random.rand(N, N, 2, 2)

    # Xarray version
    dims = ("y", "x")
    S_dict = dict(
        hh=xr.DataArray(S[..., 0, 0], dims=dims),
        hv=xr.DataArray(S[..., 0, 1], dims=dims),
        vh=xr.DataArray(S[..., 1, 0], dims=dims),
        vv=xr.DataArray(S[..., 1, 1], dims=dims),
    )
    Sx = xr.Dataset(S_dict, attrs=dict(poltype="S"))
    T4x = S_to_T4(Sx)

    # test ouput shapes and types
    assert T4x.poltype == "T4"
    assert all(T4x[var].shape == (N, N) for var in T4x.data_vars)
    assert all(T4x[var].dtype == "float32" for var in ["m11", "m22", "m33"])
    assert all(T4x[var].dtype == "complex64" for var in ["m12", "m13", "m23"])

def test_T3_to_C3():

    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    # fake T3 matrix
    T3 = vec_to_mat(v)

    # Xarray version
    dims = ("y", "x")
    T3_dict = dict(
        m11=xr.DataArray(T3[..., 0, 0].real.astype("float32"), dims=dims),
        m22=xr.DataArray(T3[..., 1, 1].real.astype("float32"), dims=dims),
        m33=xr.DataArray(T3[..., 2, 2].real.astype("float32"), dims=dims),
        m12=xr.DataArray(T3[..., 0, 1].astype("complex64"), dims=dims),
        m13=xr.DataArray(T3[..., 0, 2].astype("complex64"), dims=dims),
        m23=xr.DataArray(T3[..., 1, 2].astype("complex64"), dims=dims),
    )
    T3x = xr.Dataset(T3_dict, attrs=dict(poltype="T3"))
    C3x = T3_to_C3(T3x)
    # test ouput shapes and types
    assert C3x.poltype == "C3"
    assert all(C3x[var].shape == (N, N) for var in C3x.data_vars)
    assert all(C3x[var].dtype == "float32" for var in ["m11", "m22", "m33"])
    assert all(C3x[var].dtype == "complex64" for var in ["m12", "m13", "m23"])

def test_T4_to_C4():

    N = 128
    D = 4
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    # fake T3 matrix
    T4 = vec_to_mat(v)

    # Xarray version
    dims = ("y", "x")
    T4_dict = dict(
        m11=xr.DataArray(T4[..., 0, 0].real.astype("float32"), dims=dims),
        m22=xr.DataArray(T4[..., 1, 1].real.astype("float32"), dims=dims),
        m33=xr.DataArray(T4[..., 2, 2].real.astype("float32"), dims=dims),
        m44=xr.DataArray(T4[..., 2, 2].real.astype("float32"), dims=dims),
        m12=xr.DataArray(T4[..., 0, 1].astype("complex64"), dims=dims),
        m13=xr.DataArray(T4[..., 0, 2].astype("complex64"), dims=dims),
        m14=xr.DataArray(T4[..., 0, 3].astype("complex64"), dims=dims),
        m23=xr.DataArray(T4[..., 1, 2].astype("complex64"), dims=dims),
        m24=xr.DataArray(T4[..., 1, 3].astype("complex64"), dims=dims),
        m34=xr.DataArray(T4[..., 2, 3].astype("complex64"), dims=dims),
    )
    T4x = xr.Dataset(T4_dict, attrs=dict(poltype="T4"))
    C4x = T4_to_C4(T4x)
    # test ouput shapes and types
    assert C4x.poltype == "C4"
    assert all(C4x[var].shape == (N, N) for var in C4x.data_vars)
    assert all(C4x[var].dtype == "float32" for var in ["m11", "m22", "m33", "m44"])
    assert all(C4x[var].dtype == "complex64" for var in ["m12", "m13", "m14", "m23", "m23", "m34"])

def test_C3_to_T3():

    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    # fake T3 matrix
    C3 = vec_to_mat(v)

    # Xarray version
    dims = ("y", "x")
    C3_dict = dict(
        m11=xr.DataArray(C3[..., 0, 0].real.astype("float32"), dims=dims),
        m22=xr.DataArray(C3[..., 1, 1].real.astype("float32"), dims=dims),
        m33=xr.DataArray(C3[..., 2, 2].real.astype("float32"), dims=dims),
        m12=xr.DataArray(C3[..., 0, 1].astype("complex64"), dims=dims),
        m13=xr.DataArray(C3[..., 0, 2].astype("complex64"), dims=dims),
        m23=xr.DataArray(C3[..., 1, 2].astype("complex64"), dims=dims),
    )
    C3x = xr.Dataset(C3_dict, attrs=dict(poltype="C3"))
    T3x = C3_to_T3(C3x)
    # test ouput shapes and types
    assert T3x.poltype == "T3"
    assert all(T3x[var].shape == (N, N) for var in T3x.data_vars)
    assert all(T3x[var].dtype == "float32" for var in ["m11", "m22", "m33"])
    assert all(T3x[var].dtype == "complex64" for var in ["m12", "m13", "m23"])

def test_C4_to_T4():

    N = 128
    D = 4
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    # fake T3 matrix
    C4 = vec_to_mat(v)

    # Xarray version
    dims = ("y", "x")
    C4_dict = dict(
        m11=xr.DataArray(C4[..., 0, 0].real.astype("float32"), dims=dims),
        m22=xr.DataArray(C4[..., 1, 1].real.astype("float32"), dims=dims),
        m33=xr.DataArray(C4[..., 2, 2].real.astype("float32"), dims=dims),
        m44=xr.DataArray(C4[..., 2, 2].real.astype("float32"), dims=dims),
        m12=xr.DataArray(C4[..., 0, 1].astype("complex64"), dims=dims),
        m13=xr.DataArray(C4[..., 0, 2].astype("complex64"), dims=dims),
        m14=xr.DataArray(C4[..., 0, 3].astype("complex64"), dims=dims),
        m23=xr.DataArray(C4[..., 1, 2].astype("complex64"), dims=dims),
        m24=xr.DataArray(C4[..., 1, 3].astype("complex64"), dims=dims),
        m34=xr.DataArray(C4[..., 2, 3].astype("complex64"), dims=dims),
    )
    C4x = xr.Dataset(C4_dict, attrs=dict(poltype="C4"))
    T4x = C4_to_T4(C4x)
    # test ouput shapes and types
    assert T4x.poltype == "T4"
    assert all(T4x[var].shape == (N, N) for var in T4x.data_vars)
    assert all(T4x[var].dtype == "float32" for var in ["m11", "m22", "m33", "m44"])
    assert all(T4x[var].dtype == "complex64" for var in ["m12", "m13", "m14", "m23", "m23", "m34"])

def test_boxcar():

    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)

    M = vec_to_mat(v)

    # Xarray version
    dims = ("y", "x")
    M_dict = dict(
        m11=xr.DataArray(M[..., 0, 0].real.astype("float32"), dims=dims),
        m22=xr.DataArray(M[..., 1, 1].real.astype("float32"), dims=dims),
        m33=xr.DataArray(M[..., 2, 2].real.astype("float32"), dims=dims),
        m12=xr.DataArray(M[..., 0, 1].astype("complex64"), dims=dims),
        m13=xr.DataArray(M[..., 0, 2].astype("complex64"), dims=dims),
        m23=xr.DataArray(M[..., 1, 2].astype("complex64"), dims=dims),
    )
    Mx = xr.Dataset(M_dict, attrs={"poltype": "C3"})
    param = {"img": Mx, "dim_az": 5, "dim_rg": 3}
    M_box = boxcar(**param)

    # test ouput shapes and types
    assert all(M_box[var].shape == (N, N) for var in Mx.data_vars)
    assert all(M_box[var].dtype == "float32" for var in ["m11", "m22", "m33"])
    assert all(M_box[var].dtype == "complex64" for var in ["m12", "m13", "m23"])
