import numpy as np
from polsarpro.util import vec_to_mat, S_to_C3, S_to_C3_dask
from polsarpro.util import T3_to_C3, T3_to_C3_dask
from polsarpro.util import boxcar, boxcar_dask


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
    for fun in [S_to_C3, S_to_C3_dask]:
        C3 = fun(S)
        assert C3.shape == (N, N, 3, 3)
        # C3 has to be Hermitian
        assert np.allclose(C3.transpose((0, 1, 3, 2)), C3.conj())
        assert np.allclose(C3.diagonal(axis1=2, axis2=3).imag, 0)


def test_T3_to_C3():

    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    # fake T3 matrix
    T3 = vec_to_mat(v)
    for fun in [T3_to_C3, T3_to_C3_dask]:
        C3 = fun(T3)
        assert C3.shape == (N, N, 3, 3)
        # C3 has to be Hermitian
        assert np.allclose(C3.transpose((0, 1, 3, 2)), C3.conj())
        assert np.allclose(C3.diagonal(axis1=2, axis2=3).imag, 0)
        # span must remain the same
        assert np.allclose(
            C3.diagonal(axis1=2, axis2=3).sum().real,
            T3.diagonal(axis1=2, axis2=3).sum().real,
        )


def test_boxcar():

    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)

    M = vec_to_mat(v)
    for fun in [boxcar, boxcar_dask]:
        param = {"img": M, "dim_az": 5, "dim_rg": 3}
        M_box = fun(**param)
        assert M_box.shape == M.shape
        assert M_box.dtype == M.dtype
        # output has to be Hermitian
        assert np.allclose(M_box.transpose((0, 1, 3, 2)), M_box.conj())
        assert np.allclose(M_box.diagonal(axis1=2, axis2=3).imag, 0)

