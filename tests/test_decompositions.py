import numpy as np
import pytest
from polsarpro.util import vec_to_mat, T3_to_C3, C3_to_T3
from polsarpro.decompositions import freeman, freeman_dask
from polsarpro.decompositions import h_a_alpha, h_a_alpha_dask 


@pytest.mark.filterwarnings("ignore:invalid")
def test_freeman():
    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    S = np.random.rand(N, N, 2, 2) + 1j * np.random.rand(N, N, 2, 2)
    # fake T3 matrix
    T3 = vec_to_mat(v)
    C3 = T3_to_C3(T3)
    for fun in [freeman, freeman_dask]:
        for poltype, input_data in zip(["C3", "T3", "S"], [C3, T3, S]):
            Ps, Pd, Pv = fun(
                input_data=input_data, input_poltype=poltype, boxcar_size=[5, 5]
            )

            assert all((it.shape == input_data.shape[:2] for it in [Ps, Pd, Pv]))
            assert all((it.dtype == "float32" for it in [Ps, Pd, Pv]))

@pytest.mark.filterwarnings("ignore:invalid")
def test_h_a_alpha():
    N = 128
    D = 3
    v = np.random.randn(N, N, D) + 1j * np.random.randn(N, N, D)
    S = np.random.randn(N, N, 2, 2) + 1j * np.random.randn(N, N, 2, 2).astype("complex64")
    # fake input matrices
    C3 = vec_to_mat(v).astype("complex64")
    T3 = C3_to_T3(C3)
    for fun in [h_a_alpha, h_a_alpha_dask]:
        for poltype, input_data in zip(["C3", "T3", "S"], [C3, T3, S]):
            outputs = fun(
                input_data=input_data, input_poltype=poltype, boxcar_size=[5, 5], flags=("alpha", "entropy", "anisotropy", "betas")
            )
            h, a, alpha, betas = outputs["entropy"], outputs["anisotropy"], outputs["alpha"], outputs["betas"]

            assert all((it.shape == input_data.shape[:2] for it in [h, a, alpha]))
            assert betas.shape ==  input_data.shape[:2] + (3,)
            assert all((it.dtype == "float32" for it in [h, a, alpha]))
