import numpy as np
import pytest
import warnings
from polsarpro.util import vec_to_mat
from polsarpro.decompositions import freeman, freeman_dask

@pytest.mark.filterwarnings("ignore:invalid")
def test_freeman():
    N = 128
    D = 3
    v = np.random.rand(N, N, D) + 1j * np.random.rand(N, N, D)
    # fake T3 matrix
    T3 = vec_to_mat(v)
    for fun in [freeman, freeman_dask]:
        Ps, Pd, Pv = fun(input_data=T3, input_poltype="T3", boxcar_size=[5, 5])

        assert all((it.shape == T3.shape[:2] for it in [Ps, Pd, Pv]))
        assert all((it.dtype == "float32" for it in [Ps, Pd, Pv]))