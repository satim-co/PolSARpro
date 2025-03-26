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
    S = np.random.rand(N, N, 2, 2) + 1j * np.random.rand(N, N, 2, 2)
    # fake T3 matrix
    T3 = vec_to_mat(v)
    for fun in [freeman, freeman_dask]:
        for poltype, input_data in zip(["T3", "S"], [T3, S]):
            Ps, Pd, Pv = fun(
                input_data=input_data, input_poltype=poltype, boxcar_size=[5, 5]
            )
            print(Ps.dtype)

            assert all((it.shape == input_data.shape[:2] for it in [Ps, Pd, Pv]))
            assert all((it.dtype == "float32" for it in [Ps, Pd, Pv]))
