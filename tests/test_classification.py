import pytest
from polsarpro.classification import wishart_h_a_alpha

@pytest.mark.parametrize(
    "synthetic_poldata", ["S", "C3", "C4", "T3", "T4", "h_a_alpha"], indirect=True
)
def test_wishart_h_a_alpha(synthetic_poldata):
    input_data = synthetic_poldata
    for _, ds in input_data.items():
        poltype = wishart_h_a_alpha(ds)