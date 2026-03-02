import pytest
from polsarpro.classification import wishart_h_a_alpha, _h_alpha_classifier

@pytest.mark.parametrize(
    "synthetic_poldata", ["S", "C3", "C4", "T3", "T4", "h_a_alpha"], indirect=True
)
def test_wishart_h_a_alpha(synthetic_poldata):
    input_data = synthetic_poldata
    for _, ds in input_data.items():
        poltype = wishart_h_a_alpha(ds)

@pytest.mark.parametrize(
    "synthetic_poldata", ["h_a_alpha"], indirect=True
)
def test_h_alpha_classifier(synthetic_poldata):
    ds = synthetic_poldata["h_a_alpha"]
    res = _h_alpha_classifier(ds)

    assert res.dtype == int
    assert res.shape == ds.entropy.shape
    assert res.min() >= 1
    assert res.max() <= 9