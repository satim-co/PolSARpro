import pytest
import numpy as np
from polsarpro.classification import wishart_h_a_alpha, _h_alpha_classifier

@pytest.mark.parametrize(
    "synthetic_poldata", ["S", "C3", "C4", "T3", "T4"], indirect=True
)
def test_wishart_h_a_alpha(synthetic_poldata):
    input_data = synthetic_poldata
    for poltype, ds in input_data.items():
        result = wishart_h_a_alpha(ds)
        
        # Check output type
        assert isinstance(result, type(ds)), f"Output should be {type(ds)}, got {type(result)}"
        
        # Check output shape matches input spatial dimensions
        assert result['label'].shape == ds.m11.shape if 'm11' in ds else ds.hh.shape, \
            f"Output shape {result['label'].shape} should match input spatial shape"
        
        # Check class values are integers in range [1, 9]
        class_data = result['label'].values
        assert np.issubdtype(class_data.dtype, np.integer), \
            f"Class values should be integers, got {class_data.dtype}"
        assert class_data.min() >= 1, f"Min class value should be >= 1, got {class_data.min()}"
        assert class_data.max() <= 8, f"Max class value should be <= 8, got {class_data.max()}"
        
        # Check attributes
        assert result.attrs.get('poltype') == 'wishart_h_a_alpha', \
            "Output poltype should be 'wishart_h_a_alpha'"

@pytest.mark.parametrize(
    "synthetic_poldata", ["S", "C3", "C4", "T3", "T4"], indirect=True
)
def test_wishart_h_a_alpha_with_ha_result(synthetic_poldata):
    """Test wishart_h_a_alpha with pre-computed h_a_alpha_result."""
    from polsarpro.decompositions import h_a_alpha
    
    input_data = synthetic_poldata
    for poltype, ds in input_data.items():
        # Compute h_a_alpha decomposition first
        ha_result = h_a_alpha(ds, boxcar_size=[5, 5], flags=("entropy", "anisotropy", "alpha"))
        
        # Use pre-computed result
        result = wishart_h_a_alpha(ds, h_a_alpha_result=ha_result)
        
        # Check output type
        assert isinstance(result, type(ds)), f"Output should be {type(ds)}, got {type(result)}"
        
        # Check output shape matches input spatial dimensions
        assert result['label'].shape == ds.m11.shape if 'm11' in ds else ds.hh.shape, \
            f"Output shape {result['label'].shape} should match input spatial shape"
        
        # Check class values are integers in range [1, 9]
        class_data = result['label'].values
        assert np.issubdtype(class_data.dtype, np.integer), \
            f"Class values should be integers, got {class_data.dtype}"
        assert class_data.min() >= 1, f"Min class value should be >= 1, got {class_data.min()}"
        assert class_data.max() <= 9, f"Max class value should be <= 9, got {class_data.max()}"

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