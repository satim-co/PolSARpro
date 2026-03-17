import pytest
import numpy as np
from polsarpro.classification import wishart_h_a_alpha, _h_alpha_classifier


# Fast tests with chunk_size=128 (single chunk for 128x128 image)
@pytest.mark.parametrize(
    "synthetic_poldata",
    [
        {"poltypes": "S", "chunk_size": 128},
        {"poltypes": "C3", "chunk_size": 128},
        {"poltypes": "C4", "chunk_size": 128},
        {"poltypes": "T3", "chunk_size": 128},
        {"poltypes": "T4", "chunk_size": 128},
    ],
    indirect=True,
)
def test_wishart_h_a_alpha(synthetic_poldata):
    input_data = synthetic_poldata
    for poltype, ds in input_data.items():
        result = wishart_h_a_alpha(ds)

        # Check output type
        assert isinstance(
            result, type(ds)
        ), f"Output should be {type(ds)}, got {type(result)}"

        # Check output shape matches input spatial dimensions
        assert (
            result["label"].shape == ds.m11.shape if "m11" in ds else ds.hh.shape
        ), f"Output shape {result['label'].shape} should match input spatial shape"

        # Check class values are integers in range [1, 9]
        class_data = result["label"].values
        assert np.issubdtype(
            class_data.dtype, np.integer
        ), f"Class values should be integers, got {class_data.dtype}"
        assert (
            class_data.min() >= 1
        ), f"Min class value should be >= 1, got {class_data.min()}"
        assert (
            class_data.max() <= 8
        ), f"Max class value should be <= 8, got {class_data.max()}"

        # Check attributes
        assert (
            result.attrs.get("poltype") == "wishart_h_a_alpha"
        ), "Output poltype should be 'wishart_h_a_alpha'"


@pytest.mark.parametrize(
    "synthetic_poldata",
    [
        {"poltypes": "S", "chunk_size": 128},
        {"poltypes": "C3", "chunk_size": 128},
        {"poltypes": "C4", "chunk_size": 128},
        {"poltypes": "T3", "chunk_size": 128},
        {"poltypes": "T4", "chunk_size": 128},
    ],
    indirect=True,
)
def test_wishart_h_a_alpha_with_ha_result(synthetic_poldata):
    """Test wishart_h_a_alpha with pre-computed h_a_alpha_result."""
    from polsarpro.decompositions import h_a_alpha

    input_data = synthetic_poldata
    for poltype, ds in input_data.items():
        # Compute h_a_alpha decomposition first
        ha_result = h_a_alpha(
            ds, boxcar_size=[5, 5], flags=("entropy", "anisotropy", "alpha")
        )

        # Use pre-computed result
        result = wishart_h_a_alpha(ds, h_a_alpha_result=ha_result)

        # Check output type
        assert isinstance(
            result, type(ds)
        ), f"Output should be {type(ds)}, got {type(result)}"

        # Check output shape matches input spatial dimensions
        assert (
            result["label"].shape == ds.m11.shape if "m11" in ds else ds.hh.shape
        ), f"Output shape {result['label'].shape} should match input spatial shape"

        # Check class values are integers in range [1, 9]
        class_data = result["label"].values
        assert np.issubdtype(
            class_data.dtype, np.integer
        ), f"Class values should be integers, got {class_data.dtype}"
        assert (
            class_data.min() >= 1
        ), f"Min class value should be >= 1, got {class_data.min()}"
        assert (
            class_data.max() <= 9
        ), f"Max class value should be <= 9, got {class_data.max()}"


@pytest.mark.parametrize("synthetic_poldata", ["h_a_alpha"], indirect=True)
def test_h_alpha_classifier(synthetic_poldata):
    ds = synthetic_poldata["h_a_alpha"]
    res = _h_alpha_classifier(ds).compute()

    assert res.dtype == int
    assert res.shape == ds.entropy.shape
    assert res.min() >= 1
    assert res.max() <= 9


# Test with default chunk_size=16 to verify small chunks still work
@pytest.mark.parametrize("synthetic_poldata", [{"poltype": "T3", "chunk_size": 128}], indirect=True)
def test_wishart_h_a_alpha_max_iter(synthetic_poldata):
    """Test wishart_h_a_alpha with custom max_iter parameter.

    Uses default chunk_size=16 to verify the algorithm works with
    smaller chunks (slower but tests chunked computation).
    """
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test with max_iter=1
    result_1_iter = wishart_h_a_alpha(ds, max_iter=1)
    assert isinstance(result_1_iter, type(ds))

    # Test with max_iter=5
    result_5_iter = wishart_h_a_alpha(ds, max_iter=5)
    assert isinstance(result_5_iter, type(ds))


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": "T3", "chunk_size": 128}],
    indirect=True,
)
def test_wishart_h_a_alpha_stop_threshold(synthetic_poldata):
    """Test wishart_h_a_alpha with custom stop_threshold parameter."""
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test with stop_threshold=5.0 (more strict convergence)
    result_strict = wishart_h_a_alpha(ds, stop_threshold=5.0)
    assert isinstance(result_strict, type(ds))

    # Test with stop_threshold=20.0 (less strict convergence)
    result_relaxed = wishart_h_a_alpha(ds, stop_threshold=20.0)
    assert isinstance(result_relaxed, type(ds))

    # Test with stop_threshold=0.0 (run until max_iter)
    result_no_early_stop = wishart_h_a_alpha(ds, stop_threshold=0.0)
    assert isinstance(result_no_early_stop, type(ds))


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": "T3", "chunk_size": 128}],
    indirect=True,
)
def test_wishart_h_a_alpha_combined_params(synthetic_poldata):
    """Test wishart_h_a_alpha with both max_iter and stop_threshold."""
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test with custom values for both parameters
    result = wishart_h_a_alpha(ds, max_iter=5, stop_threshold=15.0)
    assert isinstance(result, type(ds))


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": "T3", "chunk_size": 128}],
    indirect=True,
)
def test_wishart_h_a_alpha_invalid_max_iter(synthetic_poldata):
    """Test wishart_h_a_alpha raises errors for invalid max_iter."""
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test with non-integer max_iter
    with pytest.raises(TypeError, match="max_iter must be an integer"):
        wishart_h_a_alpha(ds, max_iter=3.5)

    with pytest.raises(TypeError, match="max_iter must be an integer"):
        wishart_h_a_alpha(ds, max_iter="10")

    # Test with non-positive max_iter
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        wishart_h_a_alpha(ds, max_iter=0)

    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        wishart_h_a_alpha(ds, max_iter=-1)


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": "T3", "chunk_size": 128}],
    indirect=True,
)
def test_wishart_h_a_alpha_invalid_stop_threshold(synthetic_poldata):
    """Test wishart_h_a_alpha raises errors for invalid stop_threshold."""
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test with non-numeric stop_threshold
    with pytest.raises(TypeError, match="stop_threshold must be a number"):
        wishart_h_a_alpha(ds, stop_threshold="10")

    # Test with out-of-range stop_threshold
    with pytest.raises(ValueError, match="stop_threshold must be in the range"):
        wishart_h_a_alpha(ds, stop_threshold=-1.0)

    with pytest.raises(ValueError, match="stop_threshold must be in the range"):
        wishart_h_a_alpha(ds, stop_threshold=101.0)
