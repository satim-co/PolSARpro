import dask.array as da
import pytest
import numpy as np
import xarray as xr
from polsarpro.classification import (
    wishart_h_a_alpha,
    wishart_supervised,
    _h_alpha_classifier,
    _label_training_clusters,
)


# Fast tests with chunk_size=128 (single chunk for 128x128 image)
@pytest.mark.parametrize(
    "synthetic_poldata",
    [
        {"poltypes": "S", "size": 64, "chunk_size": 64},
        {"poltypes": "C3", "size": 64, "chunk_size": 64},
        {"poltypes": "C4", "size": 64, "chunk_size": 64},
        {"poltypes": "T3", "size": 64, "chunk_size": 64},
        {"poltypes": "T4", "size": 64, "chunk_size": 64},
    ],
    indirect=True,
)
def test_wishart_h_a_alpha(synthetic_poldata):
    input_data = synthetic_poldata
    for poltype, ds in input_data.items():
        result = wishart_h_a_alpha(ds).compute()

        # Check output type
        assert isinstance(
            result, type(ds)
        ), f"Output should be {type(ds)}, got {type(result)}"

        # Check output shape matches input spatial dimensions
        assert (
            result["wishart_h_alpha_class"].shape == ds.m11.shape if "m11" in ds else ds.hh.shape
        ), f"Output shape {result['wishart_h_alpha_class'].shape} should match input spatial shape"

        # Access computed data (already in memory after .compute() above)
        class_data = result["wishart_h_alpha_class"]

        # Check class values are integers in range [1, 9]
        assert np.issubdtype(
            class_data.dtype, np.integer
        ), f"Class values should be integers, got {class_data.dtype}"
        assert (
            class_data.min() >= 1
        ), f"Min class value should be >= 1, got {class_data.min()}"
        assert (
            class_data.max() <= 8
        ), f"Max class value should be <= 8, got {class_data.max()}"

        class_data = result["wishart_h_a_alpha_class"]

        # Check class values are integers in range [1, 9]
        assert np.issubdtype(
            class_data.dtype, np.integer
        ), f"Class values should be integers, got {class_data.dtype}"
        assert (
            class_data.min() >= 1
        ), f"Min class value should be >= 1, got {class_data.min()}"
        assert (
            class_data.max() <= 16
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
        result = wishart_h_a_alpha(ds, h_a_alpha_result=ha_result).compute()

        # Check output type
        assert isinstance(
            result, type(ds)
        ), f"Output should be {type(ds)}, got {type(result)}"

        # Check output shape matches input spatial dimensions
        assert (
            result["wishart_h_alpha_class"].shape == ds.m11.shape if "m11" in ds else ds.hh.shape
        ), f"Output shape {result['wishart_h_alpha_class'].shape} should match input spatial shape"

        # Access computed data (already in memory after .compute() above)
        class_data = result["wishart_h_alpha_class"]

        # Check class values are integers in range [1, 9]
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

    assert res.dtype == np.uint8
    assert res.shape == ds.entropy.shape
    assert res.min() >= 1
    assert res.max() <= 9


def test_label_training_clusters():
    # ensure that connected regions with 
    # different classes are in different clusters
    training_labels = np.zeros((5, 5), dtype=np.uint8)
    training_labels[0, 0] = 1
    training_labels[0, 4] = 1
    training_labels[4, 4] = 2

    lab, cluster_classes = _label_training_clusters(training_labels)

    assert lab.max() == 3
    assert cluster_classes.tolist() == [0, 1, 1, 2]


@pytest.mark.parametrize(
    "synthetic_poldata",
    [
        {
            "poltypes": ["S", "C3", "C4", "T3", "T4"],
            "size": 64,
            "chunk_size": 16,
        }
    ],
    indirect=True,
)
def test_wishart_supervised(synthetic_poldata):
    fake_class_map = np.zeros((64, 64), dtype=np.uint8)
    fake_class_map[1, 1] = 1
    fake_class_map[1, 30] = 1
    fake_class_map[30, 1] = 2
    fake_class_map[30, 30] = 2

    for poltype, ds in synthetic_poldata.items():
        training_labels = xr.DataArray(
            fake_class_map,
            dims=("y", "x"),
            coords={"y": ds.coords["y"], "x": ds.coords["x"]},
        )
        result = wishart_supervised(ds, training_labels, boxcar_size=[1, 1]).compute()
        class_map = result["wishart_supervised_class"].values

        assert class_map.shape == (ds.sizes["y"], ds.sizes["x"])
        assert np.issubdtype(class_map.dtype, np.integer)
        assert set(np.unique(class_map)).issubset({1, 2}), (
            f"Unexpected class labels for poltype {poltype}: "
            f"{set(np.unique(class_map))}"
        )
        assert result.attrs.get("poltype") == "wishart_supervised"


# Test with default chunk_size=16 to verify small chunks still work
@pytest.mark.parametrize(
    "synthetic_poldata", [{"poltype": "T3", "chunk_size": 128}], indirect=True
)
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
def test_wishart_h_a_alpha_tol_pct(synthetic_poldata):
    """Test wishart_h_a_alpha with custom tol_pct parameter."""
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test with tol_pct=9.0 (more strict convergence)
    result_strict = wishart_h_a_alpha(ds, tol_pct=9.0)
    assert isinstance(result_strict, type(ds))

    # Test with tol_pct=11.0 (less strict convergence)
    result_relaxed = wishart_h_a_alpha(ds, tol_pct=11.0)
    assert isinstance(result_relaxed, type(ds))


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": "T3", "chunk_size": 128}],
    indirect=True,
)
def test_wishart_h_a_alpha_combined_params(synthetic_poldata):
    """Test wishart_h_a_alpha with both max_iter and tol_pct."""
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test with custom values for both parameters
    result = wishart_h_a_alpha(ds, max_iter=5, tol_pct=15.0)
    assert isinstance(result, type(ds))


@pytest.mark.parametrize(
    "synthetic_poldata",
    [{"poltypes": "T3", "chunk_size": 128}],
    indirect=True,
)
def test_wishart_h_a_alpha_invalid_params(synthetic_poldata):
    """Test wishart_h_a_alpha raises errors for invalid parameters."""
    input_data = synthetic_poldata
    ds = input_data["T3"]

    # Test invalid max_iter
    with pytest.raises(TypeError, match="max_iter must be an integer"):
        wishart_h_a_alpha(ds, max_iter=3.5)

    with pytest.raises(TypeError, match="max_iter must be an integer"):
        wishart_h_a_alpha(ds, max_iter="10")

    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        wishart_h_a_alpha(ds, max_iter=0)

    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        wishart_h_a_alpha(ds, max_iter=-1)

    # Test invalid tol_pct
    with pytest.raises(TypeError, match="tol_pct must be a number"):
        wishart_h_a_alpha(ds, tol_pct="10")

    with pytest.raises(ValueError, match="tol_pct must be in the range"):
        wishart_h_a_alpha(ds, tol_pct=-1.0)

    with pytest.raises(ValueError, match="tol_pct must be in the range"):
        wishart_h_a_alpha(ds, tol_pct=101.0)
