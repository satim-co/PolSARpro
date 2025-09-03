import numpy as np


def compute_error_metrics(arr_pred, arr_ref, bins=200):
    stats = {}

    eps = 1e-30

    # simple error
    err = arr_pred - arr_ref

    se = err**2
    ae = abs(err)

    # normalized absolute error
    naerr = ae / (eps + abs(arr_pred) + abs(arr_ref))

    dm = np.sqrt(np.nanmean(arr_pred) - np.nanmean(arr_ref))
    item = {"value": dm, "descr": "Difference of Means"}
    stats["rmse"] = item

    rmse = np.sqrt(np.nanmean(se) + eps)
    item = {"value": rmse, "descr": "Root Mean Square Error"}
    stats["rmse"] = item

    mae = np.nanmean(ae)
    item = {"value": mae, "descr": "Mean Absolute Error"}
    stats["mae"] = item

    mnae = np.nanmean(naerr)
    item = {"value": mnae, "descr": "Mean Normalized Absolute Error"}
    stats["mnae"] = item

    bias = np.nanmean(err)
    item = {"value": bias, "descr": "Mean Error (bias)"}
    stats["bias"] = item

    # herr = np.histogram(err, bins=bins)
    # item = {"value": herr, "descr": "Error histrogram"}
    # stats["hist_err"] = item

    p99 = np.nanpercentile(naerr, q=99)
    item = {"value": p99, "descr": "99% percentile of the Normalized Absolute Error"}
    stats["p99"] = item

    p90 = np.nanpercentile(naerr, q=90)
    item = {"value": p90, "descr": "90% percentile of the Normalized Absolute Error"}
    stats["p90"] = item

    p50 = np.nanpercentile(naerr, q=50)
    item = {"value": p50, "descr": "50% percentile of the Normalized Absolute Error"}
    stats["p50"] = item


