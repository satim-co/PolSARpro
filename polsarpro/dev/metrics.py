import numpy as np
import xarray
import pandas as pd
from collections import OrderedDict
import warnings

# these are convenience function used only for development


# returns a dict with various error metrics for two arrays (predicted and reference)
def compute_error_metrics(arr_pred, arr_ref):
    stats = OrderedDict()

    eps = 1e-30

    # simple error
    err = arr_pred - arr_ref

    # squared and absolute errors
    se = err**2
    ae = abs(err)

    # normalized absolute error (between 0 and 1)
    naerr = ae / (eps + abs(arr_pred) + abs(arr_ref))

    # ---- Global statistics

    dm = np.nanmean(arr_pred) - np.nanmean(arr_ref)
    item = {"value": dm, "descr": "Difference of Means"}
    stats["dm"] = item

    rmse = np.sqrt(np.nanmean(se) + eps)
    item = {"value": rmse, "descr": "Root Mean Square Error"}
    stats["rmse"] = item

    bias = np.nanmean(err)
    item = {"value": bias, "descr": "Mean Error (bias)"}
    stats["bias"] = item

    mae = np.nanmean(ae)
    item = {"value": mae, "descr": "Mean Absolute Error"}
    stats["mae"] = item

    mnae = np.nanmean(naerr)
    item = {"value": mnae, "descr": "Mean Normalized Absolute Error"}
    stats["mnae"] = item

    p99 = np.nanpercentile(naerr, q=99)
    item = {"value": p99, "descr": "99% percentile of the Normalized Absolute Error"}
    stats["p99"] = item

    p90 = np.nanpercentile(naerr, q=90)
    item = {"value": p90, "descr": "90% percentile of the Normalized Absolute Error"}
    stats["p90"] = item

    p50 = np.nanpercentile(naerr, q=50)
    item = {"value": p50, "descr": "50% percentile of the Normalized Absolute Error"}
    stats["p50"] = item

    return stats


# returns a dataframe that summarizes statistics for all variables
def summarize_metrics(
    out_py: xarray.Dataset,
    out_c: dict,
    short_titles: bool = False,
    verbose: bool = True,
):

    data_vars = list(out_py.data_vars)

    # dataframes used to collect results and to merge
    dfs = []

    for var in data_vars:
        # check that python variable appears in C data
        if var not in out_c:
            warnings.warn(f"Skipping variable '{var}'! Not found in C outputs.")
            continue
        else:
            if verbose:
                print(f"Computing variable '{var}'.")
        stats = compute_error_metrics(out_py[var], out_c[var])
        if short_titles:
            tmp = {k: [stats[k]["value"]] for k in stats}
        else:
            tmp = {stats[k]["descr"]: [stats[k]["value"]] for k in stats}
        df = pd.DataFrame.from_dict(tmp, orient="columns")
        df.index = [var]
        dfs.append(df)

    return pd.concat(dfs)

# helper function to visualize errors
def visualize_errors(out_py, out_c, clip=True):
    import matplotlib.pyplot as plt

    for var in out_py.data_vars:
        if var not in out_c:
            Warning(f"Skipping variable '{var}'! Not found in C outputs.")
        if np.iscomplexobj(out_py[var]):
            img_c = np.abs(out_c[var])
            img_py = np.abs(out_py[var])
            err = abs(out_py[var] - out_c[var])
            t_err = "Absolute difference: python - C"
        else:
            img_c = out_c[var]
            img_py = out_py[var]
            err = out_py[var] - out_c[var]
            t_err = "Difference: python - C"

        if clip:
            m = 0.5 * (np.nanmean(img_py) + np.nanmean(img_py))
            img_c = img_c.clip(0, 2*m) 
            img_py = img_py.clip(0, 2*m) 

        plt.figure(figsize=(10, 6))
        plt.suptitle(var)
        plt.subplot(131)
        plt.imshow(err[::8], interpolation="none")
        plt.title(t_err)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04, location="bottom")
        plt.subplot(132)
        plt.imshow(img_c[::8], interpolation="none")
        plt.title("C")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04, location="bottom")
        plt.subplot(133)
        plt.imshow(img_py[::8], interpolation="none")
        plt.colorbar(fraction=0.046, pad=0.04, location="bottom")
        plt.title("python")
        plt.axis("off")