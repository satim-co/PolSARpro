# this script may be used to measure running time of any function
# since bench.timeit is not accurate for dask based processing

from pathlib import Path
from polsarpro.io import open_netcdf_beam
from polsarpro.decompositions import h_a_alpha
import time

# change to your data paths
# original dataset
input_alos_slc = Path("/data/psp/test_files/SAN_FRANCISCO_ALOS1_slc.nc")
input_alos_slc_re = Path("/data/psp/test_files/SAN_FRANCISCO_ALOS1_slc_rechunked.nc")

# input files from C
input_test_dir = Path("/data/psp/test_files/input/h_a_alpha_decomposition/")

# output files from C
output_test_dir = Path("/data/psp/res/h_a_alpha_c")

n_runs = 3
# %%
test_path = "/data/psp/test_files/test.zarr"
slc = open_netcdf_beam(input_alos_slc)
for _ in range(n_runs):
    start_time = time.perf_counter()
    res_slc = h_a_alpha(slc, boxcar_size=[7 ,7]).compute()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Function took {total_time:.4f} seconds')