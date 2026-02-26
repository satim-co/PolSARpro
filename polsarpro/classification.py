import numpy as np
import xarray as xr
import dask.array as da
from polsarpro.util import boxcar, C3_to_T3, S_to_C3, S_to_T3, C4_to_T4, T3_to_C3
from polsarpro.auxil import validate_dataset

def wishart_h_a_alpha(input_data):
    # If input == S, C3, T3 -> (S_to_C3) -> compute h_a_alpha

    # If input == h_a_alpha, check if H A Alpha are present

    # Define areas 

    # classify using areas (assign)

    # until no change
    # use for loops on class indices
    # compute centers (class wise averages)
    # invert center matrices -> use inv 
    # compute log(det) of centers
    # distance to all classes
    # use hardcoded product trace -> make a _trace{3,4}_hm1hm2 function
    pass