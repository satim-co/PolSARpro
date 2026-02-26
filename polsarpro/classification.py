"""
This code is part of the Python PolSARpro software:

"A re-implementation of selected PolSARPro functions in Python,
following the scientific recommendations of PolInSAR 2021"

developed within an ESA funded project with SATIM.

Author: Olivier D'Hondt, 2026.
Scientific advisors: Armando Marino and Eric Pottier.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-----

# Description: module containing polarimetric decomposition functions

"""

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