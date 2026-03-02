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
from polsarpro.decompositions import h_a_alpha

def wishart_h_a_alpha(input_data, boxcar_size=[5, 5]):

    poltype = validate_dataset(input_data, allowed_poltypes=("C3", "T3", "C4", "T4", "S", "h_a_alpha"))


    # If input == S, C3, T3 -> (S_to_C3) -> boxcar -> compute ds_ha = h_a_alpha with flags=("entropy", "anisotropy", "alpha")

    # If input == h_a_alpha, check if H A Alpha are present
    # ds_ha = input_data

    # classify using areas (assign)
    # _h_alpha_classifier(ds_ha)

    # until no change
    # use for loops on class indices
    # compute centers (class wise averages)
    # invert center matrices -> use inv 
    # compute log(det) of centers
    # distance to all classes
    # use hardcoded product trace -> make a _trace{3,4}_hm1hm2 function

def _h_alpha_classifier(ds_ha):
    """
    Classify pixels based on H-Alpha decomposition using decision boundaries.
    
    Parameters
    ----------
    ds_ha : xarray.Dataset
        Dataset containing 'entropy' and 'alpha' variables.
    
    Returns
    -------
    dask.array.Array
        Classification map with integer class labels (1-9).
    """
    alpha = ds_ha.alpha.data
    H = ds_ha.entropy.data
    
    # Decision boundaries
    lim_al1 = 55.0
    lim_al2 = 50.0
    lim_al3 = 48.0
    lim_al4 = 42.0
    lim_al5 = 40.0
    lim_H1 = 0.9
    lim_H2 = 0.5
    
    # Thresholds (C boolean logic: 0 is false, non-zero is true)
    # In Python/dask: True=1, False=0, so we use direct boolean operations
    a1 = (alpha <= lim_al1)
    a2 = (alpha <= lim_al2)
    a3 = (alpha <= lim_al3)
    a4 = (alpha <= lim_al4)
    a5 = (alpha <= lim_al5)
    
    h1 = (H <= lim_H1)
    h2 = (H <= lim_H2)
    
    # Regions
    # In Python/dask: ~ for NOT, & for AND, | for OR
    r1 = (~a3) & h2
    r2 = a3 & (~a4) & h2
    r3 = a4 & h2
    r4 = (~a2) & h1 & (~h2)
    r5 = a2 & (~a5) & h1 & (~h2)
    r6 = a5 & h1 & (~h2)
    r7 = (~a1) & (~h1)
    r8 = a1 & (~a5) & (~h1)
    r9 = a5 & (~h1)  # Non feasible region
    
    # Compute class labels (1-9)
    # Each region contributes its class number where the region is True
    class_map = (1 * r1 + 2 * r2 + 3 * r3 + 4 * r4 + 
                 5 * r5 + 6 * r6 + 7 * r7 + 8 * r8 + 9 * r9)
    
    return class_map
