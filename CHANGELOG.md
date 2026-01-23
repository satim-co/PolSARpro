# 2026.1.2
## New feature
- `open_netcdf_beam` is now able to read calibrated BIOMASS files exported from SNAP.

## Bugfix
- `pauli_rgb` replicates input dimensions instead of looking for y and x. It now works with geocoded data too.

# 2026.1.1
## New feature
- Tutorial notebook to process BIOMASS data on the MAAP platform

## Bugfix and improvements 
- Added some missing S matrix channels in `polmat_to_netcdf`
- Updated the unit test accordingly
- Improved the unit test by using simulated data 
- Simulated data fixture now returns chunked datasets 
- Added a link to the sample dataset in the docs 

# 2026.1.0
## New features 
- Cameron decomposition 
- Pauli RGB visualization for S, C3 and T3 matrices (`util.pauli_rgb`)
- Multilooking operator (`util.multilook`) for all C and T matrices 
- Updated docs 
## Other improvements 
- Moved the legacy PSP bin reading functions to the `dev` module
- Created `conftest.py` to share fake data generation across all unit tests 

# 2025.12.2
## Features
- Freeman-Durden decomposition
- Yamaguchi 4 & 3 component decompositions
- Touzi's TSVM decomposition 
- Van Zyl (1992) 3 component decomposition
- Refined Lee speckle filter 
- PWF (Polarimetric Whitening Filter)
- New function: `polmat_to_netcdf` to write polarimetric matrices to NetCDF files (complex values are not natively handled in NetCDF ). Files written with this function can be read with `open_netcdf_beam`
- PolSARPro is now available as a package on the conda-forge channel
- Updated San Francisco ALOS-1 datasets with the latest JAXA version. Data will be distributed on STEP.
## Improvements 
- Function for automatic data generation to test more polarimetric types for each algorithm (unit tests)
- Add test notebooks for all decompositions for geocoded data 
- Protecting decompositions against division by zero warnings when using geocoded data 

# 2025.12.1
## Bugfixes
- Updated `pyptoject.toml` for conda-forge packaging.

# 2025.12.0
## Bugfixes
- Updated dependencies to match conda recipe.

# 2025.10.0
## Features 
- Definition of and Xarray dataset based structure including: named matrix elements, coordinates (geocoded or SAR geometry), polarimetric type (e.g. S, T3, C4)
- Reader for NetCDF-BEAM files exported from SNAP
- Documentation using mkdocs and hosted on ReadTheDocs https://polsarpro.readthedocs.io
    - Installation guide
    - Tutorials 
    - API reference
    - Theoretical background
- Adapted H/A/Alpha decomposition for the new data structure
- Visualization of data points in the H/Alpha plane  
- Conversion utilities between different polarimetric representations (e.g. S, T3, C4)
- Boxcar filter using 2-D convolutions
- Tutorial notebooks for basic usage and H/A/Alpha
- Updated installation instructions 
## Misc
- Unit tests for decompositions, utilities and readers
- Data validation function ensuring correct structure of input datasets
