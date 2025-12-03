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
