<p>
    <img src="https://raw.githubusercontent.com/satim-co/PolSARpro/refs/heads/main/docs/polsarpro_logo_dark.svg" alt="PyPolSARpro logo" width="360">
</p>
<p>
    <img src="https://raw.githubusercontent.com/odhondt/PolSARpro/dev-cycle-ms4/docs/collage_lanscape.png" alt="Collage of polarimetric SAR outputs and visualizations" width="900">
</p>

# PyPolSARPro

_"Re-implementation of selected PolSARpro functions in Python, following the scientific recommendations of PolInSAR 2021 (Work In Progress)."_

- [Source code](https://github.com/satim-co/PolSARpro/)
- [Documentation](https://polsarpro.readthedocs.io/)
- [Sample data](https://step.esa.int/auxdata/PolSARpro/SAN_FRANCISCO_ALOS1_slc.nc) — The original data used for this product have been supplied by JAXA’s ALOS-2 sample product.


[![Conda Version](https://img.shields.io/conda/vn/conda-forge/polsarpro.svg)](https://anaconda.org/conda-forge/polsarpro) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/polsarpro.svg)](https://anaconda.org/conda-forge/polsarpro)  

## Features

PyPolSARPro is actively developed, with more features planned for the future.

Currently the available features are:

| Category | Features |
| --- | --- |
| Decompositions | <ul><li>Cameron</li><li>Freeman–Durden (3-component)</li><li>Freeman (2-component)</li><li>H/A/alpha</li><li>Yamaguchi (3- and 4-component)</li><li>Touzi TSVM</li><li>Van Zyl</li></ul> |
| Speckle filtering | <ul><li>Refined Lee</li><li>Polarimetric Whitening Filter</li></ul> |
| Polarisation | <ul><li>Synthesis</li><li>Polarimetric signatures</li><li>Orientation compensation</li></ul> |
| Classification | <ul><li>Wishart H/A/alpha</li><li>Supervised Wishart</li></ul> |
| Surface inversion | <ul><li>Dubois model</li><li>Oh model</li></ul> |
| Averaging | <ul><li>Boxcar filtering</li><li>Multilooking</li></ul> |
| Visualization | <ul><li>H–alpha plane</li><li>Pauli RGB</li><li>Polarimetric signature plots</li></ul> |
| Matrix utilities | <ul><li>Conversions between scattering, covariance, and coherency representations</li></ul> |
| Data access | <ul><li>SNAP NetCDF-BEAM support</li><li>Offline BIOMASS Level-1a reader</li></ul> |
| BIOMASS workflows | <ul><li>Tutorial for [local products](notebooks/biomass-tutorial.ipynb)</li><li>Tutorial for [discovery/access through the MAAP STAC API](notebooks/maap-biomass-stac-api.ipynb)</li></ul> |

PyPolSARPro is powered by xarray and Dask, combining structured, labelled data
with lazy and parallel processing. This supports scalable workflows and
interoperability with the scientific Python ecosystem. PyPolSARPro also supports
BIOMASS data and is compatible with the MAAP ecosystem.

## Installation Guidelines

### Recommended: conda-forge
This is the simplest and most reliable installation method.

- Install the `conda` package manager (recommended: **miniforge**).
- Create a dedicated environment to avoid dependency conflicts:
```bash
conda create -n polsarpro
conda activate polsarpro
```
- Install the package from the `conda-forge` channel:
```bash
conda install conda-forge::polsarpro
```

### Using a cloned repository
Choose this approach if you want access to the source code.

- Clone the repository from GitHub and move into the project root.
- Install `conda` (recommended: **miniforge**).
- Create and activate the environment:  
```bash
conda env create -f environment.yaml
conda activate psp
```
- Add the toolbox to your `PYTHONPATH`:    
```bash
export PYTHONPATH="${PYTHONPATH}:/mypath/to/polsarpro/source"
```
- To verify the installation, run `pytest` from the main directory. All tests should pass.

## Getting Started

Read this [tutorial](https://polsarpro.readthedocs.io/en/latest/quickstart-tutorial/).

Tutorial notebooks may be downloaded from the [notebooks](https://github.com/satim-co/PolSARpro/tree/main/notebooks) directory on Github.

The ALOS-1 image used in the tutorials may be downloaded [here](https://step.esa.int/auxdata/PolSARpro/SAN_FRANCISCO_ALOS1_slc.nc). The original data used for this product have been supplied by JAXA’s ALOS-2 sample product.

## Scientific evaluation

The routines have been compared with the PolSARPro C code and eventual numerical differences are negligible. If you would like to look at the comparison reports, please contact [Dr Armando Marino](https://www.stir.ac.uk/people/894087) at the University of Stirling, Scotland, UK.
