# PolSARPro

## Installation guidelines
- Clone the repository, type `git checkout feature/pythonic-freeman` and follow the instructions below 

### To use as a devcontainer in VSCode

- This method is recommended, especially for development because it installs all dependencies and sets up environments.
- Moreover jupyter notebooks are also supported directly in VSCode.
- For development purposes, the docker container installs the C version of PolSARpro. This makes it possible to run the different decompositions from the command line, e.g. `freeman_decomposition.exe` to process some data and compare its outputs with the ones of the python version.
- To build the container, it is required to download the zip file of the original PolSARpro in the main directory. This file can be found at https://ietr-lab.univ-rennes1.fr/polsarpro-bio/Linux/PolSARpro_v6.0.4_Biomass_Edition_Linux_Installer_20250122.zip
- This assumes `docker` and `docker compose` are installed on your system.
- Edit `docker-compose.yml` to set volume paths that suit your needs. 
- Open the directory in VSCode
- Then Ctrl-Shift-P (Cmd-Shift-P on Mac) and look for `Rebuild and reopen in container`. This will build the dev container and the development environment may be used.

### To install with conda package manager

- Install conda (recommended: miniforge)
- Create and activate the environment
```bash
conda env create -f environment.yaml
conda activate psp 
```
- Add the toolbox path to `PYTHONPATH`
```bash
export PYTHONPATH="${PYTHONPATH}:/mypath/to/polsarpo/source"
```
- To check that the module is working, `pytest` can be run from the main directory. All tests should pass.

### To run the docker container without VSCode (untested)
- This assumes `docker` and `docker compose` are installed on your system.
- Edit `docker-compose.yml` to set volume paths that suit your needs. 
- From the main directory type `docker compose up -d`
- To stop the container `docker compose down`

## Quickstart
- Start the jupyter notebook at `notebooks/test_freeman_real_data.ipynb` to see example usage of the Freeman function.
- The `freeman` function is a baseline function we used as a reference, which only uses `numpy` arrays. The `freeman_dask` function uses dask arrays and dask features allowing to automatically process the data by chunks and in parallel.
