# PolSARPro

_"Re-implementation of selected PolSARpro functions in Python, following the scientific recommendations of PolInSAR 2021 (Work In Progress)."_

#### Important note:
This is a temporary documentation that is describing features in a development branch.
This documentation will become official when the branch is merged into `main`.

## Installation guidelines
- Clone the repository and follow the instructions below 

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

### To use as a devcontainer in VSCode

- This method is recommended for development or advanced users who are familiar with docker.
- Moreover jupyter notebooks are also supported directly in VSCode.
- For development purposes, the docker container installs the C version of PolSARpro. This makes it possible to run the different decompositions from the command line, e.g. `freeman_decomposition.exe` to process some data and compare its outputs with the ones of the python version.
- To build the container, it is required to download the zip file of the original PolSARpro in the main directory. This file can be found at https://ietr-lab.univ-rennes1.fr/polsarpro-bio/Linux/PolSARpro_v6.0.4_Biomass_Edition_Linux_Installer_20250122.zip
- This assumes `docker` and `docker compose` are installed on your system.
- Edit `docker-compose.yml` to set volume paths that suit your needs. 
- Open the directory in VSCode
- Then Ctrl-Shift-P (Cmd-Shift-P on Mac) and look for `Rebuild and reopen in container`. This will build the dev container and the development environment may be used.

### To run the docker container without VSCode (untested)
- This assumes `docker` and `docker compose` are installed on your system.
- Edit `docker-compose.yml` to set volume paths that suit your needs. 
- From the main directory type `docker compose up -d`
- To stop the container `docker compose down`

## Getting started
Read this [tutorial](https://polsarpro.readthedocs.io/en/latest/quickstart-tutorial/) or use the tutorial notebook in the `notebooks/folder`.
