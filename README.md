# Installation guidelines
- Clone the repository and follow the instructions below.

## To use as a devcontainer in VSCode

- This method is recommended because it installs all dependencies and sets up environments.
- This assumes `docker` and `docker compose` are installed on your system.
- Edit `docker-compose.yml` to set volume paths that suit your needs. 
- Open the directory in VSCode
- Then Ctrl-Shift-P (Cmd-Shift-P on Mac) and look for `Rebuild and reopen in container`. This will build the dev container and the development environment may be used.

## To install with conda package manager

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
- To check that the module is working, `pytest` can be run from the `eo_tools` directory. All tests should pass.
- Note: with this installation method, the visualization using TiTiler cannot be used.

## To run the docker container without VSCode (untested)

- This assumes `docker` and `docker compose` are installed on your system.
- Edit `docker-compose.yml` to set volume paths that suit your needs. 
- From the main directory type `docker compose up -d`
- To stop the container `docker compose down`