# Conda release-test devcontainer

This container is for testing the conda-forge package as an installed user would see it.

It starts from a plain Miniforge image with only the base conda environment. It bind-mounts the local `README.md`, `data/`, `docs/`, `notebooks/`, and `tests/` into `/polsarpro-dev`, but it does not copy or mount the `polsarpro/` source package and does not set `PYTHONPATH`.

After opening the devcontainer, follow the main README conda-forge install flow:

```sh
conda create -n polsarpro
conda activate polsarpro
conda install conda-forge::polsarpro
```

Then run tests from `/polsarpro-dev`:

```sh
pytest tests
```
