# Conda release-test devcontainer

This container is for testing the conda-forge package as an installed user would see it.

It bind-mounts the local `data/`, `docs/`, `notebooks/`, and `tests/` folders into `/polsarpro-dev`, but it does not copy or mount the `polsarpro/` source package and does not set `PYTHONPATH`.

After opening the devcontainer, install the released package manually:

```sh
conda install -c conda-forge polsarpro
```

Then run tests from `/polsarpro-dev`:

```sh
pytest tests
```
