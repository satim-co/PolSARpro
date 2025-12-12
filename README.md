# PolSARPro

_"Re-implementation of selected PolSARpro functions in Python, following the scientific recommendations of PolInSAR 2021 (Work In Progress)."_

## Installation Guidelines

### Install from conda-forge (recommended)
This is the simplest and most reliable installation method.

1. Install the `conda` package manager (recommended: **miniforge**).
2. Create a dedicated environment to avoid dependency conflicts:
   ```bash
   conda create -n polsarpro
   conda activate polsarpro
   ```
3. Install the package from the `conda-forge` channel:
   ```bash
   conda install conda-forge::polsarpro
   ```

---

### Install with conda using a cloned repository
Choose this approach if you want access to the source code.

1. Clone the repository from GitHub and move into the project root.
2. Install `conda` (recommended: **miniforge**).
3. Create and activate the environment:
   ```bash
   conda env create -f environment.yaml
   conda activate psp
   ```
4. Add the toolbox to your `PYTHONPATH`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/mypath/to/polsarpro/source"
   ```
5. To verify the installation, run `pytest` from the main directory. All tests should pass.

---

## Development Environment (optional)

These instructions are intended for contributors or advanced users who want to work with the project's development tooling. They rely on a Docker configuration that mirrors the maintainer’s own setup and may require adjustments depending on your environment.

### VSCode Devcontainer
- Provides a ready-to-use environment for development and testing.
- Includes the C version of PolSARpro for running native decompositions.
- Requires placing the official PolSARpro ZIP archive in the project root before building the container.
- Requires Docker and Docker Compose.

Steps:
1. Adjust volume paths in `docker-compose.yml` to match your system.
2. Open the project in VSCode.
3. Use **Rebuild and Reopen in Container** (Ctrl+Shift+P / Cmd+Shift+P) to launch the devcontainer.

### Running the Docker container outside VSCode
- Intended only for users comfortable managing containers manually.
- Requires adjusting `docker-compose.yml` first.

Commands:
```bash
docker compose up -d
docker compose down
```

---

## Getting Started

Read the tutorial:  
https://polsarpro.readthedocs.io/en/latest/quickstart-tutorial/

Or open the tutorial notebook in the `notebooks/` directory.