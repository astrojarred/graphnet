# AGENTS.md

## Cursor Cloud specific instructions

GraphNeT is a **Python library** (not a web app). There is no long-running dev server; development is done via scripts under `examples/` and tests under `tests/`.

### Python version

The package supports **Python 3.9–3.11** (`README.md`). On Ubuntu 24.04 the default `python3` is 3.12, so use **Python 3.11** (e.g. `ppa:deadsnakes/ppa` → `python3.11`, `python3.11-venv`) and a project venv at `.venv/`.

Activate the environment:

```bash
source /workspace/.venv/bin/activate
```

### Install / refresh dependencies

Match CI (`.github/actions/install/action.yml`): editable install with PyTorch Geometric CPU wheels and `jammy_flows` from GitHub. See the VM **update script** for the exact pip commands used on startup.

### Lint

Uses **pre-commit** (`.pre-commit-config.yaml`), same as `.github/workflows/code-quality.yml`:

```bash
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

### Tests (without IceTray)

Full `pytest tests/` expects **IceCube IceTray** (Docker image `icecube/icetray:icetray-devel-v1.13.0-ubuntu22.04-2025-02-12`; job `build-icetray` in `.github/workflows/build.yml`).

For a typical cloud VM **without** IceTray, run the same example suite as `build-matrix-examples`:

```bash
pytest tests/examples --ignore=tests/examples/01_icetray/
```

I3-specific tests under `tests/data/` and `tests/deployment/` require IceTray.

### Run / demo

Example scripts are invoked directly, e.g.:

```bash
python examples/02_data/01_read_dataset.py sqlite
```

Bundled fixtures live under `data/tests/` (SQLite, Parquet, etc.).

### Optional services

| Service | When needed |
|---------|-------------|
| IceTray / `icecube` | I3 readers, `tests/examples/01_icetray/`, deployment tests |
| wandb | Only if passing `--wandb` to training examples |
| ERDA / network downloads | Some dataset tests (`tests/datasets/test_erda_hosted_dataset.py`) |

### Gotchas

- Import warnings for missing `icecube` / `km3net` are expected without those optional stacks.
- `tests/data/test_dataconverters_and_datasets.py::test_dataconverter` needs IceTray and pre-converted I3 fixtures; do not treat its failure as a broken venv if IceTray is absent.
- PyTorch Geometric wheels must match the installed torch version; use the `-f https://data.pyg.org/whl/torch-<version>+cpu.html` find-links URL from CI.
