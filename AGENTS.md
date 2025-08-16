# Repository Guidelines

## Project Structure & Module Organization
- `offshore_methane/`: Python package (algorithms, masking, EE/GCS helpers, orchestrator).
- `tests/`: Pytest suite (files named `test_*.py`).
- `notebooks/`: Jupyter examples for exploration and diagnostics.
- `data/`: Small reference inputs (e.g., `sites.csv`).
- `docs/`: Additional documentation.

## Build, Test, and Development Commands
- Create env: `mamba env create -f environment.yml && conda activate methane`
- Editable install: `pip install -e .`
- Lint (auto-fix): `ruff . --fix`
- Run pre-commit locally: `pre-commit run -a`
- Tests: `pytest -q`
- Orchestrator (main pipeline): `python -m offshore_methane.orchestrator`

## Coding Style & Naming Conventions
- Python 3.10+. Line length 88 (ruff). Use 4-space indentation.
- Modules/functions: `snake_case`; classes: `CamelCase`; constants: `UPPER_SNAKE`.
- Prefer type hints and concise docstrings for public functions.
- Keep modules cohesive: EE helpers in `ee_utils.py`, masking in `masking.py`, pipeline in `orchestrator.py`.

## Testing Guidelines
- Place unit tests under `tests/` with names like `test_mbsp.py`.
- Use `pytest` assertions; keep tests fast and deterministic.
- Mock external services (Earth Engine, GCS, network). Avoid live API calls.
- Add tests for new logic and edge cases (e.g., uniform scenes, parameter extremes).

## Commit & Pull Request Guidelines
- Commits: short imperative summary (e.g., "add sgi model filter"). Group related changes.
- Reference issues/PRs where relevant (e.g., `#30`, `Fixes #123`).
- PRs must include: clear description, rationale, test updates, and any config changes (`config.py`). Include screenshots/links for notebook visuals when helpful.
- CI expectations: pass `ruff`, `pre-commit`, and `pytest` before requesting review.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` (loaded via `python-dotenv`) and local credential stores for EE/GCS.
- Review `offshore_methane/config.py` before runs: update site lists, date ranges, export backends, and thresholds.
- Large outputs should target buckets or EE assets per `EXPORT_PARAMS`; avoid bloating the repo.

## Architecture Overview
- Workflow: build masks (`masking.py`) → compute MBSP (`mbsp.py`) → polygonize/export (`algos.py`, `ee_utils.py`) → orchestrate in parallel (`orchestrator.py`).
- Start exploration with `notebooks/` and `ee_utils.quick_view()`; promote stable logic into the package with tests.

## Notebook-Style Execution
- Cell markers: files containing `# %%` are executed as virtual notebook cells.
- `main()` behavior: invoked without CLI parsing in this mode; do not use `argparse` or rely on `sys.argv` inside `main()` for these files.
- Keep `main()` param-driven (e.g., `main(eid=None, sid=None, stats=False)`), don't provide a separate `cli()` wrapper.
- Goal: ensure the same module runs interactively in cells.
