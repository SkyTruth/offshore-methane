# Offshore Methane Pilot

This repository hosts experimental code and notebooks for SkyTruth's offshore methane detection pilot. The goal is to evaluate satellite-based methods for identifying methane plumes near oil and gas infrastructure.

## Project Structure

- `src/` – Python modules and utilities.
- `notebooks/` – Exploratory Jupyter notebooks.
- `data/` – Input data or small examples.
- `docs/` – Additional documentation and references.
- `tests/` – Automated tests.

## Setup

1. Create a conda environment and install dependencies using
   [mamba](https://mamba.readthedocs.io/en/latest/):

   ```bash
   mamba env create -f environment.yml
   conda activate methane
   ```

2. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

3. Run tests with `pytest`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. All contributions should pass linting via `ruff` and unit tests before submission.

## License

This project is licensed under the [MIT License](LICENSE).

## References

Additional background and links can be found in [docs/references.md](docs/references.md).
