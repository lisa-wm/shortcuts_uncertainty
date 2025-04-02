# Trust Me, I Know the Way: Predictive Uncertainty in the Presence of Shortcut Learning

Code repository for eponymous publicaation at :sparkles: ICLR 2025 Workshop on Spurious Correlation and Shortcut Learning: Foundations and Solutions :sparkles:

## Setup

From within some virtual environment, package dependencies can be installed via `poetry install`.

Adapt the local path to the root directory in `root_path.py`.


## Structure

- `configs/` contains experiment configuration files (`.yml`) to specify global settings like batch size.
- `shortcuts/` contains the source code, in particular:
  - `config.py`, `utils.py`: helper classes and functions;
  - `data.py`, `datamodules.py`: MNIST3, CMNIST3, PMNIST3 datasets plus corresponding `PyTorch Lightning` data modules to facilitate training;
  - `models.py`, `ensemble.py`: model architectures and ensemble training;
  - `eval.py`: performance evaluation and estimation of uncertainty components.
  
## Usage

Run from CLI via `poetry run python main.py -expid="$EXPID" -rs="$RSEED"` 

:arrow_right: e.g., `$EXPID=mnist3`, `$RSEED=42`
