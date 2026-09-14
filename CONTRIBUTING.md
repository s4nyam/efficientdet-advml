# Contributing

## Setup

```bash
git clone https://github.com/s4nyam/efficientdet-advml.git
cd efficientdet-advml
python -m venv .venv && source .venv/bin/activate
make install-dev          # installs everything + the pre-commit hooks
make test
```

## Before you open a pull request

```bash
make lint        # ruff check + format check
make test        # pytest
make typecheck   # mypy
```

Pre-commit runs the same checks plus a secret scan. Please do not skip it —
this repository has already shipped a credential once.

## Conventions

- **Tests are required** for new behaviour. Each test should check one thing
  and say in its name what that thing is.
- **Docstrings** in Google style with `Args:` and `Returns:`. Say why, not
  just what — a shape assertion is obvious from the code, the reason the shape
  has to hold is not.
- **Line length 100**, enforced by `ruff format`.
- Never commit dataset files, credentials or checkpoints over 5 MB.

## Reporting a discrepancy with the paper

These are welcome and useful. Please include:

1. Which table or figure.
2. What the repository produces instead, with the command you ran.
3. Your environment: Python, torch, OS.

Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) first — several known
discrepancies are already documented there.

## Adding a new experiment

1. Add a YAML file to `configs/`.
2. Put the training log in `results/<run_name>/results.csv` so
   `deepseanet report` finds it.
3. Note the split strategy, input size and epoch count in the config. Those
   three dominate any comparison, and leaving them implicit is how the
   confusion in §2 of the reproducibility notes happened in the first place.
