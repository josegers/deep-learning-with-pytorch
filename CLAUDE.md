# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package management (Python 3.14+, dependencies in `pyproject.toml`).

```bash
# Install dependencies
uv sync

# Run a script
uv run python train.py
uv run python tensor_test.py
uv run python network.py
```

## Architecture

This is an early-stage deep learning learning project using PyTorch.

- **`network.py`** — Defines `GalaxyClassifier(nn.Module)`: a 2-layer MLP (3 input features → 8 hidden neurons → 2 output classes). Classes: 0 = Spiral, 1 = Elliptical.
- **`train.py`** — Generates synthetic galaxy data (100 spirals + 100 ellipticals as offset Gaussian clusters), trains `GalaxyClassifier` using `CrossEntropyLoss` + Adam optimizer for 100 epochs, then runs a sample inference.
- **`tensor_test.py`** — Environment check: prints PyTorch version, detects compute device (CUDA → MPS → CPU), and demonstrates basic tensor operations.
- **`main.py`** — Unused PyCharm boilerplate.

The training data is fully synthetic (no external datasets). Device detection follows the pattern in `tensor_test.py` — note that `train.py` currently does not move tensors to GPU; that would be a natural next step.
