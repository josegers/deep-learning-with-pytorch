# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package management (Python 3.14+, dependencies in `pyproject.toml`).

```bash
# Install dependencies
uv sync

# Run a script
uv run python train.py          # MLP training (CUDA required)
uv run python cnn_train.py      # CNN training, saves galaxy_cnn.pt (CUDA required)
uv run python cnn_predictor.py  # CNN inference, requires galaxy_cnn.pt
uv run python tensor_test.py
uv run python network.py
```

## Architecture

This is an early-stage deep learning learning project using PyTorch. There are two parallel model pipelines, both classifying galaxies as Spiral (0) or Elliptical (1) on fully synthetic data.

### MLP Pipeline (tabular features)
- **`network.py`** — `GalaxyClassifier(nn.Module)`: 2-layer MLP (3 input features → 8 hidden → 2 output classes).
- **`train.py`** — Generates 100 spirals + 100 ellipticals as offset Gaussian clusters (3 features), trains `GalaxyClassifier` with `CrossEntropyLoss` + Adam for 100 epochs on CUDA.

### CNN Pipeline (image-based)
- **`cnn_network.py`** — `GalaxyCNN(nn.Module)`: Conv2d(1→16, 3×3) + MaxPool2d → flatten → Linear(4096→64) → Linear(64→2). Expects 32×32 grayscale images `[B, 1, 32, 32]`.
- **`cnn_train.py`** — Generates 100 synthetic 32×32 images per class (spirals: bright diagonal; ellipticals: bright 4×4 center), trains `GalaxyCNN` on CUDA, saves weights to `galaxy_cnn.pt`.
- **`cnn_predictor.py`** — Loads `galaxy_cnn.pt` and exposes `predict_galaxy(model, image_tensor) -> (class_name, confidence)` using `model.eval()` + `torch.no_grad()`.

### Other
- **`tensor_test.py`** — Environment check: PyTorch version, device detection (CUDA → MPS → CPU), basic tensor ops.
- **`main.py`** — Unused PyCharm boilerplate.

Both training scripts hardcode `torch.device("cuda")` — a CUDA GPU is required to run them.
