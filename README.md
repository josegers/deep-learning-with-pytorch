# Deep Learning with PyTorch

A learning project exploring PyTorch fundamentals through galaxy classification.

## Files

| File | Description |
|------|-------------|
| `network.py` | Defines `GalaxyClassifier`: a 2-layer MLP for tabular input (3 features → 8 hidden → 2 classes) |
| `train.py` | Trains `GalaxyClassifier` on synthetic tabular galaxy data (offset Gaussian clusters) using CUDA |
| `cnn_network.py` | Defines `GalaxyCNN`: a CNN for 32×32 grayscale images (Conv2d + MaxPool → FC layers → 2 classes) |
| `cnn_train.py` | Trains `GalaxyCNN` on synthetic image data and saves weights to `galaxy_cnn.pt` |
| `cnn_predictor.py` | Loads `galaxy_cnn.pt` and runs inference, returning a class name and confidence score |
| `tensor_test.py` | Environment check: prints PyTorch version, detects compute device, and demos basic tensor ops |
| `galaxy_cnn.pt` | Saved weights from the most recent `cnn_train.py` run |
| `main.py` | Unused boilerplate |

## Setup

```bash
uv sync
uv run python cnn_train.py     # train CNN, saves galaxy_cnn.pt (requires CUDA)
uv run python cnn_predictor.py # run inference against saved weights
uv run python train.py         # train MLP (requires CUDA)
```
