# Project Overview

A learning project exploring PyTorch fundamentals through galaxy classification. It implements two primary neural network architectures:
- A Multi-Layer Perceptron (MLP) for tabular data (`GalaxyClassifier` and `LargeGalaxyClassifier` in `network.py`).
- A Convolutional Neural Network (CNN) for grayscale images (`GalaxyCNN` in `cnn_network.py`).

The project is structured around synthetic data generation, training, and inference, managed with `uv` and specifically configured to utilize CUDA for GPU acceleration.

# Building and Running

The project relies on the `uv` package manager. Ensure `uv` is installed to manage dependencies and run the scripts.

**Dependencies:**
- `uv sync`: Sync and install all required packages (PyTorch and Torchvision).

**Execution Commands:**
- `uv run python cnn_train.py`: Trains the `GalaxyCNN` on synthetic image data and saves the trained weights to `galaxy_cnn.pt`. Requires CUDA.
- `uv run python cnn_predictor.py`: Loads the saved model weights (`galaxy_cnn.pt`) and performs inference, outputting class names and confidence scores.
- `uv run python train.py`: Trains the MLP models on synthetic tabular galaxy data. Requires CUDA.
- `uv run python tensor_test.py`: Validates the environment setup by printing the PyTorch version, detecting the compute device, and demonstrating basic tensor operations.

# Development Conventions

- **Educational Focus:** Code is heavily commented and structured to act as a learning resource. It explains foundational deep learning concepts, such as tensor shapes, layer definitions, forward passes, and even includes analogies to concepts in other languages (e.g., C# arrays).
- **Architecture:** All neural network architectures inherit from `torch.nn.Module`. 
- **Type Hinting:** Method signatures consistently use type hints, particularly for PyTorch tensors (e.g., `x: torch.Tensor`).
- **Entry Points:** Most `.py` files contain a `main()` block that can be executed directly to demonstrate the module's functionality, often passing "dummy" tensors through the network to verify shapes and outputs.
- **Hardware Acceleration:** The project explicitly targets CUDA 12.8 (as defined in `pyproject.toml`) and is designed to leverage GPU compute for training scripts.