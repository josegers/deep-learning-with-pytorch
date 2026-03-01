import torch


def main():
    print("--- PyTorch Environment Check ---")
    print(f"PyTorch Version: {torch.__version__}")

    # 1. Hardware Check
    # 'cuda' is Nvidia's GPU compute engine.
    # 'mps' is Apple's Silicon (M1/M2/M3) compute engine.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Active Compute Device: {device.upper()}")

    # 2. The Tensor (PyTorch's core data structure)
    # Tensors are essentially multi-dimensional arrays, similar to C# multidimensional arrays
    # or NumPy arrays, but heavily optimized for AI math.
    my_tensor = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ], device=device)  # We explicitly tell PyTorch to load this data into the active device's memory

    print("\n--- The Tensor ---")
    print(my_tensor)
    print(f"Shape: {my_tensor.shape}")

    # 3. Vectorized Math
    print("\n--- Math Operation (Squared) ---")
    squared = my_tensor ** 2
    print(squared)


if __name__ == "__main__":
    main()