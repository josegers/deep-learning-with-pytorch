import torch
import torch.nn as nn
import torch.nn.functional as F

class GalaxyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # --- FEATURE EXTRACTION ---
        # Conv2d looks for 2D patterns.
        # in_channels=1 (Grayscale image), out_channels=16 (It learns 16 different feature filters)
        # kernel_size=3 means it uses a 3x3 sliding window.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # MaxPool2d shrinks the image by half. It takes the "brightest" pixel in a 2x2 grid.
        # This keeps the most important features but drastically reduces memory usage.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- CLASSIFICATION HEAD ---
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Slide the 3x3 window, apply ReLU, then shrink the image
        x = self.pool(F.relu(self.conv1(x)))

        # 2. Flatten the 2D matrix into a 1D array
        # C# Analogy: This is like casting a int[,] to a single int[]
        # The '-1' tells PyTorch to automatically figure out the batch size.
        x = x.view(-1, 16 * 16 * 16)

        # 3. Standard classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    model = GalaxyCNN()
    print("--- Vision Network Architecture ---")
    print(model)

    print("\n--- Testing the Image Pipeline ---")
    # Tensors for images always have 4 dimensions: [Batch_Size, Channels, Height, Width]
    # Let's create 1 fake grayscale image that is 32x32 pixels.
    # We fill it with random noise (like TV static).
    dummy_image = torch.randn(1, 1, 32, 32)

    print(f"Input Shape: {dummy_image.shape}")

    # Pass the image through the CNN
    output = model(dummy_image)

    print(f"Output Shape: {output.shape}")
    print(f"Raw Logits: {output}")


if __name__ == "__main__":
    main()