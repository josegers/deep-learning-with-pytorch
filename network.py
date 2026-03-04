import torch
import torch.nn as nn
import torch.nn.functional as F

class GalaxyClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        #First layer: 3 input features map to 8 internal neurons
        self.layer1 = nn.Linear(in_features=3, out_features=8)

        #The output layer: take 8 internal neurons and map to 2 outputs (Spiral or Elliptical)
        self.layer2 = nn.Linear(in_features=8, out_features=2)

    #The foreward method is automatically called when we pass data into the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Pass data through the first layer
        x = self.layer1(x)

        #Apply the activation function
        x = F.relu(x)

        #Pass through the final output layer
        x = self.layer2(x)

        return x

class LargeGalaxyClassifier(nn.Module):
    """Deeper, wider network to stress the GPU and show meaningful speedup."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    model = GalaxyClassifier()

    print("--- Network Architecture ---")
    # Printing the model shows you its internal structure
    print(model)

    print("\n--- Testing the Forward Pass ---")
    # Let's create a "dummy" galaxy with 3 random features
    # Shape is [1, 3] -> 1 row of data, 3 columns (features)
    dummy_galaxy = torch.tensor([[0.5, 1.2, 0.1]])

    # Pass the data through the network.
    # Notice we don't call model.forward(dummy_galaxy), we just call the object directly.
    raw_output = model(dummy_galaxy)

    print(f"Input: {dummy_galaxy}")
    print(f"Output: {raw_output}")


if __name__ == "__main__":
    main()

