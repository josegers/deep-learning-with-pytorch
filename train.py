import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network import GalaxyClassifier

def main():
    print("--- 1. PREPARING DATA ---")
    # Let's create 100 Spiral Galaxies (Label 0) and 100 Elliptical Galaxies (Label 1)

    # Spirals: Let's say they have higher values for feature 1 and 2
    spirals_X = torch.randn(100, 3) + torch.tensor([2.0, 2.0, 0.0])
    spirals_y = torch.zeros(100, dtype=torch.long) # label 0 = spiral

    # Ellipticals: Let's say they have lower values
    ellipticals_x = torch.randn(100, 3) - torch.tensor([2.0, 2.0, 0.0])
    ellipticals_y = torch.ones(100, dtype=torch.long)  # Label 1 = elliptical

    # Combine them
    X = torch.cat((spirals_X, ellipticals_x), dim=0)
    y = torch.cat((spirals_y, ellipticals_y), dim=0)

    print(f"Dataset ready: {X.shape[0]} galaxies.")

    print("\n--- 2. INITIALIZING MODEL ---")
    model = GalaxyClassifier()

    # The Loss Function (How wrong are we?)
    criterion = nn.CrossEntropyLoss()

    # The Optimizer (The updater). lr is "Learning Rate" (how big of a step to take)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\n--- 3. STARTING TRAINING LOOP ---")
    epochs = 100  # How many times we show the full dataset to the model

    for epoch in range(epochs):
        # Step 1: Clear old gradients
        # C# Analogy: Resetting a state variable at the top of a loop
        optimizer.zero_grad()

        # Step 2: Forward Pass (Make predictions)
        outputs = model(X)

        # Step 3: Calculate Loss
        loss = criterion(outputs, y)

        # Step 4: Backward Pass (Calculate the gradients)
        loss.backward()

        # Step 5: Optimizer Step (Update the weights)
        optimizer.step()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {loss.item():.4f}")

    print("\n--- 4. TRAINING COMPLETE ---")

    # Let's test it on a brand new "Spiral" galaxy
    new_galaxy = torch.tensor([[2.5, 2.1, 0.5]])
    raw_prediction = model(new_galaxy)
    print(f"\nRaw prediction: {raw_prediction}")

    # Convert logits to probabilities
    probabilities = F.softmax(raw_prediction, dim=1)

    # We use argmax to find the index of the highest value (0 for Spiral, 1 for Elliptical)
    predicted_class = torch.argmax(raw_prediction, dim=1).item()

    class_names = ["Spiral", "Elliptical"]
    print(f"Test Galaxy Prediction: {class_names[predicted_class]}")

if __name__ == "__main__":
    main()

