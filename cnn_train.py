import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cnn_network import GalaxyCNN

cuda_device = torch.device("cuda")

def main():
    print("--- 1. PREPARING DATA ---")
    # Let's create 100 Spiral Galaxies (Label 0) and 100 Elliptical Galaxies (Label 1)

    # Spirals: Let's say they have higher values for feature 1 and 2
    #dummy_image = torch.randn(1, 1, 32, 32)

    spirals_X = torch.randn(100, 1, 32, 32)
    # 1. Draw a "Spiral" feature: A bright diagonal line
    # We iterate through the 32x32 grid and brighten the x=y coordinates
    for i in range(32):
        spirals_X[:, 0, i, i] += 2.0
    spirals_y = torch.zeros(100, dtype=torch.long) # label 0 = spiral

    # Ellipticals: Let's say they have lower values
    ellipticals_X = torch.randn(100, 1, 32, 32)
    # 2. Draw an "Elliptical" feature: A bright core in the exact center
    # We slice a 4x4 grid in the middle (pixels 14 to 18) and brighten them
    ellipticals_X[:, 0, 14:18, 14:18] += 2.0
    ellipticals_y = torch.ones(100, dtype=torch.long)  # Label 1 = elliptical

    # Combine them
    X = torch.cat((spirals_X, ellipticals_X), dim=0).to(cuda_device)
    y = torch.cat((spirals_y, ellipticals_y), dim=0).to(cuda_device)


    print(f"Dataset ready: {X.shape[0]} galaxies.")

    print("\n--- 2. INITIALIZING MODEL ---")
    model = GalaxyCNN().to(cuda_device)

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
    new_galaxy = torch.randn(1, 1, 32, 32).to(cuda_device)
    # for i in range(32):
    #     new_galaxy[:, 0, i, i] += 2.0

    raw_prediction = model(new_galaxy).to(cuda_device)
    probabilities = F.softmax(raw_prediction, dim=1)
    print(f"\nRaw prediction: {raw_prediction}, Probabilities: {probabilities}")

    # We use argmax to find the index of the highest value (0 for Spiral, 1 for Elliptical)
    predicted_class = torch.argmax(raw_prediction, dim=1).item()

    class_names = ["Spiral", "Elliptical"]
    print(f"Test Galaxy Prediction: {class_names[predicted_class]}")

    print("\n--- 5. SAVING THE MODEL ---")
    # .pt or .pth is the standard file extension for PyTorch models
    torch.save(model.state_dict(), "galaxy_cnn.pt")
    print("Model weights saved to 'galaxy_cnn.pt'")


if __name__ == "__main__":
    main()

