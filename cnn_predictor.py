import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Import the architecture
from cnn_network import GalaxyCNN

def predict_galaxy(model: nn.Module, image_tensor: torch.Tensor) -> tuple[str, float]:
    """
    Simulates a production API endpoint.
    Takes a 4D image tensor, runs inference, and returns a human-readable result.
    """
    class_names = ["Spiral", "Elliptical"]

    # Golden Rule 1: Set to evaluation mode
    model.eval()

    # Golden Rule 2: Turn off the gradient engine
    with torch.no_grad():
        # Make the prediction
        logits = model(image_tensor)

        # Convert raw logits to percentages (0.0 to 1.0)
        probabilities = F.softmax(logits, dim=1)

        # Find the winning class index (0 or 1)
        winning_index = torch.argmax(probabilities, dim=1).item()

        # Extract the confidence score for that specific class
        confidence = probabilities[0][winning_index].item()

    return class_names[winning_index], confidence


def main():
    print("--- STARTING TELESCOPE INFERENCE SERVICE ---")

    # 2. INITIALIZE THE EMPTY SHELL
    # C# Analogy: var myBrain = new GalaxyCNN();
    model = GalaxyCNN()

    # 3. LOAD THE BRAIN
    # C# Analogy: myBrain.LoadState(File.ReadAllBytes("galaxy_cnn.pt"));
    try:
        # weights_only=True is a security best practice when loading pickled data
        model.load_state_dict(torch.load("galaxy_cnn.pt", weights_only=True))
        print("Model weights loaded successfully. Ready for requests.")
    except FileNotFoundError:
        print("CRITICAL: 'galaxy_cnn.pt' not found. Run the training script first.")
        return

    print("\n--- RECEIVING NEW TELEMETRY ---")
    # 4. CREATE A TEST IMAGE
    # Let's explicitly draw a diagonal line so it should confidently predict "Spiral"
    incoming_image = torch.randn(1, 1, 32, 32)
    for i in range(32):
        incoming_image[:, 0, i, i] += 2.0

    print("Processing 32x32 image...")

    # 5. RUN INFERENCE
    prediction, confidence = predict_galaxy(model, incoming_image)

    print(f"\nResult: {prediction.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()