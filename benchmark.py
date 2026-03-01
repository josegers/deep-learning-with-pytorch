import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network import GalaxyClassifier

# --- Config ---
N_SAMPLES = 10_000   # galaxies per class
EPOCHS = 500
LR = 0.01


def make_dataset(device: torch.device):
    spirals_X = torch.randn(N_SAMPLES, 3) + torch.tensor([2.0, 2.0, 0.0])
    spirals_y = torch.zeros(N_SAMPLES, dtype=torch.long)

    ellipticals_X = torch.randn(N_SAMPLES, 3) - torch.tensor([2.0, 2.0, 0.0])
    ellipticals_y = torch.ones(N_SAMPLES, dtype=torch.long)

    X = torch.cat((spirals_X, ellipticals_X), dim=0).to(device)
    y = torch.cat((spirals_y, ellipticals_y), dim=0).to(device)
    return X, y


def train(device: torch.device) -> dict:
    X, y = make_dataset(device)

    model = GalaxyClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Warm-up pass so device initialisation doesn't skew timing
    with torch.no_grad():
        _ = model(X[:1])
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Accuracy on the training set
    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1)
        accuracy = (preds == y).float().mean().item()

    return {
        "device": str(device),
        "elapsed_s": elapsed,
        "final_loss": loss.item(),
        "accuracy": accuracy,
    }


def print_results(results: list[dict]) -> None:
    col = 14
    header = f"{'Device':<{col}} {'Time (s)':>10} {'Loss':>10} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['device']:<{col}} "
            f"{r['elapsed_s']:>10.3f} "
            f"{r['final_loss']:>10.4f} "
            f"{r['accuracy']:>9.1%}"
        )

    if len(results) == 2:
        a, b = results[0], results[1]
        speedup = a["elapsed_s"] / b["elapsed_s"]
        faster = b["device"] if speedup > 1 else a["device"]
        ratio = max(speedup, 1 / speedup)
        print(f"\n{faster} is {ratio:.2f}x faster")


def main():
    print(f"Benchmark — {N_SAMPLES * 2:,} samples, {EPOCHS} epochs\n")

    devices_to_test: list[torch.device] = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices_to_test.append(torch.device("mps"))
    else:
        print("No GPU detected — running CPU only.\n")

    results = []
    for device in devices_to_test:
        print(f"Running on {device}...", flush=True)
        results.append(train(device))

    print()
    print_results(results)


if __name__ == "__main__":
    main()
