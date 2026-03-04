import time
import torch
import torch.nn as nn
import torch.optim as optim

from network import GalaxyClassifier, LargeGalaxyClassifier

LR = 0.01

CONFIGS = [
    {
        "label": "Small network  (3 -> 8 -> 2)",
        "model_cls": GalaxyClassifier,
        "n_samples": 10_000,
        "epochs": 500,
    },
    {
        "label": "Large network  (3 -> 128 -> 512 -> 1024 -> 512 -> 128 -> 2)",
        "model_cls": LargeGalaxyClassifier,
        "n_samples": 50_000,
        "epochs": 200,
    },
]


def make_dataset(n_samples: int, device: torch.device):
    spirals_X = torch.randn(n_samples, 3) + torch.tensor([2.0, 2.0, 0.0])
    spirals_y = torch.zeros(n_samples, dtype=torch.long)

    ellipticals_X = torch.randn(n_samples, 3) - torch.tensor([2.0, 2.0, 0.0])
    ellipticals_y = torch.ones(n_samples, dtype=torch.long)

    X = torch.cat((spirals_X, ellipticals_X), dim=0).to(device)
    y = torch.cat((spirals_y, ellipticals_y), dim=0).to(device)
    return X, y


def train(model_cls, n_samples: int, epochs: int, device: torch.device) -> dict:
    X, y = make_dataset(n_samples, device)

    model = model_cls().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Warm-up pass so device initialisation doesn't skew timing
    with torch.no_grad():
        _ = model(X[:1])
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1)
        accuracy = (preds == y).float().mean().item()

    return {
        "device": str(device),
        "elapsed_s": elapsed,
        "final_loss": loss.item(),
        "accuracy": accuracy,
    }


def print_section(label: str, n_samples: int, epochs: int, results: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {n_samples * 2:,} samples  |  {epochs} epochs")
    print(f"{'=' * 60}")

    col = 14
    header = f"  {'Device':<{col}} {'Time (s)':>10} {'Loss':>10} {'Accuracy':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['device']:<{col}} "
            f"{r['elapsed_s']:>10.3f} "
            f"{r['final_loss']:>10.4f} "
            f"{r['accuracy']:>9.1%}"
        )

    if len(results) == 2:
        a, b = results[0], results[1]
        speedup = a["elapsed_s"] / b["elapsed_s"]
        faster = b["device"] if speedup > 1 else a["device"]
        ratio = max(speedup, 1 / speedup)
        print(f"\n  >> {faster} is {ratio:.2f}x faster")


def main():
    devices: list[torch.device] = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    else:
        print("No GPU detected — running CPU only.")

    for cfg in CONFIGS:
        results = []
        for device in devices:
            print(f"Running {cfg['label']} on {device}...", flush=True)
            results.append(
                train(cfg["model_cls"], cfg["n_samples"], cfg["epochs"], device)
            )
        print_section(cfg["label"], cfg["n_samples"], cfg["epochs"], results)


if __name__ == "__main__":
    main()
