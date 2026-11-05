"""Reproduce the MLP training diagnostics from lab1.docx on CPU."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dlnn_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CONFIGS = {
    "A": (6, "Sigmoid", "xavier", "Adam", .003),
    "B": (6, "Tanh", "xavier", "Adam", .003),
    "C": (6, "ReLU", "xavier", "Adam", .003),
    "D": (6, "ReLU", "he", "Adam", .003),
    "E": (6, "ReLU", "he", "SGD", .05),
    "F": (1, "ReLU", "he", "Adam", .003),
}
SEEDS = (11, 29)
EPOCHS = 100
BATCH_SIZE = 128


class MLP(nn.Module):
    def __init__(self, depth, activation, initialization):
        super().__init__()
        self.hidden = nn.ModuleList()
        input_width = 2
        for _ in range(depth):
            layer = nn.Linear(input_width, 32)
            if initialization == "he":
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            else:
                nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            input_width = 32
        self.output = nn.Linear(32, 2)
        nn.init.xavier_normal_(self.output.weight, gain=1.0)
        nn.init.zeros_(self.output.bias)
        self.activation = {"Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "ReLU": nn.ReLU}[activation]()

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.output(x)


def prepare_data():
    x, y = make_moons(n_samples=2400, noise=.25, random_state=2026)
    x_rest, x_test, y_rest, y_test = train_test_split(
        x, y, test_size=.2, random_state=2026, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=.25, random_state=2026, stratify=y_rest)
    scaler = StandardScaler().fit(x_train)
    scaled = [scaler.transform(a).astype(np.float32) for a in (x_train, x_val, x_test)]
    labels = [torch.as_tensor(a, dtype=torch.long) for a in (y_train, y_val, y_test)]
    tensors = [(torch.as_tensor(a), b) for a, b in zip(scaled, labels)]
    info = {name: {"size": len(label), "class_counts": np.bincount(label.numpy(), minlength=2).tolist()}
            for name, label in zip(("train", "validation", "test"), labels)}
    info["train_scaled_mean"] = scaled[0].mean(axis=0).tolist()
    info["train_scaled_std_population"] = scaled[0].std(axis=0).tolist()
    assert [len(t[0]) for t in tensors] == [1440, 480, 480]
    assert all(min(v["class_counts"]) > 0 for v in (info[k] for k in ("train", "validation", "test")))
    assert np.allclose(scaled[0].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(scaled[0].std(axis=0), 1, atol=1e-6)
    plt.figure(figsize=(6, 4))
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=9, cmap="coolwarm", alpha=.65)
    plt.xlabel("Feature 1 (original)"); plt.ylabel("Feature 2 (original)")
    plt.title("Training data: make_moons")
    plt.tight_layout(); plt.savefig(FIGURES / "train_scatter.png", dpi=160); plt.close()
    return tensors, info


def evaluate(model, data, criterion):
    model.eval()
    x, y = data
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    assert np.isfinite(loss)
    return loss, accuracy


def gradient_norms(model, control, criterion, config, seed, epoch):
    model.eval()
    model.zero_grad(set_to_none=True)
    x, y = control
    loss = criterion(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    rows = []
    for index, layer in enumerate(model.hidden, 1):
        grad = layer.weight.grad
        assert grad is not None and grad.shape == layer.weight.shape and torch.isfinite(grad).all()
        rows.append({"config": config, "seed": seed, "epoch": epoch,
                     "layer": index, "gradient_norm": torch.linalg.matrix_norm(grad).item()})
    model.zero_grad(set_to_none=True)
    return rows


def check_shapes(data):
    x, y = data
    criterion = nn.CrossEntropyLoss(reduction="mean")
    checks = {}
    for depth in (1, 6):
        torch.manual_seed(11)
        model = MLP(depth, "ReLU", "he")
        logits = model(x[:8])
        loss = criterion(logits, y[:8])
        assert x[:8].shape == (8, 2) and logits.shape == (8, 2) and loss.ndim == 0
        expected = (2 * 32 + 32) + (depth - 1) * (32 * 32 + 32) + (32 * 2 + 2)
        assert sum(p.numel() for p in model.parameters()) == expected
        loss.backward()
        for layer in model.hidden:
            assert layer.weight.grad is not None
            assert layer.weight.grad.shape == layer.weight.shape
            assert torch.isfinite(layer.weight.grad).all()
        checks[str(depth)] = {"parameters": expected, "input_shape": [8, 2],
                              "output_shape": [8, 2], "loss_shape": []}
    return checks


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_results(metrics, gradients, selected):
    for field, label in (("val_loss", "Validation loss"), ("val_accuracy", "Validation accuracy")):
        plt.figure(figsize=(8, 5))
        for config in CONFIGS:
            rows = [r for r in metrics if r["config"] == config and r["seed"] == 11]
            plt.plot([r["epoch"] for r in rows], [r[field] for r in rows], label=config)
        plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(ncol=3); plt.grid(alpha=.25)
        plt.tight_layout(); plt.savefig(FIGURES / f"{field}_seed11.png", dpi=160); plt.close()
    for config in ("A", "C", "D"):
        plt.figure(figsize=(7, 4.5))
        for epoch in (0, 20, 100):
            rows = [r for r in gradients if r["config"] == config and r["seed"] == 11 and r["epoch"] == epoch]
            positive = [(r["layer"], r["gradient_norm"]) for r in rows if r["gradient_norm"] > 0]
            if positive:
                plt.plot([p[0] for p in positive], [p[1] for p in positive], marker="o", label=f"Epoch {epoch}")
        plt.yscale("log"); plt.xticks(range(1, 7)); plt.xlabel("Hidden layer")
        plt.ylabel("Weight gradient Frobenius norm"); plt.title(f"Config {config}, seed 11")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(FIGURES / f"gradients_{config}_seed11.png", dpi=160); plt.close()
    config, seed = selected
    rows = [r for r in metrics if r["config"] == config and r["seed"] == seed]
    plt.figure(figsize=(8, 5))
    plt.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label="Train")
    plt.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.title(f"Selected run: {config}, seed {seed}")
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIGURES / "selected_losses.png", dpi=160); plt.close()


def main():
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    data, info = prepare_data()
    checks = check_shapes(data[0])
    criterion = nn.CrossEntropyLoss(reduction="mean")
    control = (data[0][0][:128], data[0][1][:128])
    metrics, gradients, models = [], [], {}
    for config, (depth, activation, initialization, optimizer_name, lr) in CONFIGS.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = MLP(depth, activation, initialization)
            optimizer = (torch.optim.Adam(model.parameters(), lr=lr) if optimizer_name == "Adam"
                         else torch.optim.SGD(model.parameters(), lr=lr, momentum=0))
            loader = DataLoader(TensorDataset(*data[0]), batch_size=BATCH_SIZE, shuffle=True,
                                generator=torch.Generator().manual_seed(10000 + seed))
            def record(epoch):
                train_loss, train_accuracy = evaluate(model, data[0], criterion)
                val_loss, val_accuracy = evaluate(model, data[1], criterion)
                metrics.append({"config": config, "seed": seed, "epoch": epoch,
                                "train_loss": train_loss, "train_accuracy": train_accuracy,
                                "val_loss": val_loss, "val_accuracy": val_accuracy})
            record(0)
            if config in ("A", "C", "D"):
                gradients.extend(gradient_norms(model, control, criterion, config, seed, 0))
            for epoch in range(1, EPOCHS + 1):
                model.train()
                for x, y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(x), y)
                    assert torch.isfinite(loss)
                    loss.backward(); optimizer.step()
                record(epoch)
                if config in ("A", "C", "D") and epoch in (20, 100):
                    gradients.extend(gradient_norms(model, control, criterion, config, seed, epoch))
            models[(config, seed)] = model
            print(f"{config}/{seed}: validation loss {metrics[-1]['val_loss']:.5f}", flush=True)
    final = [r for r in metrics if r["epoch"] == 100]
    means = {c: float(np.mean([r["val_loss"] for r in final if r["config"] == c])) for c in CONFIGS}
    best_config = min(CONFIGS, key=lambda c: (means[c], c))
    best_seed = min(SEEDS, key=lambda s: (next(r["val_loss"] for r in final if r["config"] == best_config and r["seed"] == s), s))
    test_loss, test_accuracy = evaluate(models[(best_config, best_seed)], data[2], criterion)
    summary = {"device": "cpu", "data": info, "shape_checks": checks,
               "mean_val_loss": means, "selected_config": best_config, "selected_seed": best_seed,
               "test_loss": test_loss, "test_accuracy": test_accuracy,
               "final_metrics": final}
    write_csv(RESULTS / "metrics.csv", metrics,
              ["config", "seed", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
    write_csv(RESULTS / "gradients.csv", gradients,
              ["config", "seed", "epoch", "layer", "gradient_norm"])
    write_csv(RESULTS / "final_comparison.csv", final,
              ["config", "seed", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(metrics, gradients, (best_config, best_seed))
    print(f"Selected {best_config}/{best_seed}; test loss {test_loss:.5f}, accuracy {test_accuracy:.5f}")


if __name__ == "__main__":
    main()
