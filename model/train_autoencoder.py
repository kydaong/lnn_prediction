"""Train an LNN (liquid time-constant) autoencoder on normal-operation
compressor windows, then calibrate a reconstruction-error anomaly threshold
from the held-out validation split.

Usage:
    .venv/Scripts/python.exe model/train_autoencoder.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ncps.torch import CfC
import mlflow

FEATURES_DIR = Path("data/features")
OUT_DIR = Path("model/artifacts")


class LNNAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, latent_dim: int, decoder_units: int):
        super().__init__()
        self.encoder = CfC(input_size=n_sensors, units=latent_dim, return_sequences=False, batch_first=True)
        self.decoder = CfC(
            input_size=latent_dim, units=decoder_units, proj_size=n_sensors, return_sequences=True, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = x.shape[1]
        latent, _ = self.encoder(x)
        latent_seq = latent.unsqueeze(1).expand(-1, window, -1)
        recon, _ = self.decoder(latent_seq)
        return recon


def load_windows(name: str) -> np.ndarray:
    return np.load(FEATURES_DIR / f"{name}_windows.npz")["X"]


def per_window_error(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((recon - x) ** 2).mean(dim=(1, 2))


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = per_window_error(recon, batch).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
        n += len(batch)
    return total_loss / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    errors = []
    for (batch,) in loader:
        batch = batch.to(device)
        recon = model(batch)
        window_errors = per_window_error(recon, batch)
        errors.append(window_errors.cpu().numpy())
        total_loss += window_errors.sum().item()
        n += len(batch)
    return total_loss / n, np.concatenate(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--decoder-units", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience", type=int, default=10, help="Early-stop after this many epochs with no val improvement"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile of val reconstruction error used as the anomaly threshold",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(FEATURES_DIR / "feature_columns.json") as f:
        feature_meta = json.load(f)
    n_sensors = len(feature_meta["sensor_tags"])

    train_X = load_windows("train")
    val_X = load_windows("val")
    print(f"train windows: {train_X.shape}, val windows: {val_X.shape}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_X)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_X)), batch_size=args.batch_size, shuffle=False)

    model = LNNAutoencoder(n_sensors, args.latent_dim, args.decoder_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment("lnn_autoencoder")
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, _ = evaluate(model, val_loader, device)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"epoch {epoch:03d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                    break

        model.load_state_dict(best_state)
        _, val_errors = evaluate(model, val_loader, device)
        threshold = float(np.percentile(val_errors, args.threshold_percentile))
        print(f"Calibrated anomaly threshold ({args.threshold_percentile}th pct of val error): {threshold:.5f}")

        config = {
            "n_sensors": n_sensors,
            "latent_dim": args.latent_dim,
            "decoder_units": args.decoder_units,
            "window_minutes": feature_meta["window_minutes"],
            "sensor_tags": feature_meta["sensor_tags"],
            "threshold": threshold,
            "threshold_percentile": args.threshold_percentile,
            "best_val_loss": best_val_loss,
        }
        torch.save({"model_state": best_state, "config": config}, OUT_DIR / "autoencoder.pt")
        with open(OUT_DIR / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("anomaly_threshold", threshold)
        mlflow.log_artifact(str(OUT_DIR / "autoencoder.pt"))
        mlflow.log_artifact(str(OUT_DIR / "config.json"))

    print(f"Saved model, config, and anomaly threshold to {OUT_DIR}/")


if __name__ == "__main__":
    main()
