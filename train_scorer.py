"""
Phase 2.2: Train MLP banger scorer on cached MERT embeddings.
Input: 768-dim MERT embeddings
Output: banger score 0-10
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import argparse
import time


class BangerDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, scores: np.ndarray):
        self.embeddings = torch.FloatTensor(embeddings)
        self.scores = torch.FloatTensor(scores)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.scores[idx]


class BangerScorer(nn.Module):
    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data(embeddings_dir: str):
    """Load cached MERT embeddings and labels."""
    embeddings = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    with open(os.path.join(embeddings_dir, "metadata.json")) as f:
        metadata = json.load(f)

    scores = np.array([label["banger_score"] for label in metadata["labels"]])
    genres = [label["genre"] for label in metadata["labels"]]

    print(f"Loaded {len(scores)} tracks with {embeddings.shape[1]}-dim embeddings")
    print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"Score mean: {scores.mean():.2f}, std: {scores.std():.2f}")

    return embeddings, scores, genres, metadata


def train_model(
    embeddings: np.ndarray,
    scores: np.ndarray,
    device: str = "mps",
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 20,
):
    """Train the banger scorer MLP."""

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, scores, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\nSplit sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    train_dataset = BangerDataset(X_train, y_train)
    val_dataset = BangerDataset(X_val, y_val)
    test_dataset = BangerDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = BangerScorer(input_dim=embeddings.shape[1], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Model size: {param_count * 4 / 1024 / 1024:.1f} MB (float32)\n")

    # Training loop
    best_val_mae = float("inf")
    best_epoch = 0
    train_losses = []
    val_maes = []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(y_batch.numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_mae = np.mean(np.abs(val_preds - val_true))
        val_maes.append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), "scorer_model.pt")

        if (epoch + 1) % 20 == 0:
            corr, _ = spearmanr(val_preds, val_true)
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_losses[-1]:.4f} | "
                  f"Val MAE: {val_mae:.3f} | Val Corr: {corr:.3f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
        if epoch - best_epoch > patience:
            print(f"\nEarly stopping at epoch {epoch+1} (best was {best_epoch+1})")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("scorer_model.pt", weights_only=True))
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            test_preds.extend(pred.cpu().numpy())
            test_true.extend(y_batch.numpy())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    test_mae = np.mean(np.abs(test_preds - test_true))
    test_corr, _ = spearmanr(test_preds, test_true)

    print(f"\n{'='*50}")
    print(f"RESULTS (best model from epoch {best_epoch+1})")
    print(f"{'='*50}")
    print(f"Test MAE:  {test_mae:.3f}")
    print(f"Test Corr: {test_corr:.3f}")
    print(f"Val MAE:   {best_val_mae:.3f}")
    print(f"{'='*50}")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.scatter(test_true, test_preds, alpha=0.5, s=10)
    ax2.plot([0, 10], [0, 10], "r--", label="Perfect prediction")
    ax2.set_xlabel("Actual Banger Score")
    ax2.set_ylabel("Predicted Banger Score")
    ax2.set_title(f"Test Set: MAE={test_mae:.3f}, Corr={test_corr:.3f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    print("Saved training_results.png")

    return model, test_mae, test_corr


def main():
    parser = argparse.ArgumentParser(description="Train banger scorer MLP")
    parser.add_argument("--embeddings-dir", default="data/embeddings",
                        help="Path to cached MERT embeddings")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, mps, cuda, or cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load data
    embeddings, scores, genres, metadata = load_data(args.embeddings_dir)

    # Train
    model, test_mae, test_corr = train_model(
        embeddings, scores,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
