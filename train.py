"""
Autoresearch: Trainable file (AGENT-MODIFIABLE).
This file defines the model architecture and training loop.
The agent can modify this file to experiment with different approaches.

Current approach: MLP with BatchNorm and Dropout
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from prepare import load_splits

# ============================================================
# CONFIG — Agent can modify these
# ============================================================
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 30
TIME_BUDGET = 300  # 5 minutes max training time
# ============================================================


def build_model(input_dim: int) -> nn.Module:
    """Build the scorer model. Agent can modify architecture."""
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(DROPOUT),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(DROPOUT),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(DROPOUT / 2),

        nn.Linear(128, 1),
    )


def train():
    """Main training loop."""
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load data
    data = load_splits()
    input_dim = data["embedding_dim"]

    # Create dataloaders
    train_ds = TensorDataset(
        torch.FloatTensor(data["X_train"]),
        torch.FloatTensor(data["y_train"]),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(data["X_val"]),
        torch.FloatTensor(data["y_val"]),
    )
    test_ds = TensorDataset(
        torch.FloatTensor(data["X_test"]),
        torch.FloatTensor(data["y_test"]),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = build_model(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())

    # Training
    best_val_mae = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Time budget check
        if time.time() - start_time > TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

        # Train
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze(-1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                pred = model(X).squeeze(-1)
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(y.numpy())

        val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_true)))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), "scorer_model_best.pt")

        if epoch - best_epoch > PATIENCE:
            break

    # Evaluate on test set
    model.load_state_dict(torch.load("scorer_model_best.pt", weights_only=True))
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X).squeeze(-1)
            test_preds.extend(pred.cpu().numpy())
            test_true.extend(y.numpy())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    test_mae = np.mean(np.abs(test_preds - test_true))
    test_corr, _ = spearmanr(test_preds, test_true)

    elapsed = time.time() - start_time

    # Log results
    result = {
        "val_mae": float(best_val_mae),
        "test_mae": float(test_mae),
        "test_corr": float(test_corr),
        "best_epoch": int(best_epoch),
        "total_epochs": int(epoch + 1),
        "elapsed_seconds": float(elapsed),
        "param_count": int(param_count),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "patience": PATIENCE,
        },
    }

    # Save to results directory
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = f"results/run_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary for the agent to read
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Val MAE:     {best_val_mae:.4f}")
    print(f"Test MAE:    {test_mae:.4f}")
    print(f"Test Corr:   {test_corr:.4f}")
    print(f"Best Epoch:  {best_epoch + 1}")
    print(f"Params:      {param_count:,}")
    print(f"Time:        {elapsed:.1f}s")
    print(f"Saved:       {result_path}")
    print(f"{'='*50}")

    return result


if __name__ == "__main__":
    train()
