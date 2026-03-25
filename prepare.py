"""
Autoresearch: Data preparation (READ-ONLY — do not modify).
Loads cached MERT embeddings and provides train/val/test splits.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

EMBEDDINGS_DIR = "data/embeddings"


def load_splits(random_state=42, test_size=0.15, val_size=0.15):
    """Load MERT embeddings and return train/val/test splits.

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        genres_train, genres_val, genres_test, metadata
    """
    embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"))
    with open(os.path.join(EMBEDDINGS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    scores = np.array([label["banger_score"] for label in metadata["labels"]])
    genres = np.array([label["genre"] for label in metadata["labels"]])

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test, g_trainval, g_test = train_test_split(
        embeddings, scores, genres,
        test_size=test_size,
        random_state=random_state,
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(
        X_trainval, y_trainval, g_trainval,
        test_size=val_ratio,
        random_state=random_state,
    )

    print(f"Data loaded: {len(scores)} total tracks")
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"  Score mean: {scores.mean():.2f}, std: {scores.std():.2f}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "genres_train": g_train, "genres_val": g_val, "genres_test": g_test,
        "embedding_dim": embeddings.shape[1],
        "metadata": metadata,
    }
