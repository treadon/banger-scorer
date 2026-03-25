"""
Phase 2.1: Extract MERT embeddings from FMA audio files.
Runs MERT-v1-330M on each track and caches 1024-dim embeddings to disk.

Uses concurrent loading to pipeline CPU (MP3 decode) and GPU (MERT inference).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import AutoModel, AutoFeatureExtractor
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time


def load_fma_metadata(metadata_dir: str) -> pd.DataFrame:
    """Load FMA tracks.csv and extract popularity (listens) + genre info."""
    tracks_path = os.path.join(metadata_dir, "tracks.csv")
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    df = pd.DataFrame({
        "track_id": tracks.index,
        "listens": tracks[("track", "listens")].values,
        "interest": tracks[("track", "interest")].values,
        "genre_top": tracks[("track", "genre_top")].values,
        "subset": tracks[("set", "subset")].values,
    })

    df = df[df["subset"] == "small"].copy()
    df["listens"] = pd.to_numeric(df["listens"], errors="coerce").fillna(0)
    df = df[df["listens"] > 0].copy()

    log_listens = np.log1p(df["listens"])
    df["banger_score"] = (log_listens - log_listens.min()) / (log_listens.max() - log_listens.min()) * 10.0

    print(f"Loaded {len(df)} tracks from FMA-Small", flush=True)
    print(f"Banger score: mean={df['banger_score'].mean():.2f}, std={df['banger_score'].std():.2f}", flush=True)
    print(f"Genres: {df['genre_top'].value_counts().to_dict()}", flush=True)

    return df


def get_audio_path(audio_dir: str, track_id: int) -> str:
    """Get the file path for a track ID in FMA directory structure."""
    tid_str = str(track_id).zfill(6)
    return os.path.join(audio_dir, tid_str[:3], f"{tid_str}.mp3")


def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    """Load and resample audio file. Returns numpy array or None on failure."""
    try:
        wav, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        # Truncate to 30 seconds
        max_samples = target_sr * 30
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        return wav
    except Exception:
        return None


def extract_embeddings(
    df: pd.DataFrame,
    audio_dir: str,
    output_dir: str,
    device: str = "mps",
    num_workers: int = 4,
):
    """Run MERT on all tracks with pipelined CPU loading + GPU inference."""

    print(f"\nLoading MERT-v1-330M on {device}...", flush=True)
    model_name = "m-a-p/MERT-v1-330M"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    target_sr = feature_extractor.sampling_rate
    print(f"Sample rate: {target_sr}Hz", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    # Build list of (track_id, audio_path, banger_score, genre)
    tasks = []
    for _, row in df.iterrows():
        track_id = row["track_id"]
        audio_path = get_audio_path(audio_dir, track_id)
        if os.path.exists(audio_path):
            tasks.append({
                "track_id": track_id,
                "audio_path": audio_path,
                "banger_score": float(row["banger_score"]),
                "genre": row["genre_top"] if pd.notna(row["genre_top"]) else "Unknown",
                "listens": int(row["listens"]),
            })

    print(f"Found {len(tasks)} audio files out of {len(df)} tracks", flush=True)

    # Process tracks one at a time: load audio (CPU) → MERT inference (GPU)
    print(f"\nProcessing tracks: load + embed sequentially...", flush=True)
    embeddings = {}
    labels = {}
    failed = 0
    start_time = time.time()

    for i, task in enumerate(tasks):
        track_id = task["track_id"]

        # Load audio (CPU)
        wav = load_audio(task["audio_path"], target_sr)
        if wav is None:
            failed += 1
            continue

        # Feature extraction (CPU)
        inputs = feature_extractor(
            wav,
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # MERT forward pass (GPU)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

        embeddings[track_id] = embedding
        labels[track_id] = {
            "banger_score": task["banger_score"],
            "genre": task["genre"],
            "listens": task["listens"],
        }

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = len(embeddings) / elapsed
            remaining = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(f"  [{len(embeddings)}/{len(tasks)}] "
                  f"{rate:.1f} tracks/s, {failed} failed, "
                  f"ETA: {remaining/60:.1f}m", flush=True)

    total_time = time.time() - start_time
    print(f"\nDone: {len(embeddings)} tracks in {total_time:.1f}s "
          f"({len(embeddings)/total_time:.1f} tracks/s), {failed} failed", flush=True)
    load_time = 0
    infer_time = total_time

    # Save
    track_ids = sorted(embeddings.keys())
    embedding_matrix = np.array([embeddings[tid] for tid in track_ids])
    labels_list = [labels[tid] for tid in track_ids]

    np.save(os.path.join(output_dir, "embeddings.npy"), embedding_matrix)

    metadata = {
        "track_ids": track_ids,
        "labels": labels_list,
        "embedding_dim": int(embedding_matrix.shape[1]),
        "num_tracks": len(track_ids),
        "model": "m-a-p/MERT-v1-330M",
        "sample_rate": target_sr,
        "device": device,
        "audio_load_time_s": load_time,
        "inference_time_s": infer_time,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = load_time + infer_time
    print(f"\nSaved: {embedding_matrix.shape} embeddings to {output_dir}", flush=True)
    print(f"Total time: {total_time/60:.1f} minutes", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract MERT embeddings from FMA dataset")
    parser.add_argument("--audio-dir", default="data/fma_small")
    parser.add_argument("--metadata-dir", default="data/fma_metadata")
    parser.add_argument("--output-dir", default="data/embeddings")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4, help="Threads for audio loading")
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

    print(f"Device: {device}", flush=True)
    df = load_fma_metadata(args.metadata_dir)
    extract_embeddings(df, args.audio_dir, args.output_dir, device=device, num_workers=args.workers)


if __name__ == "__main__":
    main()
