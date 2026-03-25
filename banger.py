"""
Phase 3: End-to-end banger pipeline.
Generate N songs with ACE-Step → score with MERT + trained MLP → keep the best.

Usage:
    python banger.py --prompt "upbeat pop song about summer" --generate 10 --keep 3
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import time
import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor
from pathlib import Path

# Import the scorer model class
from train_scorer import BangerScorer


def load_scorer(model_path: str, device: str, input_dim: int = 768):
    """Load the trained banger scorer."""
    model = BangerScorer(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_mert(device: str):
    """Load MERT encoder for audio embedding."""
    model_name = "m-a-p/MERT-v1-330M"
    print("Loading MERT encoder...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, feature_extractor


def embed_audio(audio_path: str, mert_model, feature_extractor, device: str) -> np.ndarray:
    """Extract MERT embedding from a WAV file."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to MERT's expected rate
    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    # Truncate to 30s max for consistency
    max_samples = target_sr * 30
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = mert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    return embedding


def generate_song(prompt: str, lyrics: str, seed: int, output_path: str, ace_step_dir: str):
    """Generate a song using ACE-Step CLI."""
    cmd = [
        os.path.join(ace_step_dir, ".venv", "bin", "python"),
        os.path.join(ace_step_dir, "cli.py"),
        "--prompt", prompt,
        "--seed", str(seed),
        "--output", output_path,
    ]
    if lyrics:
        cmd.extend(["--lyrics", lyrics])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ace_step_dir,
        timeout=300,
    )
    return result.returncode == 0


def generate_song_direct(prompt: str, lyrics: str, seed: int, output_path: str, ace_step_dir: str):
    """Generate a song using ACE-Step by importing directly."""
    script = f"""
import sys
sys.path.insert(0, '{ace_step_dir}')
import torch
import numpy as np

# Set seed
torch.manual_seed({seed})
np.random.seed({seed})

from acestep.core.generation.handler import GenerationHandler

handler = GenerationHandler()
handler.initialize(
    checkpoint_dir='{ace_step_dir}/checkpoints',
    lm_model_path='acestep-5Hz-lm-1.7B',
    backend='pt',
    device='auto',
)

result = handler.generate(
    prompt='''{prompt.replace("'", "\\'")}''',
    lyrics='''{lyrics.replace("'", "\\'")}''' if '''{lyrics}''' else None,
    duration=60,
    num_diffusion_steps=8,
)

# Save the audio
import soundfile as sf
audio = result['audio']
sr = result.get('sample_rate', 48000)
sf.write('{output_path}', audio, sr)
print(f'Generated: {output_path}')
"""

    result = subprocess.run(
        [os.path.join(ace_step_dir, ".venv", "bin", "python"), "-c", script],
        capture_output=True,
        text=True,
        cwd=ace_step_dir,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"  Generation failed: {result.stderr[:200]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate and filter songs by banger score")
    parser.add_argument("--prompt", required=True, help="Song generation prompt")
    parser.add_argument("--lyrics", default="", help="Optional lyrics")
    parser.add_argument("--generate", type=int, default=10, help="Number of songs to generate")
    parser.add_argument("--keep", type=int, default=3, help="Number of top songs to keep")
    parser.add_argument("--scorer-model", default="scorer_model.pt", help="Path to scorer weights")
    parser.add_argument("--ace-step-dir", default="../ace-step/repo", help="Path to ACE-Step repo")
    parser.add_argument("--output-dir", default="output", help="Directory for output songs")
    parser.add_argument("--device", default="auto")
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

    print(f"Device: {device}")
    print(f"Generating {args.generate} songs, keeping top {args.keep}")
    print(f"Prompt: {args.prompt}")
    print()

    # Load models
    mert_model, feature_extractor = load_mert(device)
    scorer = load_scorer(args.scorer_model, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "all_candidates")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate and score songs
    results = []
    total_start = time.time()

    for i in range(args.generate):
        seed = 42 + i
        song_path = os.path.join(temp_dir, f"candidate_{i:03d}_seed{seed}.wav")

        print(f"[{i+1}/{args.generate}] Generating seed={seed}...")
        gen_start = time.time()

        success = generate_song_direct(
            args.prompt, args.lyrics, seed, song_path, args.ace_step_dir
        )

        if not success or not os.path.exists(song_path):
            print(f"  FAILED - skipping")
            continue

        gen_time = time.time() - gen_start

        # Score the song
        score_start = time.time()
        embedding = embed_audio(song_path, mert_model, feature_extractor, device)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)
        with torch.no_grad():
            score = scorer(embedding_tensor).item()
        score = max(0, min(10, score))  # Clamp to 0-10
        score_time = time.time() - score_start

        results.append({
            "path": song_path,
            "seed": seed,
            "score": score,
            "gen_time": gen_time,
            "score_time": score_time,
        })

        print(f"  Score: {score:.2f}/10 | Gen: {gen_time:.1f}s | Score: {score_time:.1f}s")

    # Rank and keep top songs
    results.sort(key=lambda x: x["score"], reverse=True)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"RESULTS — Generated {len(results)} songs in {total_time/60:.1f} minutes")
    print(f"{'='*60}")

    for i, r in enumerate(results):
        marker = " ★ KEEPER" if i < args.keep else ""
        print(f"  #{i+1} Score: {r['score']:.2f}/10 | seed={r['seed']}{marker}")

        # Copy keepers to output root
        if i < args.keep:
            import shutil
            dest = os.path.join(args.output_dir, f"banger_{i+1:02d}_score{r['score']:.1f}.wav")
            shutil.copy2(r["path"], dest)
            print(f"       → {dest}")

    # Save results log
    log_path = os.path.join(args.output_dir, "results.json")
    with open(log_path, "w") as f:
        json.dump({
            "prompt": args.prompt,
            "total_generated": len(results),
            "kept": args.keep,
            "total_time_minutes": total_time / 60,
            "results": results,
        }, f, indent=2)
    print(f"\nResults log: {log_path}")


if __name__ == "__main__":
    main()
