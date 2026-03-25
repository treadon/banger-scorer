"""
Generate N songs with ACE-Step, score each with the banger scorer, keep the best.

ACE-Step runs in its own venv (Python 3.12 required).
Scoring runs in the banger-scorer venv.
"""

import os
import sys
import time
import shutil
import json
import subprocess
import glob
import numpy as np
import torch
import librosa
from transformers import AutoModel, AutoFeatureExtractor
from train_scorer import BangerScorer

ACE_STEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ace-step")


def init_scorer(device="mps"):
    """Load MERT + trained scorer."""
    print("Loading MERT encoder...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    mert = mert.to(device).eval()

    scorer = BangerScorer(input_dim=1024)
    scorer.load_state_dict(torch.load("scorer_model.pt", weights_only=True, map_location=device))
    scorer = scorer.to(device).eval()
    print("Scorer ready.", flush=True)

    return mert, fe, scorer


def score_audio(audio_path, mert, fe, scorer, device="mps"):
    """Score a single audio file."""
    wav, _ = librosa.load(audio_path, sr=fe.sampling_rate, mono=True)
    wav = wav[:fe.sampling_rate * 30]

    inputs = fe(wav, sampling_rate=fe.sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = mert(**inputs).last_hidden_state.mean(dim=1)
        score = scorer(emb).item()

    return max(0, min(10, score))


def find_ace_step_python():
    """Find ACE-Step's Python interpreter."""
    # Check for venv in the submodule
    candidates = [
        os.path.join(ACE_STEP_DIR, ".venv", "bin", "python"),
        os.path.join(ACE_STEP_DIR, ".venv", "bin", "python3"),
        # Fallback: check the original install location
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ace-step", "repo", ".venv", "bin", "python"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def generate_song(python_path, caption, lyrics, seed, output_dir, duration=120):
    """Generate a single song via subprocess."""
    os.makedirs(output_dir, exist_ok=True)

    script = f'''
import os, sys, time
os.environ["TORCHAUDIO_USE_BACKEND"] = "ffmpeg"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = "{ACE_STEP_DIR}"
sys.path.insert(0, project_root)

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.gpu_config import get_gpu_config, set_global_gpu_config
from acestep.model_downloader import ensure_lm_model

gpu_config = get_gpu_config()
set_global_gpu_config(gpu_config)

checkpoint_dir = os.path.join(project_root, "checkpoints")

dit_handler = AceStepHandler()
_, success = dit_handler.initialize_service(
    project_root=project_root, config_path="acestep-v15-turbo",
    device="auto", use_flash_attention=False, compile_model=False,
    offload_to_cpu=False, offload_dit_to_cpu=False,
    quantization=None, use_mlx_dit=True,
)
assert success, "DiT init failed"

llm_handler = LLMHandler()
try:
    ensure_lm_model(model_name="acestep-5Hz-lm-1.7B", checkpoints_dir=checkpoint_dir)
except: pass

_, lm_ok = llm_handler.initialize(
    checkpoint_dir=checkpoint_dir, lm_model_path="acestep-5Hz-lm-1.7B",
    backend="pt", device="auto", offload_to_cpu=False, dtype=None,
)

params = GenerationParams(
    task_type="text2music",
    caption="""{caption.replace('"', '\\"')}""",
    lyrics="""{lyrics.replace('"', '\\"')}""",
    vocal_language="en", bpm=90, keyscale="C minor",
    duration={duration}, inference_steps=8, shift=3.0,
    seed={seed}, thinking=lm_ok,
    use_cot_metas=True, use_cot_caption=True, use_cot_language=True,
)
config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[{seed}], audio_format="wav")

result = generate_music(
    dit_handler=dit_handler, llm_handler=llm_handler,
    params=params, config=config, save_dir="{output_dir}",
)

if result.success and result.audios:
    print("OUTPUT_PATH:" + result.audios[0].get("path", ""))
else:
    print("GENERATION_FAILED")
'''

    result = subprocess.run(
        [python_path, "-c", script],
        capture_output=True, text=True,
        cwd=ACE_STEP_DIR,
        timeout=600,
    )

    # Parse output path from stdout
    for line in result.stdout.split("\n"):
        if line.startswith("OUTPUT_PATH:"):
            path = line.replace("OUTPUT_PATH:", "").strip()
            if path and os.path.exists(path):
                return path

    # Debug on failure
    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
        print(f"  Generation subprocess failed: {stderr_tail}", flush=True)

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate songs and score them")
    parser.add_argument("--prompt", default="90s east coast hip hop, heavy bass, gritty boom bap beat, deep male vocals, Notorious BIG style flow, confident swagger, street storytelling, piano samples, vinyl crackle, hard hitting drums")
    parser.add_argument("--lyrics", default="""[Verse 1]
Yeah, uh, check it
Crown on my head, gold on my neck
Every word I spit, cash a bigger check
From the block to the top, never looked back
Spit it smooth like silk on a Cadillac track

[Chorus]
Big money, big dreams, big life
Every day a hustle, every night a fight
Stack it to the ceiling, feeling so right
Crown heavy but I wear it tight

[Verse 2]
Blowing cigar smoke through the city lights
Penthouse view but I remember the nights
When the fridge was empty and the heat was off
Now the whole world listening when I talk

[Chorus]
Big money, big dreams, big life
Every day a hustle, every night a fight
Stack it to the ceiling, feeling so right
Crown heavy but I wear it tight

[Outro]
Yeah, you know how we do
Brooklyn to the world, nothing but the truth""")
    parser.add_argument("--generate", type=int, default=10)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output-dir", default="output_biggie")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    # Find ACE-Step python
    ace_python = find_ace_step_python()
    if not ace_python:
        print("ERROR: Can't find ACE-Step Python. Run setup first:")
        print("  cd ace-step && uv sync --python 3.12")
        sys.exit(1)
    print(f"ACE-Step Python: {ace_python}", flush=True)

    # Init scorer
    mert, fe, scorer = init_scorer(args.device)

    # Generate and score
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    print(f"\n{'='*60}", flush=True)
    print(f"Generating {args.generate} songs, keeping top {args.keep}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for i in range(args.generate):
        seed = 42 + i
        gen_dir = os.path.join(args.output_dir, "candidates", f"seed_{seed}")

        print(f"[{i+1}/{args.generate}] Generating seed={seed}...", flush=True)

        t0 = time.time()
        audio_path = generate_song(ace_python, args.prompt, args.lyrics, seed, gen_dir, args.duration)
        gen_time = time.time() - t0

        if not audio_path:
            print(f"  FAILED ({gen_time:.1f}s)", flush=True)
            continue

        # Score it
        t0 = time.time()
        score = score_audio(audio_path, mert, fe, scorer, args.device)
        score_time = time.time() - t0

        results.append({
            "path": audio_path,
            "seed": seed,
            "score": score,
            "gen_time": gen_time,
            "score_time": score_time,
        })
        print(f"  Score: {score:.2f}/10 | Gen: {gen_time:.1f}s | Score: {score_time:.1f}s", flush=True)

    # Rank and keep top
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS — {len(results)} songs generated", flush=True)
    print(f"{'='*60}", flush=True)

    for i, r in enumerate(results):
        marker = " ★ KEEPER" if i < args.keep else ""
        print(f"  #{i+1} Score: {r['score']:.2f}/10 | seed={r['seed']}{marker}", flush=True)

        if i < args.keep:
            dest = os.path.join(args.output_dir, f"banger_{i+1:02d}_score{r['score']:.1f}_seed{r['seed']}.wav")
            shutil.copy2(r["path"], dest)
            print(f"       → {dest}", flush=True)

    # Save log
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"prompt": args.prompt, "results": results}, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
