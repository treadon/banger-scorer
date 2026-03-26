"""
Generate N songs with ACE-Step across varied parameters, score each, keep the best.

Each song runs in its own subprocess to avoid MPS memory/state issues.
Models reload per song (~30s overhead) but never hang.
"""

import os
import sys
import time
import shutil
import json
import subprocess
import itertools
import random
import numpy as np
import torch
import librosa
from transformers import AutoModel, AutoFeatureExtractor
from train_scorer import BangerScorer

ACE_STEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ace-step")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CAPTION_VARIANTS = [
    "90s east coast hip hop, heavy bass, boom bap drums, deep male vocals, smooth confident flow, piano samples, vinyl crackle, swagger",
    "dark 90s hip hop, orchestral strings, menacing bass, slow deliberate flow, cinematic, street storytelling, deep voice, minor key",
    "classic boom bap, jazzy piano loops, hard snare, deep male rapper, New York hip hop, smooth flow, head-nodding beat",
    "gritty street rap, heavy 808 bass, chopped soul samples, aggressive delivery, raw hip hop production, hard-hitting drums",
]

BPM_VARIANTS = [78, 85, 90, 95, 100]

KEY_VARIANTS = ["C minor", "Bb minor", "D minor", "E minor", "Ab minor"]

DEFAULT_LYRICS = """[Verse 1]
Yeah, uh, check it
Crown on my head, gold on my neck
Every word I spit, cash a bigger check
From the bottom to the top, never looked back
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
From the ground up, nothing but the truth"""

# Single-song generation script (written to disk, run in isolated subprocess)
GEN_SCRIPT = '''
import os, sys, json

os.environ["TORCHAUDIO_USE_BACKEND"] = "ffmpeg"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open(sys.argv[1]) as f:
    p = json.load(f)

sys.path.insert(0, p["ace_step_dir"])

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.gpu_config import get_gpu_config, set_global_gpu_config
from acestep.model_downloader import ensure_lm_model

gpu_config = get_gpu_config()
set_global_gpu_config(gpu_config)

checkpoint_dir = os.path.join(p["ace_step_dir"], "checkpoints")

dit_handler = AceStepHandler()
_, success = dit_handler.initialize_service(
    project_root=p["ace_step_dir"], config_path="acestep-v15-turbo",
    device="auto", use_flash_attention=False, compile_model=False,
    offload_to_cpu=False, offload_dit_to_cpu=False,
    quantization=None, use_mlx_dit=True,
)
assert success, "DiT init failed"

llm_handler = LLMHandler()
try:
    ensure_lm_model(model_name="acestep-5Hz-lm-1.7B", checkpoints_dir=checkpoint_dir)
except:
    pass

_, lm_ok = llm_handler.initialize(
    checkpoint_dir=checkpoint_dir, lm_model_path="acestep-5Hz-lm-1.7B",
    backend="pt", device="auto", offload_to_cpu=False, dtype=None,
)

params = GenerationParams(
    task_type="text2music",
    caption=p["caption"],
    lyrics=p["lyrics"],
    vocal_language="en", bpm=p["bpm"], keyscale=p["key"],
    duration=p["duration"], inference_steps=8, shift=3.0,
    seed=p["seed"], thinking=lm_ok,
    use_cot_metas=True, use_cot_caption=True, use_cot_language=True,
)
config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[p["seed"]], audio_format="wav")

result = generate_music(
    dit_handler=dit_handler, llm_handler=llm_handler,
    params=params, config=config, save_dir=p["output_dir"],
)

if result.success and result.audios:
    print("OUTPUT_PATH:" + result.audios[0].get("path", ""))
else:
    print("GENERATION_FAILED:" + str(getattr(result, "error", "unknown")))
'''


def find_ace_step_python():
    candidates = [
        os.path.join(ACE_STEP_DIR, ".venv", "bin", "python"),
        os.path.join(PROJECT_DIR, "..", "ace-step", "repo", ".venv", "bin", "python"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def generate_one_song(ace_python, caption, lyrics, seed, bpm, key, output_dir, duration=120):
    """Generate one song in an isolated subprocess. Clean process per song."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Write params
    params_file = os.path.join(output_dir, "params.json")
    with open(params_file, "w") as f:
        json.dump({
            "caption": caption, "lyrics": lyrics, "seed": seed,
            "bpm": bpm, "key": key, "duration": duration,
            "output_dir": output_dir,
            "ace_step_dir": os.path.abspath(ACE_STEP_DIR),
        }, f)

    # Write generation script
    script_file = os.path.join(output_dir, "gen.py")
    with open(script_file, "w") as f:
        f.write(GEN_SCRIPT)

    # Run in isolated subprocess
    result = subprocess.run(
        [ace_python, script_file, params_file],
        capture_output=True, text=True,
        cwd=os.path.abspath(ACE_STEP_DIR),
        timeout=300,
    )

    for line in result.stdout.split("\n"):
        if line.startswith("OUTPUT_PATH:"):
            path = line.replace("OUTPUT_PATH:", "").strip()
            if path and os.path.exists(path):
                return path

    if result.returncode != 0:
        stderr_tail = result.stderr[-200:] if result.stderr else ""
        print(f"    error: {stderr_tail}", flush=True)

    return None


def init_scorer(device="mps"):
    print("Loading MERT + scorer...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    mert = mert.to(device).eval()

    scorer = BangerScorer(input_dim=1024)
    scorer.load_state_dict(torch.load("scorer_model.pt", weights_only=True, map_location=device))
    scorer = scorer.to(device).eval()
    print("Ready.", flush=True)
    return mert, fe, scorer


def score_audio(audio_path, mert, fe, scorer, device="mps"):
    wav, _ = librosa.load(audio_path, sr=fe.sampling_rate, mono=True)
    wav = wav[:fe.sampling_rate * 30]
    inputs = fe(wav, sampling_rate=fe.sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = mert(**inputs).last_hidden_state.mean(dim=1)
        score = scorer(emb).item()
    return max(0, min(10, score))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate songs across varied params and score them")
    parser.add_argument("--generate", type=int, default=20)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lyrics", default=None, help="Custom lyrics (uses default Biggie lyrics if not set)")
    args = parser.parse_args()

    ace_python = find_ace_step_python()
    if not ace_python:
        print("ERROR: Can't find ACE-Step Python. Run: cd ace-step && uv sync --python 3.12")
        sys.exit(1)

    # Build search grid
    all_combos = list(itertools.product(CAPTION_VARIANTS, BPM_VARIANTS, KEY_VARIANTS))
    random.seed(42)
    random.shuffle(all_combos)
    selected = all_combos[:args.generate]

    lyrics = args.lyrics or DEFAULT_LYRICS
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Generate (one subprocess per song)
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 1: Generating {len(selected)} songs", flush=True)
    print(f"  Search: {len(CAPTION_VARIANTS)} styles x {len(BPM_VARIANTS)} BPMs x {len(KEY_VARIANTS)} keys", flush=True)
    print(f"  One subprocess per song (stable, no MPS hangs)", flush=True)
    print(f"{'='*60}\n", flush=True)

    generated = []
    for i, (caption, bpm, key) in enumerate(selected):
        seed = 42 + i
        gen_dir = os.path.join(output_dir, "candidates", f"gen_{i:03d}")
        caption_short = caption[:50] + "..."

        print(f"[{i+1}/{len(selected)}] seed={seed} bpm={bpm} key={key}", flush=True)
        print(f"    {caption_short}", flush=True)

        t0 = time.time()
        path = generate_one_song(ace_python, caption, lyrics, seed, bpm, key, gen_dir, args.duration)
        gen_time = time.time() - t0

        if path:
            generated.append({"path": path, "seed": seed, "bpm": bpm, "key": key, "caption": caption, "gen_time": gen_time})
            print(f"    OK ({gen_time:.0f}s)", flush=True)
        else:
            print(f"    FAILED ({gen_time:.0f}s)", flush=True)

    print(f"\nGenerated {len(generated)}/{len(selected)} songs", flush=True)

    if not generated:
        print("No songs generated. Exiting.")
        sys.exit(1)

    # Phase 2: Score all
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 2: Scoring {len(generated)} songs", flush=True)
    print(f"{'='*60}\n", flush=True)

    mert, fe, scorer = init_scorer(args.device)

    results = []
    for g in generated:
        score = score_audio(g["path"], mert, fe, scorer, args.device)
        g["score"] = score
        results.append(g)
        print(f"  {score:.2f}/10 | bpm={g['bpm']} key={g['key']} seed={g['seed']}", flush=True)

    # Rank
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RANKINGS", flush=True)
    print(f"Score range: {results[-1]['score']:.2f} to {results[0]['score']:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    for i, r in enumerate(results):
        is_top = i < args.keep
        is_bottom = i >= len(results) - args.keep
        marker = " ★ BEST" if is_top else (" ✗ WORST" if is_bottom else "")
        print(f"  #{i+1} {r['score']:.2f}/10 | bpm={r['bpm']} key={r['key']} seed={r['seed']}{marker}", flush=True)

        if is_top:
            fname = f"best_{i+1:02d}_score{r['score']:.1f}_bpm{r['bpm']}_{r['key'].replace(' ', '')}_seed{r['seed']}.wav"
            dest = os.path.join(output_dir, fname)
            shutil.copy2(r["path"], dest)
        elif is_bottom:
            rank_from_bottom = len(results) - i
            fname = f"worst_{rank_from_bottom:02d}_score{r['score']:.1f}_bpm{r['bpm']}_{r['key'].replace(' ', '')}_seed{r['seed']}.wav"
            dest = os.path.join(output_dir, fname)
            shutil.copy2(r["path"], dest)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"results": [{k: v for k, v in r.items() if k != "path"} for r in results]}, f, indent=2)

    print(f"\nTop {args.keep} + bottom {args.keep} saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
