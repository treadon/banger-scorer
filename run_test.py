"""
General test runner. Takes a test config JSON and runs generation + scoring.
Usage: python run_test.py test_configs/test03.json
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
from generate_and_score import (
    find_ace_step_python, generate_one_song, init_scorer, score_audio,
    ACE_STEP_DIR,
)


def run_test(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    test_name = cfg["test_name"]
    test_id = cfg["test_id"]
    captions = cfg["captions"]
    bpms = cfg["bpms"]
    keys = cfg["keys"]
    lyrics = cfg["lyrics"]
    language = cfg.get("language", "en")
    duration = cfg.get("duration", 120)
    num_generate = cfg.get("num_generate", 20)
    keep = cfg.get("keep", 5)
    seed_start = cfg.get("seed_start", 200)

    ace_python = find_ace_step_python()
    if not ace_python:
        print("ERROR: Can't find ACE-Step Python.")
        sys.exit(1)

    # Build search grid
    all_combos = list(itertools.product(captions, bpms, keys))
    random.seed(seed_start)
    random.shuffle(all_combos)
    selected = all_combos[:num_generate]

    output_dir = os.path.abspath(f"{test_id}/output")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Generate
    print(f"\n{'='*60}", flush=True)
    print(f"{test_id.upper()}: {test_name} — {num_generate} songs", flush=True)
    print(f"  {len(captions)} styles x {len(bpms)} BPMs x {len(keys)} keys", flush=True)
    print(f"  Language: {language}", flush=True)
    print(f"{'='*60}\n", flush=True)

    generated = []
    for i, (caption, bpm, key) in enumerate(selected):
        seed = seed_start + i
        gen_dir = os.path.join(output_dir, "candidates", f"gen_{i:03d}")

        print(f"[{i+1}/{num_generate}] seed={seed} bpm={bpm} key={key}", flush=True)
        print(f"    {caption[:55]}...", flush=True)

        t0 = time.time()
        path = generate_one_song(ace_python, caption, lyrics, seed, bpm, key, gen_dir, duration)
        gen_time = time.time() - t0

        if path:
            generated.append({"path": path, "seed": seed, "bpm": bpm, "key": key, "caption": caption, "gen_time": gen_time})
            print(f"    OK ({gen_time:.0f}s)", flush=True)
        else:
            print(f"    FAILED ({gen_time:.0f}s)", flush=True)

    print(f"\nGenerated {len(generated)}/{num_generate} songs", flush=True)

    if not generated:
        print("No songs generated.")
        return

    # Phase 2: Score
    print(f"\n{'='*60}", flush=True)
    print(f"Scoring {len(generated)} songs", flush=True)
    print(f"{'='*60}\n", flush=True)

    mert, fe, scorer = init_scorer("mps")

    results = []
    for g in generated:
        score = score_audio(g["path"], mert, fe, scorer, "mps")
        g["score"] = score
        results.append(g)
        print(f"  {score:.2f}/10 | bpm={g['bpm']} key={g['key']} seed={g['seed']}", flush=True)

    results.sort(key=lambda x: x["score"], reverse=True)

    # Convert to MP3
    print(f"\nConverting to MP3...", flush=True)
    for r in results:
        mp3_path = r["path"].replace(".wav", ".mp3")
        subprocess.run(["ffmpeg", "-i", r["path"], "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path, "-y", "-loglevel", "error"])
        os.remove(r["path"])
        r["path"] = mp3_path

    # Save best + worst
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RANKINGS — {test_id.upper()}: {test_name}", flush=True)
    print(f"Score range: {results[-1]['score']:.2f} to {results[0]['score']:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    for i, r in enumerate(results):
        is_top = i < keep
        is_bottom = i >= len(results) - keep
        marker = " ★ BEST" if is_top else (" ✗ WORST" if is_bottom else "")
        print(f"  #{i+1} {r['score']:.2f}/10 | bpm={r['bpm']} key={r['key']} seed={r['seed']}{marker}", flush=True)

        if is_top:
            fname = f"best_{i+1:02d}_score{r['score']:.1f}_bpm{r['bpm']}_{r['key'].replace(' ', '')}_seed{r['seed']}.mp3"
            shutil.copy2(r["path"], os.path.join(output_dir, fname))
        elif is_bottom:
            rank_from_bottom = len(results) - i
            fname = f"worst_{rank_from_bottom:02d}_score{r['score']:.1f}_bpm{r['bpm']}_{r['key'].replace(' ', '')}_seed{r['seed']}.mp3"
            shutil.copy2(r["path"], os.path.join(output_dir, fname))

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "test_id": test_id,
            "test_name": test_name,
            "language": language,
            "num_generated": len(generated),
            "num_failed": num_generate - len(generated),
            "score_min": results[-1]["score"],
            "score_max": results[0]["score"],
            "score_mean": np.mean([r["score"] for r in results]),
            "results": [{k: v for k, v in r.items() if k != "path"} for r in results],
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    run_test(sys.argv[1])
