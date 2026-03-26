"""
Test 02: Upbeat pop/dance songs across varied styles.
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
    ACE_STEP_DIR, GEN_SCRIPT,
)

CAPTION_VARIANTS = [
    "upbeat pop, catchy melody, bright synths, female vocals, danceable, feel-good summer anthem, major key, polished production",
    "upbeat dance pop, four on the floor beat, euphoric synth lead, energetic, club-ready, big chorus, hands in the air",
    "feel-good funk pop, groovy bass, upbeat rhythm guitar, claps, sunny disposition, infectious hook, retro vibes",
    "upbeat electronic pop, pulsing beat, shimmering arpeggios, soaring vocals, anthemic chorus, festival energy, bright and airy",
]

BPM_VARIANTS = [110, 118, 124, 128, 135]

KEY_VARIANTS = ["C major", "G major", "D major", "A major", "F major"]

LYRICS = """[Verse 1]
Wake up, the sun is calling out your name
Step outside and feel the fire in the flame
Every morning is a chance to start again
Throw your worries to the wind and let them fade

[Pre-Chorus]
Can you feel it rising, feel it in your bones
Electric running through you, you are not alone

[Chorus]
Light it up, light it up, let the whole world see
We are alive and we are free
Turn it up, turn it up, let the music breathe
This is where we're meant to be

[Verse 2]
Dancing in the glow of neon city lights
Every heartbeat syncing up to feel the vibe
We don't need a reason, we don't need a sign
Just the rhythm and the moment, feeling fine

[Chorus]
Light it up, light it up, let the whole world see
We are alive and we are free
Turn it up, turn it up, let the music breathe
This is where we're meant to be

[Bridge]
Let it go, let it flow
Feel the beat down in your soul
Nothing stopping us tonight

[Outro]
Light it up, light it up
We are alive and we are free"""


def main():
    ace_python = find_ace_step_python()
    if not ace_python:
        print("ERROR: Can't find ACE-Step Python.")
        sys.exit(1)

    # Build search grid
    all_combos = list(itertools.product(CAPTION_VARIANTS, BPM_VARIANTS, KEY_VARIANTS))
    random.seed(99)
    random.shuffle(all_combos)
    selected = all_combos[:20]

    output_dir = os.path.abspath("test02/output")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Generate
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 02: Upbeat Pop/Dance — 20 songs", flush=True)
    print(f"  {len(CAPTION_VARIANTS)} styles x {len(BPM_VARIANTS)} BPMs x {len(KEY_VARIANTS)} keys", flush=True)
    print(f"{'='*60}\n", flush=True)

    generated = []
    for i, (caption, bpm, key) in enumerate(selected):
        seed = 100 + i
        gen_dir = os.path.join(output_dir, "candidates", f"gen_{i:03d}")

        print(f"[{i+1}/20] seed={seed} bpm={bpm} key={key}", flush=True)
        print(f"    {caption[:55]}...", flush=True)

        t0 = time.time()
        path = generate_one_song(ace_python, caption, LYRICS, seed, bpm, key, gen_dir, duration=120)
        gen_time = time.time() - t0

        if path:
            generated.append({"path": path, "seed": seed, "bpm": bpm, "key": key, "caption": caption, "gen_time": gen_time})
            print(f"    OK ({gen_time:.0f}s)", flush=True)
        else:
            print(f"    FAILED ({gen_time:.0f}s)", flush=True)

    print(f"\nGenerated {len(generated)}/20 songs", flush=True)

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

    # Convert all to MP3
    print(f"\nConverting to MP3...", flush=True)
    for r in results:
        mp3_path = r["path"].replace(".wav", ".mp3")
        subprocess.run(["ffmpeg", "-i", r["path"], "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path, "-y", "-loglevel", "error"])
        os.remove(r["path"])
        r["path"] = mp3_path

    # Rankings
    keep = 5
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RANKINGS — TEST 02 (Upbeat Pop/Dance)", flush=True)
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
        json.dump({"test": "02_upbeat_pop_dance", "results": [{k: v for k, v in r.items() if k != "path"} for r in results]}, f, indent=2)

    print(f"\nTop {keep} + bottom {keep} saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
