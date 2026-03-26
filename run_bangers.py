"""
Banger-optimized generation run.
Uses only the parameter combos that scored highest in our 200-song test.
"""

import os
import sys
import time
import shutil
import json
import subprocess
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

# Only the styles/params that scored highest
BANGER_CAPTIONS = [
    # EDM styles (dominated top 10)
    "melodic techno, driving beat, atmospheric pads, dark and dreamy, analog synth textures, pulsing bass, hypnotic rhythm, rave energy, euphoric breakdown",
    "deep house, groovy bassline, warm chord stabs, subtle chopped vocal samples, hypnotic four-on-the-floor, late night underground club, immersive",
    "progressive house, massive building pads, euphoric synth lead, hands-in-the-air drop, festival anthem, emotional melody, soaring energy, cinematic build",
    # Punjabi/Bhangra fusion with electronic (top 2 genres combined)
    "modern bhangra fusion with electronic production, heavy dhol over house beat, tumbi melody with synth layers, high energy dance, festival ready, euphoric",
    "Punjabi dance anthem, dhol and 808 fusion, electronic drops with traditional instruments, massive energy, celebratory, club meets wedding party",
    # Bollywood electronic fusion
    "Bollywood club banger, heavy desi bass meets EDM drop, female vocals with electronic production, cinematic strings, dance floor energy, massive chorus",
    "modern Indian electronic, tabla and synth fusion, driving beat, atmospheric, minor key, dark and intense, Bollywood meets techno",
    # Dark electronic (our #1 was "dark and dreamy")
    "dark electronic, industrial textures, relentless driving beat, distorted bass, menacing atmosphere, minor key, intense, underground rave",
    "ambient techno, ethereal pads, deep sub bass, minimal percussion building to massive drop, space, darkness, transcendent",
    # The exact style of our #1 song with variations
    "melodic techno, atmospheric textures, dark and dreamy, driving four-on-the-floor, deep reverb, analog warmth, hypnotic groove, peak time",
]

# Only the BPMs and keys that scored highest
BANGER_BPMS = [126, 128, 130, 135, 138]
BANGER_KEYS = ["Eb minor", "F minor", "Bb minor", "C minor", "D minor"]

# Minimal lyrics for EDM (mostly instrumental with vocal hooks)
EDM_LYRICS = """[Build]
Feel the bass beneath your feet
Let the rhythm take control
Every heartbeat syncs the beat
Lose yourself and free your soul

[Drop]

[Verse]
Lasers cutting through the haze
Lost inside this endless maze
We don't need a reason at all
Just the sound, just the call

[Build]
Hands up, reach for the sky
Feel the energy rise
This is where we come alive

[Drop]

[Outro]
One more time
Feel it rise"""

# Punjabi/fusion lyrics
FUSION_LYRICS = """[Verse 1]
Dhol vajje bass drop heavy
Saari duniya nachdi ready
East meets west on the dance floor
Give me more give me more

[Hook]
Bass drop dhol drop
Never gonna stop
Light it up light it up
Take it to the top

[Verse 2]
Tumbi da sound speaker rattle
Every beat a sonic battle
Traditional meets the future
Feel the bass feel the rupture

[Hook]
Bass drop dhol drop
Never gonna stop
Light it up light it up
Take it to the top

[Drop]

[Outro]
Never gonna stop
Take it to the top"""


def main():
    ace_python = find_ace_step_python()
    if not ace_python:
        print("ERROR: Can't find ACE-Step Python.")
        sys.exit(1)

    num_generate = 30
    keep = 5

    # Build combos — weight toward the highest-scoring parameter combos
    combos = []
    for caption in BANGER_CAPTIONS:
        for bpm in BANGER_BPMS:
            for key in BANGER_KEYS:
                combos.append((caption, bpm, key))

    random.seed(777)
    random.shuffle(combos)
    selected = combos[:num_generate]

    output_dir = os.path.abspath("bangers_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"BANGER RUN: Optimized for highest scores", flush=True)
    print(f"  {len(BANGER_CAPTIONS)} caption styles (all high-scoring)", flush=True)
    print(f"  BPMs: {BANGER_BPMS} (sweet spots only)", flush=True)
    print(f"  Keys: {BANGER_KEYS} (minor keys only)", flush=True)
    print(f"  Generating {num_generate} songs, keeping top {keep}", flush=True)
    print(f"{'='*60}\n", flush=True)

    generated = []
    for i, (caption, bpm, key) in enumerate(selected):
        seed = 2000 + i
        gen_dir = os.path.join(output_dir, "candidates", f"gen_{i:03d}")

        # Pick lyrics based on style
        if any(kw in caption.lower() for kw in ["bhangra", "punjabi", "dhol", "tumbi", "desi", "bollywood", "indian"]):
            lyrics = FUSION_LYRICS
        else:
            lyrics = EDM_LYRICS

        print(f"[{i+1}/{num_generate}] seed={seed} bpm={bpm} key={key}", flush=True)
        print(f"    {caption[:55]}...", flush=True)

        t0 = time.time()
        path = generate_one_song(ace_python, caption, lyrics, seed, bpm, key, gen_dir, duration=120)
        gen_time = time.time() - t0

        if path:
            generated.append({
                "path": path, "seed": seed, "bpm": bpm, "key": key,
                "caption": caption, "gen_time": gen_time,
            })
            print(f"    OK ({gen_time:.0f}s)", flush=True)
        else:
            print(f"    FAILED ({gen_time:.0f}s)", flush=True)

    print(f"\nGenerated {len(generated)}/{num_generate} songs", flush=True)

    if not generated:
        print("No songs generated.")
        sys.exit(1)

    # Score all
    print(f"\n{'='*60}", flush=True)
    print(f"Scoring {len(generated)} songs", flush=True)
    print(f"{'='*60}\n", flush=True)

    mert, fe, scorer = init_scorer("mps")

    for g in generated:
        score = score_audio(g["path"], mert, fe, scorer, "mps")
        g["score"] = score
        print(f"  {score:.2f}/10 | bpm={g['bpm']} key={g['key']} seed={g['seed']}", flush=True)

    generated.sort(key=lambda x: x["score"], reverse=True)

    # Convert to MP3
    print(f"\nConverting to MP3...", flush=True)
    for g in generated:
        mp3_path = g["path"].replace(".wav", ".mp3")
        subprocess.run(["ffmpeg", "-i", g["path"], "-codec:a", "libmp3lame", "-b:a", "192k",
                        mp3_path, "-y", "-loglevel", "error"])
        os.remove(g["path"])
        g["path"] = mp3_path

    # Results
    print(f"\n{'='*60}", flush=True)
    print(f"BANGER RESULTS", flush=True)
    print(f"Score range: {generated[-1]['score']:.2f} to {generated[0]['score']:.2f}", flush=True)
    print(f"Mean: {np.mean([g['score'] for g in generated]):.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    for i, g in enumerate(generated):
        is_top = i < keep
        marker = " ★ BANGER" if is_top else ""
        print(f"  #{i+1} {g['score']:.2f}/10 | bpm={g['bpm']} key={g['key']} seed={g['seed']}{marker}", flush=True)
        if is_top:
            print(f"       {g['caption'][:60]}...", flush=True)
            fname = f"BANGER_{i+1:02d}_score{g['score']:.1f}_bpm{g['bpm']}_{g['key'].replace(' ', '')}_seed{g['seed']}.mp3"
            shutil.copy2(g["path"], os.path.join(output_dir, fname))

    # Compare to previous test averages
    print(f"\n{'='*60}", flush=True)
    print(f"COMPARISON TO PREVIOUS TESTS", flush=True)
    print(f"{'='*60}", flush=True)
    banger_mean = np.mean([g["score"] for g in generated])
    banger_max = generated[0]["score"]
    print(f"  Previous best single score: 5.29 (EDM test)", flush=True)
    print(f"  This run best score:        {banger_max:.2f}", flush=True)
    print(f"  Previous highest mean:      3.77 (Punjabi)", flush=True)
    print(f"  This run mean:              {banger_mean:.2f}", flush=True)
    print(f"  Previous overall mean:      3.17 (200 songs)", flush=True)
    print(f"  Improvement over random:    {((banger_mean / 3.17) - 1) * 100:.0f}%", flush=True)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "run": "banger_optimized",
            "num_generated": len(generated),
            "score_min": generated[-1]["score"],
            "score_max": generated[0]["score"],
            "score_mean": float(banger_mean),
            "results": [{k: v for k, v in g.items() if k != "path"} for g in generated],
        }, f, indent=2)

    print(f"\nBangers saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
