"""
Run all tests (03-10) sequentially, then compile best-overall across all tests.
Tests 01 and 02 are already complete.
"""

import os
import sys
import json
import shutil
import glob
import time


def run_test(config_path):
    """Run a single test by importing and calling run_test."""
    from run_test import run_test as _run_test
    _run_test(config_path)


def compile_best_overall():
    """Read all test results and find the best songs across all tests."""
    print(f"\n{'='*60}", flush=True)
    print(f"BEST OVERALL — Across All Tests", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_results = []

    for test_dir in sorted(glob.glob("test*/output")):
        results_file = os.path.join(test_dir, "results.json")
        if not os.path.exists(results_file):
            continue

        with open(results_file) as f:
            data = json.load(f)

        test_id = data.get("test_id", os.path.basename(os.path.dirname(test_dir)))
        test_name = data.get("test_name", test_id)

        for r in data.get("results", []):
            r["test_id"] = test_id
            r["test_name"] = test_name
            all_results.append(r)

        # Summary per test
        scores = [r["score"] for r in data.get("results", [])]
        if scores:
            print(f"  {test_id}: {test_name}", flush=True)
            print(f"    Songs: {len(scores)} | Range: {min(scores):.2f}–{max(scores):.2f} | Mean: {sum(scores)/len(scores):.2f}", flush=True)

    if not all_results:
        print("No results found.")
        return

    # Sort all results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Best overall
    os.makedirs("best_overall", exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"TOP 10 ACROSS ALL {len(all_results)} SONGS", flush=True)
    print(f"{'='*60}", flush=True)

    for i, r in enumerate(all_results[:10]):
        print(f"  #{i+1} {r['score']:.2f}/10 | {r['test_name']} | bpm={r['bpm']} key={r['key']} seed={r['seed']}", flush=True)
        print(f"       {r['caption'][:60]}...", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"BOTTOM 5 ACROSS ALL SONGS", flush=True)
    print(f"{'='*60}", flush=True)

    for i, r in enumerate(all_results[-5:]):
        rank = len(all_results) - 4 + i
        print(f"  #{rank} {r['score']:.2f}/10 | {r['test_name']} | bpm={r['bpm']} key={r['key']} seed={r['seed']}", flush=True)

    # Save summary
    summary = {
        "total_songs": len(all_results),
        "overall_score_min": all_results[-1]["score"],
        "overall_score_max": all_results[0]["score"],
        "overall_score_mean": sum(r["score"] for r in all_results) / len(all_results),
        "top_10": [{k: v for k, v in r.items()} for r in all_results[:10]],
        "bottom_5": [{k: v for k, v in r.items()} for r in all_results[-5:]],
    }

    with open("best_overall/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to best_overall/summary.json", flush=True)


def main():
    tests = [
        "test_configs/test03.json",
        "test_configs/test04.json",
        "test_configs/test05.json",
        "test_configs/test06.json",
        "test_configs/test07.json",
        "test_configs/test08.json",
        "test_configs/test09.json",
        "test_configs/test10.json",
    ]

    total_start = time.time()

    for i, config in enumerate(tests):
        test_name = os.path.basename(config).replace(".json", "")
        print(f"\n{'#'*60}", flush=True)
        print(f"# Running {test_name} ({i+1}/{len(tests)})", flush=True)
        print(f"{'#'*60}", flush=True)

        try:
            run_test(config)
        except Exception as e:
            print(f"ERROR in {test_name}: {e}", flush=True)
            continue

    total_time = time.time() - total_start
    print(f"\n\nAll tests completed in {total_time/60:.0f} minutes", flush=True)

    # Compile best overall (including test01 and test02)
    compile_best_overall()


if __name__ == "__main__":
    main()
