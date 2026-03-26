"""
Generate scatter plots for all test results:
1. One plot per test (score vs song index, colored by caption style)
2. One global plot with all tests color-coded
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150

TEST_LABELS = {
    "test01": "Hip Hop",
    "test02": "Pop/Dance",
    "test03": "R&B/Soul",
    "test04": "Latin/Reggaeton",
    "test05": "Bollywood",
    "test06": "Punjabi/Bhangra",
    "test07": "C-Pop",
    "test08": "Rock/Alt",
    "test09": "EDM",
    "test10": "Acoustic/Folk",
}

COLORS = {
    "test01": "#e6194b",
    "test02": "#3cb44b",
    "test03": "#4363d8",
    "test04": "#f58231",
    "test05": "#911eb4",
    "test06": "#42d4f4",
    "test07": "#f032e6",
    "test08": "#bfef45",
    "test09": "#fabed4",
    "test10": "#469990",
}


def load_all_results():
    all_data = {}
    for results_file in sorted(glob.glob("test*/output/results.json")):
        test_dir = results_file.split("/")[0]
        with open(results_file) as f:
            data = json.load(f)

        test_id = data.get("test_id", test_dir)
        test_name = data.get("test_name", TEST_LABELS.get(test_id, test_id))
        results = data.get("results", [])

        all_data[test_id] = {
            "name": test_name,
            "label": TEST_LABELS.get(test_id, test_name),
            "results": results,
        }
    return all_data


def plot_per_test(all_data, output_dir="plots"):
    """One scatter plot per test: score vs BPM, sized by seed."""
    os.makedirs(output_dir, exist_ok=True)

    for test_id, data in all_data.items():
        results = data["results"]
        if not results:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        scores = [r["score"] for r in results]
        bpms = [r["bpm"] for r in results]

        # Color by caption (unique captions)
        captions = list(set(r.get("caption", "")[:40] for r in results))
        caption_colors = plt.cm.Set2(np.linspace(0, 1, len(captions)))
        caption_map = {c: caption_colors[i] for i, c in enumerate(captions)}

        for r in results:
            c_key = r.get("caption", "")[:40]
            color = caption_map.get(c_key, "gray")
            ax.scatter(r["bpm"], r["score"], c=[color], s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

        ax.set_xlabel("BPM")
        ax.set_ylabel("Banger Score (0-10)")
        ax.set_title(f"{test_id.upper()}: {data['label']} — Score vs BPM\n"
                     f"Range: {min(scores):.2f}–{max(scores):.2f} | Mean: {np.mean(scores):.2f}")
        ax.set_ylim(0, 6)
        ax.axhline(y=np.mean(scores), color="red", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(scores):.2f}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add caption legend
        legend_elements = []
        for i, c in enumerate(captions):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=caption_colors[i],
                                              markersize=8, label=c[:35] + "..."))
        if len(legend_elements) <= 5:
            ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{test_id}_scatter.png"))
        plt.close()
        print(f"  Saved {test_id}_scatter.png")


def plot_global(all_data, output_dir="plots"):
    """Global scatter: all tests on one plot, color-coded by genre."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    for test_id, data in all_data.items():
        results = data["results"]
        if not results:
            continue

        scores = [r["score"] for r in results]
        # X axis: just index within each test spread out
        x_base = list(all_data.keys()).index(test_id)

        color = COLORS.get(test_id, "gray")
        label = data["label"]

        # Jitter x position within genre band
        x_positions = [x_base + np.random.uniform(-0.3, 0.3) for _ in scores]

        ax.scatter(x_positions, scores, c=color, s=60, alpha=0.7,
                   edgecolors="black", linewidth=0.3, label=f"{label} ({min(scores):.1f}–{max(scores):.1f})")

        # Mean line
        ax.plot([x_base - 0.4, x_base + 0.4], [np.mean(scores), np.mean(scores)],
                color=color, linewidth=2, alpha=0.8)

    ax.set_xticks(range(len(all_data)))
    ax.set_xticklabels([d["label"] for d in all_data.values()], rotation=45, ha="right")
    ax.set_ylabel("Banger Score (0-10)")
    ax.set_title("Banger Scores Across All 200 Songs — 10 Genres\n"
                 "Horizontal lines = genre mean | Dots = individual songs")
    ax.set_ylim(0, 6)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_scatter.png"))
    plt.close()
    print("  Saved global_scatter.png")

    # Also make a box plot version
    fig, ax = plt.subplots(figsize=(14, 8))

    box_data = []
    box_labels = []
    box_colors = []
    for test_id, data in all_data.items():
        scores = [r["score"] for r in data["results"]]
        if scores:
            box_data.append(scores)
            box_labels.append(data["label"])
            box_colors.append(COLORS.get(test_id, "gray"))

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Banger Score (0-10)")
    ax.set_title("Score Distribution by Genre — Box Plot\n"
                 "Red diamond = mean | Line = median | Whiskers = 1.5×IQR")
    ax.set_ylim(0, 6)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_boxplot.png"))
    plt.close()
    print("  Saved global_boxplot.png")


def main():
    print("Loading results...", flush=True)
    all_data = load_all_results()
    print(f"Found {len(all_data)} tests\n", flush=True)

    print("Generating per-test scatter plots...", flush=True)
    plot_per_test(all_data)

    print("\nGenerating global plots...", flush=True)
    plot_global(all_data)

    print("\nDone! All plots in plots/", flush=True)


if __name__ == "__main__":
    main()
