"""
Generate comprehensive plots for the banger-scorer project.
Organized into folders by category.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['figure.facecolor'] = 'white'

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

GENRE_COLORS = {
    "test01": "#e6194b",
    "test02": "#3cb44b",
    "test03": "#4363d8",
    "test04": "#f58231",
    "test05": "#911eb4",
    "test06": "#42d4f4",
    "test07": "#f032e6",
    "test08": "#9A6324",
    "test09": "#800000",
    "test10": "#469990",
}


def load_all_results():
    all_data = {}
    for results_file in sorted(glob.glob("test*/output/results.json")):
        test_dir = results_file.split("/")[0]
        with open(results_file) as f:
            data = json.load(f)
        test_id = data.get("test_id", test_dir)
        all_data[test_id] = {
            "name": TEST_LABELS.get(test_id, test_id),
            "results": data.get("results", []),
        }
    return all_data


def load_training_data():
    """Load MERT embedding metadata for training distribution plots."""
    meta_path = "data/embeddings/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta
    return None


# ============================================================
# 1. OVERVIEW PLOTS (plots/overview/)
# ============================================================

def plot_global_scatter(all_data, out_dir):
    """All 200 songs, color-coded by genre."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))

    for test_id, data in all_data.items():
        results = data["results"]
        if not results:
            continue
        scores = [r["score"] for r in results]
        x_base = list(all_data.keys()).index(test_id)
        color = GENRE_COLORS.get(test_id, "gray")
        x_positions = [x_base + np.random.RandomState(i).uniform(-0.3, 0.3) for i in range(len(scores))]
        ax.scatter(x_positions, scores, c=color, s=60, alpha=0.7,
                   edgecolors="black", linewidth=0.3,
                   label=f"{data['name']} ({min(scores):.1f}–{max(scores):.1f})")
        ax.plot([x_base - 0.4, x_base + 0.4], [np.mean(scores), np.mean(scores)],
                color=color, linewidth=2.5, alpha=0.8)

    ax.set_xticks(range(len(all_data)))
    ax.set_xticklabels([d["name"] for d in all_data.values()], rotation=45, ha="right")
    ax.set_ylabel("Banger Score (0-10)")
    ax.set_title("Banger Scores Across All 200 Songs — 10 Genres\nHorizontal lines = genre mean")
    ax.set_ylim(0, 6)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_scatter.png"))
    plt.close()


def plot_global_boxplot(all_data, out_dir):
    """Box plot comparing distributions."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))

    box_data = []
    box_labels = []
    box_colors = []
    for test_id, data in all_data.items():
        scores = [r["score"] for r in data["results"]]
        if scores:
            box_data.append(scores)
            box_labels.append(data["name"])
            box_colors.append(GENRE_COLORS.get(test_id, "gray"))

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Banger Score (0-10)")
    ax.set_title("Score Distribution by Genre — Box Plot\nRed diamond = mean | Line = median")
    ax.set_ylim(0, 6)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_boxplot.png"))
    plt.close()


def plot_genre_ranking_bar(all_data, out_dir):
    """Horizontal bar chart: genres ranked by mean score."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    genre_stats = []
    for test_id, data in all_data.items():
        scores = [r["score"] for r in data["results"]]
        if scores:
            genre_stats.append({
                "name": data["name"],
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores),
                "color": GENRE_COLORS.get(test_id, "gray"),
            })

    genre_stats.sort(key=lambda x: x["mean"])

    y_pos = range(len(genre_stats))
    means = [g["mean"] for g in genre_stats]
    stds = [g["std"] for g in genre_stats]
    colors = [g["color"] for g in genre_stats]
    names = [g["name"] for g in genre_stats]

    bars = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7,
                   edgecolor="black", linewidth=0.5, capsize=4)

    # Add min-max markers
    for i, g in enumerate(genre_stats):
        ax.plot(g["min"], i, "v", color="black", markersize=6, alpha=0.5)
        ax.plot(g["max"], i, "^", color="black", markersize=6, alpha=0.5)
        ax.text(g["mean"] + g["std"] + 0.15, i, f'{g["mean"]:.2f}', va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Banger Score (0-10)")
    ax.set_title("Genre Ranking by Mean Score\nBar = mean ± std | ▼ = min | ▲ = max")
    ax.set_xlim(0, 6)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "genre_ranking.png"))
    plt.close()


def plot_top_bottom_comparison(all_data, out_dir):
    """Bar chart: top 10 vs bottom 10 songs."""
    os.makedirs(out_dir, exist_ok=True)

    all_songs = []
    for test_id, data in all_data.items():
        for r in data["results"]:
            r["genre"] = data["name"]
            r["test_id"] = test_id
            all_songs.append(r)

    all_songs.sort(key=lambda x: x["score"], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top 10
    top = all_songs[:10]
    y_pos = range(len(top))
    colors = [GENRE_COLORS.get(t["test_id"], "gray") for t in top]
    labels = [f'{t["genre"]} | bpm={t["bpm"]} {t["key"]}' for t in top]
    scores = [t["score"] for t in top]

    ax1.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Banger Score")
    ax1.set_title("Top 10 Songs")
    ax1.set_xlim(0, 6)
    ax1.invert_yaxis()
    for i, s in enumerate(scores):
        ax1.text(s + 0.05, i, f"{s:.2f}", va="center", fontsize=9)

    # Bottom 10
    bottom = all_songs[-10:]
    y_pos = range(len(bottom))
    colors = [GENRE_COLORS.get(t["test_id"], "gray") for t in bottom]
    labels = [f'{t["genre"]} | bpm={t["bpm"]} {t["key"]}' for t in bottom]
    scores = [t["score"] for t in bottom]

    ax2.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Banger Score")
    ax2.set_title("Bottom 10 Songs")
    ax2.set_xlim(0, 6)
    ax2.invert_yaxis()
    for i, s in enumerate(scores):
        ax2.text(s + 0.05, i, f"{s:.2f}", va="center", fontsize=9)

    plt.suptitle("Best vs Worst Songs Across All 200", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_vs_bottom.png"), bbox_inches="tight")
    plt.close()


# ============================================================
# 2. PER-GENRE PLOTS (plots/per_genre/)
# ============================================================

def plot_per_genre_scatter(all_data, out_dir):
    """Score vs BPM for each genre, colored by caption style."""
    os.makedirs(out_dir, exist_ok=True)

    for test_id, data in all_data.items():
        results = data["results"]
        if not results:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        scores = [r["score"] for r in results]
        bpms = [r["bpm"] for r in results]

        captions = list(set(r.get("caption", "")[:40] for r in results))
        cmap = plt.cm.Set2(np.linspace(0, 1, max(len(captions), 1)))
        caption_map = {c: cmap[i] for i, c in enumerate(captions)}

        legend_handles = []
        for i, c in enumerate(captions):
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=cmap[i], markersize=8,
                                             label=c[:35] + "..."))

        for r in results:
            c_key = r.get("caption", "")[:40]
            ax.scatter(r["bpm"], r["score"], c=[caption_map.get(c_key, "gray")],
                       s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

        ax.set_xlabel("BPM")
        ax.set_ylabel("Banger Score")
        ax.set_title(f"{test_id.upper()}: {data['name']}\n"
                     f"Range: {min(scores):.2f}–{max(scores):.2f} | Mean: {np.mean(scores):.2f}")
        ax.set_ylim(0, 6)
        ax.axhline(y=np.mean(scores), color="red", linestyle="--", alpha=0.5)
        if len(legend_handles) <= 5:
            ax.legend(handles=legend_handles, loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{test_id}_score_vs_bpm.png"))
        plt.close()


def plot_per_genre_key_heatmap(all_data, out_dir):
    """Heatmap: average score by BPM × Key for each genre."""
    os.makedirs(out_dir, exist_ok=True)

    for test_id, data in all_data.items():
        results = data["results"]
        if not results:
            continue

        # Collect unique BPMs and keys
        bpms = sorted(set(r["bpm"] for r in results))
        keys = sorted(set(r["key"] for r in results))

        if len(bpms) < 2 or len(keys) < 2:
            continue

        grid = np.full((len(keys), len(bpms)), np.nan)
        counts = np.zeros((len(keys), len(bpms)))

        for r in results:
            bi = bpms.index(r["bpm"])
            ki = keys.index(r["key"])
            if np.isnan(grid[ki, bi]):
                grid[ki, bi] = r["score"]
            else:
                grid[ki, bi] = (grid[ki, bi] * counts[ki, bi] + r["score"]) / (counts[ki, bi] + 1)
            counts[ki, bi] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=1.5, vmax=5.5)
        ax.set_xticks(range(len(bpms)))
        ax.set_xticklabels(bpms)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys)
        ax.set_xlabel("BPM")
        ax.set_ylabel("Key")
        ax.set_title(f"{test_id.upper()}: {data['name']} — Score by BPM × Key")

        for i in range(len(keys)):
            for j in range(len(bpms)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if grid[i, j] < 2.5 or grid[i, j] > 4.5 else "black")

        plt.colorbar(im, ax=ax, label="Banger Score")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{test_id}_bpm_key_heatmap.png"))
        plt.close()


# ============================================================
# 3. ANALYSIS PLOTS (plots/analysis/)
# ============================================================

def plot_bpm_vs_score_global(all_data, out_dir):
    """All songs: BPM vs score, colored by genre."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    for test_id, data in all_data.items():
        results = data["results"]
        if not results:
            continue
        bpms = [r["bpm"] for r in results]
        scores = [r["score"] for r in results]
        ax.scatter(bpms, scores, c=GENRE_COLORS.get(test_id, "gray"), s=50, alpha=0.6,
                   edgecolors="black", linewidth=0.3, label=data["name"])

    ax.set_xlabel("BPM")
    ax.set_ylabel("Banger Score")
    ax.set_title("BPM vs Banger Score — All 200 Songs")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bpm_vs_score_global.png"))
    plt.close()


def plot_key_analysis(all_data, out_dir):
    """Average score by musical key across all genres."""
    os.makedirs(out_dir, exist_ok=True)

    key_scores = defaultdict(list)
    for test_id, data in all_data.items():
        for r in data["results"]:
            key_scores[r["key"]].append(r["score"])

    keys = sorted(key_scores.keys(), key=lambda k: np.mean(key_scores[k]), reverse=True)
    means = [np.mean(key_scores[k]) for k in keys]
    stds = [np.std(key_scores[k]) for k in keys]
    counts = [len(key_scores[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color major vs minor differently
    colors = ["#4CAF50" if "major" in k else "#2196F3" for k in keys]
    bars = ax.bar(range(len(keys)), means, yerr=stds, color=colors, alpha=0.7,
                  edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Banger Score")
    ax.set_title("Score by Musical Key (All Genres Combined)\nGreen = major | Blue = minor")
    ax.set_ylim(0, 5)

    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(i, m + stds[i] + 0.1, f"n={c}", ha="center", fontsize=7)

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "key_analysis.png"))
    plt.close()


def plot_caption_style_analysis(all_data, out_dir):
    """Score distribution by caption style keywords."""
    os.makedirs(out_dir, exist_ok=True)

    # Extract keyword categories from captions
    keyword_scores = defaultdict(list)
    keywords_to_check = [
        ("electronic", ["electronic", "synth", "EDM", "house", "techno"]),
        ("acoustic", ["acoustic", "folk", "fingerpicked", "campfire"]),
        ("dark/menacing", ["dark", "menacing", "gritty", "aggressive"]),
        ("upbeat/bright", ["upbeat", "bright", "feel-good", "euphoric", "sunny"]),
        ("bass-heavy", ["bass", "808", "heavy bass", "boom bap"]),
        ("orchestral", ["orchestral", "strings", "cinematic"]),
        ("traditional", ["dhol", "tumbi", "bhangra", "Bollywood"]),
        ("jazzy", ["jazzy", "jazz", "piano loops", "Rhodes"]),
    ]

    for test_id, data in all_data.items():
        for r in data["results"]:
            caption = r.get("caption", "").lower()
            for category, kws in keywords_to_check:
                if any(kw.lower() in caption for kw in kws):
                    keyword_scores[category].append(r["score"])

    categories = sorted(keyword_scores.keys(), key=lambda k: np.mean(keyword_scores[k]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    bp_data = [keyword_scores[c] for c in categories]
    bp = ax.boxplot(bp_data, tick_labels=categories, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5})

    cmap = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Banger Score")
    ax.set_title("Score by Caption Style Keywords\n(songs can appear in multiple categories)")
    ax.set_ylim(0, 6)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    for i, c in enumerate(categories):
        ax.text(i + 1, -0.3, f"n={len(keyword_scores[c])}", ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "caption_style_analysis.png"))
    plt.close()


def plot_score_histogram_global(all_data, out_dir):
    """Histogram of all 200 scores."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    all_scores = []
    for data in all_data.values():
        all_scores.extend([r["score"] for r in data["results"]])

    ax.hist(all_scores, bins=25, color="#4CAF50", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=np.mean(all_scores), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(all_scores):.2f}")
    ax.axvline(x=np.median(all_scores), color="blue", linestyle="--", linewidth=2,
               label=f"Median: {np.median(all_scores):.2f}")
    ax.set_xlabel("Banger Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — All {len(all_scores)} Generated Songs")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_histogram.png"))
    plt.close()


def plot_major_vs_minor(all_data, out_dir):
    """Compare major vs minor key scores."""
    os.makedirs(out_dir, exist_ok=True)

    major_scores = []
    minor_scores = []
    for data in all_data.values():
        for r in data["results"]:
            if "major" in r["key"]:
                major_scores.append(r["score"])
            elif "minor" in r["key"]:
                minor_scores.append(r["score"])

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot([major_scores, minor_scores], tick_labels=["Major Keys", "Minor Keys"],
                    patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6})
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#2196F3")
    bp["boxes"][1].set_alpha(0.6)

    ax.set_ylabel("Banger Score")
    ax.set_title(f"Major vs Minor Keys\nMajor: n={len(major_scores)}, mean={np.mean(major_scores):.2f} | "
                 f"Minor: n={len(minor_scores)}, mean={np.mean(minor_scores):.2f}")
    ax.set_ylim(0, 6)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "major_vs_minor.png"))
    plt.close()


# ============================================================
# 4. TRAINING DATA PLOTS (plots/training/)
# ============================================================

def plot_training_distribution(meta, out_dir):
    """Distribution of banger scores in FMA training data."""
    os.makedirs(out_dir, exist_ok=True)

    scores = [l["banger_score"] for l in meta["labels"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(scores, bins=50, color="#FF9800", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axvline(x=np.mean(scores), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(scores):.2f}")
    ax1.axvline(x=np.median(scores), color="blue", linestyle="--", linewidth=2,
                label=f"Median: {np.median(scores):.2f}")
    ax1.set_xlabel("Banger Score (0-10)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"FMA Training Data Distribution\n{len(scores)} tracks")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # CDF
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    ax2.plot(sorted_scores, cdf, color="#FF9800", linewidth=2)
    ax2.axhline(y=50, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(x=np.median(scores), color="blue", linestyle="--", alpha=0.5)
    # Mark key percentiles
    for pct in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(scores, pct)
        ax2.plot(val, pct, "ro", markersize=5)
        ax2.annotate(f"  {pct}th: {val:.1f}", (val, pct), fontsize=8)

    ax2.set_xlabel("Banger Score (0-10)")
    ax2.set_ylabel("Cumulative % of Tracks")
    ax2.set_title("CDF — Where Do Scores Fall?\nOnly 8.4% score ≥5, only 0.6% score ≥7")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_distribution.png"))
    plt.close()


def plot_training_genre_distribution(meta, out_dir):
    """Score distribution per genre in training data."""
    os.makedirs(out_dir, exist_ok=True)

    genre_scores = defaultdict(list)
    for l in meta["labels"]:
        genre_scores[l["genre"]].append(l["banger_score"])

    genres = sorted(genre_scores.keys(), key=lambda g: np.mean(genre_scores[g]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot([genre_scores[g] for g in genres], tick_labels=genres,
                    patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5})

    cmap = plt.cm.Set3(np.linspace(0, 1, len(genres)))
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Banger Score (0-10)")
    ax.set_title("FMA Training Data: Score Distribution by Genre\nThis is what the scorer learned 'popular' means")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_genre_distribution.png"))
    plt.close()


def plot_generated_vs_training(all_data, meta, out_dir):
    """Compare generated song scores against training data distribution."""
    os.makedirs(out_dir, exist_ok=True)

    training_scores = [l["banger_score"] for l in meta["labels"]]
    generated_scores = []
    for data in all_data.values():
        generated_scores.extend([r["score"] for r in data["results"]])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(training_scores, bins=50, alpha=0.5, color="#FF9800", label=f"FMA Training (n={len(training_scores)})",
            density=True, edgecolor="black", linewidth=0.3)
    ax.hist(generated_scores, bins=20, alpha=0.6, color="#4CAF50", label=f"AI Generated (n={len(generated_scores)})",
            density=True, edgecolor="black", linewidth=0.5)

    ax.axvline(x=np.mean(training_scores), color="#FF9800", linestyle="--", linewidth=2,
               label=f"FMA mean: {np.mean(training_scores):.2f}")
    ax.axvline(x=np.mean(generated_scores), color="#4CAF50", linestyle="--", linewidth=2,
               label=f"Generated mean: {np.mean(generated_scores):.2f}")

    ax.set_xlabel("Banger Score (0-10)")
    ax.set_ylabel("Density")
    ax.set_title("AI-Generated Music vs Real Music (FMA)\nWhere do generated songs fall on the quality scale?")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "generated_vs_training.png"))
    plt.close()


# ============================================================
# 5. TIMING PLOTS (plots/timing/)
# ============================================================

def plot_generation_times(all_data, out_dir):
    """Generation time distribution."""
    os.makedirs(out_dir, exist_ok=True)

    all_times = []
    genre_times = {}
    for test_id, data in all_data.items():
        times = [r.get("gen_time", 0) for r in data["results"] if r.get("gen_time", 0) > 0]
        if times:
            all_times.extend(times)
            genre_times[data["name"]] = times

    if not all_times:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of all generation times
    ax1.hist(all_times, bins=20, color="#2196F3", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axvline(x=np.mean(all_times), color="red", linestyle="--",
                label=f"Mean: {np.mean(all_times):.0f}s")
    ax1.set_xlabel("Generation Time (seconds)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Song Generation Time Distribution\n{len(all_times)} songs")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Per-genre times
    if genre_times:
        labels = sorted(genre_times.keys(), key=lambda k: np.mean(genre_times[k]))
        bp = ax2.boxplot([genre_times[l] for l in labels], tick_labels=labels,
                         patch_artist=True, showmeans=True,
                         meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5})
        for patch in bp["boxes"]:
            patch.set_facecolor("#2196F3")
            patch.set_alpha(0.5)
        ax2.set_ylabel("Generation Time (seconds)")
        ax2.set_title("Generation Time by Genre")
        ax2.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "generation_times.png"))
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    # Clean old plots
    import shutil
    if os.path.exists("plots"):
        shutil.rmtree("plots")

    print("Loading data...", flush=True)
    all_data = load_all_results()
    meta = load_training_data()
    print(f"  {len(all_data)} tests loaded", flush=True)
    if meta:
        print(f"  {meta['num_tracks']} training tracks loaded", flush=True)

    print("\n1. Overview plots...", flush=True)
    plot_global_scatter(all_data, "plots/overview")
    plot_global_boxplot(all_data, "plots/overview")
    plot_genre_ranking_bar(all_data, "plots/overview")
    plot_top_bottom_comparison(all_data, "plots/overview")
    plot_score_histogram_global(all_data, "plots/overview")

    print("2. Per-genre plots...", flush=True)
    plot_per_genre_scatter(all_data, "plots/per_genre")
    plot_per_genre_key_heatmap(all_data, "plots/per_genre")

    print("3. Analysis plots...", flush=True)
    plot_bpm_vs_score_global(all_data, "plots/analysis")
    plot_key_analysis(all_data, "plots/analysis")
    plot_caption_style_analysis(all_data, "plots/analysis")
    plot_major_vs_minor(all_data, "plots/analysis")

    if meta:
        print("4. Training data plots...", flush=True)
        plot_training_distribution(meta, "plots/training")
        plot_training_genre_distribution(meta, "plots/training")
        plot_generated_vs_training(all_data, meta, "plots/training")

    print("5. Timing plots...", flush=True)
    plot_generation_times(all_data, "plots/timing")

    # Count total plots
    total = sum(len(files) for _, _, files in os.walk("plots") if files)
    print(f"\nDone! {total} plots generated in plots/", flush=True)
    for root, dirs, files in sorted(os.walk("plots")):
        for f in sorted(files):
            print(f"  {os.path.join(root, f)}")


if __name__ == "__main__":
    main()
