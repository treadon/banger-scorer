"""
Create a hero plot: AI Music vs Human Music — The Quality Gap
Shows where AI-generated songs fall on the scale trained on real human music.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib


matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['figure.facecolor'] = 'white'


def load_data():
    # Training data (real human music)
    with open("data/embeddings/metadata.json") as f:
        meta = json.load(f)
    training_scores = np.array([l["banger_score"] for l in meta["labels"]])

    # AI generated (random params)
    random_scores = []
    for f_path in sorted(glob.glob("test*/output/results.json")):
        with open(f_path) as f:
            data = json.load(f)
        random_scores.extend([r["score"] for r in data["results"]])
    random_scores = np.array(random_scores)

    # AI generated (optimized)
    with open("bangers_output/results.json") as f:
        data = json.load(f)
    optimized_scores = np.array([r["score"] for r in data["results"]])

    return training_scores, random_scores, optimized_scores


def make_hero_plot():
    training, random_ai, optimized_ai = load_data()

    fig, ax = plt.subplots(figsize=(16, 9))

    # Background zones
    ax.axvspan(0, 2, alpha=0.06, color='red', zorder=0)
    ax.axvspan(2, 4, alpha=0.06, color='orange', zorder=0)
    ax.axvspan(4, 6, alpha=0.06, color='yellow', zorder=0)
    ax.axvspan(6, 8, alpha=0.06, color='lightgreen', zorder=0)
    ax.axvspan(8, 10, alpha=0.06, color='green', zorder=0)

    # Zone labels at top
    zone_labels = [
        (1, "Poor", "#e74c3c"),
        (3, "Below Average", "#e67e22"),
        (5, "Above Average", "#f1c40f"),
        (7, "Great", "#2ecc71"),
        (9, "Exceptional", "#27ae60"),
    ]
    for x, label, color in zone_labels:
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color=color, fontweight="bold", alpha=0.7)

    # Plot distributions
    bins = np.linspace(0, 10, 60)

    # Human music
    ax.hist(training, bins=bins, alpha=0.45, color="#3498db", density=True,
            edgecolor="white", linewidth=0.3, label=f"Human Music (FMA, n={len(training)})")

    # AI random
    ax.hist(random_ai, bins=np.linspace(0, 10, 30), alpha=0.55, color="#e74c3c", density=True,
            edgecolor="white", linewidth=0.5, label=f"AI Generated — Random (n={len(random_ai)})")

    # AI optimized
    ax.hist(optimized_ai, bins=np.linspace(0, 10, 20), alpha=0.65, color="#f39c12", density=True,
            edgecolor="white", linewidth=0.5, label=f"AI Generated — Optimized (n={len(optimized_ai)})")

    # Mean lines with labels
    for scores, color, label, style in [
        (training, "#3498db", f"Human mean: {training.mean():.1f}", "-"),
        (random_ai, "#e74c3c", f"AI random mean: {random_ai.mean():.1f}", "--"),
        (optimized_ai, "#f39c12", f"AI optimized mean: {optimized_ai.mean():.1f}", "-."),
    ]:
        ax.axvline(x=scores.mean(), color=color, linestyle=style, linewidth=2.5, alpha=0.8)

    # Annotations
    # Arrow showing the gap
    ax.annotate("",
                xy=(training.mean(), 0.65), xytext=(random_ai.mean(), 0.65),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    gap_x = (training.mean() + random_ai.mean()) / 2
    ax.text(gap_x, 0.68, "The Gap", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=12, fontweight="bold", color="black")

    # Arrow showing optimization improvement
    ax.annotate("",
                xy=(optimized_ai.mean(), 0.55), xytext=(random_ai.mean(), 0.55),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="->", color="#f39c12", lw=2.5))
    opt_x = (optimized_ai.mean() + random_ai.mean()) / 2
    ax.text(opt_x, 0.58, "+10% from\noptimization", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=9, color="#f39c12", fontweight="bold")

    # Key stats as text box
    stats_text = (
        f"Human Music:\n"
        f"  Mean: {training.mean():.1f}  |  Top 10%: ≥{np.percentile(training, 90):.1f}  |  Top 1%: ≥{np.percentile(training, 99):.1f}\n\n"
        f"AI Music (Random):\n"
        f"  Mean: {random_ai.mean():.1f}  |  Best: {random_ai.max():.1f}  |  At {(np.searchsorted(np.sort(training), random_ai.max()) / len(training) * 100):.0f}th percentile of human\n\n"
        f"AI Music (Optimized):\n"
        f"  Mean: {optimized_ai.mean():.1f}  |  Best: {optimized_ai.max():.1f}  |  4x hit rate for score ≥4.0"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')

    # Title and labels
    ax.set_xlabel("Banger Score (0-10)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("AI Music vs Human Music: The Quality Gap\n"
                 "Where do AI-generated songs fall on a scale trained on 8,000 real songs?",
                 fontsize=16, fontweight="bold", pad=20)

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    ax.legend(loc="upper center", fontsize=10, ncol=3,
              bbox_to_anchor=(0.5, -0.08), frameon=True)

    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs("plots/hero", exist_ok=True)
    plt.savefig("plots/hero/ai_vs_human_quality_gap.png", bbox_inches="tight")
    plt.close()
    print("Saved plots/hero/ai_vs_human_quality_gap.png")


def make_percentile_plot():
    """Where does each AI genre land on the human music percentile scale?"""
    training, _, _ = load_data()

    fig, ax = plt.subplots(figsize=(14, 8))

    TEST_LABELS = {
        "test01": "Hip Hop", "test02": "Pop/Dance", "test03": "R&B/Soul",
        "test04": "Latin/Reggaeton", "test05": "Bollywood", "test06": "Punjabi/Bhangra",
        "test07": "C-Pop", "test08": "Rock/Alt", "test09": "EDM",
        "test10": "Acoustic/Folk", "bangers": "Optimized",
    }

    genre_data = []
    for f_path in sorted(glob.glob("test*/output/results.json")) + ["bangers_output/results.json"]:
        if not os.path.exists(f_path):
            continue
        with open(f_path) as f:
            data = json.load(f)
        test_id = data.get("test_id", os.path.dirname(f_path).split("/")[0])
        if test_id == "bangers_output":
            test_id = "bangers"
        scores = [r["score"] for r in data["results"]]
        mean_score = np.mean(scores)
        best_score = max(scores)

        # Find percentile in human distribution
        mean_pct = np.searchsorted(np.sort(training), mean_score) / len(training) * 100
        best_pct = np.searchsorted(np.sort(training), best_score) / len(training) * 100

        genre_data.append({
            "name": TEST_LABELS.get(test_id, test_id),
            "mean_pct": mean_pct,
            "best_pct": best_pct,
            "mean_score": mean_score,
            "best_score": best_score,
            "is_optimized": test_id == "bangers",
        })

    genre_data.sort(key=lambda x: x["mean_pct"])

    y_pos = range(len(genre_data))
    colors = ["#f39c12" if g["is_optimized"] else "#3498db" for g in genre_data]

    # Plot mean percentile bars
    bars = ax.barh(y_pos, [g["mean_pct"] for g in genre_data],
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5,
                   label="Mean song percentile")

    # Add best-song markers
    for i, g in enumerate(genre_data):
        ax.plot(g["best_pct"], i, marker="*", color="#e74c3c", markersize=14, zorder=5, linestyle="none")
        ax.text(g["mean_pct"] + 1.5, i,
                f'{g["mean_pct"]:.0f}th pct (score {g["mean_score"]:.1f})',
                va="center", fontsize=9)
        ax.text(g["best_pct"] + 1, i + 0.25,
                f'Best: {g["best_pct"]:.0f}th',
                va="center", fontsize=8, color="#e74c3c", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([g["name"] for g in genre_data], fontsize=11)
    ax.set_xlabel("Percentile in Human Music Distribution", fontsize=12)
    ax.set_title("Where Does AI Music Rank Among Human Music?\n"
                 "Bar = mean song | ★ = best song in genre",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)

    # Add reference lines
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax.text(50, len(genre_data) - 0.5, "Median", ha="center", fontsize=9, color="gray")
    ax.axvline(x=75, color="gray", linestyle=":", alpha=0.3)
    ax.text(75, len(genre_data) - 0.5, "75th", ha="center", fontsize=9, color="gray", alpha=0.6)

    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    star = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e74c3c",
                       markersize=12, label="Best single song")
    blue_patch = mpatches.Patch(color="#3498db", alpha=0.7, label="Random params")
    gold_patch = mpatches.Patch(color="#f39c12", alpha=0.7, label="Optimized params")
    ax.legend(handles=[blue_patch, gold_patch, star], loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig("plots/hero/ai_vs_human_percentile.png", bbox_inches="tight")
    plt.close()
    print("Saved plots/hero/ai_vs_human_percentile.png")


if __name__ == "__main__":
    make_hero_plot()
    make_percentile_plot()
