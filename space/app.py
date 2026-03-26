"""
Banger Scorer — HuggingFace Space
Upload any song, get a banger score 0-10.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import gradio as gr
from transformers import AutoModel, AutoFeatureExtractor
from huggingface_hub import hf_hub_download


class BangerScorer(nn.Module):
    def __init__(self, input_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Load models on startup
print("Loading MERT encoder (this takes ~30s on CPU)...")
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "m-a-p/MERT-v1-330M", trust_remote_code=True
)
mert_model = AutoModel.from_pretrained(
    "m-a-p/MERT-v1-330M", trust_remote_code=True
)
mert_model.eval()
print("MERT loaded.")

print("Loading banger scorer...")
model_path = hf_hub_download(
    repo_id="treadon/banger-scorer",
    filename="scorer_model.pt",
)
scorer = BangerScorer(input_dim=1024)
scorer.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
scorer.eval()
print("Scorer loaded.")

GENRE_BENCHMARKS = {
    "EDM": {"mean": 3.71, "best": 5.29},
    "Punjabi/Bhangra": {"mean": 3.77, "best": 4.26},
    "Bollywood": {"mean": 3.53, "best": 4.38},
    "C-Pop": {"mean": 3.20, "best": 4.47},
    "Latin/Reggaeton": {"mean": 3.19, "best": 3.90},
    "Pop/Dance": {"mean": 3.05, "best": 4.31},
    "Rock/Alt": {"mean": 3.03, "best": 3.66},
    "Hip Hop": {"mean": 2.92, "best": 3.38},
    "Acoustic/Folk": {"mean": 2.63, "best": 3.31},
    "R&B/Soul": {"mean": 2.62, "best": 3.21},
}

FMA_PERCENTILES = [
    (1.0, 5), (1.5, 10), (2.0, 20), (2.5, 35),
    (3.0, 45), (3.5, 58), (4.0, 70), (4.5, 78),
    (5.0, 84), (5.5, 89), (6.0, 93), (6.5, 96),
    (7.0, 97.5), (8.0, 99), (9.0, 99.8), (10.0, 100),
]


def get_percentile(score):
    for threshold, pct in FMA_PERCENTILES:
        if score <= threshold:
            return pct
    return 100


def get_rating(score):
    if score >= 5.0:
        return "Certified Banger", "🔥🔥🔥"
    elif score >= 4.0:
        return "Strong Track", "🔥🔥"
    elif score >= 3.5:
        return "Above Average", "🔥"
    elif score >= 3.0:
        return "Average", "👍"
    elif score >= 2.5:
        return "Below Average", "😐"
    elif score >= 2.0:
        return "Weak", "👎"
    else:
        return "Not a Banger", "💀"


def score_song(audio_path):
    if audio_path is None:
        return "No audio uploaded", "", ""

    try:
        # Load audio
        wav, _ = librosa.load(audio_path, sr=feature_extractor.sampling_rate, mono=True)

        # Truncate to 30s
        max_samples = feature_extractor.sampling_rate * 30
        if len(wav) > max_samples:
            wav = wav[:max_samples]

        # MERT encoding
        inputs = feature_extractor(
            wav, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = mert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            score = scorer(embedding).item()

        score = max(0, min(10, score))
        rating, emoji = get_rating(score)
        percentile = get_percentile(score)

        # Build result
        score_text = f"# {emoji} {score:.1f} / 10\n### {rating}"

        # Context
        context_lines = [
            f"**Percentile:** Better than {percentile:.0f}% of real music in FMA dataset",
            "",
            "**How this compares to our AI-generated test songs:**",
            "",
            "| Genre | Mean Score | Your Song |",
            "|-------|-----------|-----------|",
        ]
        for genre, stats in GENRE_BENCHMARKS.items():
            marker = " ◀" if abs(score - stats["mean"]) < 0.3 else ""
            context_lines.append(
                f"| {genre} | {stats['mean']:.2f} | {marker} |"
            )

        context_text = "\n".join(context_lines)

        details = (
            f"**Score:** {score:.2f}/10\n\n"
            f"**Rating:** {rating}\n\n"
            f"**FMA Percentile:** {percentile:.0f}th — "
            f"{'above average' if percentile > 50 else 'below average'} "
            f"compared to 8,000 real songs\n\n"
            f"*Scored using MERT-v1-330M embeddings + trained MLP. "
            f"The scorer was trained on FMA play counts and favors "
            f"high-energy, rhythmically driven music. "
            f"Scores are most useful for relative ranking, not absolute judgment.*"
        )

        return score_text, context_text, details

    except Exception as e:
        return f"Error: {str(e)}", "", ""


# Build the Gradio interface
with gr.Blocks(
    title="Banger Scorer",
    theme=gr.themes.Base(primary_hue="orange"),
) as demo:
    gr.Markdown(
        """
        # 🔥 Banger Scorer
        ### Rate any song 0-10 on banger potential

        Upload an MP3 or WAV and the model will score it using
        [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) audio embeddings + a trained MLP.

        *Trained on 8,000 songs from the Free Music Archive. Runs on CPU — scoring takes ~30 seconds.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload a song",
                type="filepath",
                sources=["upload", "microphone"],
            )
            score_btn = gr.Button("Score it!", variant="primary", size="lg")

        with gr.Column(scale=1):
            score_output = gr.Markdown(label="Score")
            context_output = gr.Markdown(label="Context")

    details_output = gr.Markdown(label="Details")

    score_btn.click(
        fn=score_song,
        inputs=[audio_input],
        outputs=[score_output, context_output, details_output],
    )

    gr.Markdown(
        """
        ---
        **How it works:** Your audio is encoded by MERT (a 330M-parameter music understanding model)
        into a 1024-dimensional embedding, then scored by a tiny MLP trained on music popularity data.

        **Limitations:** The scorer favors high-energy electronic music and underrates
        mellow genres (R&B, folk). Use scores for relative comparison, not absolute judgment.

        **Links:**
        [GitHub](https://github.com/treadon/banger-scorer) |
        [Blog Post](https://riteshkhanna.com/banger-scorer) |
        [Dataset: 230 AI Songs](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs) |
        [Dataset: MERT Embeddings](https://huggingface.co/datasets/treadon/fma-mert-embeddings) |
        [Model](https://huggingface.co/treadon/banger-scorer)
        """
    )

demo.launch()
