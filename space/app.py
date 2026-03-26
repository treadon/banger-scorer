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
print("Scorer loaded. Ready!")


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


def get_percentile(score):
    percentiles = [
        (1.0, 5), (1.5, 10), (2.0, 20), (2.5, 35),
        (3.0, 45), (3.5, 58), (4.0, 70), (4.5, 78),
        (5.0, 84), (5.5, 89), (6.0, 93), (7.0, 97.5),
        (8.0, 99), (10.0, 100),
    ]
    for threshold, pct in percentiles:
        if score <= threshold:
            return pct
    return 100


def score_song(audio):
    """Score a single audio file. Takes filepath from gr.Audio."""
    if audio is None:
        return "## Upload a song first"

    # gr.Audio with type="filepath" gives us a string path
    audio_path = audio

    try:
        wav, _ = librosa.load(audio_path, sr=feature_extractor.sampling_rate, mono=True)
        max_samples = feature_extractor.sampling_rate * 30
        if len(wav) > max_samples:
            wav = wav[:max_samples]

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

        result = f"""# {emoji} {score:.1f} / 10

### {rating}

---

**Better than {percentile:.0f}%** of real songs in the FMA dataset (8,000 tracks)

| How this compares | Mean Score |
|---|---|
| EDM (top AI genre) | 3.71 |
| Punjabi/Bhangra | 3.77 |
| Bollywood | 3.53 |
| Pop/Dance | 3.05 |
| Hip Hop | 2.92 |
| R&B/Soul | 2.62 |
| **Your song** | **{score:.2f}** |

---

*Scored using MERT-v1-330M embeddings + trained MLP. The scorer favors high-energy, rhythmically driven music. Best used for relative ranking, not absolute judgment.*
"""
        return result

    except Exception as e:
        return f"## Error\n\n{str(e)}"


with gr.Blocks(
    title="Banger Scorer",
    theme=gr.themes.Soft(primary_hue="orange"),
) as demo:
    gr.Markdown(
        """
        # 🔥 Banger Scorer
        ### Upload any song. Get a banger score from 0 to 10.

        Uses [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) (330M param music model) + a trained MLP.
        Runs on CPU — scoring takes ~30 seconds.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Drop a song here",
                type="filepath",
            )
            score_btn = gr.Button(
                "🔥 Score it!",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            result_output = gr.Markdown(
                value="## Upload a song and press Score it!",
                label="Result",
            )

    score_btn.click(
        fn=score_song,
        inputs=audio_input,
        outputs=result_output,
        api_name="score",
    )

    gr.Markdown(
        """
        ---
        **How it works:** Audio → MERT encoder (1024-dim embedding) → MLP scorer → 0-10 score

        **Limitations:** Favors high-energy electronic music. Underrates mellow genres. Use for relative comparison.

        [GitHub](https://github.com/treadon/banger-scorer) ·
        [Blog](https://riteshkhanna.com/blog/banger-scorer) ·
        [230 AI Songs](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs) ·
        [Model](https://huggingface.co/treadon/banger-scorer)
        """
    )

demo.launch()
