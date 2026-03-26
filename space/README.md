---
title: Banger Scorer
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Rate any song 0-10 on banger potential
---

# Banger Scorer

Upload any song (MP3/WAV) and get a banger score from 0-10.

Uses [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M) audio embeddings + a trained MLP scorer.

Trained on 8,000 songs from the [Free Music Archive](https://github.com/mdeff/fma).

[GitHub](https://github.com/treadon/banger-scorer) | [Blog](https://riteshkhanna.com/blog/banger-scorer) | [Full Writeup](https://github.com/treadon/banger-scorer/blob/main/WRITEUP.md)
