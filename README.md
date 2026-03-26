# Banger Scorer

**Can AI learn what makes a song a banger?**

AI music generation is technically competent but emotionally flat. Generate 50 songs and maybe 3-5 are actually good. This project builds an automated quality scorer that rates songs 0-10, then uses it to filter AI-generated music — keeping only the bangers.

We generated **230 songs across 10 genres in 6 languages**, scored them all, analyzed what the scorer favors, and used that data to **4x the hit rate** of producing high-scoring tracks.

Everything runs locally on a MacBook Pro. No cloud GPUs. No API costs.

## AI Music vs Human Music: The Quality Gap

We trained our scorer on 8,000 real songs from the Free Music Archive, then used it to evaluate 230 AI-generated songs. The results tell a clear story:

![AI vs Human Music Quality Gap](https://raw.githubusercontent.com/treadon/banger-scorer/main/plots/hero/ai_vs_human_quality_gap.png)

**AI music lives in the middle of the human distribution.** The average AI-generated song scores at the 42nd-67th percentile of real music depending on genre — not bad, but not exceptional. Real music has a long tail of truly great songs (scores 7-10) that AI can't yet reach.

The best AI songs — melodic techno, Punjabi bhangra, Bollywood dance numbers — reach the **94th percentile** of human music. But those are outliers. Most AI output is competent mediocrity: it sounds like music, has all the right parts, but doesn't move you.

![Where AI ranks among human music by genre](https://raw.githubusercontent.com/treadon/banger-scorer/main/plots/hero/ai_vs_human_percentile.png)

**The key insight: genre matters more than anything else.** AI excels at genres with repetitive structure and strong beats (EDM, Punjabi, Bollywood) and struggles with genres requiring emotional subtlety (R&B, acoustic folk). Choosing the right genre is the single biggest lever for AI music quality.

**The solution: don't make AI music better — make the filtering smarter.** By scoring and ranking AI output automatically, we can surface the best 10-20% and discard the rest. Data-driven parameter optimization then pushes the hit rate even higher.

![Score distribution across all genres](https://raw.githubusercontent.com/treadon/banger-scorer/main/plots/overview/global_scatter.png)

### Listen for Yourself

Hear the best and worst songs from each genre on HuggingFace (built-in audio player):

**[Browse and play all 230 songs on HuggingFace](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs?row=0)**

| | Song | Score | Genre | BPM | Key |
|---|------|-------|-------|-----|-----|
| #1 | [Best overall](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs/viewer?row=0&sort%5Bcolumn%5D=banger_score&sort%5Bdirection%5D=desc) | **5.29** | EDM | 130 | Eb minor |
| #2 | [Runner up](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs/viewer?row=1&sort%5Bcolumn%5D=banger_score&sort%5Bdirection%5D=desc) | **4.90** | EDM | 138 | F minor |
| #3 | [3rd place](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs/viewer?row=2&sort%5Bcolumn%5D=banger_score&sort%5Bdirection%5D=desc) | **4.87** | EDM | 126 | Eb minor |
| #last | [Worst overall](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs/viewer?row=0&sort%5Bcolumn%5D=banger_score&sort%5Bdirection%5D=asc) | **1.98** | Pop | 128 | G major |

> Sort by `banger_score` descending on HuggingFace to hear the best first, or ascending for the worst. Can you tell the difference?

## How It Works

```
Training:
  FMA audio (8K tracks) → MERT encoder (frozen, 330M params) → 1024-dim embeddings → MLP scorer → score 0-10

Inference:
  ACE-Step generates song → MERT encodes it → MLP scores it → keep or discard
```

1. **[MERT](https://huggingface.co/m-a-p/MERT-v1-330M)** — a 330M-parameter model pretrained on 160K hours of music. It "understands" music: rhythm, harmony, timbre, structure. We freeze it and use it as a feature extractor.

2. **MLP Scorer** — a tiny neural network (1024→512→256→128→1) trained in **30 seconds** on [FMA](https://github.com/mdeff/fma) play count data. Learns which MERT embeddings correspond to popular music.

3. **[ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5)** — open-source music generator that produces full songs with vocals in ~2 minutes on Apple Silicon.

4. **Generate → Score → Filter** — generate a batch of songs with varied parameters (BPM, key, style), score each one, keep the best.

## Results

### Scorer Performance

| Metric | Value | Target |
|--------|-------|--------|
| Test MAE | **0.858** | < 1.5 |
| Spearman Correlation | **0.468** | > 0.4 |
| Training Time | **30 seconds** | — |
| Model Size | **2.6 MB** | < 50 MB |

### 10-Genre Test (200 songs)

![Genre ranking by mean score](https://raw.githubusercontent.com/treadon/banger-scorer/main/plots/overview/genre_ranking.png)

| Genre | Language | Score Range | Mean |
|-------|----------|-------------|------|
| EDM | EN | 2.80–**5.29** | 3.71 |
| Punjabi/Bhangra | PA | 2.79–4.26 | **3.77** |
| Bollywood | HI | 2.70–4.38 | 3.53 |
| C-Pop | ZH | 2.13–4.47 | 3.20 |
| Latin/Reggaeton | ES | 2.36–3.90 | 3.19 |
| Pop/Dance | EN | 1.98–4.31 | 3.05 |
| Rock/Alt | EN | 2.02–3.66 | 3.03 |
| Hip Hop | EN | 2.52–3.38 | 2.92 |
| Acoustic/Folk | EN | 2.03–3.31 | 2.63 |
| R&B/Soul | EN | 2.14–3.21 | 2.62 |

### Data-Driven Optimization

Using the analysis to select only high-scoring parameters:

![Optimization impact — random vs optimized](https://raw.githubusercontent.com/treadon/banger-scorer/main/plots/overview/optimization_impact.png)

| Metric | Random (200 songs) | Optimized (30 songs) |
|--------|-------------------|---------------------|
| Mean score | 3.17 | **3.48** (+10%) |
| Songs >= 3.5 | 20% | **60%** (3x) |
| Songs >= 4.0 | 5% | **20%** (4x) |

### AI vs Real Music

![AI generated vs real music score distribution](https://raw.githubusercontent.com/treadon/banger-scorer/main/plots/training/generated_vs_training.png)

AI-generated music clusters at the FMA median (~3.2). Real music has a long tail up to 10. The best AI song (5.29) sits at the 67th percentile — above average, not exceptional. There's still a quality gap, but the generate-and-filter approach makes the most of what's available.

## Key Findings

- **Energy = popularity**: The scorer learned that driving beats and loud production correlate with FMA play counts
- **Minor keys > major keys**: 3.26 mean vs 3.03 — all top songs were in minor keys
- **BPM sweet spots**: EDM peaks at 126-138, Punjabi at 95-105, Pop at 124-128
- **Parameter variation > seed variation**: Changing BPM/key/style produces more score diversity than just re-rolling the random seed
- **The bottleneck is data quality, not model architecture**: Better labels (Spotify engagement metrics, human preferences) would help more than a fancier neural network

## Quick Start

```bash
# Clone with ACE-Step submodule
git clone --recursive https://github.com/treadon/banger-scorer.git
cd banger-scorer

# Setup scorer environment
python3 -m venv venv && source venv/bin/activate
pip install torch transformers librosa soundfile numpy pandas scikit-learn matplotlib

# Setup ACE-Step (requires Python 3.12)
cd ace-step && uv sync --python 3.12 && cd ..

# Download FMA data
cd data
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -L -o fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip
python3 -c "import zipfile; zipfile.ZipFile('fma_metadata.zip').extractall('.')"
python3 -c "import zipfile; zipfile.ZipFile('fma_small.zip').extractall('.')"
cd ..

# Extract MERT embeddings (~100 min, or use pre-computed from HuggingFace)
python embed_dataset.py

# Train scorer (~30 seconds)
python train_scorer.py

# Generate and score songs
python generate_and_score.py --generate 20 --keep 5

# Run the optimized banger generator
python run_bangers.py

# Run full 10-genre test suite (~5 hours)
python run_all_tests.py
```

**Skip the 100-minute embedding step** — use our pre-computed embeddings:
```python
from datasets import load_dataset
ds = load_dataset("treadon/fma-mert-embeddings")
```

## Project Structure

```
banger-scorer/
├── ace-step/                  # ACE-Step 1.5 music generator (git submodule)
├── data/                      # FMA audio, metadata, cached embeddings
├── plots/
│   ├── overview/              # Global scatter, boxplot, ranking, histogram
│   ├── per_genre/             # Score vs BPM + BPM×Key heatmaps per genre
│   ├── analysis/              # BPM correlation, key analysis, major vs minor
│   ├── training/              # FMA distribution, AI vs real comparison
│   └── timing/                # Generation time analysis
├── test_configs/              # JSON configs for each genre test
├── test01/ ... test10/        # Results per genre
├── bangers_output/            # Optimized banger run results
├── embed_dataset.py           # Extract MERT embeddings from audio
├── train_scorer.py            # Train the MLP scorer
├── generate_and_score.py      # Generate + score + rank pipeline
├── run_bangers.py             # Optimized generation using best params
├── run_test.py                # Run a single genre test
├── run_all_tests.py           # Run all tests + compile best overall
├── make_all_plots.py          # Generate all 37 analysis plots
└── WRITEUP.md                 # Detailed technical writeup
```

## Hardware

Everything runs on a MacBook Pro M4 Pro (64GB unified memory):

| Task | Time |
|------|------|
| MERT embedding extraction (8K tracks) | ~100 min |
| Scorer training | ~30 sec |
| Song generation (per song) | ~115 sec |
| Song scoring (per song) | ~2 sec |
| Full 10-genre test (200 songs) | ~6 hours |
| All plots | ~10 sec |

## Links

- **Blog Post**: [AI Music Is Good But Not Great](https://riteshkhanna.com/banger-scorer) *(coming soon)*
- **Pre-computed Embeddings**: [treadon/fma-mert-embeddings](https://huggingface.co/datasets/treadon/fma-mert-embeddings)
- **Generated Songs Dataset**: [treadon/banger-scorer-generated-songs](https://huggingface.co/datasets/treadon/banger-scorer-generated-songs) *(coming soon)*
- **Full Technical Writeup**: [WRITEUP.md](WRITEUP.md)

## Built With

- [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) — Pretrained music understanding model
- [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) — Music generation
- [FMA](https://github.com/mdeff/fma) — Free Music Archive dataset
- [Claude Code](https://claude.ai/claude-code) — AI pair programming

## License

Apache 2.0
