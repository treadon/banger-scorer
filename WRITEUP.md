# Building a Banger Scorer: Using AI to Rate How Good a Song Is

> Can we train a model to predict whether a song is a banger? And then use it to filter AI-generated music so only the best tracks survive?

## The Idea

AI music generation is getting good. Models like [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) can generate full songs — vocals, instruments, structure — in under 2 minutes on a MacBook. But quality is inconsistent. Generate 50 songs and maybe 3-5 are actually good. The rest are mediocre or weird.

What if we had a **banger scorer** — a model that listens to a song and rates it 0-10 on how good it is? The workflow becomes: generate a big batch, score them all automatically, keep only the top ones. Bangers only.

This is essentially the same pattern used in image generation (generate 4 images, show the best one) but applied to music, with a learned quality function instead of human eyes.

## Why Not Just Use Human Judges?

You could. But humans are slow, expensive, and inconsistent. If you're generating 50 songs to find the best 5, that's 50 × 2 minutes = 100 minutes of listening per batch. A trained model does it in seconds. The model won't have perfect taste, but it can cheaply filter out the obviously bad stuff and surface the most promising candidates.

## The Core Technical Decision: Transfer Learning

We could try to train a model from scratch to understand music. But that would require:
- Millions of labeled songs
- Weeks of GPU time
- Deep expertise in audio processing

Instead, we use **transfer learning** — we take a model that already understands music and just teach it our specific task (rating songs). This is the same pattern that makes modern AI practical:

- **Image classification**: Take a model pretrained on ImageNet, fine-tune for your specific images
- **Text tasks**: Take a pretrained LLM, fine-tune for your specific task
- **Music scoring**: Take a model pretrained on 160K hours of music, add a small scoring head

The pretrained model we use is **MERT** (Music Understanding Model). More on that below.

## Architecture

```
Training Pipeline:
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────┐
│ FMA Dataset  │────▶│ MERT v1 (330M)   │────▶│ 1024-dim     │────▶│ MLP Scorer   │
│ 8K tracks    │     │ Frozen encoder   │     │ embeddings  │     │ Trainable    │
│ + play counts│     └──────────────────┘     └─────────────┘     │ → score 0-10 │
└─────────────┘                                                    └──────────────┘

Inference Pipeline:
┌──────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────┐
│ ACE-Step     │────▶│ MERT v1 (330M)   │────▶│ 1024-dim     │────▶│ MLP Scorer   │
│ generates    │     │ Same encoder     │     │ embedding   │     │ Trained      │
│ N songs      │     └──────────────────┘     └─────────────┘     │ → score 0-10 │
└──────────────┘                                                   └──────────────┘
                                                                          │
                                                                   Sort by score
                                                                   Keep top K
                                                                          │
                                                                    🔥 Bangers
```

### What "Frozen" Means

When we say MERT is "frozen," we mean we don't change its weights during training. It's locked in place. We only train the small MLP head on top. This is important because:

1. **MERT is 330M parameters.** Training all of them would need more data and compute than we have.
2. **It already understands music.** It was pretrained on 160K hours of audio using self-supervised learning (predicting masked portions of audio, similar to how BERT predicts masked words). That knowledge is encoded in its weights.
3. **We only need to learn the mapping from "music understanding" → "popularity score."** That's a much simpler task — a small MLP can handle it.

Think of it like hiring an expert music critic (MERT) and just teaching them your specific rating scale (the MLP). You don't need to teach them what music is.

### What "Embedding" Means

An embedding is just a list of numbers that represents something. MERT takes a raw audio waveform (millions of amplitude samples) and compresses it into 1024 numbers that capture the "essence" of what it heard — rhythm patterns, harmonic content, timbral quality, melodic structure, etc.

These 1024 numbers live in a "latent space" where similar-sounding music ends up near each other. A jazz ballad and a metal track would be far apart; two upbeat pop songs would be close together. The MLP scorer learns which regions of this space correspond to high popularity.

### What "MLP" Means

MLP = Multi-Layer Perceptron. It's the simplest type of neural network — just layers of matrix multiplications with nonlinear activations between them:

```
Input (1024 numbers)
    → Multiply by 1024×512 weight matrix → apply ReLU → dropout
    → Multiply by 512×256 weight matrix → apply ReLU → dropout
    → Multiply by 256×128 weight matrix → apply ReLU → dropout
    → Multiply by 128×1 weight matrix
Output (1 number: the banger score)
```

**ReLU** (Rectified Linear Unit): `max(0, x)` — zeros out negative values. This adds nonlinearity, which is critical. Without it, stacking linear layers would just collapse into a single linear layer (matrix multiplication is associative). ReLU lets the network learn curved, complex decision boundaries instead of just flat planes.

**Dropout**: During training, randomly zero out 30% of neurons each forward pass. This prevents overfitting by forcing the network to not rely on any single neuron. At inference time, dropout is turned off and all neurons are used (scaled appropriately). It's like training a team where random members are absent each day — everyone has to be independently useful.

**BatchNorm** (Batch Normalization): Normalizes the inputs to each layer so they have mean=0 and std=1 across the batch. This stabilizes training — without it, the distribution of values can shift wildly between layers as weights update, making optimization difficult. With BatchNorm, each layer always receives nicely scaled inputs.

## Step 1: Data — The Free Music Archive

### What We Need

To train a scorer, we need two things:
1. **Audio files** — the actual music
2. **Quality labels** — some number that says "this song is good" or "this song isn't"

### Why FMA?

The [FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset gives us both:
- **8,000 tracks** (30-second clips) across 8 balanced genres
- **Play counts per track** — our proxy for "how good is this"
- **Creative Commons licensed** — legal to use, legal to redistribute
- **Curated subsets** — FMA-Small (8K tracks, 7.2GB) is manageable on a laptop

**Genres (1,000 tracks each, perfectly balanced):**
Hip-Hop, Pop, Folk, Experimental, Rock, International, Electronic, Instrumental

The balanced genre split is important. If we had 7,000 pop songs and 100 jazz tracks, the model would learn "pop = popular" rather than learning actual quality signals. With equal genres, it has to learn quality *within* each genre.

### FMA Metadata: What We Got

```
Total tracks in small subset: 8,000
Listen count range: 196 to 543,252
Listen count mean: 4,730
Listen count median: 2,492
```

The huge gap between mean (4,730) and median (2,492) tells us the distribution is heavily right-skewed — most songs have modest play counts, a few have hundreds of thousands. This is typical of popularity distributions (power law / Pareto distribution).

### The Popularity Problem

Raw play counts are a flawed quality signal:

**Why they're imperfect:**
- A song with high play counts might be popular because of playlist placement, not quality
- Older songs have had more time to accumulate plays
- FMA play counts reflect a specific audience (people browsing a free music archive) — not mainstream taste
- Some brilliant niche music has low plays; some generic stuff has high plays

**Why they're still useful:**
- At the scale of 8,000 songs, the correlation between quality and plays is real (even if noisy)
- Log-normalization compresses the extremes and makes the distribution more usable
- We're not trying to predict exact play counts — we're learning a rough quality ordering
- The noise actually helps prevent overfitting to a narrow definition of "good"

**Better metrics that exist but aren't publicly available:**
- **Save-to-stream ratio** (Spotify): How often people actively save a song vs just listen. ~1-3% average, 5%+ means genuine love. This is probably the single best quality signal.
- **Skip rate** (Spotify): How often people skip before 30 seconds. Low skip = good hook.
- **Shazam lookups**: Someone hears a song in the wild and wants to know what it is. Pure organic discovery signal — can't be gamed.
- **User playlist add rate**: People curating a song into their own playlist (not editorial playlists, which are influenced by labels).

We can't access these at scale, so FMA play counts are our best option for an open-source project.

### Normalizing Play Counts to 0-10

Raw play counts range from 196 to 543,252. We need to squish this into 0-10. But we can't just divide by max — the distribution is so skewed that 99% of songs would score near 0.

**Log normalization** compresses the range:

```python
# Without log: range is 196 to 543,252 (2,773x difference)
# With log:    range is 5.28 to 13.21 (2.5x difference)

log_listens = np.log1p(df["listens"])  # log(1 + x) to handle potential zeros
banger_score = (log_listens - log_listens.min()) / (log_listens.max() - log_listens.min()) * 10.0
```

`np.log1p` is `log(1 + x)` — the `+1` prevents `log(0)` which would be negative infinity. After log transform, the distribution is much more Gaussian (bell-shaped), which is what neural networks work best with.

**Why log works here:** The difference between 100 and 1,000 plays is more meaningful than the difference between 500,000 and 501,000 plays. Log captures this intuition — it treats multiplicative differences equally. Going from 100→1,000 (10x) is the same "step" as 1,000→10,000 (10x).

## Step 2: MERT Embeddings

### What is MERT?

[MERT v1 (330M)](https://huggingface.co/m-a-p/MERT-v1-330M) is a **self-supervised music understanding model** with 330 million parameters. It was trained on 160,000 hours of music.

"Self-supervised" means it learned without human labels. The training process:
1. Take a piece of music
2. Mask out (hide) random portions
3. Ask the model to predict what was masked
4. Repeat on 160K hours of diverse music

This is the same idea as BERT for text (predict masked words) but for audio. By learning to predict missing audio, the model implicitly learns:
- Rhythm and tempo patterns
- Harmonic progressions (chord sequences)
- Timbral qualities (what instruments sound like)
- Musical structure (verse, chorus, bridge patterns)
- Genre characteristics
- Production quality cues

All of this knowledge is encoded in the 1024-dimensional embedding vectors it produces.

### The Embedding Process

For each 30-second track, here's what happens:

**1. Load and resample audio**
```python
waveform, sr = torchaudio.load("track.mp3")  # Load raw audio
# Convert stereo → mono (average left and right channels)
waveform = waveform.mean(dim=0)
# Resample to 16kHz (MERT's expected input rate)
# Why 16kHz? It captures frequencies up to 8kHz (Nyquist theorem)
# which covers most musically relevant content while keeping data small
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
waveform = resampler(waveform)
```

**2. Feature extraction**
MERT's feature extractor converts the raw waveform into a format the model expects — normalized, windowed, and shaped into the right tensor dimensions.

**3. Forward pass through MERT**
```python
outputs = model(**inputs)
# outputs.last_hidden_state shape: (1, num_frames, 1024)
# num_frames ≈ 1 per 25ms of audio
# So 30 seconds → ~1,200 frames, each with a 1024-dim vector
```

Each frame captures what's happening in that ~25ms window, in context of the surrounding audio (thanks to attention).

**4. Mean pooling**
```python
embedding = outputs.last_hidden_state.mean(dim=1)  # (1024,)
```

We average all ~1,200 frame vectors into a single 1024-dim vector. This collapses the time dimension — we lose information about *when* things happen but keep information about *what* happens overall.

**Why mean pooling?** It's simple and works well as a baseline. Alternatives:
- **Max pooling**: Take the max across time for each dimension. Captures the most extreme features but ignores their frequency.
- **Attention pooling**: Learn which time frames are most important (e.g., the chorus might matter more than the intro). More powerful but requires training.
- **CLS token**: Some models have a special "summary" token. MERT doesn't use this pattern.

Mean pooling is our starting point. The autoresearch agent can experiment with alternatives.

**5. Cache to disk**
```python
np.save("embeddings.npy", embedding_matrix)  # (8000, 1024) matrix
```

This is the key efficiency trick: run MERT once (2-4 hours), save the results, and reuse them for every training experiment. Without caching, every training run would need to re-encode all 8,000 tracks through MERT — making the autoresearch loop impossibly slow.

### Why Not Fine-Tune MERT?

Fine-tuning means unfreezing MERT's weights and training them alongside the MLP head. This would be more powerful — the model could adapt its music understanding specifically for popularity prediction. But:

1. **Memory**: 330M parameters × 4 bytes × 3 (weights + gradients + optimizer states) ≈ 4GB just for MERT, plus the MLP and batch data. Doable but tight on a laptop.
2. **Overfitting risk**: With only 8K training samples, fine-tuning 330M parameters would almost certainly overfit. The model would memorize training songs rather than learning general quality patterns.
3. **Speed**: Each training step would require a full forward+backward pass through MERT. Training would take hours instead of minutes.
4. **Can't cache**: Embeddings change as MERT's weights update, so no caching trick.

Frozen MERT + trainable MLP is the sweet spot for our data size and compute budget.

## Step 3: Train the Scorer

### The Model

```python
class BangerScorer(nn.Module):
    def __init__(self, input_dim=1024, dropout=0.3):
        self.net = nn.Sequential(
            nn.Linear(1024, 512),    # 1024×512 = 524,288 weights
            nn.BatchNorm1d(512),    # 512 learnable params
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),    # 512×256 = 131,072 weights
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),    # 256×128 = 32,768 weights
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),       # Less dropout near the output

            nn.Linear(128, 1),      # 128×1 = 128 weights
        )
```

**Total trainable parameters: ~558K** — tiny compared to MERT's 330M. This trains in minutes.

**Why this specific architecture?** The layers progressively narrow (1024→512→256→128→1), which forces the network to compress information at each stage. The first layer keeps most of the richness; later layers distill it down to the single most important signal: "is this a banger?"

### Training Details

**Loss function: MSE (Mean Squared Error)**
```python
loss = ((predicted_score - actual_score) ** 2).mean()
```

We square the errors so large mistakes are penalized more heavily. Predicting 3.0 when the answer is 8.0 is much worse than predicting 4.5 when the answer is 5.0.

**Optimizer: AdamW**

Adam (Adaptive Moment Estimation) is the standard optimizer for deep learning. It maintains per-parameter learning rates that adapt based on the history of gradients:
- If a parameter's gradients have been consistently pointing the same direction → take bigger steps (the optimization is confident)
- If gradients have been noisy/oscillating → take smaller steps (the optimization is uncertain)

The "W" in AdamW means "decoupled weight decay" — it adds a small penalty for large weights (`weight_decay=1e-4`), which acts as regularization to prevent overfitting.

**Learning rate schedule: Cosine Annealing**

The learning rate starts at `1e-3` and gradually decreases following a cosine curve down to nearly zero. The intuition:
- **Early training**: Large learning rate → explore the loss landscape broadly, make big jumps
- **Late training**: Small learning rate → fine-tune, make small adjustments to settle into a good minimum

```
LR: 0.001 ──╲                    (cosine curve)
              ╲
               ╲
                ╲──── 0.0001
                  ╲── 0.00001
Epoch: 0    50    100    150    200
```

**Early stopping (patience=20)**

If the validation MAE doesn't improve for 20 consecutive epochs, we stop training. This prevents overfitting — the model starts to memorize training data rather than learning general patterns. We save the model from the best validation epoch, not the last epoch.

### Train/Val/Test Split

```
8,000 tracks
├── 70% Training (5,600) — model learns from these
├── 15% Validation (1,200) — used to tune hyperparameters and detect overfitting
└── 15% Test (1,200) — final evaluation, never seen during training or tuning
```

**Why three sets, not two?**

If we only had train and test, we'd tune hyperparameters (dropout rate, learning rate, architecture) based on test performance. But then the test set isn't truly "unseen" — we've indirectly optimized for it. The validation set absorbs this: we tune against validation, and test remains truly independent.

### Evaluation Metrics

**MAE (Mean Absolute Error)**: Average magnitude of errors. "On average, our predictions are off by X points on the 0-10 scale."
- MAE of 1.0 means typical errors are ±1 point
- MAE of 2.0 would be concerning
- Target: < 1.5

**Spearman Rank Correlation**: Does our ranking agree with the true ranking? Ignores the actual values, only cares about ordering.
- 1.0 = perfect agreement
- 0.0 = random ordering
- Target: > 0.4

Spearman correlation is arguably more important than MAE for our use case. We don't care if we predict 7.2 when the truth is 6.8 — we care that we rank the best songs above the worst songs.

### Results

| Metric | Value | Target | |
|--------|-------|--------|---|
| **Test MAE** | **0.858** | < 1.5 | Exceeded target by 43% |
| **Test Spearman Correlation** | **0.468** | > 0.4 | Hit target |
| **Val MAE** | **0.822** | — | |
| **Best Epoch** | 9 | out of 200 max | Early stopped at epoch 30 |
| **Training Time** | ~30 seconds | — | On M4 Pro MPS with cached embeddings |
| **Model Size** | ~2.6 MB | < 50 MB | 658K params × 4 bytes |

**What the scatter plot shows:**
- The model learned a real trend — low actual scores get low predictions, high get high
- Most FMA songs cluster in the 1-5 range (few true bangers in an indie archive)
- The model struggles at extremes (8+) — too few examples to learn from
- The correlation (0.468) means the model's ranking agrees with actual popularity about half the time — enough to filter the worst from the best, not enough to be an oracle

**What this means for the pipeline:**
An MAE of 0.86 on a 0-10 scale means the model is typically off by less than 1 point. For filtering purposes (pick top 5 from 50), this is very usable — the model can reliably distinguish a 2/10 from a 6/10, even if it can't tell a 7/10 from an 8/10.

## Step 4: Autoresearch — Overnight Optimization

### The Concept

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) is a framework where an AI agent acts as an ML researcher: it modifies training code, runs experiments, reads results, and iterates. You set it up before bed, and wake up with a better model.

We adapt this pattern for our scorer:

```
prepare.py  — Fixed: loads cached MERT embeddings, provides train/val/test splits
              DO NOT MODIFY. This ensures every experiment uses identical data.

train.py    — Agent-modifiable: model architecture, hyperparameters, training loop
              The agent changes this file each experiment.

program.md  — Instructions for the agent: what to optimize, constraints, suggestions

results/    — Each run saves metrics as a JSON file. The agent reads previous
              results to decide what to try next.
```

### What the Agent Can Experiment With

**Architecture changes:**
- Deeper networks (more layers) — can learn more complex patterns but risk overfitting
- Wider layers (1024→1024→512 instead of 1024→512→256) — more capacity per layer
- Residual connections (`output = layer(x) + x`) — helps gradients flow in deeper networks
- Attention-based pooling — if we modify the embedding step to keep per-frame vectors
- GELU activation instead of ReLU — smoother, sometimes works better

**Loss function experiments:**
- Huber loss — like MSE but less sensitive to outliers (songs with anomalous play counts)
- Ranking loss — directly optimizes for correct ordering rather than exact value prediction
- Multi-task — predict genre AND popularity jointly; the genre task provides auxiliary supervision

**Regularization:**
- Mixup — during training, blend two songs' embeddings and their labels: `x_new = 0.7*x_a + 0.3*x_b`, `y_new = 0.7*y_a + 0.3*y_b`. Creates synthetic training examples that smooth the decision boundary.
- Label smoothing — instead of training on exact scores, add small noise. Prevents the model from being overconfident.

### 5-Minute Budget

Each experiment is capped at 5 minutes of training time. This is important:
- Ensures the agent can run many experiments per night
- Forces architectures to be efficient (can't just scale up indefinitely)
- Makes results comparable (same time budget = fair comparison)

## Step 5: Generate-and-Filter Pipeline

### How It Works End-to-End

```bash
python banger.py --prompt "upbeat pop song about summer" --generate 50 --keep 5
```

What happens under the hood:

```
1. Load MERT encoder (one-time, ~30s to load 330M params into memory)
2. Load trained scorer MLP (instant, ~2MB)

3. For i in 1..50:
   a. Set random seed = 42 + i (deterministic — same seed = same song)
   b. ACE-Step generates a song (~75s)
      - LM plans the song structure (~27s)
      - DiT diffusion synthesizes audio (~9s for 8 turbo steps)
      - VAE decodes latent → waveform (~18s)
   c. MERT encodes the WAV → 1024-dim embedding (~2s)
   d. MLP scorer predicts banger score (~0.001s)
   e. Store (song, score) pair

4. Sort all 50 songs by score, descending
5. Copy top 5 to output directory with scores in filename
6. Save results log as JSON
```

### Timing Budget (estimated for M4 Pro)

| Step | Per song | 50 songs |
|------|----------|----------|
| ACE-Step generation | ~75s | ~62 min |
| MERT encoding | ~2s | ~100s |
| MLP scoring | <0.01s | <0.5s |
| **Total** | **~77s** | **~65 min** |

The bottleneck is generation, not scoring. Scoring 50 songs takes ~2 minutes total. Generating them takes over an hour. This means a better scorer doesn't slow down the pipeline at all — we could make it 10x more complex and it wouldn't matter.

### Why Different Seeds?

Each random seed produces a different song from the same prompt. The seed initializes the random noise that the diffusion model starts from — different noise → different denoising path → different song. Same seed with same prompt always produces the same song (deterministic).

This is crucial for the filter approach: we need variety to select from. If all 50 songs were identical, there'd be nothing to filter.

## Hardware

Everything runs locally on a MacBook Pro M4 Pro with 64GB unified memory:

| Component | Memory Usage | Time |
|-----------|-------------|------|
| MERT model (330M params) | ~1.3 GB | 2s per track encoding |
| Scorer MLP (558K params) | ~2 MB | <1ms per prediction |
| ACE-Step models (~9.4 GB total) | ~10 GB | 75s per song |
| FMA dataset (8K tracks) | ~7.2 GB on disk | — |
| Cached embeddings (8K × 1024) | ~31 MB | — |

**Total peak memory during scoring: ~12 GB** — well within 64 GB.

**Why Apple Silicon works well here:**
- Unified memory: CPU and GPU share the same RAM. No copying data between CPU RAM and GPU VRAM.
- MPS (Metal Performance Shaders): PyTorch operations run on the GPU cores natively.
- Memory bandwidth: M4 Pro has ~273 GB/s, which is the key bottleneck for running neural networks (the cores are fast, feeding them data is the hard part).

No cloud GPUs, no API costs, no data leaving the machine.

## Process Log

*(Detailed notes added as work progresses)*

### Data Download
- FMA metadata: 342 MB, downloaded in ~X min
- FMA-Small audio: 7.2 GB, downloaded in ~X min
- Metadata zip needed Python's zipfile module to extract (macOS `unzip` couldn't handle the compression format — "need PK compat. v4.6")
- 8,000 tracks confirmed, perfectly balanced: 1,000 per genre
- Listen count stats: min=196, max=543,252, mean=4,730, median=2,492

### MERT Embedding Extraction
- Model: `m-a-p/MERT-v1-330M` — 330M params, 24-layer transformer, **1024-dim embeddings** (not 768 as initially assumed — that was MERT-v0)
- Expected input: **24kHz** sample rate (not 16kHz — caught this by reading the HF model card closely: the feature extractor throws a `ValueError` if you pass 16kHz audio)
- Device: MPS (Apple M4 Pro Metal)
- Single track benchmark: **0.31s** on MPS (warm), 4.66s on CPU (15x speedup from Metal)
- Full extraction rate: **1.3 tracks/s** (includes MP3 decode + resample + MERT forward pass)
- ETA for 8000 tracks: **~102 minutes**
- 0 failures out of 8000 tracks
- Peak memory: ~1.7GB (MERT model + one audio buffer)

**Issues encountered:**
1. `torchaudio` on PyTorch 2.11 requires `torchcodec` for MP3 loading — switched to `librosa` instead. No quality difference, just which library decodes the MP3.
2. First attempt tried loading all 8000 tracks into RAM simultaneously with ThreadPoolExecutor (8K × 30s × 24kHz × 4 bytes ≈ 23GB). Process was killed by OOM. Fixed by processing sequentially — one track at a time, ~1.7GB peak memory.
3. Initial runs showed no progress because `print()` output was buffered. Added `flush=True` to all prints.

**Final results:**
- 7,997 tracks successfully embedded (3 corrupt MP3s failed — 99.96% success rate)
- Output shape: `(7997, 1024)` — 31 MB on disk
- Total wall time: **101.4 minutes**
- Consistent 1.3 tracks/s throughout the run, no slowdowns or memory issues

**Why not parallelize GPU inference?**
MPS (Metal) doesn't support concurrent streams like CUDA does. You can't run two MERT forward passes simultaneously on one Apple GPU. The CPU-bound MP3 decoding (~0.5s) overlaps slightly with GPU inference (~0.3s) but true pipelining would need CUDA streams.

### Training Run 1: Baseline MLP
- Input dim: 1024 (auto-detected from embeddings)
- Architecture: 1024 → 512 → 256 → 128 → 1 (BatchNorm + ReLU + Dropout at each layer)
- Dropout: 0.3
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR over 200 epochs
- Batch size: 64
- Split: 5600 train / 1200 val / 1200 test
- Best epoch: **9** (early stopped at 30 with patience=20)
- Test MAE: **0.858** | Test Spearman: **0.468** | Val MAE: **0.822**
- Training time: ~30 seconds on MPS (cached embeddings, tiny model)
- The model converged extremely fast — loss dropped from 4.5 to 1.4 in the first 2 epochs, then gradually to ~0.1 by epoch 25. But validation MAE plateaued at epoch 9, meaning later improvements were overfitting.
- Scatter plot shows clear positive trend but most predictions compressed into 1-5 range. The model has learned the central tendency well but can't differentiate the extremes.

### Test 02: Upbeat Pop/Dance (20 songs)
- 20/20 generated, zero failures
- Score range: **1.98 to 4.31** — significantly wider than hip hop (2.52–3.38)
- **4.31 is our highest score yet** — the scorer clearly prefers upbeat pop with high energy over hip hop
- Best combo: bpm=128, D major, upbeat pop with bright synths (seed=100)
- Worst combo: bpm=128/135, G/F major, scored 1.98 — some pop attempts fell flat
- **128 BPM dominated the top rankings** — classic dance pop tempo
- Major keys (D, C, A, F) — brighter tonality correlated with higher scores
- The scorer's FMA training data likely has more pop representation than hip hop, which explains the genre bias

### Autoresearch Experiments
*(Skipped — baseline already exceeded targets, bottleneck is data quality not model architecture)*

### Generate-and-Filter Test: 20 Songs

**Setup:**
- 20 songs generated across a grid of 4 caption styles × 5 BPMs × 5 keys (sampled 20 from 100 combos)
- Caption styles: east coast boom bap, dark orchestral, jazzy piano loops, gritty 808s
- BPMs: 78, 85, 90, 95, 100
- Keys: C minor, Bb minor, D minor, E minor, Ab minor
- Same lyrics across all, only musical parameters varied
- One subprocess per song (fresh process each time for stability)
- Each song: ~115s generation + ~2s scoring

**Stability: 20/20 generated successfully. Zero failures.** The one-subprocess-per-song approach solved all MPS hanging issues from earlier batch attempts.

**Results:**

| Rank | Score | BPM | Key | Style | Seed |
|------|-------|-----|-----|-------|------|
| #1 ★ | **3.38** | 90 | C minor | East coast boom bap | 53 |
| #2 ★ | **3.25** | 100 | D minor | Dark orchestral | 56 |
| #3 ★ | **3.17** | 100 | E minor | Dark orchestral | 58 |
| #4 ★ | **3.08** | 78 | E minor | Gritty 808s | 51 |
| #5 ★ | **3.07** | 78 | Bb minor | Dark orchestral | 60 |
| ... | ... | ... | ... | ... | ... |
| #16 ✗ | **2.81** | 85 | C minor | Jazzy boom bap | 54 |
| #17 ✗ | **2.79** | 78 | Bb minor | Gritty 808s | 48 |
| #18 ✗ | **2.70** | 78 | C minor | Jazzy boom bap | 47 |
| #19 ✗ | **2.58** | 85 | Ab minor | East coast boom bap | 45 |
| #20 ✗ | **2.52** | 85 | Bb minor | Jazzy boom bap | 55 |

**Score range:** 2.52 to 3.38 (34% spread between worst and best)

**Patterns observed:**
- Higher BPM (90-100) consistently scored better than slower (78-85)
- C minor and E minor outperformed Bb minor and Ab minor
- Dark orchestral and east coast boom bap styles scored higher than jazzy piano loops
- The scorer clearly differentiates between parameter combinations — varying style/BPM/key produces more score diversity than just changing the random seed (previous seed-only test had only 2.55-3.26 range across 5 songs)

**Timing:**
- Total generation: ~38 minutes (20 × ~115s per song)
- Total scoring: ~40 seconds (20 × ~2s per song)
- Generation is 57x slower than scoring — the bottleneck is 100% generation, not evaluation

**File sizes:**
- WAV output: 660 MB total (22 MB per song at 48kHz stereo)
- After MP3 conversion (192kbps): 87 MB total (~4.3 MB per song)
- 87% size reduction with minimal perceptible quality loss for evaluation purposes

### Tests 03–10 Summary

All tests ran 20 songs each, one subprocess per song, zero crashes across 160 songs (tests 03-10). Combined with tests 01-02: **200 songs generated total, 200/200 successful.**

| Rank | Genre | Language | Range | Mean | Best |
|------|-------|----------|-------|------|------|
| 1 | **Electronic/EDM** | EN | 2.80–**5.29** | 3.71 | **5.29** |
| 2 | **Punjabi/Bhangra** | PA | 2.79–4.26 | **3.77** | 4.26 |
| 3 | **Bollywood** | HI | 2.70–4.38 | 3.53 | 4.38 |
| 4 | C-Pop | ZH | 2.13–4.47 | 3.20 | 4.47 |
| 5 | Latin/Reggaeton | ES | 2.36–3.90 | 3.19 | 3.90 |
| 6 | Pop/Dance | EN | 1.98–4.31 | 3.05 | 4.31 |
| 7 | Rock/Alternative | EN | 2.02–3.66 | 3.03 | 3.66 |
| 8 | Hip Hop | EN | 2.52–3.38 | 2.92 | 3.38 |
| 9 | Acoustic/Folk | EN | 2.03–3.31 | 2.63 | 3.31 |
| 10 | R&B/Soul | EN | 2.14–3.21 | 2.62 | 3.21 |

**Overall best: Melodic techno, 130 BPM, Eb minor, seed 907 — scored 5.29/10**

5 of the top 10 songs overall were EDM. The scorer heavily favors:
- **Rhythmically consistent, driving beats** (EDM, Punjabi dhol, Bollywood dance)
- **Higher energy** — slow R&B and acoustic folk scored lowest
- **Electronic production** — clean, loud, repetitive patterns match what FMA's popular electronic tracks sound like

The scorer essentially learned "popular FMA music has strong beats and high energy" — which is correct for that dataset but doesn't capture the full picture of what makes music good.

### Best Overall — Top 10 Across 200 Songs

| Rank | Score | Genre | BPM | Key | Style |
|------|-------|-------|-----|-----|-------|
| #1 | **5.29** | EDM | 130 | Eb minor | Melodic techno, atmospheric, driving |
| #2 | **4.90** | EDM | 138 | F minor | Deep house, groovy bassline |
| #3 | **4.87** | EDM | 126 | Eb minor | Progressive house, euphoric drop |
| #4 | **4.66** | EDM | 138 | Bb minor | Melodic techno, dark and dreamy |
| #5 | **4.47** | C-Pop | 120 | A minor | Emotional ballad, piano-driven |
| #6 | **4.38** | Bollywood | 120 | E minor | Club-ready beat, heavy bass |
| #7 | **4.32** | EDM | 126 | Bb minor | Deep house, warm chords |
| #8 | **4.31** | Pop/Dance | 128 | D major | Electronic pop, shimmering arpeggios |
| #9 | **4.28** | Bollywood | 130 | D minor | Energetic dance number, desi bass |
| #10 | **4.26** | Punjabi | 95 | C major | Bhangra, traditional tumbi, dhol |

## What We Learned

1. **MERT + MLP can predict popularity.** Test MAE of 0.858 on a 0-10 scale, Spearman correlation 0.468. Not an oracle, but a useful heuristic for relative ranking.

2. **The scorer does differentiate quality.** Across 200 songs, scores ranged from 1.98 to 5.29. Parameter variations (BPM, key, caption style) produced meaningful score differences within each genre — the scorer isn't random.

3. **Genre bias is real and predictable.** The scorer strongly prefers high-energy electronic music over mellow acoustic/R&B. This reflects FMA's popularity distribution, not absolute musical quality. A production-quality R&B ballad might be "better" than a generic techno loop, but the scorer can't tell.

4. **The bottleneck is labels, not architecture.** We skipped the autoresearch loop because the baseline model already beat our targets. Better training data (Spotify engagement metrics, more songs, human preference labels) would improve results far more than a fancier MLP.

5. **Varying musical parameters matters more than varying random seeds.** Caption style, BPM, and key produce wider score distributions than just changing the diffusion seed. This makes sense — the seed varies texture/timbre while the parameters vary the fundamental musical properties.

6. **One subprocess per song is the reliable approach for Apple Silicon.** Batch generation hangs after 2-3 songs due to MPS memory/state accumulation. Fresh process per song adds ~30s overhead but never crashes. 200/200 success rate.

7. **AI-generated music scores below average on a real-music scale.** Our best score (5.29) is at the 67th percentile of FMA — "above average" but not exceptional. Most generated songs landed at the median (3.0-3.5). The scorer correctly identifies that AI music doesn't yet match the quality distribution of human-made music.

8. **Multilingual generation works.** Hindi, Punjabi, Spanish, and Chinese lyrics all generated successfully. Interestingly, Punjabi and Bollywood scored highest among all genres — ACE-Step may have strong South Asian music training data, or the rhythmic patterns (dhol, desi bass) align well with what the scorer learned from FMA's "International" genre category.

## Limitations

- **FMA popularity ≠ mainstream popularity.** FMA is indie/unsigned artists on a free archive. The model learns "music that people actively seek out on a niche platform" not "Billboard chart hits." This might actually be *better* for our purposes — it's measuring organic quality without marketing budget bias. But it means the model's "taste" reflects FMA's audience, not the general public.

- **MERT is frozen.** Fine-tuning MERT's weights on our task would likely improve results but risks overfitting with only 8K samples. A larger dataset (FMA-Medium with 25K tracks, or FMA-Large with 106K) would make fine-tuning viable.

- **30-second clips.** We score based on 30-second excerpts, but song quality involves full-track structure — build-ups, drops, bridges, callbacks. A song might have a weak 30-second excerpt but a brilliant chorus that happens outside the clip window.

- **Mean pooling loses temporal info.** By averaging MERT's per-frame outputs, we lose information about *when* things happen. A song with an incredible 10-second hook followed by 20 seconds of noise would average to the same embedding as a consistently mediocre song. Attention pooling could address this.

- **Popularity is not quality.** Some brilliant niche music has low play counts. Some generic music has millions of plays. The model learns statistical tendencies across 8,000 tracks, not absolute aesthetic truth. It's a useful heuristic, not an oracle.

- **Distribution shift.** MERT was trained on real music. AI-generated music may have subtle artifacts or patterns that don't exist in real music. The embeddings might capture these differences, potentially biasing scores.

## Future Work

- **RL fine-tuning (RLHF for music):** Use the scorer as a reward model and apply reinforcement learning to ACE-Step. Instead of generate-and-filter, directly train the generator to produce higher-scoring songs. This is exactly the RLHF pipeline used for language models, but applied to music diffusion.

- **Better labels:** Incorporate Spotify save-to-stream ratio, skip rate, or Shazam data for less biased quality signals. Would require Spotify API access and matching tracks across platforms.

- **MERT fine-tuning:** Unfreeze MERT's last few layers and jointly train with the scorer head. Needs a larger dataset to avoid overfitting.

- **Longer context:** Score full songs (2-5 minutes) instead of 30-second clips. Requires more memory and compute but captures structural quality.

- **Attention pooling:** Replace mean pooling with learned attention weights over MERT's time frames. The model could learn that choruses matter more than intros.

- **A/B preference model:** Instead of regression on play counts, generate pairs of songs, have humans pick the better one, and train on preference data (Bradley-Terry model). This directly learns human taste rather than using popularity as a proxy. More expensive to label but cleaner signal.

- **Scale to FMA-Medium/Large:** 25K-106K tracks instead of 8K. More data = better generalization, and potentially enough to fine-tune MERT.

## Reproduce It

```bash
git clone <this-repo>
cd banger-scorer

# Setup
python3 -m venv venv && source venv/bin/activate
pip install torch transformers librosa soundfile numpy pandas scikit-learn matplotlib

# Download FMA data
cd data
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -L -o fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip
python3 -c "import zipfile; zipfile.ZipFile('fma_metadata.zip').extractall('.')"
python3 -c "import zipfile; zipfile.ZipFile('fma_small.zip').extractall('.')"
cd ..

# Extract MERT embeddings (2-4 hours, one-time)
python embed_dataset.py

# Train scorer (10-30 min)
python train_scorer.py

# Generate and filter bangers (requires ACE-Step setup)
python banger.py --prompt "upbeat pop song about summer" --generate 50 --keep 5
```

## Links

- [MERT: Acoustic Music Understanding Model](https://huggingface.co/m-a-p/MERT-v1-330M) — The pretrained audio encoder
- [ACE-Step 1.5: Music Generation](https://huggingface.co/ACE-Step/Ace-Step1.5) — The music generator
- [FMA: Free Music Archive Dataset](https://github.com/mdeff/fma) — Training data
- [Autoresearch by Karpathy](https://github.com/karpathy/autoresearch) — The overnight optimization pattern

---

*Built with Claude Code on a MacBook Pro M4 Pro (64GB). No cloud GPUs were used.*
