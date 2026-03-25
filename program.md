# Banger Scorer — Autoresearch Program

## Objective
Build the best possible model for predicting song "banger potential" (0-10 score) from MERT audio embeddings.

## Setup
- `prepare.py` — DO NOT MODIFY. Provides `load_splits()` which returns train/val/test data with 768-dim MERT embeddings and 0-10 banger scores.
- `train.py` — MODIFY THIS. Contains model architecture, training loop, and hyperparameters.
- Run: `python train.py` — trains the model and prints results.
- Results are logged to `results/run_TIMESTAMP.json`.

## Metrics
- **Primary: minimize val_mae** (Mean Absolute Error on validation set, 0-10 scale)
- **Secondary: maximize test_corr** (Spearman rank correlation on test set)
- Target: val_mae < 1.5, test_corr > 0.4

## Constraints
- Model must be < 50MB (needs fast inference for scoring songs)
- Inference time < 100ms per song (just the MLP, not MERT encoding)
- Training must complete within 5 minutes (TIME_BUDGET = 300s)
- Must use PyTorch, must work on MPS (Apple Silicon)

## What to Try
Explore these directions in order of likely impact:

1. **Architecture variations**
   - Deeper or wider MLPs
   - Residual connections (skip connections)
   - Attention-based pooling (if modifying embedding aggregation)
   - GELU vs ReLU activation
   - Layer normalization vs batch normalization

2. **Regularization**
   - Different dropout rates
   - Label smoothing
   - Mixup augmentation on embeddings
   - Weight decay tuning

3. **Training dynamics**
   - Learning rate: try 3e-4, 5e-4, 1e-3, 3e-3
   - Warmup + cosine schedule
   - Larger/smaller batch sizes
   - Gradient clipping

4. **Loss functions**
   - Huber loss (robust to outliers)
   - Rank-based loss (focus on ordering, not exact values)
   - Combined MSE + ranking loss

5. **Multi-task learning**
   - Predict genre as auxiliary task
   - Predict popularity bucket (classification) alongside regression

## Workflow
1. Read current results in `results/` directory
2. Decide what to change in `train.py`
3. Make the change
4. Run `python train.py`
5. Compare new results to previous runs
6. Repeat, always trying to beat the best val_mae
