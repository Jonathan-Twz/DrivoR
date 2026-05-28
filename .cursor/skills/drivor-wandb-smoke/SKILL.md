---
name: drivor-wandb-smoke
description: Debug DrivoR PyTorch Lightning and W&B logging with short smoke tests. Use when checking W&B console output, train/val step mixing, epoch metrics, or quick real-training logging behavior without full epochs.
disable-model-invocation: true
---

# DrivoR W&B Smoke Tests

## When To Use

Use this for DrivoR training logging issues:
- W&B online dashboard has no console/log output.
- `train/*` and `val/*` metrics appear mixed between epochs.
- Need a fast test of epoch/step behavior without full training.

## Key Rules

- Do not run full training for logging checks.
- For pure W&B/Lightning checks, run `scripts/training/debug_wandb_smoke.py`; it uses dummy tensors and CPU only.
- For real DrivoR logging behavior, use cached data plus tiny batch limits:

```bash
USE_WANDB=1 \
WANDB_MODE=online \
WANDB_CONSOLE=wrap \
NUM_GPUS=1 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
PREFETCH_FACTOR=1 \
LIMIT_TRAIN_BATCHES=8 \
LIMIT_VAL_BATCHES=3 \
CACHE_PATH="$PWD/exp/navsim_cache_nommcv_same_as_training" \
USE_CACHE_WITHOUT_DATASET=true \
bash scripts/training/run_drivor_bev_phase1.sh \
  ./weights/checkpoints/drivor_Nav1_25epochs.pth \
  debug_wandb_real_train_steps \
  2
```

## Expected Metric Behavior

- `train/loss`: log with `on_step=True, on_epoch=True`; W&B shows `train/loss_step` and `train/loss_epoch`.
- Other train losses: log with `on_step=True, on_epoch=False`; W&B shows original names such as `train/trajectory_loss`.
- `val/score_epoch`: log with `on_step=False, on_epoch=True`; use this for `ModelCheckpoint(monitor="val/score_epoch")`.
- Avoid `val/score` with `on_step=True,on_epoch=True` unless `val/score_step` is explicitly desired; validation batches can appear between training epochs on W&B's step axis.

## Console Output

- `WANDB_CONSOLE=off` disables stdout/stderr capture.
- Use `WANDB_CONSOLE=wrap` for online runs when the user expects logs in W&B.
- Local W&B console logs should appear under `wandb/run-*/files/output.log`.
