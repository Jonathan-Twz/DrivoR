# Production CPU scoring workers

The BEV phase-1 launcher now defaults to the measured four-A40 setup:
batch 16/GPU, three scoring workers/GPU, three DataLoader workers/GPU,
prefetch one and pinned memory. The default full training/validation limits
remain 1.0; no benchmark adapter or batch cap is needed for normal training.

The scorer pool is CPU-only, ordered and persistent across batches/epochs.
Each DDP rank starts its own three-process pool lazily. Pools are excluded
from serialization and closed on normal trainer teardown and exceptions.
The original PDM scorer, losses, validation metrics, checkpoints, learning-rate
schedule and logging are unchanged. A broken/unavailable pool logs a warning
and uses the identical serial scorer; ordinary scene errors still propagate.

## Launch and configuration

Run inside an allocation with four GPUs and at least 16 total CPUs:

```bash
SCORING_WORKERS=3 NUM_WORKERS=3 PREFETCH_FACTOR=1 \
bash scripts/training/run_drivor_bev_phase1.sh \
  /path/to/baseline.pth my-workerpool-experiment 30
```

Set dataset/token-filter, checkpoint, W&B and experiment settings as usual.
To reproduce allocation 63199495's data selection, use
`BEV_TOKEN_FILTER_FILE=exp/bev_feature_tokens/jun12_train010_val100_seed2_job61826557/combined_tokens.txt`,
`USE_CACHE_WITHOUT_DATASET=false`, `CACHE_PATH=null` and
`INCLUDE_VAL_LOGS_IN_TRAIN=false`. Use a new experiment name/UID to avoid
mixing new checkpoints or W&B runs into a completed experiment.

Environment knobs passed through to Hydra:

| Setting | Meaning | BEV phase-1 default |
| --- | --- | --- |
| `SCORING_WORKERS` | Extra CPU scoring processes per rank | 3 |
| `SCORING_WORKER_THREADS` | CPU library threads per scoring process | 1 |
| `NUM_WORKERS` | DataLoader workers per rank | 3 |
| `PREFETCH_FACTOR` | Prefetched batches per loader worker | 1 |
| `USE_RAY_SCORE` | Legacy single-GPU Ray backend, used only if scoring workers = 0 | false |

For direct Hydra launches, set `agent.config.scoring_workers=3`,
`agent.config.scoring_worker_threads=1`, `dataloader.params.num_workers=3`
and `dataloader.params.prefetch_factor=1`. The generic agent YAML keeps
`scoring_workers=0` for compatibility with other entrypoints; the production
BEV launcher and both four-GPU BEV Slurm wrappers override it to three.

`SCORING_WORKERS=0 USE_RAY_SCORE=false` restores original serial scoring.
For loader debugging, `NUM_WORKERS=0` automatically sets prefetch to null.
Workers do not add CPU capacity: 12 scorers and the loader processes share
the existing 16 allocated CPUs. Train/validation loader sets may coexist.
The three-scorer/three-loader choice is measured for 16 CPUs; retune if the
CPU/GPU ratio, scene count or batch size changes. Existing Slurm wrapper
resource/account declarations are intentionally unchanged.

## Checks

```bash
python scripts/training/test_scoring_pool.py
```

The four-GPU benchmark report is in
`../gpu_benchmark_63199495/REPORT.md`. Production integration smoke tests use
separate experiment outputs and verify multi-epoch execution and checkpoint
resume, rather than writing into the original experiment.

Integration results are in `../gpu_benchmark_63199495/PRODUCTION_INTEGRATION.md`.
The existing step-based checkpoint can require an extra boundary batch when
resuming; the pool does not promise exact dataloader-state continuation.
