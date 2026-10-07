# Great Lakes setup

Workspace: `/nfs/turbo/coe-xiaonanh/wenzhet`.
Conda environment: `/home/wenzhet/.conda/envs/drivor` (Python 3.9).
Activate with `conda activate drivor`. The launchers also select its Python directly.

The environment uses the README's PyTorch 2.1.0 / torchvision 0.16.0 CUDA 12.1
builds, Lightning 2.4.0, and timm 1.0.15. See `requirements-greatlakes.lock.txt`
for installed package versions. nuPlan and DrivoR are editable installations.
The idea worktree selects its own implementation through PYTHONPATH.
This older environment was validated on A40 (`spgpu`), not on Blackwell.
Subsequent A40 and RTX PRO 6000 training/evaluation use the separate
`/home/wenzhet/.conda/envs/drivor-blackwell` environment. PRO 6000 distributed
training passed with `NCCL_PROTO=LL` (see `run_blackwell_2gpu.sh`); the older
environment's package lock does not describe the Blackwell environment.

Validation on 2026-09-16: `pip check` passed; metric-cache and DrivoRAgent
imports passed. Slurm job `61223523` generated two training metric caches
successfully. Job `61223564` passed CUDA forward/backward on an NVIDIA A40
and the existing CPU BEV decoder/scorer parity tests.

## Training metric cache (no BEV features required)

From DrivoR, create the Slurm output directory and submit:

```bash
mkdir -p exp/cache_slurm
sbatch scripts/training/cache_metrics.sbatch
```

This uses account `xiaonanh0`, CPU partition `standard`, 32 disjoint log shards,
and at most 8 concurrent tasks, each with 1 CPU and 12 GB RAM. Existing cache
files are reused. The input token set includes both navtrain and the copied
full BEV token list; tokens still must pass the scene filter's history/future
and route requirements. Cache output is `exp/train_metric_cache`.

The historical September production array was `61223702`; after its initial eight shards
progressed without errors, its runtime concurrency was raised to 32. The
dependent verification job is `61224160`. Submission defaults remain eight
concurrent shards. These job IDs record the current run, not completion.

```bash
squeue -j 61223702,61224160
```

After submission, queue `sbatch --dependency=afterok:JOB_ID
scripts/training/finalize_metric_cache.sbatch` to verify on a compute node
once every shard succeeds.
It verifies eligible scene coverage and file presence, loads sample pickles,
and consolidates all shard CSV files into one manifest. It archives shard
manifests under `metadata/shards` because the legacy loader reads only one CSV.
Do not train from partial shard metadata. A successful verification writes
`exp/train_metric_cache/metadata/verification.json`.

For small tests, `CACHE_PATH=... bash metric_caching.sh
train_test_split.scene_filter.max_scenes=2` runs sequentially. Use a Slurm
allocation for computation; the top-level launcher now invokes **training**
metric caching instead of the old navtest entry point.

## BEV training

BEV tensors remain a separate prerequisite:
`../navsim_bev_feature/exports_pretrained/trainval/**/*_decoder_neck.pt`.
The copied token list does not substitute for these tensors.

GPU experiments reuse an allocation that you obtain manually (typically seven
days, under your chosen account). Do **not** submit the GPU templates with
`sbatch`: their historical `.sbatch` filenames are retained, but they now run
with `bash` and create an `srun --jobid` step inside the existing allocation.
The account, partition, memory and allocation expiry come from that job, not
from the experiment script. CPU cache-array `.sbatch` scripts still submit
independent CPU jobs; they are not GPU experiment launchers.

After BEV files and the verified training metric cache are present, for example:

```bash
export ALLOCATION_ID=YOUR_RUNNING_JOB_ID
export EXPECTED_ACCOUNT=xiaonanh0  # optional check; use your allocation's account
export EXPECTED_RUN_HOURS=40      # estimate for this experiment, not a new time request
export BEV_TOKEN_FILTER_FILE=/absolute/path/to/your/combined_tokens.txt
# Read-only validation/command preview; will refuse a busy allocation:
DRY_RUN=1 bash scripts/training/train_bev_decoder_baseline_4gpu.sbatch
# Run in a persistent terminal (e.g. tmux) on the login host:
bash scripts/training/train_bev_decoder_baseline_4gpu.sbatch
```

The four-GPU baseline is decoder-only BEV LoRA16, gate 0.1, zero ego-motion
dropout, batch 16/GPU (global 64), online W&B, and held-out validation. The
matching dropout template is `train_bev_ego_dropout_4gpu.sbatch` (10% dropout,
one warmup epoch). `run_blackwell_2gpu.sh` uses two PRO 6000 GPUs, batch 32/GPU
and the required NCCL workaround. The one-GPU `train_bev.sbatch` and eight-GPU
`train_bev_ego_dropout_8gpu.sbatch` are alternate resource presets, not matched
four-GPU comparisons. All use raw loading rather than requiring a feature cache.

The common launcher checks a RUNNING, user-owned, single-node allocation,
optional account match, sufficient GPUs/CPUs, and enough remaining time for
`EXPECTED_RUN_HOURS` plus a five-minute reserve. CPU count defaults to allocated
CPUs divided by GPU ranks; `CPUS_PER_TASK` can reduce it. Any other active
workload step causes a refusal, rather than sharing GPUs with an experiment
or evaluation. These scripts do not cancel/release the allocation when the
experiment ends. A step's wall-time cap is the allocation time remaining minus
the reserve; a runtime estimate is not a guarantee that training will finish.

`RUN_MODE=smoke` limits training/validation batches and defaults to a 15-minute
estimated budget. Use different `EXPERIMENT`/`EXPERIMENT_UID` values for smoke
and full training. Full runs require an explicit duration estimate; ~40 hours
describes the observed fixed-10%-data four-A40 run, not full-data training.
Use the same fixed token file for matched comparisons. Set `TRAIN_CKPT_PATH`
to resume an existing run; otherwise each run starts from the pretrained baseline.
Capture logs with your persistent terminal/launcher, for example
`bash scripts/training/train_bev_decoder_baseline_4gpu.sbatch > experiment.log 2>&1`.

Dataset feature caching is optional and must wait for BEV tensors. Its launcher
is `scripts/training/run_dataset_caching.sh`; the configuration must match the
training schema, especially trajectory length and BEV feature options.
