---
name: drivor-bev-eval
description: >-
  Evaluate DrivoR BEV checkpoints (scorer-side, decoder-side, residual refiner,
  or proposal-world) on NAVSIM v1 PDMS and diagnose structure/gate mismatches.
  Use when running navtest eval, when load_state_dict fails, or when checking
  whether the BEV branch is actually used. For v2 EPDMS use drivor-navsim-v2-eval.
disable-model-invocation: true
---

# DrivoR BEV Checkpoint Eval & Gate Diagnostics

## When To Use

- Running **NAVSIM v1** navtest PDM eval on a BEV phase-1 checkpoint.
- For **NAVSIM v2** (navhard_two_stage / EPDMS), use skill `drivor-navsim-v2-eval`.
- Eval crashes with `RuntimeError: ... size mismatch for ..._lora.* ...`.
- Want to confirm the BEV branch is contributing (inspect LayerScale gate).

## Eval Scripts (v1 navtest)

| BEV injection site | Script | Key flags |
|--------------------|--------|-----------|
| **Scorer** (original) | `scripts/evaluation/run_drivor_bev_evaluation.sh` | `use_bev_in_scorer=true` (default) |
| **Decoder** (trajectory generator) | `scripts/evaluation/run_drivor_bev_decoder_evaluation.sh` | `use_bev_in_scorer=false`, `use_bev_in_decoder=true` |
| **Residual proposal refiner** | `scripts/evaluation/run_drivor_bev_residual_proposal_refiner_evaluation.sh` | `use_bev_residual_proposal_refiner=true` |
| **Proposal-conditioned world refiner** | `scripts/evaluation/run_drivor_proposal_world_evaluation.sh` | `use_proposal_world_refiner=true`; all other BEV injection flags false |

All run from the **DrivoR tree** with env **`drivoR-share`**. Ensure `NAVSIM_DEVKIT_ROOT=DrivoR` (not navsim).

## Key Rules

- **Eval structure params MUST match the checkpoint's training run.**
  - Scorer-BEV: `scorer_bev.lora_rank` (yaml default 8; trained runs often 16).
  - Decoder-BEV: `decoder_bev.lora_rank` (default in eval script: 16).
  - Other shape-changing params: `tf_d_model`, `ref_num`, `bev_channels`, `proposal_num`.
  - `init_gate` does NOT affect shape (gamma loaded from ckpt); `lora_rank` must match.
- Env vars exposed by scripts:
  - Scorer: `SCORER_BEV_LORA_RANK`, `SCORER_BEV_INIT_GATE`
  - Decoder: `DECODER_BEV_LORA_RANK`, `CKPT_PATH`, `EXPERIMENT_NAME`
  - Proposal-world: `CKPT_PATH`, `EXPERIMENT_NAME`, `WORLD_LAYERS`,
    `WORLD_HEADS`, `WORLD_FFN_DIM`, `WORLD_ROLLOUT_STEPS`, and
    `PROPOSAL_CHUNK_SIZE`
- On golduck, do not mix `nvidia-smi` indices with CUDA's default
  FASTEST_FIRST ordering. Either use `CUDA_VISIBLE_DEVICES=0,1,2,3` with the
  default ordering, or set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and then use the
  physical A100 indices `CUDA_VISIBLE_DEVICES=0,1,2,4`. The proposal-world
  launcher uses the second convention to exclude the display GPU.

## Run Eval

```bash
# Scorer-BEV (default checkpoint)
bash scripts/evaluation/run_drivor_bev_evaluation.sh

# Decoder-BEV LoRA16 (reference run)
bash scripts/evaluation/run_drivor_bev_decoder_evaluation.sh

# Custom ckpt
CKPT_PATH=exp/ke/.../checkpoints/last.ckpt \
DECODER_BEV_LORA_RANK=16 \
bash scripts/evaluation/run_drivor_bev_decoder_evaluation.sh

# Proposal-conditioned world refiner
CKPT_PATH=exp/ke/.../checkpoints/best-epoch=3-step=7844.ckpt \
EXPERIMENT_NAME=drivoR_nav1-proposal-world-best-epoch3 \
bash scripts/evaluation/run_drivor_proposal_world_evaluation.sh
```

## Reference Results

Decoder-BEV LoRA16 `last.ckpt` on navtest: **PDMS = 0.9322**

Proposal-conditioned world refiner `best-epoch=3-step=7844.ckpt` on the full
NAVSIM v1 navtest split (2026-09-03): **PDMS = 0.934903**, with 12,146 valid
scenarios and 0 failures. Submetrics: NC `0.989791`, DAC `0.988721`, EP
`0.895495`, TTC `0.967479`, comfort `0.999918`, DDC `0.973078`. The same-protocol
pretrained baseline is `0.936905`, so the result is `-0.002002`; the largest
regression is ego progress (`-0.003925`). See `docs/evaluation-ledger.md` for
the checkpoint and CSV paths.

The first proposal-world attempt failed on logical rank 3 with a 4 GB display
GPU OOM because `CUDA_VISIBLE_DEVICES=0,1,2,4` was used without PCI bus ordering.
This is a device-enumeration failure, not a model-memory failure. The corrected
four-A100 run took about 24 minutes wall time, including startup, inference, and
Ray scoring.

## Diagnose The BEV Gate

**Scorer** (`cross_attn_bev_ls.gamma`) or **Decoder** (`trajectory_decoder.layers.*.cross_attn_bev_ls.gamma`):

```bash
/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python - <<'PY'
import torch, re
ck = "exp/ke/<exp>/<uid>/checkpoints/last.ckpt"
sd = torch.load(ck, map_location="cpu"); sd = sd.get("state_dict", sd)
for pat in ["cross_attn_bev_ls.gamma", "trajectory_decoder"]:
    keys = [k for k in sd if pat in k and "gamma" in k]
    if not keys: continue
    print(f"--- {pat} ---")
    for k in sorted(keys):
        g = sd[k].float()
        print(f"  {k}: meanabs={g.abs().mean():.5f} L2={g.norm():.4f}")
PY
```

With `init_gate=0.0`, trained scorer runs show |gamma| ≈ 0.01–0.02 → BEV ≈ 1–2% of residual.

## Missing-BEV Check (eval split)

navtest tokens without a BEV `.pt` fall back to zeros. Features:
`navsim_bev_feature/exports_pretrained/test/**/*_decoder_neck.pt` (`bev_data_split=test`).

## NAVSIM v2

v2 EPDMS is **not** covered here. See `drivor-navsim-v2-eval` and:
- Scorer: `navsim/scripts/evaluation/run_drivoR_pdm_score_v2.sh`
- Decoder: `navsim/scripts/evaluation/run_drivoR_pdm_score_v2_decoder_bev.sh` (env `drivoR-share`)
