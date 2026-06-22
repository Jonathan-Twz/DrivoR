---
name: drivor-bev-eval
description: >-
  Evaluate DrivoR BEV phase-1 checkpoints (scorer-side or decoder-side) on NAVSIM
  v1 PDMS and diagnose LoRA/gate mismatches. Use when running navtest eval, when
  load_state_dict fails with LoRA size mismatch, or when checking whether the BEV
  branch is actually used. For v2 EPDMS use drivor-navsim-v2-eval.
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

Both run from **DrivoR tree** with env **`drivoR-share`**. Ensure `NAVSIM_DEVKIT_ROOT=DrivoR` (not navsim).

## Key Rules

- **Eval structure params MUST match the checkpoint's training run.**
  - Scorer-BEV: `scorer_bev.lora_rank` (yaml default 8; trained runs often 16).
  - Decoder-BEV: `decoder_bev.lora_rank` (default in eval script: 16).
  - Other shape-changing params: `tf_d_model`, `ref_num`, `bev_channels`, `proposal_num`.
  - `init_gate` does NOT affect shape (gamma loaded from ckpt); `lora_rank` must match.
- Env vars exposed by scripts:
  - Scorer: `SCORER_BEV_LORA_RANK`, `SCORER_BEV_INIT_GATE`
  - Decoder: `DECODER_BEV_LORA_RANK`, `CKPT_PATH`, `EXPERIMENT_NAME`

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
```

## Reference Result (Jun 2026)

Decoder-BEV LoRA16 `last.ckpt` on navtest: **PDMS = 0.9322**

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
