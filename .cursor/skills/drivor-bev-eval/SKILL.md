---
name: drivor-bev-eval
description: Evaluate DrivoR BEV phase-1 checkpoints and diagnose the BEV cross-attention gate. Use when running navtest PDM eval on a BEV checkpoint, when load_state_dict fails with LoRA size mismatch, or when checking whether the BEV branch is actually used (cross_attn_bev_ls.gamma).
disable-model-invocation: true
---

# DrivoR BEV Checkpoint Eval & Gate Diagnostics

## When To Use

- Running **NAVSIM v1** navtest PDM eval on a BEV phase-1 checkpoint.
- For **NAVSIM v2** (navhard_two_stage / EPDMS), use skill `drivor-navsim-v2-eval` instead — different repo, paths, and Hydra overrides.
- Eval crashes with `RuntimeError: ... size mismatch for ..._lora.* ...`.
- Want to confirm the BEV branch is contributing (inspect LayerScale gate).

## Key Rules

- **Eval structure params MUST match the checkpoint's training run.** The most common break is `scorer_bev.lora_rank`: the ckpt has rank 16 but the yaml default is 8 → all `*_lora.*` weights mismatch (`[16,256]` vs `[8,256]`).
  - Other shape-changing params to keep matched: `tf_d_model`, `ref_num`, `bev_channels`, `proposal_num`.
  - `init_gate` does NOT affect shape (gamma is loaded from the ckpt), but keep it matched for clarity.
- The eval script `scripts/evaluation/run_drivor_bev_evaluation.sh` exposes env vars:
  - `SCORER_BEV_LORA_RANK` (default 16), `SCORER_BEV_INIT_GATE` (default 0.1).

## Run Eval

```bash
# Default checkpoint (16-lora / 0.1-gate)
bash scripts/evaluation/run_drivor_bev_evaluation.sh

# Evaluate an older rank=8 checkpoint
SCORER_BEV_LORA_RANK=8 bash scripts/evaluation/run_drivor_bev_evaluation.sh
```

Edit `CKPT_PATH` / `EXPERIMENT_NAME` near the top of the script to point at the run you want.

## Diagnose The BEV Gate (cross_attn_bev_ls.gamma)

The BEV-only cross-attn is gated by a per-channel LayerScale `cross_attn_bev_ls.gamma`
(init = `init_gate`). Tiny gamma → BEV barely used.

```bash
/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python - <<'PY'
import torch, re
ck = "exp/ke/<exp>/<uid>/checkpoints/best-....ckpt"
sd = torch.load(ck, map_location="cpu"); sd = sd.get("state_dict", sd)
li = lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1))
for k in sorted([x for x in sd if "cross_attn_bev_ls.gamma" in x], key=li):
    g = sd[k].float()
    print(f"layer {li(k)} meanabs={g.abs().mean():.5f} L2={g.norm():.4f} max|.|={g.abs().max():.4f}")
PY
```

Reference: with `init_gate=0.0`, a fully trained run shows |gamma| ≈ 0.01–0.02
(L2 ≈ 0.2–0.4 over 256 dims) → BEV contributes only ~1–2% of the residual.
Larger gamma after training `init_gate=0.1` means BEV is being used more.

## Missing-BEV Check (eval split)

navtest tokens without a BEV `.pt` fall back to `_empty_bev_tensor()` (zeros), which
dilutes any BEV gain. The eval features live under
`navsim_bev_feature/exports_pretrained/test/**/*_decoder_neck.pt` (`bev_data_split=test`).
If results look flat, verify coverage of navtest tokens before concluding BEV doesn't help.

## NAVSIM v2

v2 EPDMS is **not** covered by this skill. See `drivor-navsim-v2-eval` and `navsim/scripts/evaluation/run_drivoR_pdm_score_v2.sh`.
