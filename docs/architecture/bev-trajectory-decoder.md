# BEV-Aware Trajectory Decoder

This document is the canonical design note for the BEV injection added to the
DrivoR trajectory generator. It is intended for maintainers and coding agents.

## Purpose

The original DrivoR trajectory decoder generates and refines 64 trajectory
proposals from ego-conditioned proposal tokens and camera/lidar scene tokens.
The BEV-aware decoder adds precomputed spatial BEV features directly to this
generation path while preserving the pretrained decoder at initialization.

The current phase-1 launcher uses decoder-only injection:

```text
use_bev_feature=true
use_bev_in_scorer=false
use_bev_in_decoder=true
decoder_bev.init_gate=0.0
decoder_bev.lora_rank=16
```

## Data Flow

```text
precomputed BEV                    original DrivoR inputs
(B, 256, 128, 128)                ego state + scene tokens
        |                                  |
        v                                  v
  BevTokenizer                    initial proposal tokens
  patch conv 8x8                           |
  16x16 -> pool 8x8                        |
  position embedding                       |
  LayerNorm                                |
        |                                  |
        +---------- 64 x 256 tokens -------+
                                           |
                         4 BEV-aware decoder blocks
                                           |
                         per-layer trajectory heads
                                           |
                              64 trajectory proposals
```

`BevTokenizer` converts each `(256, 128, 128)` feature map into 64 tokens with
dimension 256. The token order corresponds to the flattened 8x8 BEV grid.

## Decoder Block

The original block is:

```text
x = x + SelfAttention(LN(x))
x = x + SceneCrossAttention(LN(x), LN(scene))
x = x + MLP(LN(x))
```

The replacement keeps those modules and state-dict names, then inserts one
BEV cross-attention residual:

```text
x = x + frozen_self_attention(x)  + side_self_adapter(x)
x = x + frozen_scene_attention(x) + side_scene_adapter(x, scene)
x = x + gamma * BEVCrossAttention(LN(x), LN(bev_tokens))
x = x + frozen_mlp(x)             + side_mlp_adapter(x)
```

For BEV cross-attention, trajectory proposal tokens are queries and BEV tokens
are keys and values. There is one independent BEV attention module and one
channel-wise LayerScale gate `gamma` in each decoder block.

Implementation: `navsim/agents/drivoR/layers/bev_decoder_blocks.py`.

## Low-Rank Side Adapters

The modules named `*_lora` are parallel low-dimensional adapters. They do not
modify or merge low-rank deltas into the pretrained Q/K/V weight matrices, so
they are not conventional LoRA in the strict sense.

- `self_attn_lora`: low-dimensional attention among proposal tokens.
- `cross_attn_lora`: proposal queries attending to original scene tokens.
- `mlp_lora`: `LayerNorm -> down projection -> GELU -> up projection`.

The attention adapter projects Q/K/V from model dimension `d` into
`rank * num_heads`, performs scaled dot-product attention there, and projects
back to `d`. The output projection is initialized to zero. The MLP adapter's up
projection is also initialized to zero. Therefore all adapter residuals are
exactly zero before training.

These adapters help the frozen original decoder adjust around the new BEV
signal. The new BEV cross-attention itself is a full attention module, not a
LoRA adapter.

## Initialization And Gradients

With `decoder_bev.init_gate=0.0`:

- Model output exactly matches the original pretrained decoder at startup.
- The gate receives a nonzero gradient on the first backward pass.
- BEV attention and `BevTokenizer` gradients are zero while the gate is exactly
  zero; they begin receiving gradients after the optimizer moves the gate.
- Side-adapter output projections receive gradients immediately, while their
  input projections begin receiving gradients after the output projections move.

This staged activation is intentional, but a very slow LR warmup can leave BEV
nearly inactive early in training. Inspect
`trajectory_decoder.layers.*.cross_attn_bev_ls.gamma` when diagnosing this.

## Phase-1 Trainability

When `freeze_pretrained_except_bev_scorer=true`, the optimizer includes:

```text
bev_tokenizer.*
trajectory_decoder.layers.*.cross_attn_bev*
trajectory_decoder.layers.*.self_attn_lora*
trajectory_decoder.layers.*.cross_attn_lora*
trajectory_decoder.layers.*.mlp_lora*
```

The original decoder, backbone, trajectory heads, and scorer remain frozen.
Despite the legacy config name `freeze_pretrained_except_bev_scorer`, this
policy supports both scorer-side and decoder-side BEV modules.

For the current 4-layer, rank-16 configuration, approximately 5.44M parameters
are trainable: 4.21M in the tokenizer, 1.06M in full BEV attention, and 0.17M
in side adapters.

## Checkpoint Compatibility

Every original `TransformerDecoder` parameter retains the same module path and
shape in `BevAwareTrajectoryDecoder`. Loading a baseline checkpoint with
`strict=False` should report only these expected missing groups:

```text
bev_tokenizer.*
trajectory_decoder.layers.*.cross_attn_bev*
trajectory_decoder.layers.*.*_lora*
```

`decoder_bev.lora_rank` changes parameter shapes and must match when loading a
fine-tuned checkpoint. `decoder_bev.init_gate` does not change shapes because
the trained gate value is loaded from the checkpoint.

Run the compatibility test after changing the decoder:

```bash
/mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python \
  scripts/training/test_drivor_bev_decoder_parity.py
```

Required properties tested:

1. Every original decoder key exists with the same shape.
2. Gate zero produces the same output as the original decoder.
3. No BEV tokens produce the same output as the original decoder.
4. A nonzero gate with BEV tokens changes the output.

## Configuration And Entry Points

| Concern | Location |
|---|---|
| Decoder implementation | `navsim/agents/drivoR/layers/bev_decoder_blocks.py` |
| BEV tokenization | `navsim/agents/drivoR/layers/bev_tokenizer.py` |
| Model routing | `navsim/agents/drivoR/drivor_model.py` |
| Checkpoint/trainable whitelist | `navsim/agents/drivoR/drivor_agent.py` |
| Hydra defaults | `navsim/planning/script/config/common/agent/drivoR.yaml` |
| Phase-1 launcher | `scripts/training/run_drivor_bev_phase1.sh` |
| Parity test | `scripts/training/test_drivor_bev_decoder_parity.py` |

Relevant configuration:

```yaml
use_bev_feature: false
use_bev_in_scorer: true
use_bev_in_decoder: false

decoder_bev:
  init_gate: 0.0
  lora_rank: 0
  lora_dropout: 0.0
```

The launcher overrides these defaults for the current decoder experiment.

## Known Caveat: Missing BEV Features

The feature builder currently substitutes a zero tensor when a BEV file is
missing or invalid. `BevTokenizer` adds a learned positional embedding to that
tensor, so the resulting tokens are not zero and are not equivalent to
`x_bev=None`. Training uses a token filter intended to ensure BEV files exist,
but evaluation and future datasets must verify coverage.

A robust future fix should propagate a per-sample BEV-valid flag and skip or
mask BEV cross-attention for invalid samples. Do not assume zero-filled input
automatically disables the branch.

## Invariants For Future Changes

- Preserve original decoder parameter names and shapes unless intentionally
  breaking baseline-checkpoint compatibility.
- Keep BEV attention as a separate residual; do not concatenate BEV and scene
  tokens without reevaluating normalization and pretrained behavior.
- Preserve exact baseline output when the BEV gate and adapter outputs are zero.
- Keep scorer-side and decoder-side injection independently configurable.
- Update the checkpoint missing-key whitelist when adding trainable submodules.
- Match `lora_rank`, model dimension, decoder depth, BEV channels, and proposal
  count between training and evaluation.
- Run the parity test and a backward-gradient check after architectural edits.
- Verify BEV spatial orientation before changing tokenizer flattening or pooling.

