# idea_0002_01 Fast Validation

## Question

Does a proposal-conditioned BEV rollout improve frozen DrivoR proposal refinement and scoring compared with direct current-BEV residual refinement under the same small-data protocol?

## Source-Grounded Design

- WoTE duplicates the current BEV for each trajectory, injects a trajectory/action representation, and recurrently applies a Transformer encoder to produce candidate-specific future BEV states. Source: `liyingyanUCAS/WoTE`, `navsim/agents/WoTE/WoTE_model.py`, especially `_latent_world_model_processing` and `extract_reward_feature`.
- ResWorld first predicts an initial trajectory, conditions its residual latent world model on that trajectory, reconstructs a predicted future BEV by adding the residual to current BEV, and refines waypoints by attention into the predicted BEV. Source: `mengtan00/ResWorld`, `projects/mmdet3d_plugin/resworld/resworld_head.py`, lines 222-294.
- This experiment ports the intersection of those mechanisms, not either full model: frozen DrivoR proposals -> candidate-conditioned BEV Transformer -> gated trajectory delta and gated scorer context.

## Controlled Variants

| Variant | Future rollout | Proposal-conditioned | Trajectory refinement | Scorer context |
|---|---:|---:|---:|---:|
| `static_bev_refiner` | No | Path query only | Direct current-BEV cross-attention | No |
| `proposal_world` | One Transformer rollout | Yes, all 64 proposals | Candidate-future cross-attention | Candidate-future adapter |

Both variants load the same pretrained checkpoint and freeze the original encoder, trajectory decoder, trajectory heads, scorer decoder, and score heads. Only the BEV tokenizer and the selected new module are optimized.

## Small-Data Protocol

- Source token list: `trainval_decoder_neck_tokens_full_metric_covered_after_cache.txt`
- Deterministic selection: SHA256 rank of `seed:token`, seed 2
- Subset: 2,048 tokens; 1,722 train-log tokens and 326 validation-log tokens
- Split isolation: train loader is restricted to `cfg.train_logs`; validation logs are excluded from training
- Subset SHA256: `24afe47e265b15908e51b2dc1042dffc9bb029509969918caf1840584a777988`
- Per run: 4 epochs, 64 train batches/epoch, 16 validation batches/epoch
- Batch: 4/GPU, gradient accumulation 4, bf16 mixed precision
- Optimizer and LR: AdamW, configured base LR `1e-4`; effective LR is `2.5e-5` for batch 4 under the repository's square-root batch scaling
- Seed: 2
- W&B project: `drivor-world-model-fast-validation`

The fixed token file is generated at `exp/idea0002_fast/fixed_tokens_seed2_n2048.txt`; its adjacent JSON manifest records source and output hashes.

## Verification Evidence

| Check | Status |
|---|---|
| Zero-gate baseline parity | PASS |
| Nonzero candidate-specific wiring and output shapes | PASS |
| Chunked vs unchunked numerical equivalence | PASS |
| Two-step gate activation and world-model gradients | PASS |
| Baseline checkpoint compatibility | PASS |
| Freeze whitelist excludes pretrained model | PASS |

Full configuration trainable parameters: 6,086,426 of 46,874,008 (12.98%), including the BEV tokenizer and proposal-world module.

Module-only capacity is closely matched: 1,868,825 parameters for the static BEV refiner and 1,874,970 for the proposal-world refiner (+0.33%). On the same CPU screening benchmark (batch 1, 64 proposals, 64 BEV tokens), mean latency was 2.37 ms versus 45.51 ms. This is evidence of relative compute structure only; A100 latency is required before making a deployment claim.

![Structural trade-off](../figures/idea0002_01/structural_tradeoff.png)

## Results

Fill from the paired W&B runs and local checkpoints.

| Metric | Static BEV refiner | Proposal world | Delta |
|---|---:|---:|---:|
| Best `val/score_epoch` | pending | pending | pending |
| Final `val/score_epoch` | pending | pending | pending |
| Final `val/l2` | pending | pending | pending |
| Final `train/trajectory_loss` | pending | pending | pending |
| Refine gate | pending | pending | n/a |
| Score gate | n/a | pending | n/a |
| Mean module latency (A100, batch 1) | pending | pending | pending |
| Peak module memory (A100, batch 1) | pending | pending | pending |

These small-subset results are a hypothesis screen, not official NAVSIM PDMS/EPDMS evidence. A positive result should be followed by a larger controlled run and official NAVSIM evaluation.
