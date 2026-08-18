# DrivoR Evaluation Ledger

This file records NAVSIM v1 PDMS and NAVSIM v2 EPDMS evaluations that were found in the local workspace.  Use this as the canonical local results ledger before comparing experiments.

Required fields for a complete evaluation record:

- Experiment date and evaluation date.
- Checkpoint path.
- Network setup: baseline, BEV scorer, BEV decoder, decoder+scorer, residual proposal refiner, or from-scratch.
- LoRA rank/layers and init gate/alpha.
- v1 PDMS and v1 sub-scores.
- v2 EPDMS stage-one, stage-two, combined, and v2 sub-scores.
- Result CSV path and success/failure counts.

## Coverage Audit

The recent June-July BEV evaluations are now documented with metrics and result CSVs.  The pretrained baseline NAVSIM v1 result was recovered from two complete May runs whose Hydra overrides explicitly bind `weights/checkpoints/drivor_Nav1_25epochs.pth` with BEV disabled.  Some other older May/early-June CSVs only preserve metric outputs; their exact checkpoint path and training setup were not recoverable from local logs, so they remain under "Legacy Recovered Metrics" and should not be used as primary paper/table evidence without re-verifying provenance.

## Result Figures

Presentation figures generated from the verified tables in this ledger are under `docs/figures/evaluation/` in PNG and PDF formats.  Regenerate them with `scripts/viz/plot_evaluation_ledger.py`.

- `01_official_test_results`: headline v1 PDMS, v2 Stage 1, and v2 Combined EPDMS comparison.
- `02_submetric_heatmaps`: v1 and v2 Stage 2 behavior breakdown.
- `03_validation_proxy_vs_test`: validation proxy versus official test metrics; descriptive only (`n=7`).
- `04_epoch_time_and_v2_tradeoff`: recorded epoch wall time and v2 performance; not a controlled speed benchmark because run settings differ.
- `05_train_validation_metric_comparison`: parallel comparison of train loss and validation proxy metrics by model configuration.

All categorical axes use the actual model configuration rather than the training date.  The Jun29 LoRA16 `last` and `best` rows have identical metrics and are plotted once.  Legacy rows with incomplete provenance are excluded from the main figures.

## Recent Complete Results

| Experiment | Train date | Eval date | Checkpoint | Network setup | LoRA / gate | v1 PDMS | v2 stage 1 | v2 stage 2 | v2 combined EPDMS |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| Pretrained DrivoR baseline | release checkpoint | 2026-05-26 / 2026-05-27 (v1); 2026-06-03 (v2) | `weights/checkpoints/drivor_Nav1_25epochs.pth` | Original DrivoR, no BEV | none | 0.936905 | 0.809321 | 0.594511 | 0.483144 |
| Pretrained DrivoR + current BEV decoder LoRA16 fine-tune | 2026-06-12 | 2026-06-14 / 2026-06-15 | `exp/ke/Jun12-golduck-4gpu-lora16-bev-decoder/06.12_01.12/checkpoints/last.ckpt` | Frozen pretrained DrivoR; current BEV in trajectory decoder only; scorer frozen | decoder LoRA rank 16, decoder init gate not explicitly logged in eval, scorer BEV off | 0.932158 | 0.841329 | 0.584989 | 0.496626 |
| Pretrained DrivoR + current BEV residual proposal refiner | 2026-06-22 | 2026-06-24 | `exp/ke/Jun22-golduck-4gpu-0initalpha-bev-residual-proposal-refiner/06.22_05.10/checkpoints/epoch=29-step=39870.ckpt` | Frozen pretrained DrivoR; post-decoder/pre-scorer BEV residual proposal refiner | residual refiner layers 1, heads 1, alpha init 0.0, no decoder/scorer BEV | 0.931695 | 0.842178 | 0.565484 | 0.478820 |
| DrivoR from scratch + current BEV decoder LoRA16, last | 2026-06-29 | 2026-07-03 | `exp/ke/Jun29-golduck-4gpu-lora16-0initgate-bev-trajectory-decoder-refiner-from-scratch-baseline-split-30epochs/06.29_bev_decoder_from_scratch_baseline_split_30ep/checkpoints/last.ckpt` | DrivoR from scratch with current BEV in trajectory decoder only | decoder LoRA rank 16, init gate 0.0, scorer BEV off | 0.932017 | 0.836660 | 0.536875 | 0.447110 |
| DrivoR from scratch + current BEV decoder LoRA16, best | 2026-06-29 | 2026-07-04 | `exp/ke/Jun29-golduck-4gpu-lora16-0initgate-bev-trajectory-decoder-refiner-from-scratch-baseline-split-30epochs/06.29_bev_decoder_from_scratch_baseline_split_30ep/checkpoints/best-epoch=29-step=48390.ckpt` | DrivoR from scratch with current BEV in trajectory decoder only | decoder LoRA rank 16, init gate 0.0, scorer BEV off | 0.932017 | 0.836660 | 0.536875 | 0.447110 |
| DrivoR from scratch + current BEV decoder LoRA8, best | 2026-07-05 | 2026-07-08 | `exp/ke/Jul05-golduck-4gpu-lora8-0initgate-bev-decoder-from-scratch-20epochs/07.05_22.47/checkpoints/best-epoch=19-step=32260.ckpt` | DrivoR from scratch with current BEV in trajectory decoder only | decoder LoRA rank 8, init gate 0.0, scorer BEV off | 0.931384 | 0.834874 | 0.535531 | 0.448434 |
| Pretrained DrivoR + current BEV decoder+scorer LoRA16 fine-tune | 2026-07-08 | 2026-07-11 | `exp/ke/Jul08-golduck-4gpu-lora16-0initgate-bev-decoder-scorer-finetune-30epochs/07.08_23.51/checkpoints/best-epoch=0-step=1329.ckpt` | Frozen pretrained DrivoR; current BEV in decoder and scorer | decoder LoRA rank 16, scorer LoRA rank 16, init gates 0.0 / 0.0 | 0.936007 | 0.847325 | 0.547436 | 0.467831 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 2 | 2026-07-14 | 2026-07-17 | `exp/ke/Jul14-golduck-4gpu-lora16-0initgate-current-future-bev-decoder-scorer/07.14_16.16/checkpoints/best-epoch=2-step=7113.ckpt` | Frozen pretrained DrivoR; current BEV in decoder and scorer; privileged future-BEV oracle enabled during training/validation only (`future_bev_num_steps=4`, `future_bev_stride=1`, future BEV used in decoder and scorer); official eval uses current BEV only | decoder LoRA rank 16, scorer LoRA rank 16, init gates 0.0 / 0.0 | 0.933425 | 0.837930 | 0.546367 | 0.461443 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 6 | 2026-07-14 | 2026-07-21 | `exp/ke/Jul14-golduck-4gpu-lora16-0initgate-current-future-bev-decoder-scorer/07.14_16.16/checkpoints/best-epoch=6-step=16597.ckpt` | Frozen pretrained DrivoR; current BEV in decoder and scorer; privileged future-BEV oracle enabled during training/validation only (`future_bev_num_steps=4`, `future_bev_stride=1`, future BEV used in decoder and scorer); official eval uses current BEV only | decoder LoRA rank 16, scorer LoRA rank 16, init gates 0.0 / 0.0 | 0.933163 | 0.849266 | 0.561461 | 0.480632 |

For the Jul14 current+future-BEV run, epoch 6 versus epoch 2 changes v1 PDMS by `-0.000262`, v2 Stage 1 by `+0.011336`, v2 Stage 2 by `+0.015094`, and v2 Combined EPDMS by `+0.019189`.  Epoch 6 v2 Combined remains `0.002512` below the pretrained DrivoR baseline (`0.483144`).

## Train/Val Proxy Metrics

These rows align each evaluated checkpoint with the corresponding training-time and validation-time proxy metrics, then place the official test-time NAVSIM v1/v2 metrics in the same row for comparison.  Sources: checkpoint `epoch` / `global_step`, local W&B history `run-*.wandb`, W&B stdout `files/output.log`, launcher logs, and the NAVSIM v1/v2 CSVs listed below.

Timing note: `epoch wall` is the epoch delta printed as `########### Epoch N (Xs) ##########`; `val wall` is the final validation progress duration; `train wall` is `epoch wall - val wall`.  `val/score_epoch` is a validation proxy, not official PDMS/EPDMS.

| Experiment | Epoch | Step | Epoch wall | Train wall | Val wall | train/loss_epoch | val/score_epoch | val/best_score | val/lost_score | val/hit | val/top5 hit | val/l2 | val/collision | val/dac | val/progress | val/ttc | val/comfort | Test v1 PDMS | Test v2 S1 | Test v2 S2 | Test v2 combined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained DrivoR + current BEV decoder LoRA16 fine-tune | 29 | 39870 | 2:07:37 | 1:55:42 | 0:11:55 | 0.847087 | 0.941252 | 0.975063 | 0.033812 | 0.117077 | 0.375880 | 0.398444 | 0.997277 | 0.991637 | 0.882842 | 0.988721 | 0.999945 | 0.932158 | 0.841329 | 0.584989 | 0.496626 |
| Pretrained DrivoR + current BEV residual proposal refiner | 29 | 39870 | 1:15:30 | 1:03:36 | 0:11:54 | 0.867485 | 0.941686 | 0.975324 | 0.033638 | 0.107724 | 0.354643 | 0.409636 | 0.997414 | 0.991637 | 0.884166 | 0.988446 | 0.999780 | 0.931695 | 0.842178 | 0.565484 | 0.478820 |
| DrivoR from scratch + current BEV decoder LoRA16, last | 29 | 48390 | 2:58:13 | 2:33:54 | 0:24:19 | 1.251491 | 0.958703 | 0.990338 | 0.031635 | 0.062995 | 0.252421 | 0.700153 | 0.997634 | 0.991637 | 0.925446 | 0.988061 | 0.999945 | 0.932017 | 0.836660 | 0.536875 | 0.447110 |
| DrivoR from scratch + current BEV decoder LoRA16, best | 29 | 48390 | 2:58:13 | 2:33:54 | 0:24:19 | 1.251491 | 0.958703 | 0.990338 | 0.031635 | 0.062995 | 0.252421 | 0.700153 | 0.997634 | 0.991637 | 0.925446 | 0.988061 | 0.999945 | 0.932017 | 0.836660 | 0.536875 | 0.447110 |
| DrivoR from scratch + current BEV decoder LoRA8, best | 19 | 32260 | 2:54:40 | 2:31:25 | 0:23:15 | 1.354433 | 0.946079 | 0.989947 | 0.043868 | 0.042749 | 0.188160 | 0.622069 | 0.995213 | 0.989437 | 0.905152 | 0.982119 | 0.999835 | 0.931384 | 0.834874 | 0.535531 | 0.448434 |
| Pretrained DrivoR + current BEV decoder+scorer LoRA16 fine-tune | 0 | 1329 | 1:04:24 | 0:53:56 | 0:10:28 | 0.952228 | 0.951959 | 0.990435 | 0.038476 | 0.050121 | 0.187280 | 0.639529 | 0.997112 | 0.990042 | 0.913017 | 0.986356 | 0.999945 | 0.936007 | 0.847325 | 0.547436 | 0.467831 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 2 | 2 | 7113 | 9:22:04 | 8:42:11 | 0:39:53 | 1.464653 | 0.919751 | 0.968898 | 0.049147 | 0.032088 | 0.149505 | 0.620705 | 0.991616 | 0.972409 | 0.877072 | 0.970160 | 0.999924 | 0.933425 | 0.837930 | 0.546367 | 0.461443 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 6 | 6 | 16597 | 15:11:22 | 14:31:50 | 0:39:32 | 1.446993 | 0.920095 | 0.968434 | 0.048339 | 0.033155 | 0.153544 | 0.604394 | 0.991521 | 0.971608 | 0.879482 | 0.969779 | 0.999962 | 0.933163 | 0.849266 | 0.561461 | 0.480632 |

Checkpoint paths for these rows are the same as in the "Recent Complete Results" table above.  The pretrained baseline has no train/val proxy metrics in this workspace.

## NAVSIM v1 PDMS Details

Columns: `NC` = no-at-fault collisions, `DAC` = drivable-area compliance, `EP` = ego progress, `TTC` = time-to-collision within bound, `C` = comfort, `DDC` = driving-direction compliance.

| Experiment | Result CSV | Valid / failed | PDMS | NC | DAC | EP | TTC | C | DDC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained DrivoR baseline | `exp/ke/drivoR_nav1/05.26_01.09/2026.05.26.01.30.46.csv` | 12146 / 0 | 0.936905 | 0.990367 | 0.989297 | 0.899420 | 0.967150 | 1.000000 | 0.972542 |
| Pretrained DrivoR + current BEV decoder LoRA16 fine-tune | `exp/ke/drivoR_nav1_decoder_bev_lora16_last_FULL/06.14_15.45/2026.06.14.16.03.11.csv` | 12146 / 0 | 0.932158 | 0.991314 | 0.990697 | 0.881224 | 0.971760 | 0.999835 | 0.977976 |
| Pretrained DrivoR + current BEV residual proposal refiner | `exp/ke/drivoR_nav1-bev-residual-proposal-refiner-epoch29/06.24_10.22/2026.06.24.10.41.24.csv` | 12146 / 0 | 0.931695 | 0.991685 | 0.989626 | 0.882383 | 0.970690 | 1.000000 | 0.977606 |
| DrivoR from scratch + current BEV decoder LoRA16, last | `exp/ke/drivoR_nav1_Jun29_bev_decoder_from_scratch_last/07.03_14.22/2026.07.03.14.39.39.csv` | 12146 / 0 | 0.932017 | 0.986992 | 0.985263 | 0.906308 | 0.955541 | 0.999753 | 0.971143 |
| DrivoR from scratch + current BEV decoder LoRA16, best | `exp/ke/drivoR_nav1_Jun29_bev_decoder_from_scratch_best/07.04_00.28/2026.07.04.00.45.43.csv` | 12146 / 0 | 0.932017 | 0.986992 | 0.985263 | 0.906308 | 0.955541 | 0.999753 | 0.971143 |
| DrivoR from scratch + current BEV decoder LoRA8, best | `exp/ke/drivoR_nav1_Jul05_lora8_best_epoch19_gpu0123/07.08_15.23/2026.07.08.15.40.17.csv` | 12146 / 0 | 0.931384 | 0.990120 | 0.985427 | 0.893936 | 0.965091 | 0.999753 | 0.967438 |
| Pretrained DrivoR + current BEV decoder+scorer LoRA16 fine-tune | `exp/ke/drivoR_nav1_Jul08_lora16_decoder_scorer_best_epoch0_gpu0123/07.11_11.14/2026.07.11.11.32.29.csv` | 12146 / 0 | 0.936007 | 0.990120 | 0.988144 | 0.901130 | 0.965091 | 1.000000 | 0.973078 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 2 | `exp/ke/drivoR_nav1_Jul14_current_future_bev_decoder_scorer_best_epoch2/07.17_03.12/2026.07.17.03.53.53.csv` | 12146 / 0 | 0.933425 | 0.989997 | 0.988556 | 0.893291 | 0.966656 | 0.999835 | 0.972254 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 6 | `exp/ke/drivoR_nav1_Jul14_current_future_bev_decoder_scorer_best_epoch6/07.21_01.45/2026.07.21.02.27.19.csv` | 12146 / 0 | 0.933163 | 0.989709 | 0.987650 | 0.894209 | 0.966079 | 0.999835 | 0.973325 |

The baseline was independently rerun at `exp/ke/drivoR_nav1/05.27_23.35/2026.05.27.23.57.04.csv`; it also has `12146 / 0` and the same aggregate metrics.  Both runs preserve the full Hydra override list with `agent.checkpoint_path=./weights/checkpoints/drivor_Nav1_25epochs.pth`, `train_test_split=navtest`, and no BEV overrides.

## NAVSIM v2 EPDMS Details

Columns: `LK` = lane keeping, `HC` = history comfort, `EC` = two-frame extended comfort, `TLC` = traffic-light compliance.

| Experiment | Result CSV | Valid / failed | Stage 1 | Stage 2 | Combined |
|---|---|---:|---:|---:|---:|
| Pretrained DrivoR baseline | `navsim/exp/drivoR_nav2_full/2026.06.03.17.36.37/2026.06.03.19.19.49.csv` | 5912 / 0 | 0.809321 | 0.594511 | 0.483144 |
| BEV scorer legacy run | `navsim/exp/drivoR_bev_nav2_full/2026.06.04.01.27.29/2026.06.04.03.28.00.csv` | 5912 / 0 | 0.838159 | 0.548902 | 0.462490 |
| BEV scorer rank8 gate0 legacy run | `navsim/exp/drivoR_bev_nav2_rank8_gate0/2026.06.04.23.11.29/2026.06.05.01.15.41.csv` | 5912 / 0 | 0.842452 | 0.554914 | 0.467980 |
| Pretrained DrivoR + current BEV decoder LoRA16 fine-tune | `navsim/exp/drivoR_nav2_decoder_bev_lora16_last/2026.06.15.00.29.11/2026.06.15.03.14.05.csv` | 5912 / 0 | 0.841329 | 0.584989 | 0.496626 |
| Pretrained DrivoR + current BEV residual proposal refiner | `navsim/exp/drivoR_nav2_bev_residual_proposal_refiner_epoch29/2026.06.24.11.11.37/2026.06.24.13.45.59.csv` | 5912 / 0 | 0.842178 | 0.565484 | 0.478820 |
| DrivoR from scratch + current BEV decoder LoRA16, last | `navsim/exp/drivoR_nav2_Jun29_bev_decoder_from_scratch_last/2026.07.03.14.41.27/2026.07.03.17.16.01.csv` | 5912 / 0 | 0.836660 | 0.536875 | 0.447110 |
| DrivoR from scratch + current BEV decoder LoRA16, best | `navsim/exp/drivoR_nav2_Jun29_bev_decoder_from_scratch_best/2026.07.04.00.46.32/2026.07.04.03.05.08.csv` | 5912 / 0 | 0.836660 | 0.536875 | 0.447110 |
| DrivoR from scratch + current BEV decoder LoRA8, best | `navsim/exp/drivoR_nav2_Jul05_lora8_best_epoch19_gpu0123/2026.07.08.15.41.48/2026.07.08.18.16.28.csv` | 5912 / 0 | 0.834874 | 0.535531 | 0.448434 |
| Pretrained DrivoR + current BEV decoder+scorer LoRA16 fine-tune | `navsim/exp/drivoR_nav2_Jul08_lora16_decoder_scorer_best_epoch0_gpu0123/2026.07.11.11.33.22/2026.07.11.14.08.10.csv` | 5912 / 0 | 0.847325 | 0.547436 | 0.467831 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 2 | `navsim/exp/drivoR_nav2_Jul14_current_future_bev_decoder_scorer_best_epoch2/2026.07.17.03.10.28/2026.07.17.05.12.14.csv` | 5912 / 0 | 0.837930 | 0.546367 | 0.461443 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 6 | `navsim/exp/drivoR_nav2_Jul14_current_future_bev_decoder_scorer_best_epoch6/2026.07.21.01.43.54/2026.07.21.03.48.42.csv` | 5912 / 0 | 0.849266 | 0.561461 | 0.480632 |

### v2 Submetrics

| Experiment | S1 NC | S1 DAC | S1 DDC | S1 TLC | S1 EP | S1 TTC | S1 LK | S1 HC | S1 EC | S2 NC | S2 DAC | S2 DDC | S2 TLC | S2 EP | S2 TTC | S2 LK | S2 HC | S2 EC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained DrivoR baseline | 0.987778 | 0.951111 | 0.988889 | 1.000000 | 0.726299 | 0.986667 | 0.940000 | 0.975556 | 0.733333 | 0.902004 | 0.883532 | 0.918618 | 0.986224 | 0.697924 | 0.879503 | 0.500725 | 0.985213 | 0.762156 |
| BEV scorer legacy | 0.990000 | 0.966667 | 0.996667 | 1.000000 | 0.785005 | 0.986667 | 0.942222 | 0.975556 | 0.662222 | 0.857041 | 0.861923 | 0.913993 | 0.987901 | 0.796581 | 0.836907 | 0.526670 | 0.985328 | 0.664906 |
| BEV scorer rank8 gate0 legacy | 0.990000 | 0.968889 | 0.996667 | 1.000000 | 0.784932 | 0.986667 | 0.942222 | 0.975556 | 0.675556 | 0.858344 | 0.862971 | 0.920636 | 0.987820 | 0.796874 | 0.832688 | 0.530658 | 0.981215 | 0.677226 |
| Pretrained DrivoR + current BEV decoder LoRA16 fine-tune | 0.988889 | 0.971111 | 0.996667 | 1.000000 | 0.766855 | 0.988889 | 0.935556 | 0.975556 | 0.724444 | 0.883254 | 0.872065 | 0.934825 | 0.993842 | 0.754583 | 0.859066 | 0.518615 | 0.981594 | 0.733703 |
| Pretrained DrivoR + current BEV residual proposal refiner | 0.990000 | 0.971111 | 0.998889 | 1.000000 | 0.772194 | 0.988889 | 0.944444 | 0.975556 | 0.688889 | 0.869743 | 0.859456 | 0.927707 | 0.991619 | 0.767012 | 0.851680 | 0.518043 | 0.979836 | 0.691413 |
| DrivoR from scratch + current BEV decoder LoRA16, last/best | 0.991111 | 0.960000 | 0.996667 | 1.000000 | 0.798603 | 0.984444 | 0.964444 | 0.973333 | 0.657778 | 0.872222 | 0.844193 | 0.908446 | 0.989370 | 0.766800 | 0.854214 | 0.496669 | 0.973460 | 0.620977 |
| DrivoR from scratch + current BEV decoder LoRA8, best | 0.985556 | 0.966667 | 0.992222 | 0.997778 | 0.777806 | 0.986667 | 0.948889 | 0.973333 | 0.702222 | 0.856191 | 0.858177 | 0.906976 | 0.992740 | 0.736758 | 0.838047 | 0.508690 | 0.980724 | 0.717520 |
| Pretrained DrivoR + current BEV decoder+scorer LoRA16 fine-tune | 0.990000 | 0.971111 | 0.997778 | 1.000000 | 0.786832 | 0.984444 | 0.946667 | 0.975556 | 0.697778 | 0.855054 | 0.844222 | 0.919130 | 0.990897 | 0.793148 | 0.836443 | 0.516290 | 0.977976 | 0.671874 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 2 | 0.983333 | 0.973333 | 0.996667 | 0.997778 | 0.783132 | 0.982222 | 0.944444 | 0.975556 | 0.693333 | 0.870881 | 0.836706 | 0.919632 | 0.991429 | 0.775803 | 0.839679 | 0.517425 | 0.978009 | 0.705899 |
| Pretrained DrivoR + current+future BEV decoder+scorer LoRA16 fine-tune, epoch 6 | 0.990000 | 0.971111 | 0.996667 | 0.997778 | 0.790084 | 0.986667 | 0.955556 | 0.975556 | 0.711111 | 0.870729 | 0.856781 | 0.915275 | 0.988053 | 0.796639 | 0.846325 | 0.529390 | 0.972375 | 0.674714 |

## Legacy Recovered Metrics

These rows were found as local v1 CSVs, but the exact checkpoint path and network setup were not present in the result directory logs. Keep them as forensic references only.

| Eval directory | Eval CSV | PDMS | NC | DAC | EP | TTC | C | DDC | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `drivoR_nav1-best-epoch=12-step=17277-01gate-16lora/06.01_15.00` | `2026.06.01.15.21.32.csv` | 0.937918 | 0.990244 | 0.989297 | 0.905115 | 0.963939 | 1.000000 | 0.972131 | likely BEV scorer checkpoint from filename, rank/gate implied but not fully recovered |
| `drivoR_nav1-best-epoch=13-step=18606/05.29_02.14` | `2026.05.29.02.36.07.csv` | 0.938103 | 0.990326 | 0.989379 | 0.903991 | 0.965503 | 1.000000 | 0.971925 | likely BEV scorer checkpoint from filename, exact setup not fully recovered |
| `drivoR_nav1/05.21_02.03` | `2026.05.21.02.20.11.csv` | 0.462722 | 0.878643 | 0.813848 | 0.238716 | 0.867693 | 0.594846 | 0.979335 | ambiguous failed/bad early eval |
| `drivoR_nav1/05.25_23.48` | `2026.05.26.00.12.27.csv` | 0.461473 | 0.856825 | 0.839453 | 0.246113 | 0.842335 | 0.597069 | 0.984316 | ambiguous failed/bad early eval |
| `drivoR_nav1/05.28_01.40` | `2026.05.28.02.02.28.csv` | 0.937776 | 0.989832 | 0.989379 | 0.903533 | 0.965256 | 1.000000 | 0.972378 | checkpoint/setup not recovered |
| `drivoR_nav1/05.28_21.11` | `2026.05.28.21.32.33.csv` | 0.937114 | 0.989297 | 0.989462 | 0.904579 | 0.962539 | 1.000000 | 0.972090 | checkpoint/setup not recovered |

## Documentation Gaps To Fix For Future Runs

- Always preserve the exact evaluation command or Hydra overrides next to the v1 CSV.  Several v1 result directories do not contain `agent.checkpoint_path`, so the checkpoint has to be inferred from the eval name or paired v2 run.
- Include both training checkpoint provenance and evaluation CSV path in any paper/table note.
- Record whether BEV is in decoder, scorer, both, residual refiner, or future-BEV oracle.  For future-BEV runs, note whether future tokens replace current BEV tokens or are fused.
- Record structure-changing params: `decoder_bev.lora_rank`, `scorer_bev.lora_rank`, `bev_channels`, `tf_d_model`, `tf_d_ffn`, `ref_num`, `proposal_num`, and residual-refiner layer/head counts.
- Record non-shape params for clarity: init gate, residual alpha init, cache/no-cache, GPUs, batch size, workers, and W&B run.
