# DrivoR Evaluation Figures

These figures are generated from the verified values in `docs/evaluation-ledger.md`:

- `01_official_test_results`: NAVSIM v1 PDMS, v2 Stage 1, and v2 Combined EPDMS.
- `02_submetric_heatmaps`: NAVSIM v1 and v2 Stage 2 submetric breakdowns.
- `03_validation_proxy_vs_test`: training-time validation proxy versus official evaluation results.
- `04_epoch_time_and_v2_tradeoff`: recorded per-epoch train/validation wall time and v2 outcome.
- `05_train_validation_metric_comparison`: parallel train loss, validation score, driving/compliance, and proposal-diagnostic comparisons.

Generate both PNG and PDF versions from the repository root:

```bash
MPLCONFIGDIR=/tmp/drivor-matplotlib \
  /mnt/ws-frb/users/jingyuso/miniconda3/envs/drivoR-share/bin/python \
  scripts/viz/plot_evaluation_ledger.py
```

The pretrained baseline v1 PDMS and submetrics come from two complete navtest runs with identical aggregate results and explicit Hydra checkpoint provenance. The Jun29 LoRA16 `last` and `best` checkpoints have identical recorded metrics and are shown once. Legacy evaluations with incomplete provenance are excluded. Correlations are descriptive only because there are seven aligned checkpoints and `val/score_epoch` is not the official NAVSIM evaluator. Timing comparisons are also not controlled benchmarks because hardware/data-loader settings differ across runs.
