# BEV 预计算特征与 navtrain 覆盖率

- 预计算 BEV 按 `{bev_features_root}/{split}/{log_name}/{token}_{vtransform|decoder_neck}.pt` 存放。
- OpenScene **navtrain** 过滤后的 log 数量（约 1192）可能多于已预计算 BEV 的 log 目录（例如约 863）。
- **策略**：在 [`DrivoRFeatureBuilder`](../navsim/agents/drivoR/drivor_features.py) 中，若对应 `.pt` 不存在，则使用与真实特征同形状的 **零张量** 作为 `bev_feature`，训练可继续；后续可只对「有 BEV 的 log」子集训练或补算 BEV。

## 配置与两阶段微调

- 开关与路径见 [`drivoR.yaml`](../navsim/planning/script/config/common/agent/drivoR.yaml) 中 `use_bev_feature`、`bev_features_root`、`bev_data_split`、`load_checkpoint_strict`、`freeze_pretrained_except_bev`。
- **阶段一**：`freeze_pretrained_except_bev=true`，仅优化 `bev_encoder` / `bev_proj`；**阶段二**：设为 `false` 全量微调（可从阶段一 checkpoint 恢复）。
- 前向自检：[`scripts/training/test_drivor_bev_forward.py`](../scripts/training/test_drivor_bev_forward.py)。
