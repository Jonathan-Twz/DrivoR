"""BEV-conditioned residual refinement of frozen DrivoR proposals."""

from typing import Tuple

import torch
import torch.nn as nn

from navsim.agents.drivoR.timm_layers import Mlp
from navsim.agents.drivoR.transformer_decoder import Attention


class BevResidualProposalRefinerBlock(nn.Module):
    """Update one path query per proposal using global BEV cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attn_norm_q = nn.LayerNorm(dim)
        self.cross_attn_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, num_heads=num_heads, proj_drop=dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=dropout,
        )

    def forward(self, path_queries: torch.Tensor, bev_tokens: torch.Tensor) -> torch.Tensor:
        path_queries = path_queries + self.cross_attn(
            self.cross_attn_norm_q(path_queries),
            self.cross_attn_norm_kv(bev_tokens),
        )
        return path_queries + self.mlp(self.mlp_norm(path_queries))


class BevResidualProposalRefiner(nn.Module):
    """Predict gated trajectory deltas after the original trajectory decoder."""

    def __init__(self, config) -> None:
        super().__init__()
        refiner_cfg = config.get("bev_residual_proposal_refiner", {})

        def _get(key, default):
            try:
                return refiner_cfg.get(key, default)
            except Exception:
                return getattr(refiner_cfg, key, default)

        dim = int(config.tf_d_model)
        self.num_poses = int(config.num_poses)
        self.state_size = 3
        num_layers = int(_get("num_layers", 1))
        num_heads = int(_get("num_heads", 1))
        dropout = float(_get("dropout", 0.0))
        init_alpha = float(_get("init_alpha", 0.0))

        if num_layers < 1:
            raise ValueError("bev_residual_proposal_refiner.num_layers must be >= 1")

        self.layers = nn.ModuleList(
            [
                BevResidualProposalRefinerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(config.tf_d_ffn)),
            nn.GELU(),
            nn.Linear(int(config.tf_d_ffn), self.num_poses * self.state_size),
        )
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(
        self,
        base_proposals: torch.Tensor,
        path_queries: torch.Tensor,
        bev_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bev_tokens is None:
            return base_proposals, torch.zeros_like(base_proposals)

        x = path_queries
        for layer in self.layers:
            x = layer(x, bev_tokens)

        delta = self.delta_head(x).reshape(
            base_proposals.shape[0],
            base_proposals.shape[1],
            self.num_poses,
            self.state_size,
        )
        return base_proposals + self.alpha * delta, delta
