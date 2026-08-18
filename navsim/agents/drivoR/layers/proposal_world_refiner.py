"""Proposal-conditioned BEV world model with trajectory refinement adapters."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class ProposalWorldEncoder(nn.Module):
    """Roll a current BEV token grid forward under one candidate trajectory."""

    def __init__(
        self,
        d_model: int,
        num_bev_tokens: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rollout_steps: int,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_bev_tokens + 1, d_model)
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.rollout_steps = rollout_steps

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        current_bev_tokens: torch.Tensor,
        trajectory_queries: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, d_model = trajectory_queries.shape
        num_bev_tokens = current_bev_tokens.shape[1]

        action = self.action_projection(trajectory_queries).reshape(
            batch_size * num_candidates, 1, d_model
        )
        bev = current_bev_tokens[:, None].expand(
            batch_size, num_candidates, num_bev_tokens, d_model
        )
        bev = bev.reshape(batch_size * num_candidates, num_bev_tokens, d_model)
        sequence = torch.cat([action, bev], dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.shape[1]]

        for _ in range(self.rollout_steps):
            sequence = self.encoder(sequence)

        return self.output_norm(sequence[:, 1:]).reshape(
            batch_size, num_candidates, num_bev_tokens, d_model
        )


class ProposalConditionedWorldRefiner(nn.Module):
    """Predict candidate futures, refine proposals, and inform frozen scoring."""

    def __init__(self, config) -> None:
        super().__init__()
        world_cfg = config.get("proposal_world_refiner", {})

        def _get(key, default):
            try:
                return world_cfg.get(key, default)
            except Exception:
                return getattr(world_cfg, key, default)

        d_model = int(config.tf_d_model)
        num_bev_tokens = int(config.get("bev_tokenizer", {}).get("num_tokens", 64))
        num_heads = int(_get("num_heads", 4))
        num_layers = int(_get("num_layers", 2))
        ffn_dim = int(_get("ffn_dim", 512))
        dropout = float(_get("dropout", 0.0))
        rollout_steps = int(_get("rollout_steps", 1))

        if num_layers < 1 or rollout_steps < 1:
            raise ValueError("proposal world model layers and rollout steps must be >= 1")
        if d_model % num_heads != 0:
            raise ValueError("proposal world model d_model must be divisible by num_heads")

        self.world_model = ProposalWorldEncoder(
            d_model=d_model,
            num_bev_tokens=num_bev_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            rollout_steps=rollout_steps,
        )
        self.future_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.future_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)

        num_poses = int(config.num_poses)
        state_size = 3
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, num_poses * state_size),
        )
        self.scorer_adapter = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

        self.refine_gate = nn.Parameter(
            torch.tensor(float(_get("init_refine_gate", 0.0)), dtype=torch.float32)
        )
        self.score_gate = nn.Parameter(
            torch.tensor(float(_get("init_score_gate", 0.0)), dtype=torch.float32)
        )
        self.chunk_size = int(_get("proposal_chunk_size", 8))
        self.num_poses = num_poses
        self.state_size = state_size

        if self.chunk_size < 1:
            raise ValueError("proposal_world_refiner.proposal_chunk_size must be >= 1")

    def _process_chunk(
        self,
        current_bev_tokens: torch.Tensor,
        trajectory_queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        future_tokens = self.world_model(current_bev_tokens, trajectory_queries)
        batch_size, num_candidates, num_tokens, d_model = future_tokens.shape

        query = self.query_norm(trajectory_queries).reshape(
            batch_size * num_candidates, 1, d_model
        )
        future = self.future_norm(future_tokens).reshape(
            batch_size * num_candidates, num_tokens, d_model
        )
        context = self.future_attention(
            query=query, key=future, value=future, need_weights=False
        )[0]
        context = self.context_norm(context.squeeze(1)).reshape(
            batch_size, num_candidates, d_model
        )
        return context, future_tokens

    def forward(
        self,
        base_proposals: torch.Tensor,
        trajectory_queries: torch.Tensor,
        current_bev_tokens: torch.Tensor,
        return_future_tokens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if current_bev_tokens is None:
            zeros = torch.zeros_like(base_proposals)
            score_zeros = torch.zeros_like(trajectory_queries)
            return base_proposals, zeros, score_zeros, {}

        contexts = []
        futures = []
        for start in range(0, trajectory_queries.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, trajectory_queries.shape[1])
            context, future = self._process_chunk(
                current_bev_tokens, trajectory_queries[:, start:end]
            )
            contexts.append(context)
            if return_future_tokens:
                futures.append(future)

        context = torch.cat(contexts, dim=1)
        delta = self.delta_head(context).reshape(
            base_proposals.shape[0],
            base_proposals.shape[1],
            self.num_poses,
            self.state_size,
        )
        refined = base_proposals + self.refine_gate * delta
        scorer_residual = self.score_gate * self.scorer_adapter(context)

        diagnostics = {
            "proposal_world_refine_gate": self.refine_gate,
            "proposal_world_score_gate": self.score_gate,
            "proposal_world_context_norm": context.detach().norm(dim=-1).mean(),
        }
        if return_future_tokens:
            diagnostics["proposal_future_tokens"] = torch.cat(futures, dim=1)
        return refined, delta, scorer_residual, diagnostics
