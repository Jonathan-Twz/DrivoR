"""Lightweight BEV feature map tokenizer.

Converts a precomputed BEV feature map (B, C, H, W) into a sequence of tokens
(B, K, d_model) suitable as K/V for cross-attention inside the scorer.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class BevTokenizer(nn.Module):
    """Conv patchify + optional adaptive pooling + learnable positional embedding.

    Input:  (B, C, H, W)
    Output: (B, K, d_model) where K = num_tokens if set else (H/patch_size) * (W/patch_size)
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_size: int = 8,
        spatial_hw: Optional[List[int]] = None,
        num_tokens: Optional[int] = None,
        use_self_attn_block: bool = False,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(self.d_model)

        # Determine grid size after patchify (and optional pool) for pos embed
        self._pool: Optional[nn.AdaptiveAvgPool2d] = None
        if spatial_hw is not None and len(spatial_hw) == 2:
            h_after = int(spatial_hw[0]) // self.patch_size
            w_after = int(spatial_hw[1]) // self.patch_size
        else:
            h_after, w_after = None, None

        k_final: Optional[int]
        if num_tokens is not None and num_tokens > 0:
            side = int(round(float(num_tokens) ** 0.5))
            if side * side != int(num_tokens):
                raise ValueError(
                    f"bev_tokenizer.num_tokens must be a perfect square, got {num_tokens}"
                )
            self._pool = nn.AdaptiveAvgPool2d((side, side))
            k_final = side * side
        elif h_after is not None and w_after is not None:
            k_final = h_after * w_after
        else:
            k_final = None

        if k_final is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, k_final, self.d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None  # will be lazily created on first forward

        self.use_self_attn_block = bool(use_self_attn_block)
        if self.use_self_attn_block:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=int(self.d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.self_attn_block = nn.TransformerEncoder(encoder_layer, num_layers=1)
        else:
            self.self_attn_block = None

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)
        x = self.patch_embed(bev)  # (B, d_model, h', w')
        if self._pool is not None:
            x = self._pool(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, K, d_model)

        if self.pos_embed is None:
            # Lazy init to match runtime shape
            k = x.shape[1]
            pos = torch.zeros(1, k, self.d_model, device=x.device, dtype=x.dtype)
            nn.init.trunc_normal_(pos, std=0.02)
            self.pos_embed = nn.Parameter(pos)
        x = x + self.pos_embed

        x = self.norm(x)

        if self.self_attn_block is not None:
            x = self.self_attn_block(x)

        return x
