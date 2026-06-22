"""BEV-aware trajectory-decoder blocks.

This module implements BEV injection into the trajectory generator
(``TransformerDecoder``).  Each block keeps the original frozen sublayers
(self-attn, scene cross-attn, MLP) with the exact same state_dict keys as
``transformer_decoder.Block``, and inserts a new zero/gate-initialized
BEV-only cross-attention sublayer between the scene cross-attention and the
MLP (same sequential structure as the BEV-aware scorer).  Optionally,
side-LoRA adapters (reused from ``bev_scorer_blocks``) are attached in
parallel to each frozen sublayer when ``decoder_bev.lora_rank > 0``:

    x = x + drop_path(ls(self_attn(norm(x))) + self_attn_lora(x))
    x = x + drop_path(ls(cross_attn(norm_q(x), norm_kv(x_cross))) + cross_attn_lora(x, x_cross))
    x = x + drop_path(ls_bev(cross_attn_bev(norm_q_bev(x), norm_kv_bev(x_bev))))
    x = x + drop_path(ls(mlp(norm(x))) + mlp_lora(x))

With ``init_gate=0`` (LayerScale gamma) and zero-initialized LoRA output
projections the block is an exact identity extension of the original block,
so a pre-trained checkpoint reproduces the baseline outputs bit-for-bit at
initialization.
"""

from typing import Optional, Type

import torch
import torch.nn as nn

from navsim.agents.drivoR.timm_layers import Mlp, DropPath, LayerScale
from navsim.agents.drivoR.transformer_decoder import Attention
from navsim.agents.drivoR.layers.bev_scorer_blocks import SideLoRAAttn, _LowRankMlp


class BevAwareTrajDecoderBlock(nn.Module):
    """Trajectory-decoder block with an extra gated BEV cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.0,
        bev_init_gate: float = 0.0,
        lora_rank: int = 0,
        lora_dropout: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # ---- mirrors original Block (so state_dict keys match) ----
        self.self_attn_norm = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.self_attn_ls = (
            LayerScale(dim, init_values=init_values) if init_values > 0 else nn.Identity()
        )
        self.self_attn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cross_attn_norm_kv = norm_layer(dim)
        self.cross_attn_norm_q = norm_layer(dim)
        self.cross_attn = Attention(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.cross_attn_ls = (
            LayerScale(dim, init_values=init_values) if init_values > 0 else nn.Identity()
        )
        self.cross_attn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp_norm = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=None,
            bias=True,
            drop=proj_drop,
        )
        self.mlp_ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ---- new BEV cross-attention sublayer (gate-initialized) ----
        self.cross_attn_bev_norm_q = norm_layer(dim)
        self.cross_attn_bev_norm_kv = norm_layer(dim)
        self.cross_attn_bev = Attention(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.cross_attn_bev_ls = LayerScale(dim, init_values=float(bev_init_gate))
        self.cross_attn_bev_drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # ---- optional side-LoRA adapters on the frozen sublayers ----
        self.use_lora = int(lora_rank) > 0
        if self.use_lora:
            self.self_attn_lora = SideLoRAAttn(
                dim, rank=lora_rank, num_heads=num_heads, dropout=lora_dropout, cross=False
            )
            self.cross_attn_lora = SideLoRAAttn(
                dim, rank=lora_rank, num_heads=num_heads, dropout=lora_dropout, cross=True
            )
            self.mlp_lora = _LowRankMlp(dim=dim, rank=lora_rank, dropout=lora_dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        x_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- self-attn (frozen) + optional side-LoRA
        sa = self.self_attn_ls(self.self_attn(self.self_attn_norm(x)))
        if self.use_lora:
            sa = sa + self.self_attn_lora(x)
        x = x + self.self_attn_drop_path(sa)

        # --- cross-attn over scene_features (frozen) + optional side-LoRA
        ca = self.cross_attn_ls(
            self.cross_attn(self.cross_attn_norm_q(x), self.cross_attn_norm_kv(x_cross))
        )
        if self.use_lora:
            ca = ca + self.cross_attn_lora(x, x_cross)
        x = x + self.cross_attn_drop_path(ca)

        # --- BEV-only cross-attn (gated; skipped entirely when no BEV tokens)
        if x_bev is not None:
            x = x + self.cross_attn_bev_drop_path(
                self.cross_attn_bev_ls(
                    self.cross_attn_bev(
                        self.cross_attn_bev_norm_q(x), self.cross_attn_bev_norm_kv(x_bev)
                    )
                )
            )

        # --- MLP (frozen) + optional side-LoRA
        m = self.mlp_ls(self.mlp(self.mlp_norm(x)))
        if self.use_lora:
            m = m + self.mlp_lora(x)
        x = x + self.mlp_drop_path(m)

        return x


class BevAwareTrajectoryDecoder(nn.Module):
    """Drop-in replacement for ``TransformerDecoder`` with gated BEV cross-attn."""

    def __init__(self, proj_drop: float, drop_path: float, config) -> None:
        super().__init__()

        num_layers = config.ref_num
        d_model = config.tf_d_model
        num_heads = config.refiner_num_heads if hasattr(config, "refiner_num_heads") else 1
        init_values = config.refiner_ls_values if hasattr(config, "refiner_ls_values") else 0.0

        decoder_bev = config.get("decoder_bev", {}) if hasattr(config, "get") else {}

        def _get(k, default):
            try:
                return decoder_bev.get(k, default)
            except Exception:
                return getattr(decoder_bev, k, default)

        init_gate = float(_get("init_gate", 0.0))
        lora_rank = int(_get("lora_rank", 0))
        lora_dropout = float(_get("lora_dropout", 0.0))

        layers = []
        for _ in range(num_layers):
            layers.append(
                BevAwareTrajDecoderBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                    init_values=init_values,
                    bev_init_gate=init_gate,
                    lora_rank=lora_rank,
                    lora_dropout=lora_dropout,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.return_intermediate = True

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        x_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        intermediate = []
        for layer in self.layers:
            x = layer(x, x_cross, x_bev)
            if self.return_intermediate:
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return x
