"""BEV-aware scorer blocks with side-LoRA adapters.

This module implements the dual-cross-attention scorer used when
``use_bev_feature`` is enabled.  The existing frozen ``nn.MultiheadAttention``
layers are kept as-is; a low-rank side branch is added in parallel to each
of them, and a new zero-initialized BEV-only cross-attention sublayer is
inserted between the original scene cross-attention and the MLP.
"""

from typing import Optional, Type

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.drivoR.timm_layers import Mlp, DropPath, LayerScale
from navsim.agents.drivoR.transformer_decoder import Attention


class SideLoRAAttn(nn.Module):
    """Low-rank parallel side branch for attention layers.

    Computes ``out_proj(attn(A_q x, A_k kv, A_v kv))`` with a low inner
    dimension ``r``.  The output projection is zero-initialized so the full
    branch starts as an identity contribution to the frozen MHA output.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 8,
        num_heads: int = 1,
        dropout: float = 0.0,
        cross: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.num_heads = max(1, int(num_heads))
        # Project to a low-rank inner space.  We keep the inner dim equal to rank
        # per head, so total inner dim = rank * num_heads.  For rank=8 and
        # num_heads=1 we have inner dim 8.
        inner = self.rank * self.num_heads
        self.inner = inner
        self.head_dim = max(1, inner // self.num_heads)

        self.norm_q = nn.LayerNorm(self.dim)
        self.norm_kv = nn.LayerNorm(self.dim) if cross else None

        self.q_proj = nn.Linear(self.dim, inner, bias=False)
        self.k_proj = nn.Linear(self.dim, inner, bias=False)
        self.v_proj = nn.Linear(self.dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, self.dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.cross = bool(cross)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # q/k/v: kaiming uniform (small), out_proj: zero-init to guarantee
        # delta = 0 at initialization.
        nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_in = self.norm_q(x)
        if self.cross:
            assert kv is not None, "SideLoRAAttn configured as cross but kv is None"
            kv_in = self.norm_kv(kv)
        else:
            kv_in = q_in

        b, nq, _ = q_in.shape
        nk = kv_in.shape[1]
        q = self.q_proj(q_in).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(b, nq, self.inner)
        out = self.out_proj(attn)
        return self.drop(out)


class BevAwareBlock(nn.Module):
    """Transformer decoder block with dual cross-attention and side-LoRA.

    Structure (x = proposals embedded, x_cross = scene_features,
    x_bev = bev_tokens):
        x = x + drop_path(ls(self_attn(norm(x))) + side_lora_self(x))
        x = x + drop_path(ls(cross_attn(norm_q(x), norm_kv(x_cross))) + side_lora_cross(x, x_cross))
        x = x + drop_path(ls_bev(cross_attn_bev(norm_q_bev(x), norm_kv_bev(x_bev))))   # zero-init gated
        x = x + drop_path(ls(mlp(norm(x))) + side_lora_mlp(x))

    All components on the right of ``+`` that are labelled "frozen" are loaded
    from the pre-trained checkpoint.  The side-LoRA, ``cross_attn_bev`` and
    its ``LayerScale`` are the only additions and are the only trainable
    parameters inside the block when the default freeze policy is used.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.0,
        lora_rank: int = 8,
        lora_dropout: float = 0.0,
        add_new_cross_attn: bool = True,
        bev_init_gate: float = 0.0,
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

        # ---- new BEV cross-attention sublayer (zero-init gated) ----
        self.add_new_cross_attn = bool(add_new_cross_attn)
        if self.add_new_cross_attn:
            self.cross_attn_bev_norm_q = norm_layer(dim)
            self.cross_attn_bev_norm_kv = norm_layer(dim)
            self.cross_attn_bev = Attention(dim, num_heads=num_heads, proj_drop=proj_drop)
            # Force LayerScale with a (possibly zero) gate so the new branch
            # starts as identity at initialization.
            self.cross_attn_bev_ls = LayerScale(dim, init_values=float(bev_init_gate))
            self.cross_attn_bev_drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

        # ---- side-LoRA adapters on the frozen MHA layers ----
        self.self_attn_lora = SideLoRAAttn(
            dim, rank=lora_rank, num_heads=num_heads, dropout=lora_dropout, cross=False
        )
        self.cross_attn_lora = SideLoRAAttn(
            dim, rank=lora_rank, num_heads=num_heads, dropout=lora_dropout, cross=True
        )
        # MLP LoRA: a tiny residual MLP at rank r with zero-init output
        self.mlp_lora = _LowRankMlp(dim=dim, rank=lora_rank, dropout=lora_dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        x_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- self-attn (frozen) + side-LoRA
        sa = self.self_attn_ls(self.self_attn(self.self_attn_norm(x)))
        sa_lora = self.self_attn_lora(x)
        x = x + self.self_attn_drop_path(sa + sa_lora)

        # --- cross-attn over scene_features (frozen) + side-LoRA
        ca = self.cross_attn_ls(
            self.cross_attn(self.cross_attn_norm_q(x), self.cross_attn_norm_kv(x_cross))
        )
        ca_lora = self.cross_attn_lora(x, x_cross)
        x = x + self.cross_attn_drop_path(ca + ca_lora)

        # --- BEV-only cross-attn (zero-init gate)
        if self.add_new_cross_attn and x_bev is not None:
            cab = self.cross_attn_bev_ls(
                self.cross_attn_bev(
                    self.cross_attn_bev_norm_q(x), self.cross_attn_bev_norm_kv(x_bev)
                )
            )
            x = x + self.cross_attn_bev_drop_path(cab)

        # --- MLP (frozen) + side-LoRA
        m = self.mlp_ls(self.mlp(self.mlp_norm(x)))
        m_lora = self.mlp_lora(x)
        x = x + self.mlp_drop_path(m + m_lora)

        return x


class _LowRankMlp(nn.Module):
    """Tiny residual MLP at rank ``r`` with zero-init output."""

    def __init__(self, dim: int, rank: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.up(self.act(self.down(self.norm(x)))))


class BevAwareScorer(nn.Module):
    """Drop-in replacement for ``TransformerDecoderScorer`` with BEV + LoRA."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        proj_drop: float,
        drop_path: float,
        config,
    ) -> None:
        super().__init__()
        num_heads = config.refiner_num_heads if hasattr(config, "refiner_num_heads") else 1
        init_values = config.refiner_ls_values if hasattr(config, "refiner_ls_values") else 0.0

        scorer_bev = config.get("scorer_bev", {}) if hasattr(config, "get") else {}
        # OmegaConf DictConfig supports .get as well
        def _get(k, default):
            try:
                return scorer_bev.get(k, default)
            except Exception:
                return getattr(scorer_bev, k, default)

        lora_rank = int(_get("lora_rank", 8))
        lora_dropout = float(_get("lora_dropout", 0.0))
        add_new_cross_attn = bool(_get("add_new_cross_attn", True))
        init_gate = float(_get("init_gate", 0.0))

        layers = []
        for _ in range(num_layers):
            layers.append(
                BevAwareBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                    init_values=init_values,
                    lora_rank=lora_rank,
                    lora_dropout=lora_dropout,
                    add_new_cross_attn=add_new_cross_attn,
                    bev_init_gate=init_gate,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.return_intermediate = False

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        x_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, x_cross, x_bev)
        return x
