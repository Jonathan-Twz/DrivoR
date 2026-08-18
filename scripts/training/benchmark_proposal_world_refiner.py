#!/usr/bin/env python3
"""Measure proposal-world refiner parameters, latency, and peak CUDA memory."""

import argparse
import json
import statistics
import time

import torch
from omegaconf import OmegaConf

from navsim.agents.drivoR.layers.proposal_world_refiner import (
    ProposalConditionedWorldRefiner,
)
from navsim.agents.drivoR.layers.bev_residual_proposal_refiner import (
    BevResidualProposalRefiner,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--variant", choices=["proposal_world", "static_bev_refiner"], default="proposal_world"
    )
    args = parser.parse_args()

    config = OmegaConf.create(
        {
            "tf_d_model": 256,
            "tf_d_ffn": 1024,
            "num_poses": 8,
            "bev_tokenizer": {"num_tokens": 64},
            "proposal_world_refiner": {
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 512,
                "rollout_steps": 1,
                "proposal_chunk_size": args.chunk_size,
                "init_refine_gate": 1.0,
                "init_score_gate": 1.0,
                "dropout": 0.0,
            },
            "bev_residual_proposal_refiner": {
                "num_layers": 2,
                "num_heads": 4,
                "init_alpha": 1.0,
                "dropout": 0.0,
            },
        }
    )
    device = torch.device(args.device)
    if args.variant == "proposal_world":
        module = ProposalConditionedWorldRefiner(config).to(device).eval()
    else:
        module = BevResidualProposalRefiner(config).to(device).eval()
    base = torch.randn(args.batch, 64, 8, 3, device=device)
    queries = torch.randn(args.batch, 64, 256, device=device)
    bev = torch.randn(args.batch, 64, 256, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            module(base, queries, bev)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        durations = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            outputs = module(base, queries, bev)
            synchronize()
            durations.append((time.perf_counter() - start) * 1000.0)

    if args.variant == "proposal_world":
        refined, delta, score_residual, diagnostics = outputs
        score_residual_shape = list(score_residual.shape)
        context_norm = float(diagnostics["proposal_world_context_norm"])
    else:
        refined, delta = outputs
        score_residual_shape = None
        context_norm = None

    result = {
        "device": str(device),
        "variant": args.variant,
        "batch": args.batch,
        "proposal_count": 64,
        "bev_token_count": 64,
        "chunk_size": args.chunk_size,
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "mean_latency_ms": statistics.mean(durations),
        "median_latency_ms": statistics.median(durations),
        "max_latency_ms": max(durations),
        "refined_shape": list(refined.shape),
        "delta_shape": list(delta.shape),
        "score_residual_shape": score_residual_shape,
        "context_norm": context_norm,
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
