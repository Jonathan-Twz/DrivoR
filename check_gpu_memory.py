#!/usr/bin/env python3
"""Check CUDA GPU memory with PyTorch.

Usage:
  python scripts/evaluation/check_gpu_memory.py
  python scripts/evaluation/check_gpu_memory.py --gpus 0,1,2,4
"""

import argparse
import os


def _gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GPU memory through torch.")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4",
        help="Comma-separated physical GPU ids to expose before importing torch.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import torch

    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"visible_device_count={torch.cuda.device_count()}")
    print()

    if not torch.cuda.is_available():
        return

    physical_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    header = (
        f"{'visible':>7} {'physical':>8} {'name':<32} "
        f"{'free GiB':>9} {'total GiB':>9} {'allocated GiB':>13} {'reserved GiB':>12}"
    )
    print(header)
    print("-" * len(header))

    for visible_idx in range(torch.cuda.device_count()):
        physical_id = physical_ids[visible_idx] if visible_idx < len(physical_ids) else "?"
        free_bytes, total_bytes = torch.cuda.mem_get_info(visible_idx)
        allocated_bytes = torch.cuda.memory_allocated(visible_idx)
        reserved_bytes = torch.cuda.memory_reserved(visible_idx)
        name = torch.cuda.get_device_name(visible_idx)
        print(
            f"{visible_idx:>7} {physical_id:>8} {name:<32} "
            f"{_gib(free_bytes):>9.2f} {_gib(total_bytes):>9.2f} "
            f"{_gib(allocated_bytes):>13.2f} {_gib(reserved_bytes):>12.2f}"
        )


if __name__ == "__main__":
    main()
