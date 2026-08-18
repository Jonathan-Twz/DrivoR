#!/usr/bin/env python3
"""Build a deterministic NAVSIM token subset and a reproducibility manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    tokens = sorted(
        {
            line.strip()
            for line in args.source.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    )
    if args.count < 1 or args.count > len(tokens):
        raise ValueError(f"count must be in [1, {len(tokens)}], got {args.count}")

    def rank(token: str) -> str:
        return hashlib.sha256(f"{args.seed}:{token}".encode("ascii")).hexdigest()

    selected = sorted(tokens, key=rank)[: args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n")

    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": file_sha256(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "seed": args.seed,
        "count": len(selected),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
