"""Verify full scene-filter coverage and consolidate Slurm cache metadata.

The legacy MetricCacheLoader reads just one CSV, so archive shard manifests
only after a complete, verified combined manifest has been written.
"""
import csv
import json
import lzma
import os
import pickle
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parents[2]
    cache = Path(os.environ.get("CACHE_PATH", root / "exp/train_metric_cache"))
    data = Path(os.environ.get("OPENSCENE_DATA_ROOT", root.parent / "navsim_dataset"))
    cfg = yaml.safe_load((root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml").read_text())
    requested = set(cfg["tokens"])
    token_file = Path(os.environ.get("SCENE_FILTER_TOKEN_FILE", root / "exp/bev_feature_tokens/trainval_decoder_neck_tokens_full.txt"))
    requested.update(token_file.read_text().splitlines())
    expected = set()
    for name in cfg["log_names"]:
        with (data / "navsim_logs/trainval" / (name + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        history, future = cfg["num_history_frames"], cfg["num_future_frames"]
        for start in range(0, len(frames) - history - future + 1, cfg["frame_interval"]):
            frame = frames[start + history - 1]
            if frame["token"] in requested and (not cfg["has_route"] or frame["roadblock_ids"]):
                expected.add(frame["token"])
    paths = {}
    manifests = sorted((cache / "metadata").glob("*.csv"))
    if not manifests:
        raise RuntimeError("No shard manifests found")
    for manifest in manifests:
        with manifest.open() as stream:
            for row in csv.DictReader(stream):
                path = Path(row["file_name"])
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Missing/empty cache: {path}")
                token = path.parent.name
                if token in paths and paths[token] != path:
                    raise RuntimeError(f"Duplicate token in different cache paths: {token}")
                paths[token] = path
    missing = sorted(expected - paths.keys())
    if missing:
        raise RuntimeError(f"Missing {len(missing)} eligible scenes; examples: {missing[:10]}")
    # Read a deterministic sample across the output, in addition to full coverage checks.
    sample = sorted(paths)[::max(1, len(paths) // 64)]
    for token in sample:
        with lzma.open(paths[token], "rb") as stream:
            item = pickle.load(stream)
        for field in ["ego_state", "observation", "centerline", "route_lane_ids", "drivable_area_map", "pdm_progress"]:
            if not hasattr(item, field):
                raise RuntimeError(f"Invalid training cache {paths[token]}: missing {field}")
    combined = cache / "metadata/train_metric_cache_all.csv"
    temporary = combined.with_suffix(".tmp")
    with temporary.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name"])
        writer.writerows([[str(paths[token])] for token in sorted(paths)])
    temporary.replace(combined)
    archived = cache / "metadata/shards"
    archived.mkdir(exist_ok=True)
    for manifest in manifests:
        if manifest != combined:
            manifest.replace(archived / manifest.name)
    summary = {"requested_tokens": len(requested), "eligible_tokens": len(expected),
               "cached_tokens": len(paths), "missing_eligible_tokens": len(missing),
               "sampled_pickles_loaded": len(sample), "manifest": str(combined)}
    (cache / "metadata/verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
