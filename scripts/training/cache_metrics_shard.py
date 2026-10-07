"""Run an independent log shard of the standard navtrain metric cache on Slurm."""
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from navsim.planning.script.run_train_metric_caching import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    count = int(os.environ.get("CACHE_SHARDS", "32"))
    if not 0 <= index < count:
        raise ValueError(f"Invalid shard {index}/{count}")
    os.environ["NODE_RANK"] = str(index)
    with initialize_config_dir(
        config_dir=str(root / "navsim/planning/script/config/metric_caching"),
        version_base=None,
    ):
        cfg = compose(config_name="train_metric_caching", overrides=[
            "train_test_split=navtrain", "~train_test_split_synthetic", "worker=sequential",
            f"cache.cache_path={os.environ['CACHE_PATH']}",
        ])
    cfg.train_test_split.scene_filter.log_names = list(
        cfg.train_test_split.scene_filter.log_names
    )[index::count]
    token_file = os.environ.get("SCENE_FILTER_TOKEN_FILE")
    if token_file:
        # Include standard navtrain as well as the wider BEV training selection.
        extra = Path(token_file).read_text().splitlines()
        cfg.train_test_split.scene_filter.tokens = sorted(
            set(cfg.train_test_split.scene_filter.tokens)
            | {line.strip() for line in extra if line.strip() and not line.startswith("#")}
        )
    print(f"Shard {index}/{count}: {len(cfg.train_test_split.scene_filter.log_names)} logs", flush=True)
    main(cfg)
