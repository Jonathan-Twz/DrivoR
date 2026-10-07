"""Compare production pool and serial PDM outputs on real metric caches."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nuplan-devkit"))


def main():
    import numpy as np
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.agents.drivoR.score_module.compute_navsim_score import get_scores
    from navsim.agents.drivoR.score_module.scoring_pool import ScoringPool

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-cache", type=Path, default=Path(os.environ.get("NAVSIM_EXP_ROOT", ROOT / "exp")) / "train_metric_cache")
    parser.add_argument("--scenes", type=int, default=4)
    args = parser.parse_args()
    paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    tokens = sorted(paths)[:args.scenes]
    if len(tokens) != args.scenes:
        raise RuntimeError("Not enough verified metric-cache scenes")
    proposals = np.zeros((4, 8, 3), dtype=np.float32)
    for index, speed in enumerate((0., 2., 5., 10.)):
        proposals[index, :, 0] = np.arange(1, 9) * .5 * speed
    pool = ScoringPool(3)
    try:
        for test in (False, True):
            points = [dict(token=paths[token], poses=proposals.copy(), test=test) for token in tokens]
            reference = get_scores(points)
            output = pool(points)
            assert len(reference) == len(output) == args.scenes
            for serial, parallel in zip(reference, output):
                assert len(serial) == len(parallel)
                for left, right in zip(serial, parallel):
                    assert np.isfinite(left).all(), "Non-finite reference scoring output"
                    assert np.isfinite(right).all(), "Non-finite parallel scoring output"
                    np.testing.assert_array_equal(left, right)
            print(f"PASS: test={test}, {args.scenes} real scenes, all scores/corners/labels/ego areas exactly identical", flush=True)
        print("PASS: production three-worker pool reused for train and validation", flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
