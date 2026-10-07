"""CPU regression tests: python scripts/training/test_scoring_pool.py."""
import ast
from concurrent.futures.process import BrokenProcessPool
import os
from pathlib import Path
import pickle
import sys
import time
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from navsim.agents.drivoR.score_module.scoring_pool import ScoringPool


def task(point):
    # Out-of-order completion must still yield input-order outputs.
    time.sleep(point.get("delay", 0))
    return point["value"] ** 2


def failing_task(point):
    if point["value"] < 0:
        raise ValueError("invalid scene")
    return point["value"]


def worker_state(point):
    import torch
    return dict(pid=os.getpid(), threads=torch.get_num_threads(),
                cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_initialized=torch.cuda.is_initialized())


def extracted_class(relative_path, name, namespace):
    # Exercise lifecycle methods without constructing a heavyweight CUDA model.
    tree = ast.parse((ROOT / relative_path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative_path, "exec"), namespace)
    return namespace[name]


class PoolTests(unittest.TestCase):
    def test_parameter_validation(self):
        for bad in (-1, 1.5, "3", True):
            with self.assertRaises(ValueError):
                ScoringPool(bad)
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ScoringPool(3, bad)

    def test_serial_and_empty_are_lazy(self):
        serial = ScoringPool(0, score_fn=task)
        self.assertEqual(serial([{"value": 3}, {"value": 1}]), [9, 1])
        self.assertIsNone(serial._pool)
        parallel = ScoringPool(2, score_fn=task)
        self.assertEqual(parallel([]), [])
        self.assertIsNone(parallel._pool)

    def test_spawn_order_reuse_and_shutdown(self):
        pool = ScoringPool(2, score_fn=task)
        try:
            self.assertIsNone(pool._pool)
            self.assertEqual(pool([{"value": 3, "delay": .1}, {"value": 2}, {"value": 1}]), [9, 4, 1])
            executor = pool._pool
            children = list(executor._processes.values())
            self.assertEqual(pool([{"value": 4}]), [16])
            self.assertIs(pool._pool, executor)
            pool.close()
            pool.close()
            self.assertTrue(all(not child.is_alive() for child in children))
            self.assertIsNone(pool._pool)
            self.assertEqual(pool([{"value": 5}]), [25])
        finally:
            pool.close()

    def test_cpu_only_thread_limit(self):
        pool = ScoringPool(1, score_fn=worker_state)
        try:
            state = pool([{}])[0]
            self.assertNotEqual(state["pid"], os.getpid())
            self.assertEqual(state["threads"], 1)
            self.assertEqual(state["cuda_visible"], "")
            self.assertFalse(state["cuda_initialized"])
        finally:
            pool.close()

    def test_live_pool_serialization_does_not_share_executor(self):
        pool = ScoringPool(1, score_fn=task)
        other = None
        try:
            pool([{"value": 2}])
            other = pickle.loads(pickle.dumps(pool))
            self.assertIsNone(other._pool)
            self.assertEqual(other([{"value": 3}]), [9])
            self.assertIsNot(other._pool, pool._pool)
        finally:
            pool.close()
            if other is not None:
                other.close()

    def test_scene_error_is_not_silently_swallowed(self):
        pool = ScoringPool(1, score_fn=failing_task)
        try:
            with self.assertRaisesRegex(ValueError, "invalid scene"):
                pool([{"value": -1}])
            self.assertFalse(pool._serial_fallback)
        finally:
            pool.close()

    def test_broken_pool_falls_back_to_same_serial_scorer(self):
        executor = Mock()
        executor.map.side_effect = BrokenProcessPool("worker exited")
        with patch("navsim.agents.drivoR.score_module.scoring_pool.ProcessPoolExecutor", return_value=executor):
            pool = ScoringPool(3, score_fn=task)
            self.assertEqual(pool([{"value": 3}, {"value": 2}]), [9, 4])
            self.assertTrue(pool._serial_fallback)
            executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
            self.assertEqual(pool([{"value": 4}]), [16])
            executor.map.assert_called_once()

    def test_creation_failure_falls_back(self):
        with patch("navsim.agents.drivoR.score_module.scoring_pool.ProcessPoolExecutor", side_effect=OSError("no resources")):
            pool = ScoringPool(3, score_fn=task)
            self.assertEqual(pool([{"value": 2}]), [4])
            self.assertTrue(pool._serial_fallback)

    def test_cleanup_on_success_and_exception(self):
        callback_class = extracted_class("navsim/agents/drivoR/drivor_agent.py", "ScoringPoolCleanup", {"Callback": object})
        callback = callback_class()
        module = Mock()
        callback.teardown(None, module, "fit")
        callback.on_exception(None, module, RuntimeError("test"))
        self.assertEqual(module.agent.close_scoring_pool.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
