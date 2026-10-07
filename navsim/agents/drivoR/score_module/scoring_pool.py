"""Ordered, CPU-only PDM scoring with a persistent pool local to each DDP rank."""
import atexit
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import logging
import multiprocessing as mp
from numbers import Integral
import os

logger = logging.getLogger(__name__)


def score_scene(point):
    # Import in the worker, not while serializing the CUDA agent. These inputs
    # contain only a cache path, NumPy proposals and the train/validation flag.
    from .compute_navsim_score import get_sub_score

    return get_sub_score(point["token"], point["poses"], point["test"])


def _initialize_worker(threads):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Also cap libraries loaded by the spawned training entrypoint before this
    # initializer. threadpoolctl updates already-loaded BLAS/OpenMP runtimes.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    # Keep the limiter alive for the entire worker process.
    try:
        from threadpoolctl import threadpool_limits

        global _thread_limiter
        _thread_limiter = threadpool_limits(limits=threads)
    except ImportError:
        logger.warning("Install threadpoolctl to also cap already-loaded BLAS libraries")


class ScoringPool:
    """Lazy spawn avoids forking CUDA and survives checkpoints/DDP serialization.

    Zero workers scores serially. A broken/unavailable process pool falls back
    to the same serial scorer for the remainder of this process. Ordinary scene
    scoring errors propagate rather than silently returning incomplete targets.
    """

    def __init__(self, workers, threads_per_worker=1, score_fn=score_scene):
        if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 0:
            raise ValueError("scoring_workers must be a non-negative integer")
        if isinstance(threads_per_worker, bool) or not isinstance(threads_per_worker, Integral) or threads_per_worker < 1:
            raise ValueError("scoring_worker_threads must be a positive integer")
        self.workers = int(workers)
        self.threads_per_worker = int(threads_per_worker)
        self.score_fn = score_fn
        self._pool = None
        self._owner_pid = os.getpid()
        self._serial_fallback = False
        self._exit_registered = False

    def __getstate__(self):
        # Executors/queues are never serialized into a checkpoint or child rank.
        return dict(workers=self.workers, threads_per_worker=self.threads_per_worker,
                    score_fn=self.score_fn)

    def __setstate__(self, state):
        self.__init__(**state)

    def _fallback(self, points, error):
        self.close()
        self._serial_fallback = True
        logger.warning("PDM scoring pool unavailable (%s); using serial scoring in pid %s", error, os.getpid())
        return [self.score_fn(point) for point in points]

    def __call__(self, points):
        points = list(points)
        if not points:
            return []
        if self._owner_pid != os.getpid():
            # Never shut down an executor owned by a different process.
            self._pool = None
            self._owner_pid = os.getpid()
            self._serial_fallback = False
            self._exit_registered = False
        if not self.workers or self._serial_fallback:
            return [self.score_fn(point) for point in points]
        if self._pool is None:
            try:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=mp.get_context("spawn"),
                    initializer=_initialize_worker, initargs=(self.threads_per_worker,))
            except OSError as error:
                return self._fallback(points, error)
            if not self._exit_registered:
                atexit.register(self.close)
                self._exit_registered = True
        try:
            # map preserves scene order, unlike completion-order collection.
            return list(self._pool.map(self.score_fn, points, chunksize=1))
        except BrokenProcessPool as error:
            return self._fallback(points, error)

    def close(self):
        pool, self._pool = self._pool, None
        if pool is not None and self._owner_pid == os.getpid():
            pool.shutdown(wait=True, cancel_futures=True)
