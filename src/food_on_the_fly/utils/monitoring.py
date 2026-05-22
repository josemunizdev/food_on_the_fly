"""Lightweight system-metrics monitoring for training runs.

Samples CPU%, memory%, and process RSS in a background thread and logs
each sample to the active MLflow run as a timestamped metric. Designed
to be a no-op (graceful fallback) when psutil or MLflow is unavailable.
"""

from __future__ import annotations

import os
import threading
from contextlib import AbstractContextManager
from typing import Any

from food_on_the_fly.logging_config import get_logger

logger = get_logger(__name__)


class SystemMetricsLogger(AbstractContextManager["SystemMetricsLogger"]):
    """Context manager that periodically logs system metrics to MLflow.

    Usage:
        with SystemMetricsLogger(interval_seconds=2.0):
            pipeline.fit(X, y)
    """

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._step = 0
        self._psutil: Any = None
        self._proc: Any = None

    def __enter__(self) -> SystemMetricsLogger:
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(interval=None)  # prime the counter
        except ImportError:
            logger.warning("psutil not installed; skipping system metrics logging")
            return self

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            "System metrics logger started (interval=%.1fs)", self.interval_seconds
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)

    def _run(self) -> None:
        try:
            import mlflow
        except ImportError:
            return

        while not self._stop.wait(self.interval_seconds):
            try:
                cpu = self._psutil.cpu_percent(interval=None)
                vm = self._psutil.virtual_memory()
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                mlflow.log_metric("sys_cpu_percent", cpu, step=self._step)
                mlflow.log_metric("sys_mem_percent", vm.percent, step=self._step)
                mlflow.log_metric("sys_proc_rss_mb", rss_mb, step=self._step)
                self._step += 1
            except Exception as e:  # noqa: BLE001 — monitoring must never crash training
                logger.debug("System metrics sample failed: %s", e)
