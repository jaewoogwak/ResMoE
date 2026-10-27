"""Flushed progress messages for long CPU and GPU experiment steps."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


@contextmanager
def progress_step(label: str, *, heartbeat_seconds: float = 60, enabled: bool = True):
    if not enabled:
        yield
        return

    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(heartbeat_seconds):
            print(f"{label}: still running ({time.monotonic() - started:.0f}s)", flush=True)

    print(f"{label}: started", flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        print(f"{label}: failed after {time.monotonic() - started:.1f}s", flush=True)
        raise
    else:
        print(f"{label}: done in {time.monotonic() - started:.1f}s", flush=True)
    finally:
        stopped.set()
        worker.join(timeout=1)
