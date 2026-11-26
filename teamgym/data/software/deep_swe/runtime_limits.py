"""Bound heavy Docker verifications across independent queue subprocesses."""
from __future__ import annotations

import fcntl
from functools import wraps
import os
from pathlib import Path
import time


def verifier_slots(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        limit = int(os.environ.get("DEEPSWE_VERIFIER_CONCURRENCY", "0"))
        if limit <= 0:
            return function(*args, **kwargs)
        directory = Path(os.environ.get("DEEPSWE_RESOURCE_LOCK_DIR", "/tmp/deepswe-verifier-slots"))
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        while True:
            for index in range(limit):
                stream = (directory / f"slot-{index}.lock").open("a")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                waited = round(time.monotonic() - started, 3)
                try:
                    result = function(*args, **kwargs)
                    result["resource_slot_wait_seconds"] = waited
                    return result
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    stream.close()
            time.sleep(1)
    return wrapped
