"""Mutual exclusion for CLI/application runs sharing an output directory."""

from functools import wraps
from pathlib import Path
import fcntl


def locked_run(function):
    @wraps(function)
    def wrapped(before, after, output, *args, **kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "This output directory already has an active pipeline run"
                ) from None
            try:
                return function(before, after, output, *args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return wrapped
