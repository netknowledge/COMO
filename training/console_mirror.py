"""Rank-0 console mirror: stdout+stderr -> one log file, stream-safe.

Install at the EARLIEST entry point of a launcher (script-side, right after
arg parsing / output-dir resolution, BEFORE any print) so the launch banner
is captured too. Idempotent: an already-mirrored process is left alone (a
double install would double-log). The torchrun-set RANK env gates the
writers: only rank 0 / no-distributed-landing installs. Mirrors train_joint's
per-stream design: a print goes to stdout + file, a warning to stderr + file,
never duplicated within one capture (e.g. under 2>&1 | tee).
"""
import os
import sys


class _StreamTee:
    """Mirror ONE stream into a file sink without touching the other."""

    is_console_mirror = True

    def __init__(self, orig, sink):
        self._orig, self._sink = orig, sink

    def write(self, data):
        self._sink.write(data)
        return self._orig.write(data)

    def flush(self):
        self._orig.flush()
        self._sink.flush()

    def __getattr__(self, name):         # passthrough (encoding, fileno, ...)
        return getattr(self._orig, name)


def install_console_mirror(path: str) -> "str | None":
    """Mirror this process's stdout+stderr into `path` (append mode).

    Returns the mirrored path, or None when skipped (non-rank-0 writer or
    the process already has a console mirror)."""
    if getattr(sys.stdout, "is_console_mirror", False):
        return None
    rank = os.environ.get("RANK", "0")
    if rank not in ("", "0"):
        return None
    parent = os.path.dirname(os.path.abspath(str(path))) or "."
    os.makedirs(parent, exist_ok=True)
    sink = open(path, "a", buffering=1)
    import datetime
    sys.stdout = _StreamTee(sys.stdout, sink)
    sys.stderr = _StreamTee(sys.stderr, sink)
    print(f"[console] mirroring stdout+stderr -> {path} "
          f"(started {datetime.datetime.now().isoformat(timespec='seconds')}; "
          f"segments are appended per launch)", flush=True)
    return path