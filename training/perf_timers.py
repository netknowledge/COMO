"""COMO_PERF=1 env-gated per-phase wall timers for the training loop.

Enabled by `COMO_PERF=1`; without it every phase is a no-op (one env check).
Counters are per-process (each DDP rank has its own); rank 0 prints on flush.
Phases must not overlap - the training loop is single-threaded.
"""
import os
import time
from contextlib import contextmanager

_ENABLED = os.environ.get('COMO_PERF', '') == '1'
_acc: dict = {}
_flush_t = 0.0
_started = False


@contextmanager
def perf_phase(name):
    """Accumulate wall seconds spent inside the body under *name*."""
    if not _ENABLED:
        yield
        return
    global _started, _flush_t
    t0 = time.perf_counter()
    if not _started:
        _started = True
        _flush_t = t0
    try:
        yield
    finally:
        _acc[name] = _acc.get(name, 0.0) + time.perf_counter() - t0


def perf_flush(prefix=''):
    """Print per-phase shares of the wall interval since the last flush."""
    global _flush_t, _acc
    if not _ENABLED or not _started or not _acc:
        return
    now = time.perf_counter()
    window = now - _flush_t
    _flush_t = now
    acc, _acc = _acc, {}
    if window <= 0:
        return
    ranked = sorted(acc.items(), key=lambda kv: -kv[1])
    parts = ' '.join(f'{k}:{100.0 * v / window:.0f}%' for k, v in ranked)
    print(f'[perf] {prefix} {parts} window={window:.0f}s', flush=True)