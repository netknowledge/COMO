"""Hard-timeout molecule rendering, isolated in a child process.

Why: ``drawing_engine.generate_image_from_smiles`` is guarded by
``func_set_timeout(5)``, which runs the call in a thread and injects an exception
into it.  It cannot interrupt a C++ call (Indigo layout/render): the thread is
abandoned mid-flight, keeps running, and holds Indigo's internal state.  Measured
consequence: after one render that exceeded the guard, the NEXT (trivial) render
in that process took 10.0 s and tripped the guard again - so the guard does not
bound the cost, it poisons the successor call and leaks threads (eventually
aborting the process).

Here each render runs in its own OS process.  On timeout the parent SIGKILLs it,
so abandoned work dies with the process and cannot leak into the next call.  The
child calls the raw (undecorated) render function, whose own try/except still
yields the blank-image fallback - identical semantics to before.

Env knobs:
  COMO_RENDER_POOL=0        disable (fall back to the in-process call)
  COMO_RENDER_TIMEOUT=12    per-render budget in seconds.  12 s (not 3 s):
  measured bimodal slow-but-renderable molecules - profiled 10.1 s of a
  10.2 s call inside indigo layout() (2D coordinate placement, c.f.
  indigo/__init__.py:1878) on some random-augment draws, still slow with
  functional-group substitution disabled; typical renders are 20-100 ms
  and true hangs are unbounded - 12 s separates them without ever
  un-bounding a hang.
"""

from __future__ import annotations

import os
import pickle
import select
import struct
import subprocess
import sys
from typing import Any, Optional, Tuple

DEFAULT_TIMEOUT = float(os.environ.get("COMO_RENDER_TIMEOUT") or "12.0")
ENABLED = os.environ.get("COMO_RENDER_POOL", "1") not in ("0", "false", "False")

_HDR = struct.Struct("<I")
_CHILD_MODULE = "training.datasets.render_child"
_DEBUG = os.environ.get("COMO_RENDER_DEBUG") == "1"
_DEBUG_LOG = os.path.join(os.environ.get("COMO_RENDER_DEBUG_DIR", "/tmp"), f"renderpool_{os.getpid()}.log")


def _dbg(msg: str) -> None:
    if not _DEBUG:
        return
    try:
        with open(_DEBUG_LOG, "a") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass

# training/datasets/ (this file's package) must be importable by the child:
# the child runs `python -m training.datasets.render_child` and imports
# `training.drawing_engine`, so the directory containing the `training`
# package (two levels up) is what ends up on PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))


def _write_frame(fh, obj: Any) -> None:
    buf = pickle.dumps(obj, protocol=4)
    fh.write(_HDR.pack(len(buf)))
    fh.write(buf)
    fh.flush()


def _read_exact(fh, n: int) -> bytes:
    chunks = []
    while n > 0:
        b = fh.read(n)
        if not b:
            raise EOFError("child pipe closed")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def _read_frame(fh) -> Any:
    (n,) = _HDR.unpack(_read_exact(fh, _HDR.size))
    return pickle.loads(_read_exact(fh, n))


class RenderPool:
    """One warm render child per instance; SIGKILLed and restarted on timeout."""

    def __init__(self, kind: str = "smiles", timeout: float = DEFAULT_TIMEOUT):
        self.stats = {"render": 0, "ok": 0, "timeout": 0, "restart": 0, "err": 0}
        _dbg(f"pool start pid={os.getpid()} kind={kind} timeout={timeout}")
        self.kind = kind
        self.timeout = float(timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._start()

    # -- lifecycle ---------------------------------------------------------
    def _start(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = _PKG_PARENT + (os.pathsep + env["PYTHONPATH"]
                                           if env.get("PYTHONPATH") else "")
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-m", _CHILD_MODULE, self.kind],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, cwd=_HERE,
        )
        # wait for the READY frame: startup (imports) must never be charged
        # against a render's budget, otherwise an innocent molecule gets
        # blamed for a cold child and lands in the badlist
        if not self._wait_ready(float(os.environ.get("COMO_RENDER_STARTUP") or "120")):
            _dbg("child failed to become ready -> restart once")
            proc, self._proc = self._proc, None
            try:
                proc.kill()
            except Exception:
                pass
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "-m", _CHILD_MODULE, self.kind],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env, cwd=_HERE,
            )
            self._wait_ready(float(os.environ.get("COMO_RENDER_STARTUP") or "120"))

    def _wait_ready(self, timeout: float) -> bool:
        try:
            ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
            if not ready:
                return False
            msg = _read_frame(self._proc.stdout)
            return bool(msg) and msg[0] == "READY"
        except Exception:
            return False

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()           # SIGKILL: no zombie work survives
        except Exception:
            pass
        for fh in (proc.stdin, proc.stdout):
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def restart(self) -> None:
        self.stats["restart"] += 1
        _dbg(f"restart #{self.stats['restart']} (timeouts={self.stats['timeout']})")
        self._stop()
        self._start()

    # -- api ---------------------------------------------------------------
    def render(self, smiles: str, **kwargs) -> Optional[Tuple[Any, ...]]:
        """Return the drawing_engine tuple, or None if the render timed out."""
        self.stats["render"] += 1
        if self._proc is None or self._proc.poll() is not None:
            _dbg("child not alive before send -> restart")
            self.restart()
        try:
            _write_frame(self._proc.stdin, (smiles, kwargs))
            ready, _, _ = select.select([self._proc.stdout], [], [], self.timeout)
        except Exception:
            self.restart()
            return None
        if not ready:                       # hard bound reached -> kill child
            self.stats["timeout"] += 1
            _dbg(f"TIMEOUT after {self.timeout}s")
            self.restart()
            return None
        try:
            out = _read_frame(self._proc.stdout)
            self.stats["ok"] += 1
            return out
        except Exception as exc:
            self.stats["err"] += 1
            _dbg(f"read error: {type(exc).__name__}: {exc}")
            self.restart()
            return None
