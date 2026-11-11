"""Child process for render_pool: render SMILES on request, one framed reply each.

Protocol (stdin/stdout, 4-byte little-endian length prefix + pickle):
  request : ("smiles"|"wild" is fixed at launch) -> (smiles, kwargs)
  reply   : drawing_engine tuple | ("__ERR__", exc_name, exc_str)

Lifecycle: SIGKILLs itself when its spawner dies (PR_SET_PDEATHSIG), so an
aborted run cannot leak children stuck mid-import or inside an Indigo call.
"""

from __future__ import annotations

import ctypes
import os
import pickle
import signal
import struct
import sys

_HDR = struct.Struct("<I")


def _die_with_parent() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)      # PR_SET_PDEATHSIG
        if os.getppid() == 1:              # parent died before prctl landed
            os._exit(0)
    except Exception:
        pass


def _send(out, obj) -> None:
    buf = pickle.dumps(obj, protocol=4)
    out.write(_HDR.pack(len(buf)))
    out.write(buf)
    out.flush()


def _read_exact(fh, n: int) -> bytes:
    chunks = []
    while n > 0:
        b = fh.read(n)
        if not b:
            raise EOFError
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def main(kind: str) -> None:
    _die_with_parent()
    from training import drawing_engine

    fn = (drawing_engine.generate_image_from_smiles if kind == "smiles"
          else drawing_engine.generate_image_from_wild_smiles)
    raw = getattr(fn, "__wrapped__", fn)   # bypass func_timeout(5); parent enforces it
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    _send(out, ("READY",))                 # startup must not be charged to a render
    while True:
        try:
            (n,) = _HDR.unpack(_read_exact(inp, _HDR.size))
            smiles, kwargs = pickle.loads(_read_exact(inp, n))
        except (EOFError, OSError):
            return
        try:
            res = raw(smiles, **kwargs)
        except BaseException as exc:
            res = ("__ERR__", type(exc).__name__, str(exc)[:200])
        try:
            _send(out, res)
        except Exception:
            return


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smiles")
