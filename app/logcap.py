"""Size-capped log capture for the .app bundle.

Everything the process writes to stdout / stderr -- Python, and native code (Qt, MLX, PortAudio) that writes to the
file descriptors directly -- goes through a pipe to a writer thread. The thread keeps a running byte count, so the cap
is enforced on every write at the cost of an addition (no stat calls): when the file passes max_bytes it is cut to its
last keep_bytes, at start-up and at any moment while the app runs.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading

MAX_BYTES = 50 * 1024 * 1024
KEEP_BYTES = 1024 * 1024


class CappedLog:
    def __init__(self, path, max_bytes=MAX_BYTES, keep_bytes=KEEP_BYTES):
        self.path, self.max, self.keep = path, max_bytes, keep_bytes
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "ab")
        self.size = self.f.tell()
        self.trims = 0
        self._trim_if_needed()

    def write(self, data: bytes):
        self.f.write(data)
        self.f.flush()
        self.size += len(data)
        self._trim_if_needed()

    def _trim_if_needed(self):
        if self.size <= self.max:
            return
        self.f.flush()
        with open(self.path, "rb") as r:
            r.seek(max(0, self.size - self.keep))
            tail = r.read()
        nl = tail.find(b"\n")                      # start the kept part on a line boundary
        if 0 <= nl < len(tail) - 1:
            tail = tail[nl + 1:]
        self.f.seek(0)
        self.f.truncate()
        self.f.write(b"--- log cut to its last %d bytes (cap %d bytes) ---\n" % (len(tail), self.max) + tail)
        self.f.flush()
        self.size = self.f.tell()
        self.trims += 1


def capture(path, max_bytes=MAX_BYTES, keep_bytes=KEEP_BYTES):
    """Redirect fd 1 and 2 into a size-capped log at `path`. Returns the CappedLog."""
    log = CappedLog(path, max_bytes, keep_bytes)
    r, w = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(w, 1)
    os.dup2(w, 2)
    os.close(w)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)

    def pump():
        while True:
            data = os.read(r, 65536)
            if not data:
                return
            try:
                log.write(data)
            except OSError:
                pass                               # keep draining: a full pipe would block the app

    t = threading.Thread(target=pump, daemon=True, name="logcap")
    t.start()

    def drain():
        sys.stdout.flush()
        sys.stderr.flush()
        os.close(1)                                # writers gone: pump reads EOF and returns
        os.close(2)
        t.join(timeout=2)

    atexit.register(drain)
    return log
