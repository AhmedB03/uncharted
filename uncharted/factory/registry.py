"""The factory's ledger: every idea, what happened to it, and where it lives."""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Registry:
    def __init__(self, root="factory_out"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.data = self._read() if self.path.exists() else {"ideas": {}}

    def _read(self):
        for attempt in range(50):
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except PermissionError:  # Windows: another process is replacing the file right now
                time.sleep(0.1)
        return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def ideas(self):
        return self.data["ideas"]

    def project_dir(self, slug):
        return self.root / "projects" / slug

    def has_atoms(self, atoms):
        key = sorted(atoms)
        return any(sorted(i["atoms"]) == key for i in self.ideas.values())

    def unique_slug(self, slug):
        base, n = slug, 2
        while slug in self.ideas:
            slug, n = f"{base}-{n}", n + 1
        return slug

    def update(self, slug, **fields):
        # Parallel builds run in separate processes that share this file, so merge into the
        # copy on disk under a lock instead of writing back a stale in-memory snapshot.
        with self._lock():
            if self.path.exists():
                self.data = self._read()
            self.ideas.setdefault(slug, {"slug": slug, "created": now()}).update(fields, updated=now())
            self._write()

    def save(self):
        with self._lock():
            self._write()

    def _write(self):
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        for attempt in range(50):
            try:
                tmp.replace(self.path)
                return
            except PermissionError:  # Windows: another process is reading the file right now
                time.sleep(0.1)
        tmp.replace(self.path)

    @contextmanager
    def _lock(self, stale=30):
        lock = self.path.with_suffix(".lock")
        start = time.time()
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (FileExistsError, PermissionError):  # Windows: lock file is mid-delete
                if time.time() - start > stale:  # holder died mid-write; writes take milliseconds
                    lock.unlink(missing_ok=True)
                    start = time.time()
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
