"""The factory's ledger: every idea, what happened to it, and where it lives."""

import json
from datetime import datetime, timezone
from pathlib import Path


class Registry:
    def __init__(self, root="factory_out"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"ideas": {}}

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
        self.ideas.setdefault(slug, {"slug": slug, "created": now()}).update(fields, updated=now())
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
