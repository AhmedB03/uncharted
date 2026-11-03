"""Offline tests: fake GitHub client, mock atoms, clustering, DB build.

    python -m pytest tests -q
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncharted import atomize, build_db, cluster, collect  # noqa: E402


class FakeGitHub:
    def json(self, path, params=None):
        if path.endswith("/readme"):
            return None
        if path.endswith("/contents"):
            return [{"name": "package.json", "type": "file"}, {"name": "src", "type": "dir"}]
        if path.endswith("/contributors"):
            return [
                {"id": 1, "login": "alice", "type": "User", "contributions": 40},
                {"id": 2, "login": "dependabot[bot]", "type": "Bot", "contributions": 90},
                {"id": 3, "login": "typo-fixer", "type": "User", "contributions": 1},
                {"id": 4, "login": "renovate-bot", "type": "User", "contributions": 12},
            ]
        return None

    @staticmethod
    def decoded(data, max_chars):
        return None

    def raw(self, owner, name, branch, path, max_chars):
        if path == "README.md":
            return "# Demo\nA collaborative editor using yjs and react."
        if path == "package.json":
            return '{"dependencies": {"yjs": "1", "react": "18", "express": "4"}}'
        return None


ITEM = {"full_name": "acme/demo", "html_url": "https://github.com/acme/demo",
        "description": "collaborative editor", "language": "TypeScript", "topics": ["crdt"],
        "stargazers_count": 999, "created_at": "2024-03-01T00:00:00Z", "default_branch": "main"}


def test_fetch_one_filters_bots_and_drive_bys(tmp_path):
    assert collect.fetch_one(FakeGitHub(), ITEM, tmp_path, min_contributions=3) == "ok"
    meta = json.loads((tmp_path / "acme__demo" / "meta.json").read_text())
    assert [c["login"] for c in meta["contributors"]] == ["alice"]
    assert meta["manifests"] == ["package.json"]
    assert "yjs" in (tmp_path / "acme__demo" / "deps.txt").read_text()
    assert collect.fetch_one(FakeGitHub(), ITEM, tmp_path, 3) == "skip"  # resumable


def test_parse_atoms_handles_fences_and_dupes():
    reply = 'Sure!\n```json\n{"atoms": [{"atom": "CRDT based offline sync.", "role": "core"},' \
            '{"atom": "crdt based offline sync", "role": "core"}, {"atom": "x", "role": "core"}]}\n```'
    assert atomize.parse_atoms(reply) == [{"atom": "CRDT based offline sync", "role": "core"}]


def test_end_to_end_offline(tmp_path):
    repos = tmp_path / "repos"
    names = ["react yjs crdt", "react express postgres", "torch cuda triton", "torch transformers faiss",
             "tokio grpc raft", "tokio axum redis", "wgpu webgpu wasm", "bevy game wgpu"] * 3
    for i, deps in enumerate(names):
        d = repos / f"o__r{i}"
        d.mkdir(parents=True)
        (d / "deps.txt").write_text(deps)
        (d / "meta.json").write_text(json.dumps({
            "id": d.name, "full_name": f"o/r{i}", "html_url": "u", "description": "", "topics": [],
            "created_at": "2023-01-01", "language": "Go",
            "contributors": [{"id": i % 5, "login": f"dev{i % 5}", "contributions": 9}]}))
    atomize.main(["--repos", str(repos), "--mock"])
    out = tmp_path / "clusters_80.json"
    cluster.main(["--repos", str(repos), "--out", str(out), "--embedder", "tfidf",
                  "--namer", "medoid", "--min-cluster-size", "2"])
    data = json.loads(out.read_text())
    assert data["clusters"] and data["atom_assignments"]
    assert {"text", "paper_id", "cluster"} <= set(data["atom_assignments"][0])
    db = tmp_path / "u.db"
    build_db.main(["--repos", str(repos), "--clusters", str(out), "--db", str(db),
                   "--vocab", str(tmp_path / "vocab.json")])
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == len(names)
    assert conn.execute("SELECT COUNT(DISTINCT author_id) FROM paper_authors").fetchone()[0] == 5
