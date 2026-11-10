"""`demand`: query building, bug filtering and the ranked list of upvoted requests."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncharted import demand  # noqa: E402

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def issue(repo, up, title="Add a thing", labels=(), created="2020-01-01T00:00:00Z", n=[0]):
    n[0] += 1
    return {"repo": repo, "stars": 0, "title": title, "url": f"https://github.com/{repo}/issues/{n[0]}",
            "created_at": created, "body": "", "thumbs_up": up, "reactions": up, "comments": 0,
            "labels": list(labels)}


def test_query_has_one_alias_per_repo():
    q = demand.build_query(["a/b", "c/d"], 3)
    assert "r0: search" in q and "r1: search" in q and "first: 3" in q
    assert 'repo:a/b is:issue is:open sort:reactions-+1-desc' in q
    assert 'repo:a/b is:issue is:open created:>=2024-01-01 sort:reactions-+1-desc' in demand.build_query(
        ["a/b"], 3, "created:>=2024-01-01")


def test_bugs_are_left_out():
    assert demand.is_bug(issue("a/b", 5, labels=["type: Bug"]))
    assert demand.is_bug(issue("a/b", 5, title="[Bug]: it crashes"))
    assert not demand.is_bug(issue("a/b", 5, labels=["enhancement"]))


def test_top_requests_orders_caps_and_filters():
    rows = [issue("a/b", 90), issue("a/b", 80), issue("a/b", 70), issue("c/d", 60),
            issue("c/d", 500, labels=["bug"]), issue("e/f", 5),
            issue("g/h", 65, created="2025-10-01T00:00:00Z")]
    picks = demand.top_requests(rows, min_up=20, max_per_repo=2, now=NOW)
    assert [p["thumbs_up"] for p in picks] == [90, 80, 65, 60]
    assert [p["thumbs_up"] for p in demand.top_requests(rows, max_up=70, max_per_repo=0, now=NOW)] == [70, 65, 60]
    old = demand.top_requests(rows, min_up=20, min_years=1, max_per_repo=0, now=NOW)
    assert [p["thumbs_up"] for p in old] == [90, 80, 70, 60]


def test_load_and_report(tmp_path):
    (tmp_path / "a__b.json").write_text(json.dumps({"repo": "a/b", "issues": [
        {k: v for k, v in issue("a/b", 120, title="Pipe | here").items() if k not in ("repo", "stars")}]}))
    rows = demand.load(tmp_path, {"a__b": ("a/b", 42)})
    assert rows[0]["stars"] == 42 and rows[0]["repo"] == "a/b"
    text = demand.report(rows, rows, 1)
    assert "Pipe \\| here" in text and "1 have 100+" in text
