"""Factory test with a fake model: spec -> buggy draft -> test failure -> fix -> built -> gallery."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncharted import llm  # noqa: E402
from uncharted.factory import build, gallery, publish, registry, spec  # noqa: E402

SPEC = {"feasible": True, "slug": "Tally Merge!", "title": "Tally Merge", "pitch": "Merges counters.",
        "why_unusual": "Nobody pairs these.", "kind": "library", "features": ["merge"],
        "usage": "from tally_merge import merge", "tests": ["merge adds counts"]}

TESTS = """from tally_merge import merge

def test_adds():
    assert merge({"a": 1}, {"a": 2}) == {"a": 3}

def test_keeps_left():
    assert merge({"a": 1}, {}) == {"a": 1}

def test_keeps_right():
    assert merge({}, {"b": 4}) == {"b": 4}

def test_empty():
    assert merge({}, {}) == {}

def test_many():
    assert merge({"a": 1, "b": 1}, {"b": 1, "c": 1}) == {"a": 1, "b": 2, "c": 1}
"""

BUGGY = "def merge(x, y):\n    out = dict(x)\n    out.update(y)\n    return out\n"
FIXED = ("def merge(x, y):\n    out = dict(x)\n    for k, v in y.items():\n"
         "        out[k] = out.get(k, 0) + v\n    return out\n")


def fake_chat(prompt, model=None, rpm=8, **kw):
    if "invent small software projects" in prompt:
        return "```json\n" + json.dumps(SPEC) + "\n```"
    if "Implement this project" in prompt:
        return json.dumps({"files": {"tally_merge/__init__.py": "from .core import merge\n",
                                     "tally_merge/core.py": BUGGY, "tests/test_tally_merge.py": TESTS}})
    if "fails its tests" in prompt:
        assert "test_adds" in prompt  # the failure output reached the model
        return json.dumps({"files": {"tally_merge/core.py": FIXED}})
    raise AssertionError("unexpected prompt")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "chat", fake_chat)
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps({
        "clusters": {"0": {"name": "CRDT sync"}, "1": {"name": "WebGPU inference"}, "2": {"name": "event sourcing"}},
        "atom_assignments": [{"text": "x", "paper_id": f"o__r{i}", "cluster": i} for i in range(3)]}))
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"top_k_sequences": [
        {"cluster_ids": [0, 1, 2], "rank_fused": 1, "z_coherence": 1.5, "z_availability": 2.0}]}))
    return tmp_path, clusters, results


def test_factory_end_to_end(world):
    tmp, clusters, results = world
    out = tmp / "factory_out"
    spec.main(["--results", str(results), "--clusters", str(clusters), "--out", str(out), "--rpm", "999"])
    reg = registry.Registry(out)
    assert list(reg.ideas) == ["tally-merge"] and reg.ideas["tally-merge"]["status"] == "spec"

    spec.main(["--results", str(results), "--clusters", str(clusters), "--out", str(out)])
    assert len(registry.Registry(out).ideas) == 1  # same atoms are never specced twice

    build.main(["--out", str(out), "--sandbox", "local", "--rpm", "999"])
    idea = registry.Registry(out).ideas["tally-merge"]
    assert idea["status"] == "built", idea
    assert idea["iterations"] == 2 and idea["tests_passed"] == 5
    assert idea["history"][0]["failed"] > 0  # the buggy draft really failed first

    publish.main(["--org", "someone", "--out", str(out)])  # dry run: no git, no network
    assert (out / "projects" / "tally-merge" / "README.md").read_text().startswith("# Tally Merge")
    assert not (out / "projects" / "tally-merge" / ".git").exists()

    gallery.main(["--out", str(out)])
    page = (out / "gallery" / "index.html").read_text()
    assert "Tally Merge" in page and "CRDT sync" in page


def test_refuses_paths_outside_project(tmp_path):
    with pytest.raises(ValueError):
        build.apply(tmp_path, json.dumps({"files": {"../escape.py": "x"}}))
    with pytest.raises(ValueError):
        build.apply(tmp_path, json.dumps({"files": {"/etc/passwd": "x"}}))


def test_trivial_tests_are_rejected(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("import pkg\ndef test(): assert True\n")
    assert not build.tests_look_real(tmp_path, "pkg")
