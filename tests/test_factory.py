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


def test_file_block_replies_are_parsed(tmp_path):
    reply = ('Sure!\n===== FILE: pkg/__init__.py =====\nx = "a \\"quoted\\" {brace}"\n'
             "===== FILE: tests/test_pkg.py =====\n```python\nimport pkg\n```\n"
             "===== DELETE: old.py =====\n===== END =====\nthanks")
    files, deletes = build.parse_files(reply)
    assert files == {"pkg/__init__.py": 'x = "a \\"quoted\\" {brace}"', "tests/test_pkg.py": "import pkg"}
    assert deletes == ["old.py"]
    build.apply(tmp_path, reply)
    assert (tmp_path / "pkg" / "__init__.py").read_text().startswith('x = "a')


def test_hints_flag_non_stdlib_imports():
    out = "ERROR collecting tests/test_x.py\nE   ModuleNotFoundError: No module named 'pandas'"
    hints = build.stdlib_hints(out, "x")
    assert "pandas" in hints and "standard library" in hints and "ZERO tests ran" in hints
    assert build.stdlib_hints("No module named 'json'", "x") == ""


def test_foreign_imports_detected_and_purged(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("from .core import f\n")
    (tmp_path / "pkg" / "core.py").write_text("import json, os\nimport markdown\nfrom aiohttp import web\n"
                                              "def f():\n    return 1\n")
    (tmp_path / "tests" / "test_pkg.py").write_text("import pytest\nfrom pkg import f\nfrom pkg.core import f\n")
    assert build.foreign_imports(tmp_path, "pkg") == {"pkg/core.py": ["aiohttp", "markdown"]}

    def rewrite(prompt, *a, **k):
        assert "markdown" in prompt and "===== FILE: pkg/core.py =====" in prompt
        return "===== FILE: pkg/core.py =====\nimport json\ndef f():\n    return 1\n===== END ====="
    monkeypatch.setattr(llm, "chat", rewrite)

    class A:
        model, rpm = "x", 999
    assert build.purge_foreign_imports(tmp_path, "pkg", A) == {}
    assert "markdown" not in (tmp_path / "pkg" / "core.py").read_text()


def test_parse_json_tolerates_trailing_commas():
    assert llm.parse_json('here: {"a": [1, 2,], "b": "line1\nline2",}') == {"a": [1, 2], "b": "line1\nline2"}


def test_benchmark_common_triplets_come_from_real_repos(tmp_path):
    from uncharted.factory import benchmark
    clusters = tmp_path / "c.json"
    clusters.write_text(json.dumps({"clusters": {}, "atom_assignments": (
        [{"text": "t", "paper_id": "o__a", "cluster": c} for c in (0, 1, 2, 3)]
        + [{"text": "t", "paper_id": "o__b", "cluster": c} for c in (0, 1, 2)]
        + [{"text": "t", "paper_id": "o__c", "cluster": c} for c in (5, 6)])}))
    got = benchmark.common_triplets(clusters, 10, exclude=[[1, 2, 3]])
    triples = {tuple(t) for t, _ in got}
    assert triples == {(0, 1, 2), (0, 1, 3), (0, 2, 3)}  # (1,2,3) excluded, repo c too small
    assert dict((tuple(t), s) for t, s in got)[(0, 1, 2)] == 2  # shared by repos a and b


def test_fisher_one_sided():
    from uncharted.factory import benchmark
    assert benchmark.fisher_one_sided(5, 0, 0, 5) < 0.01
    assert benchmark.fisher_one_sided(0, 5, 5, 0) == 1.0
    assert abs(benchmark.fisher_one_sided(3, 2, 2, 3) - 0.5) < 0.01


def test_trivial_tests_are_rejected(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("import pkg\ndef test(): assert True\n")
    assert not build.tests_look_real(tmp_path, "pkg")
