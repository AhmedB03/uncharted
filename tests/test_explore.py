"""`explore`: novelty scoring, the minimal criterion, seeding, and a child that builds on its stone."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncharted.factory import build, explore  # noqa: E402
from uncharted.factory.registry import Registry  # noqa: E402

# techniques 0-4; 3 goes with 0 far more often than chance, 4 never appears beside 0, 1 or 2
SUPPORT = {"repo_clusters": {**{f"a{i}": [0, 1, 2] for i in range(4)}, **{f"b{i}": [0, 3] for i in range(4)},
                             **{f"c{i}": [4] for i in range(4)}, **{f"d{i}": [5] for i in range(20)}}}


def archive(tmp_path):
    reg = Registry(tmp_path / "archive")
    reg.update("stone", status="built", cluster_ids=[0, 1, 2], spec={"title": "Stone", "pitch": "p", "features": ["f"]})
    reg.update("dead", status="failed", cluster_ids=[0, 1, 3], spec={})
    return reg


def test_novelty_is_distance_to_the_nearest_sets():
    seen = [frozenset({0, 1, 2}), frozenset({7, 8, 9})]
    assert explore.novelty(frozenset({0, 1, 2}), seen, k=1) == 0
    assert explore.novelty(frozenset({0, 1, 3}), seen, k=1) == 0.5
    assert explore.novelty(frozenset({4, 5, 6}), seen, k=2) == 1
    assert explore.novelty(frozenset({1}), [], k=5) == 1


def test_candidates_extend_stones_only_and_need_a_link(tmp_path):
    cands = explore.candidates(archive(tmp_path), SUPPORT, min_repos=1)
    assert [(c["parent"], c["add"], c["cluster_ids"]) for c in cands] == [("stone", 3, [0, 1, 2, 3])]
    assert explore.candidates(archive(tmp_path), SUPPORT, exclude=[3], min_repos=1) == []
    assert explore.candidates(archive(tmp_path), SUPPORT, max_techniques=3, min_repos=1) == []


def test_pick_spreads_over_stones_and_techniques():
    cands = [{"parent": "a", "add": 1, "novelty": 0.9}, {"parent": "a", "add": 2, "novelty": 0.8},
             {"parent": "b", "add": 1, "novelty": 0.7}, {"parent": "b", "add": 3, "novelty": 0.6}]
    assert [(c["parent"], c["add"]) for c in explore.pick(cands, 3)] == [("a", 1), ("b", 3)]


def test_seed_copies_built_projects_once(tmp_path):
    src = Registry(tmp_path / "src")
    src.update("one", status="built", cluster_ids=[1, 2, 3], spec={})
    src.update("two", status="failed", cluster_ids=[4, 5, 6], spec={})
    (src.project_dir("one") / "one").mkdir(parents=True)
    (src.project_dir("one") / "one" / "__init__.py").write_text("x = 1\n")
    reg = Registry(tmp_path / "archive")
    assert (explore.seed(reg, [src.root]), explore.seed(reg, [src.root])) == (1, 0)
    assert reg.ideas["one"]["generation"] == 0 and (reg.project_dir("one") / "one" / "__init__.py").is_file()


def test_child_records_its_lineage_and_the_builder_sees_the_stone(tmp_path, monkeypatch):
    reg = archive(tmp_path)
    (reg.project_dir("stone") / "stone").mkdir(parents=True)
    (reg.project_dir("stone") / "stone" / "core.py").write_text("STONE_MARKER = 1\n")
    prompts = []
    reply = json.dumps({"feasible": True, "slug": "Next Step", "title": "t", "pitch": "p", "tests": ["a"]})
    monkeypatch.setattr(explore.llm, "chat", lambda prompt, *a, **k: prompts.append(prompt) or reply)
    cand = explore.candidates(reg, SUPPORT, min_repos=1)[0]
    slug, status = explore.write_spec(reg, cand, {3: "event sourcing"}, {3: {"o/r"}}, "some/model")
    idea = reg.ideas[slug]
    assert (slug, status, idea["parent"], idea["generation"], idea["added"]) == ("next-step", "spec", "stone", 1, 3)
    assert "Stone: p" in prompts[0] and "- event sourcing   (e.g. o/r)" in prompts[0]
    assert "STONE_MARKER" in build.parent_code(reg, idea, "next_step")
    assert build.parent_code(reg, reg.ideas["stone"], "stone") == ""
    assert explore.tree(reg, {3: "event sourcing"})[:2] == ["stone [built]", "    next-step [spec]  + event sourcing"]
