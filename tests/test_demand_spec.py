"""`demand-spec`: issue URL parsing, comment selection and the registry entry."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncharted.factory import demand_spec  # noqa: E402
from uncharted.factory.registry import Registry  # noqa: E402


def issue():
    comments = [{"body": f"comment {i}", "created_at": f"2025-01-{i + 1:02d}T00:00:00Z", "thumbs_up": up}
                for i, up in enumerate([0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
    return {"title": "Multiple config files", "url": "https://github.com/a/b/issues/7", "body": "Please {add} it",
            "thumbs_up": 39, "comments": comments}


def test_parse_url():
    assert demand_spec.parse_url("https://github.com/a/b/issues/7#issuecomment-1") == ("a", "b", 7)


def test_comments_are_top_voted_then_latest_without_repeats():
    lines = demand_spec.pick_comments(issue()["comments"], top=1, latest=2)
    assert [l.split("] ")[1] for l in lines] == ["comment 1", "comment 10", "comment 11"]
    assert lines[0].startswith("- [2025-01-02, +9]")


def test_prompt_carries_the_request_and_survives_braces():
    prompt = demand_spec.build_prompt(issue())
    assert "Multiple config files" in prompt and "39 thumbs-up" in prompt and "Please {add} it" in prompt
    assert '"feasible": true' in prompt


def test_write_spec_makes_a_buildable_registry_entry(tmp_path, monkeypatch):
    reply = json.dumps({"feasible": True, "slug": "Star Layers", "title": "t", "pitch": "p", "tests": ["a"]})
    monkeypatch.setattr(demand_spec.llm, "chat", lambda *a, **k: reply)
    reg = Registry(tmp_path)
    slug, status = demand_spec.write_spec(reg, issue(), "some/model")
    idea = reg.ideas[slug]
    assert (slug, status, idea["arm"]) == ("star-layers", "spec", "demand")
    assert idea["request_thumbs_up"] == 39 and reg.has_atoms(["https://github.com/a/b/issues/7"])
