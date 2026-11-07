from uncharted.factory import judge


def test_prompt_is_blind(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("def f():\n    return 1\n")
    idea = {"arm": "alien", "atoms": ["radix tree routing", "KCP transport"],
            "spec": {"title": "T", "pitch": "P", "kind": "cli", "features": ["a"], "usage": "u",
                     "why_unusual": "SECRET-WHY"}}
    text = judge.prompt_for(idea, tmp_path)
    assert "def f()" in text
    for leak in ("alien", "radix tree routing", "SECRET-WHY"):
        assert leak not in text


def test_mann_whitney():
    assert judge.mann_whitney_p([1, 2, 3], [1, 2, 3]) == 1.0
    assert judge.mann_whitney_p([5] * 20, [1] * 20) < 0.001
    assert judge.mann_whitney_p([], [1]) is None
