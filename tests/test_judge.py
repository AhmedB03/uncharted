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


def test_spec_prompt_is_blind_and_has_no_code(tmp_path):
    (tmp_path / "core.py").write_text("SECRET_CODE = 1\n")
    idea = {"arm": "alien", "atoms": ["radix tree routing"],
            "spec": {"title": "T", "pitch": "P", "features": ["a"], "tests": ["t1"], "why_unusual": "SECRET-WHY"}}
    text = judge.prompt_for(idea, tmp_path, specs=True)
    assert "t1" in text
    for leak in ("alien", "radix tree routing", "SECRET-WHY", "SECRET_CODE"):
        assert leak not in text


def test_combo_prompt_shows_only_atoms(tmp_path):
    idea = {"arm": "alien", "atoms": ["radix tree routing", "KCP transport", "Result types"],
            "example_repos": {"radix tree routing": ["SECRET/repo"]},
            "spec": {"title": "SECRET-TITLE", "pitch": "P", "why_unusual": "SECRET-WHY"}}
    text = judge.prompt_for(idea, tmp_path, combos=True)
    assert "KCP transport" in text
    for leak in ("alien", "SECRET"):
        assert leak not in text
