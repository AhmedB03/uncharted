"""The web build target, run against real headless Chrome with a fake model."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("playwright")

from uncharted import llm  # noqa: E402
from uncharted.factory import build, gallery, play, publish, registry, spec, web  # noqa: E402

INDEX = """<!doctype html>
<meta charset="utf-8">
<title>Blinker</title>
<style>html, body { margin: 0; height: 100%; background: #102030; } canvas { display: block; }</style>
<canvas id="c"></canvas>
<script type="module" src="./src/main.js"></script>
"""

LIFE = """export function neighbours(grid, r, c) {
  let n = 0;
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
    if ((dr || dc) && grid[r + dr] && grid[r + dr][c + dc]) n++;
  }
  return n;
}

export function step(grid) {
  return grid.map((row, r) => row.map((cell, c) => {
    const n = neighbours(grid, r, c);
    return n === 3 || (cell && n === 2) ? 1 : 0;
  }));
}
"""
BUGGY_LIFE = LIFE.replace("n === 3 || (cell && n === 2)", "n === 3")

MAIN = """import { step } from "./life.js";
const canvas = document.getElementById("c"), g = canvas.getContext("2d");
canvas.width = innerWidth; canvas.height = innerHeight;
let grid = Array.from({ length: 9 }, (_, r) => Array.from({ length: 16 }, (_, c) => (r === 4 && c > 5 && c < 9 ? 1 : 0)));
function draw() {
  const s = canvas.width / 16;
  g.fillStyle = "#102030"; g.fillRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = "#ffd166";
  grid.forEach((row, r) => row.forEach((cell, c) => cell && g.fillRect(c * s + 2, r * s + 2, s - 4, s - 4)));
}
function advance() { grid = step(grid); draw(); }
addEventListener("pointerdown", advance);
addEventListener("keydown", advance);
draw();
"""

TESTS = """import { test, assert, assertEqual, assertClose, assertThrows } from "./harness.js";
import { step, neighbours } from "../src/life.js";

const column = [[0, 1, 0], [0, 1, 0], [0, 1, 0]];
test("test_spec_01_blinker_flips", () => { assertEqual(step(column), [[0, 0, 0], [1, 1, 1], [0, 0, 0]]); });
test("test_blinker_returns", () => { assertEqual(step(step(column)), column); });
test("test_neighbours", () => { assertEqual(neighbours(column, 1, 1), 2); assertClose(neighbours(column, 1, 0), 3, 0.5); });
test("test_empty_stays_empty", async () => { assert(step([[0, 0], [0, 0]]).flat().every((x) => x === 0)); });
test("test_bad_grid_throws", () => { assertThrows(() => step(null)); });
"""

SPEC = {"feasible": True, "slug": "blinker", "title": "Blinker", "pitch": "Poke a cellular automaton.",
        "why_unusual": "Nobody pairs these.", "kind": "web", "controls": "click or press any key",
        "features": ["life"], "tests": ["one step turns the column into the row"]}


def project(tmp_path, **changes):
    files = {"index.html": INDEX, "src/life.js": LIFE, "src/main.js": MAIN, "tests/tests.js": TESTS, **changes}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return tmp_path


def test_a_working_page_passes_and_leaves_a_screenshot(tmp_path):
    passed, failed, out = web.run_tests(project(tmp_path))
    assert (passed, failed) == (6, 0), out
    assert "PASSED tests/tests.js::test_spec_01_blinker_flips" in out and f"PASSED {web.PAGE_TEST}" in out
    assert build.spec_tests_missing(out, 1) == []
    assert (tmp_path / "screenshot.png").read_bytes().startswith(b"\x89PNG")
    assert web.tests_look_real(tmp_path)


def test_a_failing_test_reports_expected_and_actual(tmp_path):
    passed, failed, out = web.run_tests(project(tmp_path, **{"src/life.js": BUGGY_LIFE}))
    assert failed >= 1 and "FAILED tests/tests.js::test_blinker_returns" in out
    assert "expected [[0,1,0],[0,1,0],[0,1,0]], got" in out
    assert build.spec_tests_missing(out, 1) == [1]


def test_requests_outside_the_project_are_refused_and_fail_the_page(tmp_path):
    leaky = INDEX + '<img src="https://example.com/sprite.png"><script src="https://cdn.example.org/lib.js"></script>'
    _, failed, out = web.run_tests(project(tmp_path, **{"index.html": leaky}))
    assert failed == 1 and f"FAILED {web.PAGE_TEST}" in out
    assert "outside its own folder" in out and "https://example.com/sprite.png" in out


def test_a_blank_page_that_ignores_input_fails(tmp_path):
    _, failed, out = web.run_tests(project(tmp_path, **{"src/main.js": 'import { step } from "./life.js";\n'}))
    assert failed == 1 and "blank when it opens" in out and "nothing on the page changed" in out


def test_page_errors_and_missing_files_are_reported(tmp_path):
    broken = MAIN.replace('"./life.js"', '"./rules.js"') + "\nnoSuchFunction();\n"
    _, failed, out = web.run_tests(project(tmp_path, **{"src/main.js": broken}))
    assert failed == 1 and "src/rules.js" in out and "do not exist" in out


def test_tests_without_assertions_or_unused_modules_do_not_count(tmp_path):
    lazy = TESTS + 'test("test_nothing", () => { step([[1]]); });\n'
    _, failed, out = web.run_tests(project(tmp_path, **{"tests/tests.js": lazy}))
    assert failed == 1 and "FAILED tests/tests.js::test_nothing - made no assertions" in out

    # The page draws and responds, but with its own copy of the rules instead of the tested module.
    inline = MAIN.replace('import { step } from "./life.js";', "const step = (g) => g.map((r) => r.map((x) => 1 - x));")
    _, failed, out = web.run_tests(project(tmp_path, **{"tests/tests.js": TESTS, "src/main.js": inline}))
    assert failed == 1 and "index.html never loads: src/life.js" in out


def test_a_broken_tests_file_is_one_clear_failure(tmp_path):
    bad = TESTS.replace("neighbours }", "neighbors }")
    passed, failed, out = web.run_tests(project(tmp_path, **{"tests/tests.js": bad}))
    assert "could not be loaded, so no test ran" in out and "neighbors" in out
    assert "ZERO tests ran" in web.hints(out)


def test_web_factory_end_to_end(tmp_path, monkeypatch):
    plays = []

    def fake_chat(prompt, model=None, rpm=8, **kw):
        if "do not want to close" in prompt:
            return json.dumps(SPEC)
        if "as one static web page" in prompt:
            assert "tests/harness.js" in prompt
            return "\n".join(f"===== FILE: {rel} =====\n{text}" for rel, text in {
                "index.html": INDEX, "src/life.js": BUGGY_LIFE, "src/main.js": MAIN, "tests/tests.js": TESTS,
            }.items()) + "\n===== END ====="
        if "play-testing" in prompt:
            plays.append(len(kw["images"]))
            return json.dumps({"seen": "A row of three yellow cells that flips on input.", "plays": True, "problems": []})
        if "fails its tests" in prompt:
            assert "TEST OUTPUT" in prompt and "test_blinker_returns" in prompt
            assert "===== tests/tests.js =====" in prompt and "===== tests/harness.js =====" not in prompt
            return f"===== FILE: src/life.js =====\n{LIFE}\n===== END ====="
        raise AssertionError("unexpected prompt")

    monkeypatch.setattr(llm, "chat", fake_chat)
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps({
        "clusters": {"0": {"name": "cellular automata"}, "1": {"name": "FM synthesis"}, "2": {"name": "rollback"}},
        "atom_assignments": [{"text": "x", "paper_id": f"o__r{i}", "cluster": i} for i in range(3)]}))
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"top_k_sequences": [
        {"cluster_ids": [0, 1, 2], "rank_fused": 1, "z_coherence": 1.5, "z_availability": 0.0}]}))
    out = tmp_path / "factory_out"

    spec.main(["--results", str(results), "--clusters", str(clusters), "--out", str(out), "--rpm", "999",
               "--max-support", "-1", "--prompt", "web"])
    # --sandbox docker would stop here if Docker were asked for: a web project never needs it.
    build.main(["--out", str(out), "--rpm", "999", "--sandbox", "auto"])
    idea = registry.Registry(out).ideas["blinker"]
    assert idea["status"] == "built", idea
    assert idea["iterations"] == 2 and idea["tests_passed"] == 6
    assert idea["target"] == "web" and idea["sandbox"] == "chrome"
    assert plays == [7] and idea["play"]["ok"] and len(idea["play"]["changes"]) == 6
    assert (out / "play" / "blinker" / "frame_7.png").read_bytes().startswith(b"\x89PNG")
    root = out / "projects" / "blinker"
    assert idea["loc"] == sum(len((root / f).read_text().splitlines()) for f in ("index.html", "src/life.js", "src/main.js"))

    publish.main(["--org", "Someone", "--out", str(out)])  # dry run
    readme = (root / "README.md").read_text()
    assert "[Play it in your browser](https://someone.github.io/blinker/)" in readme
    assert "click or press any key" in readme and "python -m pytest" not in readme
    assert (root / ".nojekyll").exists() and not (root / ".git").exists()

    gallery.main(["--out", str(out)])
    assert "Blinker" in (out / "gallery" / "index.html").read_text()


FROZEN = MAIN.replace('addEventListener("keydown", advance);', "")  # passes the page check, but keys do nothing


def build_with_reviewer(tmp_path, monkeypatch, verdicts, extra=()):
    """One web project whose first draft already passes its tests; the reviewer answers from `verdicts`."""
    seen = {"fixes": [], "reviews": 0}

    def fake_chat(prompt, model=None, rpm=8, **kw):
        if "play-testing" in prompt:
            assert "changed since screenshot 1" in prompt and "Poke a cellular automaton." in prompt
            seen["reviews"] += 1
            return json.dumps(verdicts.pop(0))
        if "as one static web page" in prompt:
            return "\n".join(f"===== FILE: {rel} =====\n{text}" for rel, text in {
                "index.html": INDEX, "src/life.js": LIFE, "src/main.js": FROZEN, "tests/tests.js": TESTS,
            }.items()) + "\n===== END ====="
        if "fails its tests" in prompt:
            seen["fixes"].append(prompt)
            return f"===== FILE: src/main.js =====\n{MAIN}\n===== END ====="
        raise AssertionError("unexpected prompt")

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = tmp_path / "factory_out"
    reg = registry.Registry(out)
    reg.update("blinker", status="spec", atoms=["a", "b", "c"], spec=SPEC, spec_prompt="web")
    build.main(["--out", str(out), "--rpm", "999", *extra])
    return registry.Registry(out).ideas["blinker"], seen


def test_play_faults_go_back_to_the_fixer(tmp_path, monkeypatch):
    bad = {"seen": "Three cells that never move.", "plays": False, "problems": ["a click does nothing; it should flip the row"]}
    good = {"seen": "The row flips to a column on a click.", "plays": True, "problems": []}
    idea, seen = build_with_reviewer(tmp_path, monkeypatch, [bad, good])
    assert idea["status"] == "built" and idea["iterations"] == 2 and idea["play"]["ok"]
    assert len(seen["fixes"]) == 1 and play.PLAY_TEST in seen["fixes"][0]
    assert "a click does nothing; it should flip the row" in seen["fixes"][0] and "Three cells that never move." in seen["fixes"][0]
    assert [h["plays"] for h in idea["history"] if "plays" in h] == [False, True]


def test_play_check_is_bounded_and_then_only_recorded(tmp_path, monkeypatch):
    bad = {"seen": "Nothing moves.", "plays": False, "problems": ["frozen"]}
    # The fix changes the files once; after that the project is the same state and is not judged again.
    idea, seen = build_with_reviewer(tmp_path, monkeypatch, [dict(bad), dict(bad)], extra=["--play-rounds", "1"])
    assert idea["status"] == "built" and idea["iterations"] == 2
    assert idea["play"]["ok"] is False and idea["play"]["problems"] == ["frozen"]
    assert len(seen["fixes"]) == 1 and seen["reviews"] == 2


def test_play_gate_off_or_broken_reviewer_never_blocks_a_build(tmp_path, monkeypatch):
    idea, seen = build_with_reviewer(tmp_path / "off", monkeypatch, [], extra=["--no-play-gate"])
    assert idea["status"] == "built" and idea["play"] is None and seen["reviews"] == 0
    idea, seen = build_with_reviewer(tmp_path / "broken", monkeypatch, ["this model cannot see pictures"])
    assert idea["status"] == "built" and idea["play"] is None and seen["fixes"] == []
