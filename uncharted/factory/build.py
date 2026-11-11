"""Build each spec: the model writes code, the tests run, failures go back, repeat.

Generated Python runs inside Docker with no network when Docker is installed.
Without Docker it runs as a local subprocess, so only do that on a machine you
do not mind an AI's code running on. Web specs (`spec --prompt web`) are built as
one static page and tested in headless Chrome with the network cut off; see web.py.

    python uncharted.py build --max 5
    python uncharted.py build --slug some-project --max-iters 8
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from uncharted import llm
from uncharted.factory import web
from uncharted.factory.registry import Registry

# Bump the tag whenever runner/Dockerfile changes: docker_ready() only builds a missing image.
IMAGE = "uncharted-runner:libs1"
RUNNER_DIR = Path(__file__).parent / "runner"

# Specs written with spec.py's v3lib prompt may use these packages (pip name -> import name).
# They are installed in the sandbox image; keep this in sync with runner/Dockerfile.
LIB_IMPORTS = {"numpy": "numpy", "cryptography": "cryptography", "aioquic": "aioquic", "networkx": "networkx",
               "lark": "lark", "pillow": "PIL", "msgpack": "msgpack", "zstandard": "zstandard",
               "pyyaml": "yaml", "tomli-w": "tomli_w"}
LIB_PROMPTS = {"v3lib"}
# Specs written with these prompts are web pages, built and tested by web.WebTarget.
WEB_PROMPTS = {"web"}


def allowed_imports(idea):
    return frozenset(LIB_IMPORTS.values()) if idea.get("spec_prompt") in LIB_PROMPTS else frozenset()


def deps_rule(allowed):
    if not allowed:
        return "- Python 3.11 standard library only."
    pips = ", ".join(sorted(LIB_IMPORTS))
    return (f"- Python 3.11 standard library plus ONLY these installed packages: {pips} (import names: "
            f"{', '.join(sorted(allowed))}). Nothing else is installed.")

WRITE_PROMPT = """Implement this project completely. It will be tested automatically.

SPEC:
{spec}

Layout rules:
- Package code in `{pkg}/` with `{pkg}/__init__.py`; a CLI also gets `{pkg}/__main__.py`
  so it runs with `python -m {pkg}`. If the spec's usage names the module differently, use
  `{pkg}` instead.
- Split the package into several modules of at most about 300 lines each (for example one per
  technique plus `cli.py`), so any single file can later be rewritten whole.
- Tests in `tests/test_{pkg}.py` using pytest. Tests must import `{pkg}` and check real
  outputs, never `assert True`.
- The spec's "tests" list is the acceptance checklist. For entry number N (counting from 1),
  write a test named `test_spec_NN_<short_name>` (two digits, e.g. `test_spec_01_parses_entry`)
  that checks exactly that behavior with the inputs and expected outputs it gives. The project
  only counts as built when every one of these passes. Other tests may use any name.
- Every feature in the spec must be wired into the real code path: a component that is built
  and then never used by the rest of the program does not count.
{deps}
- No network (local sockets on 127.0.0.1 are fine), no files outside a temp directory.
- Real, working code. No TODOs, no placeholders, no stubs.

""" + "{format}"

FILE_FORMAT = """Return every file as a block in exactly this format, with nothing else around it:

===== FILE: {pkg}/__init__.py =====
<full contents>
===== FILE: tests/test_{pkg}.py =====
<full contents>
===== END ====="""

FIX_FORMAT = """Return the full new contents of every file you change, in exactly this format:

===== FILE: path/to/file.py =====
<full contents>
===== END =====
To delete a file, add a line `===== DELETE: path =====` before END."""

FIX_PROMPT = """Your project fails its tests. Fix the CODE so the tests pass. Only change a test
if the test itself is wrong about what the spec asks for; never weaken a test to make it pass.

SPEC:
{spec}

CURRENT FILES:
{files}

{hints}{label} (truncated):
{output}

""" + "{format}"

FILE_RE = re.compile(r"^=====\s*(FILE|DELETE):\s*(\S+?)\s*=====\s*$", re.M)
END_RE = re.compile(r"^=====\s*END\s*=====\s*$", re.M)


def parse_files(reply):
    """Return (files, deletes) from a reply in file-block format, falling back to JSON.

    Small local models routinely break JSON when code contains quotes or backslashes,
    so the prompts ask for plain delimited blocks; JSON is still accepted.
    """
    marks = list(FILE_RE.finditer(reply))
    if marks:
        end = END_RE.search(reply, marks[-1].end())
        files, deletes = {}, []
        for i, m in enumerate(marks):
            if m.group(1) == "DELETE":
                deletes.append(norm_path(m.group(2)))
                continue
            stop = marks[i + 1].start() if i + 1 < len(marks) else (end.start() if end else len(reply))
            # Models often close every file with its own END line, not just the last one; left in,
            # that line is a syntax error (this broke most multi-file fix rounds until 2026-11-08).
            own_end = END_RE.search(reply, m.end(), stop)
            body = reply[m.end():own_end.start() if own_end else stop].strip("\n")
            fence = re.fullmatch(r"```[\w+-]*\n(.*?)\n?```\s*", body, re.S)
            files[norm_path(m.group(2))] = fence.group(1) if fence else body
        return files, deletes
    data = llm.parse_json(reply)
    files, deletes = data.get("files") or {}, data.get("delete") or []
    if isinstance(files, dict):
        files = {norm_path(k): v for k, v in files.items()}
    return files, [norm_path(d) for d in deletes if isinstance(d, str)]


def norm_path(rel):
    """Models sometimes write Windows-style backslash paths; treat them as forward slashes."""
    return rel.strip().replace("\\", "/")


def stdlib_hints(out, pkg, allowed=frozenset()):
    """Plain-language hints for the failures small models keep missing in raw pytest output."""
    hints = []
    for mod in sorted(set(re.findall(r"No module named '([\w.]+)'", out))):
        top = mod.split(".")[0]
        if top == pkg:
            hints.append(f"- `{mod}` does not exist: create that module or fix the import.")
        elif top not in sys.stdlib_module_names and top != "pytest" and top not in allowed:
            hints.append(f"- `{top}` is NOT in the Python standard library and is not installed. Remove every "
                         f"use of it and rewrite that code with the standard library only.")
    if "error collecting" in out.lower() or "errors during collection" in out.lower():
        hints.append("- The test file could not even be imported, so ZERO tests ran. Fix the import/syntax "
                     "error first.")
    return ("HINTS:\n" + "\n".join(hints) + "\n\n") if hints else ""


def pkg_name(slug):
    return re.sub(r"[^a-z0-9_]", "_", slug.replace("-", "_")).strip("_") or "alien"


def safe_path(root, rel):
    p = (root / rel).resolve()
    if root.resolve() not in p.parents or rel.startswith(("/", "~")) or ".." in Path(rel).parts:
        raise ValueError(f"refusing to write outside the project: {rel}")
    if not re.fullmatch(r"[\w\-./]+", rel):
        raise ValueError(f"odd file name: {rel}")
    return p


def apply(root, reply):
    files, deletes = parse_files(reply)
    if not isinstance(files, dict) or not files:
        raise ValueError("reply had no files")
    for rel in files:
        safe_path(root, rel)  # validate every path before writing any
    for rel, content in files.items():
        p = safe_path(root, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    for rel in deletes:
        p = safe_path(root, rel)
        if p.is_file():
            p.unlink()
    return list(files)


# The old 40k cap hid the end of the package (and often the whole test file) from the fixer on
# v2-sized projects: 5 of 8 failed builds were over it, and full-file rewrites of a half-seen
# module then broke collection. 200k chars is ~50k tokens, well inside the builder's context.
SNAPSHOT_LIMIT = 200000


# What counts as a project's source: Python projects hold only the first, web projects the rest.
SOURCE_SUFFIXES = (".py", ".html", ".js", ".css")


def source_files(root):
    """The files the model wrote, leaving out the test harness the factory adds to web projects."""
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES
            and "__pycache__" not in p.parts and p.relative_to(root).as_posix() not in web.FACTORY_FILES]


def snapshot(root, limit=SNAPSHOT_LIMIT, tests_last=False):
    """All source files as FILE blocks. tests_last puts package code before tests, so a reader
    with a smaller limit loses tests rather than the code. Truncation is always marked."""
    files = source_files(root)
    key = (lambda p: ("tests" in p.relative_to(root).parts, p.as_posix())) if tests_last else (lambda p: p.as_posix())
    parts = [f"===== {p.relative_to(root).as_posix()} =====\n{p.read_text(encoding='utf-8')}"
             for p in sorted(files, key=key)]
    text = "\n\n".join(parts)
    if len(text) > limit:
        text = text[:limit] + f"\n\n[... truncated: {len(text) - limit} more characters not shown]"
    return text


def state_files(root):
    """Every text file of the project (the web runner's screenshot.png is not part of its state)."""
    return [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".png"]


def save_state(root):
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in state_files(root)}


def restore_state(root, state):
    for p in state_files(root):
        if p.relative_to(root).as_posix() not in state:
            p.unlink()
    for rel, text in state.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


SPEC_TEST_RE = re.compile(r"^PASSED \S*::test_spec_(\d+)", re.M)


def spec_tests_missing(out, n_spec_tests):
    """Numbers (1-based) of spec checklist entries with no passing test_spec_NN test."""
    passed = {int(n) for n in SPEC_TEST_RE.findall(out)}
    return [i for i in range(1, n_spec_tests + 1) if i not in passed]


def docker_ready():
    if not shutil.which("docker"):
        return False
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        print(f"Building sandbox image {IMAGE} (one time)...")
        if subprocess.run(["docker", "build", "-t", IMAGE, str(RUNNER_DIR)]).returncode != 0:
            return False
    return True


def run_tests(root, sandbox, timeout=180):
    """Return (passed_count, failed_count, output)."""
    if sandbox == "docker":
        cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--cpus", "1",
               "--pids-limit", "256", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{root.resolve()}:/w", IMAGE,
               "python", "-m", "pytest", "-q", "-rfEp", "-p", "no:cacheprovider", "tests"]
        env, cwd = None, None
    else:
        # -rfEp lists every passed test by name, which spec_tests_missing() needs.
        cmd = [sys.executable, "-m", "pytest", "-q", "-rfEp", "-p", "no:cacheprovider", "tests"]
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
               "HOME": str(root.resolve()), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        # Windows cannot start sockets/asyncio without these; a bare env makes every run "fail".
        for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
            if k in os.environ:
                env[k] = os.environ[k]
        cwd = root
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 0, 1, f"Tests timed out after {timeout}s (infinite loop or blocking input?)"
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", out))
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|errors?)", out))
    if r.returncode != 0 and failed == 0:
        failed = 1  # collection error, syntax error, no tests found
    return passed, failed, out


def tests_look_real(root, pkg):
    tests = list((root / "tests").glob("test_*.py"))
    text = "\n".join(t.read_text(encoding="utf-8") for t in tests)
    asserts = len(re.findall(r"\bassert\b|pytest\.raises", text))
    trivial = len(re.findall(r"assert\s+(True|1)\b", text))
    return bool(tests) and pkg in text and asserts - trivial >= 3


def loc(root):
    return sum(len(p.read_text(encoding="utf-8").splitlines())
               for p in source_files(root) if "tests" not in p.relative_to(root).parts)


STDLIB_PROMPT = """This Python file imports {mods}, which are NOT in the Python 3.11 standard library and
are not installed. Rewrite the file so it does the same job using ONLY the standard library
(plus pytest in test files). Keep every function, class and name it exposes.

===== FILE: {rel} =====
{code}
===== END =====

""" + FIX_FORMAT


def foreign_imports(root, pkg, allowed=frozenset()):
    """{relative path: sorted third-party top-level modules} for every .py file in the project."""
    import ast
    local = {pkg, "pytest", "tests", "conftest", "_pytest"} | set(allowed)
    local |= {p.stem for p in root.rglob("*.py")} | {p.name for p in root.iterdir() if p.is_dir()}
    found = {}
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # pytest will report it
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
        bad = sorted(m for m in mods if m not in sys.stdlib_module_names and m not in local)
        if bad:
            found[p.relative_to(root).as_posix()] = bad
    return found


def purge_foreign_imports(root, pkg, args, tries=2, allowed=frozenset()):
    """Ask for a targeted, one-file rewrite of anything importing third-party packages."""
    for _ in range(tries):
        bad = foreign_imports(root, pkg, allowed)
        if not bad:
            return {}
        for rel, mods in bad.items():
            code = (root / rel).read_text(encoding="utf-8")
            try:
                ask_and_apply(root, STDLIB_PROMPT.format(mods=", ".join(mods), rel=rel, code=code), args, 0.2)
            except Exception:
                pass
    return foreign_imports(root, pkg, allowed)


# 8000 cut 800+ line projects off before their tests were written (every build then "failed"
# with no test file). The cap only costs anything when the model actually needs the room.
BUILD_MAX_TOKENS = int(os.environ.get("UNCHARTED_BUILD_MAX_TOKENS", "32000"))


def ask_and_apply(root, prompt, args, temperature):
    """One model call; if the reply cannot be parsed, ask once more with the error attached."""
    reply = llm.chat(prompt, args.model, args.rpm, temperature=temperature, max_tokens=BUILD_MAX_TOKENS)
    try:
        return apply(root, reply)
    except ValueError as e:
        if "outside the project" in str(e) or "odd file name" in str(e):
            raise
        retry = (prompt + f"\n\nYour previous reply could not be parsed ({e}). Reply again using EXACTLY "
                 "the file block format above, with no other text.")
        return apply(root, llm.chat(retry, args.model, args.rpm, temperature=temperature,
                                    max_tokens=BUILD_MAX_TOKENS))


PARENT_NOTE = """

This project extends an existing one (`explore`). The existing project's working code is below.
Start from it: keep and adapt what the spec still needs, under the new package name `{pkg}`, and
return every file of the new project in full, including the ones you carry over.

EXISTING PROJECT ({parent}):
{files}"""


def parent_code(reg, idea, pkg):
    """For an `explore` child, the code of the stone it stands on, as an addition to the write prompt."""
    parent = idea.get("parent")
    if not parent or not reg.project_dir(parent).is_dir():
        return ""
    return PARENT_NOTE.format(pkg=pkg, parent=parent, files=snapshot(reg.project_dir(parent), tests_last=True))


class PythonTarget:
    """What build_one needs to know about a kind of project; web.WebTarget is the other kind."""
    name = "python"
    sandbox = None  # decided in main(): docker or local
    output_label = "PYTEST OUTPUT"
    fix_format = FIX_FORMAT
    trivial = ("All tests pass but they look trivial: they must import the package and make at "
               "least 3 real assertions about its outputs. Strengthen the tests, then fix the code.")

    def write_prompt(self, spec, pkg, idea):
        return WRITE_PROMPT.format(spec=spec, pkg=pkg, deps=deps_rule(allowed_imports(idea)),
                                   format=FILE_FORMAT.format(pkg=pkg))

    def prepare(self, root, pkg, args, idea):
        """Runs before each test round; returns whatever it could not clean up, for the history."""
        return purge_foreign_imports(root, pkg, args, allowed=allowed_imports(idea))

    def run_tests(self, root, sandbox):
        return run_tests(root, sandbox)

    def tests_look_real(self, root, pkg):
        return tests_look_real(root, pkg)

    def hints(self, out, pkg, idea):
        return stdlib_hints(out, pkg, allowed_imports(idea))


def target_for(idea):
    return web.WebTarget() if idea.get("spec_prompt") in WEB_PROMPTS else PythonTarget()


def build_one(reg, slug, args, sandbox):
    idea = reg.ideas[slug]
    spec = json.dumps(idea["spec"], indent=2)
    pkg = pkg_name(slug)
    target = target_for(idea)
    sandbox = target.sandbox or sandbox
    root = reg.project_dir(slug)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    history = []
    prompt = target.write_prompt(spec, pkg, idea) + parent_code(reg, idea, pkg)
    try:
        ask_and_apply(root, prompt, args, temperature=0.3)
    except Exception as e:
        reg.update(slug, status="failed", fail_reason=f"first draft: {e}"[:300], iterations=0,
                   build_model=args.model)
        return "failed"

    checklist = idea["spec"].get("tests") or []
    best, best_score = None, None
    for it in range(1, args.max_iters + 1):
        leftover = target.prepare(root, pkg, args, idea)
        passed, failed, out = target.run_tests(root, sandbox)
        missing = spec_tests_missing(out, len(checklist)) if args.spec_gate else []
        if leftover:
            history.append({"iter": it, "foreign_imports": leftover})
        history.append({"iter": it, "passed": passed, "failed": failed, "spec_missing": len(missing)})
        print(f"    iter {it}: {passed} passed, {failed} failed, {len(missing)}/{len(checklist)} spec checks missing")

        # A fix round that rewrites a big module often breaks collection outright (0 passed). Never
        # build on a worse state: go back to the best one and tell the model its change was undone.
        score = (len(checklist) - len(missing), passed, -failed)
        reverted = ""
        if best_score is not None and score < best_score:
            restore_state(root, best)
            reverted = (f"NOTE: your last change made things worse ({best_score[1]} -> {passed} tests passing), "
                        "so it was undone. The files below are the better earlier version; the pytest output "
                        "is from your undone change. Make smaller, targeted edits.\n\n")
            passed, failed, out = best_out
            missing = spec_tests_missing(out, len(checklist)) if args.spec_gate else []
        else:
            best, best_score, best_out = save_state(root), score, (passed, failed, out)

        if failed == 0 and passed >= args.min_tests and not missing:
            if not target.tests_look_real(root, pkg):
                out = target.trivial
            else:
                reg.update(slug, status="built", iterations=it, tests_passed=passed, loc=loc(root),
                           history=history, package=pkg, build_model=args.model, sandbox=sandbox,
                           target=target.name,
                           spec_checks=len(checklist) if args.spec_gate else None)
                return "built"
        elif failed == 0 and passed < args.min_tests:
            out = f"Only {passed} tests ran; write at least {args.min_tests} covering the spec.\n" + out
        if missing:
            out = ("These entries of the spec's \"tests\" checklist have no passing `test_spec_NN_*` test. "
                   "Implement the behavior in the package code (not just the test) and add the test:\n"
                   + "\n".join(f"- {i:02d}: {checklist[i - 1]}" for i in missing) + "\n\n" + out)
        if it == args.max_iters:
            break
        shown = "\n".join(line for line in out.splitlines() if not line.startswith("PASSED "))
        try:
            ask_and_apply(root, reverted + FIX_PROMPT.format(
                spec=spec, files=snapshot(root), output=shown[-8000:], label=target.output_label,
                hints=target.hints(out, pkg, idea), format=target.fix_format), args, temperature=0.2)
        except Exception as e:
            history.append({"iter": it, "error": str(e)[:200]})
    if best is not None:
        restore_state(root, best)  # leave the best attempt on disk, not the last one
    reg.update(slug, status="failed", iterations=len(history), history=history, package=pkg, target=target.name,
               fail_reason="tests or spec checks still failing after max iterations", build_model=args.model)
    return "failed"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="factory_out")
    p.add_argument("--slug", default=None, help="Build just this idea")
    p.add_argument("--max", type=int, default=5, help="How many ideas to build this run")
    p.add_argument("--max-iters", type=int, default=8,
                   help="Fix rounds; v2 specs are ~650 lines and built 14/19 at 8 rounds vs 9/18 at 4")
    p.add_argument("--min-tests", type=int, default=5)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--no-spec-gate", dest="spec_gate", action="store_false",
                   help="Count a build as done when its own tests pass, without requiring one passing "
                        "test_spec_NN test per spec checklist entry (the pre-2026-11-08 behaviour)")
    p.add_argument("--sandbox", choices=["auto", "docker", "local"], default="auto")
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--rpm", type=int, default=8)
    args = p.parse_args(argv)

    reg = Registry(args.out)
    wanted = {"spec"} | ({"failed"} if args.retry_failed else set())
    queue = [args.slug] if args.slug else [s for s, i in reg.ideas.items() if i["status"] in wanted]
    queue = queue[: args.max]

    sandbox = args.sandbox
    # Web pages are tested in headless Chrome, so Docker only matters when a Python project is queued.
    if sandbox in ("auto", "docker") and any(target_for(reg.ideas[s]).name == "python" for s in queue):
        if docker_ready():
            sandbox = "docker"
        elif sandbox == "docker":
            raise SystemExit("Docker requested but not available.")
        else:
            sandbox = "local"
            print("WARNING: Docker not found, so AI written code will run directly on this machine.\n"
                  "Install Docker Desktop to sandbox it. Continuing in 5 seconds (Ctrl+C to stop).")
            import time
            time.sleep(5)

    counts = {"built": 0, "failed": 0}
    for slug in queue:
        print(f"  building {slug}: {reg.ideas[slug]['spec'].get('pitch', '')}")
        counts[build_one(reg, slug, args, sandbox)] += 1
    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
