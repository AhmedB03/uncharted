"""Build each spec: the model writes code, pytest runs, failures go back, repeat.

Generated code runs inside Docker with no network when Docker is installed.
Without Docker it runs as a local subprocess, so only do that on a machine you
do not mind an AI's code running on.

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
from uncharted.factory.registry import Registry

IMAGE = "uncharted-runner"
RUNNER_DIR = Path(__file__).parent / "runner"

WRITE_PROMPT = """Implement this project completely. It will be tested automatically.

SPEC:
{spec}

Layout rules:
- Package code in `{pkg}/` with `{pkg}/__init__.py`; a CLI also gets `{pkg}/__main__.py`
  so it runs with `python -m {pkg}`.
- Tests in `tests/test_{pkg}.py` using pytest. Cover every behavior listed in the spec's
  "tests". Tests must import `{pkg}` and check real outputs, never `assert True`.
- Python 3.11 standard library only. No network, no files outside a temp directory.
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

{hints}PYTEST OUTPUT (truncated):
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
                deletes.append(m.group(2))
                continue
            stop = marks[i + 1].start() if i + 1 < len(marks) else (end.start() if end else len(reply))
            body = reply[m.end():stop].strip("\n")
            fence = re.fullmatch(r"```[\w+-]*\n(.*?)\n?```\s*", body, re.S)
            files[m.group(2)] = fence.group(1) if fence else body
        return files, deletes
    data = llm.parse_json(reply)
    return data.get("files") or {}, data.get("delete") or []


def stdlib_hints(out, pkg):
    """Plain-language hints for the failures small models keep missing in raw pytest output."""
    hints = []
    for mod in sorted(set(re.findall(r"No module named '([\w.]+)'", out))):
        top = mod.split(".")[0]
        if top == pkg:
            hints.append(f"- `{mod}` does not exist: create that module or fix the import.")
        elif top not in sys.stdlib_module_names and top != "pytest":
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


def snapshot(root, limit=40000):
    parts = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        parts.append(f"===== {p.relative_to(root)} =====\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)[:limit]


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
               "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
        env, cwd = None, None
    else:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
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
               for p in root.rglob("*.py") if "__pycache__" not in p.parts and "tests" not in p.parts)


STDLIB_PROMPT = """This Python file imports {mods}, which are NOT in the Python 3.11 standard library and
are not installed. Rewrite the file so it does the same job using ONLY the standard library
(plus pytest in test files). Keep every function, class and name it exposes.

===== FILE: {rel} =====
{code}
===== END =====

""" + FIX_FORMAT


def foreign_imports(root, pkg):
    """{relative path: sorted third-party top-level modules} for every .py file in the project."""
    import ast
    local = {pkg, "pytest", "tests", "conftest", "_pytest"}
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


def purge_foreign_imports(root, pkg, args, tries=2):
    """Ask for a targeted, one-file rewrite of anything importing third-party packages."""
    for _ in range(tries):
        bad = foreign_imports(root, pkg)
        if not bad:
            return {}
        for rel, mods in bad.items():
            code = (root / rel).read_text(encoding="utf-8")
            try:
                ask_and_apply(root, STDLIB_PROMPT.format(mods=", ".join(mods), rel=rel, code=code), args, 0.2)
            except Exception:
                pass
    return foreign_imports(root, pkg)


def ask_and_apply(root, prompt, args, temperature):
    """One model call; if the reply cannot be parsed, ask once more with the error attached."""
    reply = llm.chat(prompt, args.model, args.rpm, temperature=temperature)
    try:
        return apply(root, reply)
    except ValueError as e:
        if "outside the project" in str(e) or "odd file name" in str(e):
            raise
        retry = (prompt + f"\n\nYour previous reply could not be parsed ({e}). Reply again using EXACTLY "
                 "the file block format above, with no other text.")
        return apply(root, llm.chat(retry, args.model, args.rpm, temperature=temperature))


def build_one(reg, slug, args, sandbox):
    idea = reg.ideas[slug]
    spec = json.dumps(idea["spec"], indent=2)
    pkg = pkg_name(slug)
    root = reg.project_dir(slug)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    history = []
    prompt = WRITE_PROMPT.format(spec=spec, pkg=pkg, format=FILE_FORMAT.format(pkg=pkg))
    try:
        ask_and_apply(root, prompt, args, temperature=0.3)
    except Exception as e:
        reg.update(slug, status="failed", fail_reason=f"first draft: {e}"[:300], iterations=0,
                   build_model=args.model)
        return "failed"

    for it in range(1, args.max_iters + 1):
        leftover = purge_foreign_imports(root, pkg, args)
        passed, failed, out = run_tests(root, sandbox)
        if leftover:
            history.append({"iter": it, "foreign_imports": leftover})
        history.append({"iter": it, "passed": passed, "failed": failed})
        print(f"    iter {it}: {passed} passed, {failed} failed")
        if failed == 0 and passed >= args.min_tests:
            if not tests_look_real(root, pkg):
                out = ("All tests pass but they look trivial: they must import the package and make at "
                       "least 3 real assertions about its outputs. Strengthen the tests, then fix the code.")
            else:
                reg.update(slug, status="built", iterations=it, tests_passed=passed, loc=loc(root),
                           history=history, package=pkg, build_model=args.model, sandbox=sandbox)
                return "built"
        elif failed == 0:
            out = f"Only {passed} tests ran; write at least {args.min_tests} covering the spec.\n" + out
        if it == args.max_iters:
            break
        try:
            ask_and_apply(root, FIX_PROMPT.format(spec=spec, files=snapshot(root), output=out[-6000:],
                                                  hints=stdlib_hints(out, pkg), format=FIX_FORMAT),
                          args, temperature=0.2)
        except Exception as e:
            history.append({"iter": it, "error": str(e)[:200]})
    reg.update(slug, status="failed", iterations=len(history), history=history, package=pkg,
               fail_reason="tests still failing after max iterations", build_model=args.model)
    return "failed"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="factory_out")
    p.add_argument("--slug", default=None, help="Build just this idea")
    p.add_argument("--max", type=int, default=5, help="How many ideas to build this run")
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--min-tests", type=int, default=5)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--sandbox", choices=["auto", "docker", "local"], default="auto")
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--rpm", type=int, default=8)
    args = p.parse_args(argv)

    sandbox = args.sandbox
    if sandbox in ("auto", "docker"):
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

    reg = Registry(args.out)
    wanted = {"spec"} | ({"failed"} if args.retry_failed else set())
    queue = [args.slug] if args.slug else [s for s, i in reg.ideas.items() if i["status"] in wanted]
    counts = {"built": 0, "failed": 0}
    for slug in queue[: args.max]:
        print(f"  building {slug}: {reg.ideas[slug]['spec'].get('pitch', '')}")
        counts[build_one(reg, slug, args, sandbox)] += 1
    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
