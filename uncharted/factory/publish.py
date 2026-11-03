"""Ship built projects to GitHub, clearly labeled as AI invented and AI built.

Dry run by default: it shows what it would push. Add --yes to really publish.
Capped per day so the org never looks like spam.

    gh auth login                                   # once
    python uncharted.py publish --org your-org            # preview
    python uncharted.py publish --org your-org --yes      # ship up to 3
"""

import argparse
import shutil
import subprocess
from datetime import datetime, timezone

from uncharted.factory.registry import Registry, now

README = """# {title}

> **Invented and built by AI.** This project was proposed by [Uncharted]({uncharted_url}), a model
> trained to find technique combinations that make sense but that no developer community is
> positioned to build. A language model ({build_model}) then wrote the code and iterated until
> its own test suite passed. No human wrote or reviewed this code. Treat it as an experiment.

{pitch}

**Why this is unusual:** {why_unusual}

## The alien combination

| Technique | Normally found in |
| --- | --- |
{atom_rows}

Coherence z score **{coherence_z:+.2f}** (how well these fit together) and unavailability z score
**{unavailability_z:+.2f}** (how far they sit from any existing community).

## Features

{features}

## Usage

Requires Python 3.11+, standard library only.

```
{usage}
```

Run the tests:

```
pip install pytest && python -m pytest -q
```

## Build record

* {tests_passed} tests passing, {loc} lines of code
* Reached passing tests after {iterations} build iteration(s)
* Published {date}

If this turns out to be useful, a star tells the experiment that this kind of combination works.
"""


def gh(*args, cwd=None):
    return subprocess.run(["gh", *args], cwd=cwd, capture_output=True, text=True)


def write_readme(root, idea, uncharted_url):
    spec = idea["spec"]
    examples = idea.get("example_repos", {})
    rows = "\n".join(
        f"| {a} | {', '.join(f'[{r}](https://github.com/{r})' for r in examples.get(a, [])) or 'n/a'} |"
        for a in idea["atoms"])
    (root / "README.md").write_text(README.format(
        title=spec.get("title", idea["slug"]), uncharted_url=uncharted_url,
        build_model=idea.get("build_model", "an LLM"), pitch=spec.get("pitch", ""),
        why_unusual=spec.get("why_unusual", ""), atom_rows=rows,
        coherence_z=idea["coherence_z"], unavailability_z=idea["unavailability_z"],
        features="\n".join(f"* {f}" for f in spec.get("features", [])),
        usage=spec.get("usage", f"python -m {idea.get('package', '')}"),
        tests_passed=idea.get("tests_passed", 0), loc=idea.get("loc", 0),
        iterations=idea.get("iterations", 1), date=datetime.now(timezone.utc).date().isoformat(),
    ), encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    (root / "LICENSE").write_text(MIT.format(year=datetime.now(timezone.utc).year))


def published_today(reg):
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(1 for i in reg.ideas.values() if (i.get("published_at") or "").startswith(today))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", required=True, help="GitHub user or org to publish under")
    p.add_argument("--out", default="factory_out")
    p.add_argument("--per-day", type=int, default=3)
    p.add_argument("--uncharted-url", default="https://github.com/AhmedB03/uncharted")
    p.add_argument("--yes", action="store_true", help="Actually publish (default is a dry run)")
    args = p.parse_args(argv)

    reg = Registry(args.out)
    ready = [s for s, i in reg.ideas.items() if i["status"] == "built"]
    room = max(0, args.per_day - published_today(reg))
    if not ready:
        raise SystemExit("Nothing built yet. Run build first.")
    if room == 0:
        raise SystemExit(f"Already published {args.per_day} today; the cap keeps the org from looking like spam.")
    if args.yes and not shutil.which("gh"):
        raise SystemExit("Install the GitHub CLI (https://cli.github.com) and run `gh auth login` first.")

    for slug in ready[:room]:
        idea, root = reg.ideas[slug], reg.project_dir(slug)
        write_readme(root, idea, args.uncharted_url)
        name = f"{args.org}/{slug}"
        if not args.yes:
            print(f"  would publish {name}: {idea['spec'].get('pitch', '')}")
            continue
        for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "."],
                    ["git", "commit", "-q", "-m", "Initial commit: invented and built by Uncharted"]):
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            if r.returncode != 0 and "nothing to commit" not in r.stdout:
                raise SystemExit(f"git failed in {root}: {r.stderr or r.stdout}")
        r = gh("repo", "create", name, "--public", "--source", ".", "--push",
               "--description", f"[AI invented] {idea['spec'].get('pitch', '')}"[:340], cwd=root)
        if r.returncode != 0:
            print(f"  failed {name}: {r.stderr.strip()}")
            continue
        gh("repo", "edit", name, "--add-topic", "ai-generated", "--add-topic", "uncharted-factory")
        reg.update(slug, status="published", repo=name, repo_url=f"https://github.com/{name}",
                   published_at=now(), stars=0)
        print(f"  published https://github.com/{name}")
    if not args.yes:
        print("Dry run. Add --yes to publish.")


MIT = """MIT License

Copyright (c) {year} Uncharted Factory contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


if __name__ == "__main__":
    main()
