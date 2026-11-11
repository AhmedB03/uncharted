"""Ship built projects to GitHub, clearly labeled as AI invented and AI built, credited to you.

Dry run by default: it shows what it would push. Add --yes to really publish.
Capped per day so the org never looks like spam.

    gh auth login                                                    # once
    python uncharted.py publish --org your-org                       # preview
    python uncharted.py publish --org your-org --slug gradlens --yes # ship one
"""

import argparse
import shutil
import subprocess
from datetime import datetime, timezone

from uncharted.factory.registry import Registry, now

README = """# {title}

**Created by {author_md}** with [Uncharted]({uncharted_url}).

> **Invented and built by AI, published by a human.** {author} built and ran Uncharted, a system that
> finds technique combinations that make sense but that no existing project combines. A language model
> ({spec_model}) designed this project from one such combination, and another ({build_model}) wrote
> the code and iterated until its own test suite passed. {author} chose it for publication; no human
> wrote the code line by line. Treat it as an experiment.

{pitch}

**Why this is unusual:** {why_unusual}
{closest}
## The combination

| Technique | Normally found in |
| --- | --- |
{atom_rows}

{evidence}

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


# A web project (spec --prompt web) is a page, so its README leads with the link that plays it.
WEB_README = """# {title}

**[Play it in your browser]({page_url})** (nothing to install, no account, works offline)

[![{title}](screenshot.png)]({page_url})

{pitch}

**How to play:** {controls}

**Created by {author_md}** with [Uncharted]({uncharted_url}).

> **Invented and built by AI, published by a human.** {author} built and ran Uncharted, a system that
> finds technique combinations that make sense but that no existing project combines. A language model
> ({spec_model}) designed this page from one such combination, and another ({build_model}) wrote
> the code and iterated until its tests passed in a real browser. {author} chose it for publication;
> no human wrote the code line by line. Treat it as an experiment.

**Why this is unusual:** {why_unusual}
{closest}
## The combination

| Technique | Normally found in |
| --- | --- |
{atom_rows}

{evidence}

## What is in it

{features}

## Run it yourself

It is one static page of plain JavaScript: no dependencies and no build step.

```
python -m http.server 8000
```

Then open http://localhost:8000 to play, or http://localhost:8000/tests/ to run the tests in your browser.

## Build record

* {tests_passed} checks passing, {loc} lines of code
* Reached passing tests after {iterations} build iteration(s)
* Published {date}

If you had fun with it, a star tells the experiment that this kind of combination works.
"""


def is_web(idea):
    return idea.get("target") == "web"


def page_url(org, slug):
    return f"https://{org.lower()}.github.io/{slug}/"


def gh(*args, cwd=None):
    return subprocess.run(["gh", *args], cwd=cwd, capture_output=True, text=True)


def git_name():
    r = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True)
    return r.stdout.strip() or None


def pretty_model(m):
    """openrouter/anthropic/claude-sonnet-5.5 -> Claude Sonnet 5.5"""
    if not m:
        return "an LLM"
    name = m.rsplit("/", 1)[-1]
    return " ".join(w if w[:1].isdigit() else w.capitalize() for w in name.split("-"))


def evidence_text(idea):
    parts = []
    if "repo_support" in idea or (idea.get("arm") == "alien" and idea.get("spec_prompt") == "v2"):
        parts.append("Checked against about 3,000 popular GitHub repos: none of them combines all three.")
    if "coherence_z" in idea and "unavailability_z" in idea:
        parts.append(f"Coherence z score **{idea['coherence_z']:+.2f}** (how well these fit together) and "
                     f"unavailability z score **{idea['unavailability_z']:+.2f}** (how far they sit from any "
                     "existing community).")
    return " ".join(parts)


def write_readme(root, idea, uncharted_url, author, author_url=None, page=None):
    """README, LICENSE and .gitignore. `page` is the live URL of a web project."""
    spec = idea["spec"]
    examples = idea.get("example_repos", {})
    rows = "\n".join(
        f"| {a} | {', '.join(f'[{r}](https://github.com/{r})' for r in examples.get(a, [])) or 'n/a'} |"
        for a in idea["atoms"])
    closest = spec.get("closest_existing")
    handle = author_url.rstrip("/").rsplit("/", 1)[-1] if author_url else None
    (root / "README.md").write_text((WEB_README if page else README).format(
        page_url=page, controls=spec.get("controls", "click, drag or press a key"),
        title=spec.get("title", idea["slug"]), uncharted_url=uncharted_url, author=author,
        author_md=f"{author} ([{handle}]({author_url}))" if author_url else author,
        spec_model=pretty_model(idea.get("spec_model")), build_model=pretty_model(idea.get("build_model")),
        pitch=spec.get("pitch", ""), why_unusual=spec.get("why_unusual", ""),
        closest=f"\n**Closest existing tool:** {closest}\n" if closest else "",
        atom_rows=rows, evidence=evidence_text(idea),
        features="\n".join(f"* {f}" for f in spec.get("features", [])),
        usage=spec.get("usage", f"python -m {idea.get('package', '')}"),
        tests_passed=idea.get("tests_passed", 0), loc=idea.get("loc", 0),
        iterations=idea.get("iterations", 1), date=datetime.now(timezone.utc).date().isoformat(),
    ), encoding="utf-8")
    if page:
        (root / ".nojekyll").write_text("")  # GitHub Pages: serve the files as they are
    else:
        (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    (root / "LICENSE").write_text(MIT.format(year=datetime.now(timezone.utc).year, author=author))


def published_today(reg):
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(1 for i in reg.ideas.values() if (i.get("published_at") or "").startswith(today))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", required=True, help="GitHub user or org to publish under")
    p.add_argument("--out", default="factory_out")
    p.add_argument("--slug", nargs="+", default=None, help="Publish just these built ideas")
    p.add_argument("--author", default=None, help="Who gets credit (default: git config user.name)")
    p.add_argument("--author-url", default=None, help="Default: https://github.com/<org>")
    p.add_argument("--per-day", type=int, default=3)
    p.add_argument("--uncharted-url", default="https://github.com/AhmedB03/uncharted")
    p.add_argument("--yes", action="store_true", help="Actually publish (default is a dry run)")
    args = p.parse_args(argv)
    author = args.author or git_name() or args.org
    author_url = args.author_url or f"https://github.com/{args.org}"

    reg = Registry(args.out)
    ready = [s for s, i in reg.ideas.items() if i["status"] == "built" and (not args.slug or s in args.slug)]
    if args.slug and len(ready) != len(set(args.slug)):
        raise SystemExit(f"Not built (or not in {args.out}): {', '.join(sorted(set(args.slug) - set(ready)))}")
    room = max(0, args.per_day - published_today(reg))
    if not ready:
        raise SystemExit("Nothing built yet. Run build first.")
    if room == 0:
        raise SystemExit(f"Already published {args.per_day} today; the cap keeps the org from looking like spam.")
    if args.yes and not shutil.which("gh"):
        raise SystemExit("Install the GitHub CLI (https://cli.github.com) and run `gh auth login` first.")

    for slug in ready[:room]:
        idea, root = reg.ideas[slug], reg.project_dir(slug)
        page = page_url(args.org, slug) if is_web(idea) else None
        write_readme(root, idea, args.uncharted_url, author, author_url, page)
        name = f"{args.org}/{slug}"
        if not args.yes:
            print(f"  would publish {name} (credited to {author}): {idea['spec'].get('pitch', '')}"
                  + (f"\n    playable at {page}" if page else ""))
            continue
        for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "."],
                    ["git", "commit", "-q", "-m", "Initial commit: invented and built with Uncharted"]):
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            if r.returncode != 0 and "nothing to commit" not in r.stdout:
                raise SystemExit(f"git failed in {root}: {r.stderr or r.stdout}")
        r = gh("repo", "create", name, "--public", "--source", ".", "--push",
               "--description", f"[AI invented] {idea['spec'].get('pitch', '')}"[:340], cwd=root)
        if r.returncode != 0:
            print(f"  failed {name}: {r.stderr.strip()}")
            continue
        gh("repo", "edit", name, "--add-topic", "ai-generated", "--add-topic", "uncharted-factory")
        if page:
            # Serve the repo root with GitHub Pages, so the README's link plays the page.
            r = gh("api", "-X", "POST", f"repos/{name}/pages", "-f", "source[branch]=main", "-f", "source[path]=/")
            if r.returncode != 0:
                print(f"  could not turn on GitHub Pages for {name} (do it in the repo settings): {r.stderr.strip()}")
            gh("repo", "edit", name, "--homepage", page, "--add-topic", "browser-game")
        reg.update(slug, status="published", repo=name, repo_url=f"https://github.com/{name}",
                   published_at=now(), stars=0, **({"page_url": page} if page else {}))
        print(f"  published https://github.com/{name}" + (f" (plays at {page} once Pages has deployed)" if page else ""))
    if not args.yes:
        print("Dry run. Add --yes to publish.")


MIT = """MIT License

Copyright (c) {year} {author}

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
