"""Collect popular GitHub repos: metadata, README, dependency manifests, contributors.

Each repo becomes repos/{owner}__{name}/ with meta.json, readme.md and deps.txt.
Reruns skip repos that are already complete.

    python uncharted.py collect --target 2000 --out repos
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from uncharted.github import GitHub

LANGUAGES = [
    "Python", "TypeScript", "JavaScript", "Go", "Rust", "Java", "C++", "C",
    "C#", "Kotlin", "Swift", "Ruby", "PHP", "Scala", "Elixir", "Zig",
    "Dart", "Julia", "Haskell", "Lua",
]

MANIFESTS = [
    "requirements.txt", "pyproject.toml", "setup.py", "package.json", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "Gemfile",
    "composer.json", "mix.exs", "Package.swift", "pubspec.yaml", "CMakeLists.txt",
    "Project.toml", "build.zig", "stack.yaml",
]

README_CHARS = 12000
MANIFEST_CHARS = 4000
BOT = re.compile(r"(\[bot\]|-bot$|^dependabot|^renovate|^github-actions)", re.I)


def repo_id(full_name):
    return full_name.replace("/", "__")


def search(gh, target, min_stars, pushed_after, languages):
    """Spread the target across languages so the corpus is not all Python and JS.

    GitHub search returns at most 1000 results per query, so each language is
    capped there; we then fill any leftover quota round robin.
    """
    quota = max(1, target // len(languages))
    seen, picked = set(), []
    for lang in languages:
        q = (f'language:"{lang}" stars:>={min_stars} pushed:>={pushed_after} '
             f"fork:false archived:false")
        got = 0
        for page in range(1, 11):
            data = gh.json("/search/repositories",
                           {"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
            if not data or not data.get("items"):
                break
            for item in data["items"]:
                if item["full_name"] in seen:
                    continue
                seen.add(item["full_name"])
                picked.append(item)
                got += 1
                if got >= quota:
                    break
            if got >= quota or len(data["items"]) < 100:
                break
        print(f"  {lang:<11} {got} repos")
        if len(picked) >= target:
            break
    return picked[:target]


def fetch_one(gh, item, out_dir, min_contributions):
    owner, name = item["full_name"].split("/")
    d = out_dir / repo_id(item["full_name"])
    if (d / "meta.json").exists():
        return "skip"
    d.mkdir(parents=True, exist_ok=True)

    branch = item.get("default_branch") or "main"
    readme = gh.decoded(gh.json(f"/repos/{owner}/{name}/readme"), README_CHARS)
    if readme is None:
        readme = next((t for t in (gh.raw(owner, name, branch, f, README_CHARS)
                                   for f in ("README.md", "readme.md", "README.rst", "README")) if t), "")

    # One API call to list the root, then manifest bodies come from raw.githubusercontent.com,
    # which does not count against the REST rate limit.
    root = gh.json(f"/repos/{owner}/{name}/contents")
    if isinstance(root, list):
        present = {f["name"] for f in root if isinstance(f, dict) and f.get("type") == "file"}
        candidates = [m for m in MANIFESTS if m in present]
    else:
        present, candidates = set(), MANIFESTS  # listing blocked: probe raw URLs directly
    deps = []
    for m in candidates:
        text = gh.raw(owner, name, branch, m, MANIFEST_CHARS)
        if text:
            present.add(m)
            deps.append(f"===== {m} =====\n{text}")

    contribs = gh.json(f"/repos/{owner}/{name}/contributors", {"per_page": 50}) or []
    contributors = [
        {"id": c["id"], "login": c["login"], "contributions": c["contributions"]}
        for c in contribs
        if isinstance(c, dict) and c.get("type") == "User"
        and not BOT.search(c.get("login", ""))
        and c.get("contributions", 0) >= min_contributions
    ]

    meta = {
        "id": repo_id(item["full_name"]),
        "full_name": item["full_name"],
        "html_url": item["html_url"],
        "description": item.get("description") or "",
        "language": item.get("language"),
        "topics": item.get("topics", []),
        "stars": item.get("stargazers_count", 0),
        "created_at": item.get("created_at"),
        "pushed_at": item.get("pushed_at"),
        "license": (item.get("license") or {}).get("spdx_id"),
        "manifests": [m for m in MANIFESTS if m in present],
        "contributors": contributors,
    }
    (d / "readme.md").write_text(readme, encoding="utf-8")
    (d / "deps.txt").write_text("\n\n".join(deps), encoding="utf-8")
    # meta.json last, so a crash mid repo gets refetched on the next run
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return "ok"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--out", default="repos")
    p.add_argument("--min-stars", type=int, default=300)
    p.add_argument("--pushed-after", default="2025-01-01")
    p.add_argument("--languages", nargs="*", default=LANGUAGES)
    p.add_argument("--min-contributions", type=int, default=3,
                   help="Drop contributors below this many commits (filters drive by typo fixes)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gh = GitHub()

    print(f"Searching for {args.target} repos...")
    items = search(gh, args.target, args.min_stars, args.pushed_after, args.languages)
    (out_dir / "_search.json").write_text(json.dumps([i["full_name"] for i in items], indent=1))
    print(f"Found {len(items)} repos, fetching details with {args.workers} workers")

    counts = {"ok": 0, "skip": 0, "error": 0}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_one, gh, i, out_dir, args.min_contributions): i for i in items}
        for n, f in enumerate(as_completed(futures), 1):
            try:
                counts[f.result()] += 1
            except Exception as e:
                counts["error"] += 1
                print(f"  error on {futures[f]['full_name']}: {e}")
            if n % 100 == 0:
                print(f"  {n}/{len(items)} {counts}")
    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
