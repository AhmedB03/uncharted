"""Demand first: the most upvoted open requests across the corpus repos.

Technique combinations say what could be built; they do not say that anyone wants it. A request
that has collected thumbs-up for years and is still open is direct evidence of unmet demand.

    python uncharted.py demand            # fetch (resumable) and write the report
    python uncharted.py demand --no-fetch # rebuild the report from what is on disk

The very top of that list is crowded: old, famous requests tend to have outside tools already.
For the middle of the distribution, fetch deeper and newer into its own directory:

    python uncharted.py demand --query "created:>=2024-11-09 reactions:>=10" --per-repo 20         --data data/demand_recent --out results/demand_recent.md --max-up 150 --max-per-repo 5

For every repo under `--repos` this asks GitHub for its open issues with the most thumbs-up
(GraphQL search, several repos per request; needs GITHUB_TOKEN or a logged-in `gh`). One JSON file
per repo goes to `data/demand/`, and the report lists the top requests by thumbs-up with bug
reports left out.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

GRAPHQL = "https://api.github.com/graphql"
ISSUE_FIELDS = """
      nodes { ... on Issue {
        title url createdAt bodyText
        comments { totalCount }
        up: reactions(content: THUMBS_UP) { totalCount }
        reactions { totalCount }
        labels(first: 8) { nodes { name } }
      } }"""
BUG_WORDS = ("bug", "regression", "crash", "defect", "broken")
BODY_CHARS = 4000


def token():
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if t:
        return t
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def corpus_repos(dirs):
    """{repo id: (full_name, stars)} from every meta.json under the given directories."""
    out = {}
    for d in dirs:
        for meta in sorted(Path(d).glob("*/meta.json")):
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if m.get("full_name"):
                out.setdefault(m.get("id") or meta.parent.name, (m["full_name"], m.get("stars") or 0))
    return out


def build_query(full_names, per_repo, extra=""):
    """`extra` is more search qualifiers, e.g. "created:>=2024-11-09 reactions:>=10"."""
    parts = []
    for i, name in enumerate(full_names):
        q = json.dumps(" ".join(f"repo:{name} is:issue is:open {extra}".split()) + " sort:reactions-+1-desc")
        parts.append(f"  r{i}: search(query: {q}, type: ISSUE, first: {per_repo}) {{{ISSUE_FIELDS}\n  }}")
    return "query {\n" + "\n".join(parts) + "\n}"


def post(session, query, tries=6):
    """GraphQL data dict (possibly partial), or None when every try failed."""
    for attempt in range(tries):
        try:
            r = session.post(GRAPHQL, json={"query": query}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        # only the error list counts: issue text that mentions rate limits is not a throttle
        limited = r.status_code in (403, 429) or (
            r.status_code == 200 and any(e.get("type") == "RATE_LIMITED" for e in r.json().get("errors") or []))
        if limited:
            wait = int(r.headers.get("Retry-After") or 0)
            if not wait and r.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time() + 2
            wait = max(wait, 30 * (attempt + 1))
            print(f"  rate limited, sleeping {int(wait)}s", flush=True)
            time.sleep(min(wait, 900))
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            return None
        return r.json().get("data")
    return None


def clean(node):
    return {
        "title": node["title"],
        "url": node["url"],
        "created_at": node["createdAt"],
        "body": (node.get("bodyText") or "")[:BODY_CHARS],
        "thumbs_up": node["up"]["totalCount"],
        "reactions": node["reactions"]["totalCount"],
        "comments": node["comments"]["totalCount"],
        "labels": [l["name"] for l in node["labels"]["nodes"]],
    }


def fetch(repos, out_dir, per_repo=5, batch=10, pause=1.0, extra=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(rid, name) for rid, (name, _) in repos.items() if not (out_dir / f"{rid}.json").exists()]
    print(f"{len(repos)} repos, {len(todo)} to fetch", flush=True)
    if not todo:
        return
    tok = token()
    if not tok:
        raise SystemExit("No GitHub token: set GITHUB_TOKEN or run `gh auth login`.")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {tok}", "User-Agent": "uncharted"})
    failed = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        data = post(session, build_query([n for _, n in chunk], per_repo, extra))
        for i, (rid, name) in enumerate(chunk):
            res = (data or {}).get(f"r{i}")
            if res is None:  # request failed, or this repo is gone or private; retried on the next run
                failed += 1
                continue
            issues = [clean(n) for n in res["nodes"] if n and n.get("url")]
            (out_dir / f"{rid}.json").write_text(
                json.dumps({"repo": name, "issues": issues}, ensure_ascii=False, indent=1), encoding="utf-8")
        done = min(start + batch, len(todo))
        if done % (batch * 10) == 0 or done == len(todo):
            print(f"  fetched {done}/{len(todo)} repos ({failed} failed)", flush=True)
        time.sleep(pause)


def is_bug(issue):
    labels = " ".join(issue["labels"]).lower()
    return any(w in labels for w in BUG_WORDS) or issue["title"].lower().lstrip("[ ").startswith("bug")


def years_open(issue, now=None):
    now = now or datetime.now(timezone.utc)
    made = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
    return (now - made).days / 365.25


def load(out_dir, repos=None):
    """Every stored issue as a flat list, each with its repo name and star count."""
    rows = []
    for f in sorted(Path(out_dir).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        stars = (repos or {}).get(f.stem, (None, 0))[1]
        for issue in d["issues"]:
            rows.append({**issue, "repo": d["repo"], "stars": stars})
    return rows


def top_requests(rows, top=200, min_up=20, min_years=0.0, max_per_repo=2, now=None, max_up=0):
    """Most upvoted open non-bug issues, best first, with at most `max_per_repo` from one repo."""
    keep = [r for r in rows if r["thumbs_up"] >= min_up and not is_bug(r) and years_open(r, now) >= min_years
            and (not max_up or r["thumbs_up"] <= max_up)]
    keep.sort(key=lambda r: (-r["thumbs_up"], r["url"]))
    seen, out = {}, []
    for r in keep:
        if max_per_repo and seen.get(r["repo"], 0) >= max_per_repo:
            continue
        seen[r["repo"]] = seen.get(r["repo"], 0) + 1
        out.append(r)
        if len(out) >= top:
            break
    return out


def report(picks, rows, n_repos):
    lines = [
        "# Most upvoted open requests in the corpus",
        "",
        f"{len(rows)} issues from {n_repos} repos; {sum(r['thumbs_up'] >= 100 for r in rows)} have 100+ thumbs-up. "
        "Bug reports are left out.",
        "",
        "| # | 👍 | Years open | Comments | Repo | Request |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(picks, 1):
        title = r["title"].replace("|", "\\|")[:110]
        lines.append(f"| {i} | {r['thumbs_up']} | {years_open(r):.1f} | {r['comments']} | {r['repo']} | "
                     f"[{title}]({r['url']}) |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos", nargs="+", default=["repos", "repos_big"], help="Directories of collected repos")
    p.add_argument("--data", default="data/demand")
    p.add_argument("--out", default="results/demand.md")
    p.add_argument("--per-repo", type=int, default=5, help="Issues to fetch per repo")
    p.add_argument("--batch", type=int, default=10, help="Repos per GraphQL request")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--min-up", type=int, default=20, help="Thumbs-up needed to be listed")
    p.add_argument("--max-up", type=int, default=0, help="Leave out requests with more thumbs-up (0: no cap)")
    p.add_argument("--query", default="",
                   help='Extra GitHub search qualifiers for the fetch, e.g. "created:>=2024-11-09 reactions:>=10"; '
                        "use a separate --data directory per query")
    p.add_argument("--min-years", type=float, default=0.0, help="Only requests open at least this long")
    p.add_argument("--max-per-repo", type=int, default=2, help="0 turns the per-repo cap off")
    args = p.parse_args(argv)

    repos = corpus_repos(args.repos)
    data = Path(args.data)
    if not args.no_fetch:
        fetch(repos, data, args.per_repo, args.batch, extra=args.query)
    rows = load(data, repos)
    if not rows:
        raise SystemExit(f"No issues under {data}; run without --no-fetch.")
    picks = top_requests(rows, args.top, args.min_up, args.min_years, args.max_per_repo, max_up=args.max_up)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report(picks, rows, len(list(data.glob("*.json")))), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} issues, {len(picks)} listed -> {out}")


if __name__ == "__main__":
    main()
