"""Spec a standalone tool from an upvoted GitHub request instead of from a technique combination.

    python uncharted.py demand-spec https://github.com/starship/starship/issues/6555 --out factory_out/demand_v1
    python uncharted.py build --out factory_out/demand_v1

The spec writer reads the issue (title, body, and the most upvoted and most recent comments) and
designs a tool that runs beside the project, since the request asks for a change nobody outside the
project can make. The registry entry has the same shape `build` and `judge` read, with arm "demand".
"""

import argparse
import os
import re

import requests

from uncharted import demand, llm
from uncharted.factory.registry import Registry
from uncharted.factory.spec import DEFAULT_SPEC_MODEL, slugify

PROMPT = """You design a small standalone tool that answers a real, upvoted feature request.

The request below is open on GitHub with {thumbs_up} thumbs-up. The maintainers have not shipped it,
and you cannot change their project. Design a separate tool that gives the people in this thread
what they are asking for, working beside the project as it is today.

REQUEST: {title}
{url}

{body}

WHAT PEOPLE IN THE THREAD SAY (most upvoted first, then the latest):
{comments}

Work in this order, and be honest at each step:
1. Say who is asking and what they do today instead (the workaround in the thread, if any).
2. Decide what the smallest tool is that they would install and run on their own real files or
   setup, on day one. It must work with the project's real formats and conventions as they are.
3. Name the closest existing tool and what yours does that it does not. If a tool named in the
   thread already solves this well, set "feasible" to false and say which.

Realness rules (a reviewer will read the code):
- It must do the real job on real inputs. Never call anything "simulated", "mock" or "toy".
- If the job needs an outside program or the network at run time (git, an HTTP API), the tool
  calls it for real, through one small function that tests replace with a fake.

Hard constraints (the project is built and tested automatically):
- Python 3.11, standard library ONLY (plus pytest for tests). If a format has no writer in the
  standard library (e.g. TOML), write the small part of it the tool needs.
- Tests run with no network access and no outside programs, and must be deterministic.
- About 300 to 900 lines of code. A CLI tool.
- If no honest version fits these limits, set "feasible" to false and explain why.

Return only JSON:
{{
  "feasible": true,
  "user": "who is asking and what they do today",
  "closest_existing": "the closest existing tool",
  "slug": "short-kebab-case-name",
  "title": "Readable Name",
  "pitch": "one sentence: who it is for, and what it lets them do that they cannot do today",
  "why_unusual": "one sentence: why this does not exist yet",
  "kind": "cli",
  "features": ["3 to 6 concrete features, each naming the real mechanism it uses"],
  "usage": "an example command a user would run on their own inputs",
  "tests": ["6 to 12 specific behaviors the test suite must check"]
}}

Each entry in "tests" must be a concrete input -> expected output check a unit test can assert
exactly. Never list performance, "handles large inputs", logging, "no network access", or "uses
only the standard library": those cannot be asserted."""

ISSUE_QUERY = """query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) { issue(number: $number) {
    title url bodyText
    up: reactions(content: THUMBS_UP) { totalCount }
    comments(last: 60) { nodes { bodyText createdAt up: reactions(content: THUMBS_UP) { totalCount } } }
  } }
}"""


def parse_url(url):
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)
    if not m:
        raise SystemExit(f"Not a GitHub issue URL: {url}")
    return m[1], m[2], int(m[3])


def fetch_issue(url):
    owner, name, number = parse_url(url)
    r = requests.post(demand.GRAPHQL, timeout=60,
                      json={"query": ISSUE_QUERY, "variables": {"owner": owner, "name": name, "number": number}},
                      headers={"Authorization": f"Bearer {demand.token()}", "User-Agent": "uncharted"})
    issue = ((r.json().get("data") or {}).get("repository") or {}).get("issue") if r.status_code == 200 else None
    if not issue:
        raise SystemExit(f"Could not read {url}: HTTP {r.status_code}")
    return {"title": issue["title"], "url": issue["url"], "body": issue["bodyText"] or "",
            "thumbs_up": issue["up"]["totalCount"],
            "comments": [{"body": c["bodyText"] or "", "created_at": c["createdAt"], "thumbs_up": c["up"]["totalCount"]}
                         for c in issue["comments"]["nodes"]]}


def pick_comments(comments, top=6, latest=4, chars=1200):
    """The most upvoted comments, then the most recent ones not already picked, as prompt lines."""
    order = sorted(range(len(comments)), key=lambda i: -comments[i]["thumbs_up"])[:top]
    order += [i for i in range(max(0, len(comments) - latest), len(comments)) if i not in order]
    lines = []
    for i in order:
        c = comments[i]
        text = re.sub(r"\s+", " ", c["body"]).strip()[:chars]
        if text:
            lines.append(f"- [{c['created_at'][:10]}, +{c['thumbs_up']}] {text}")
    return lines


def build_prompt(issue, body_chars=5000):
    return PROMPT.format(title=issue["title"], url=issue["url"], thumbs_up=issue["thumbs_up"],
                         body=issue["body"].strip()[:body_chars],
                         comments="\n".join(pick_comments(issue["comments"])) or "(no comments)")


def write_spec(reg, issue, model, rpm=8, max_tokens=16000):
    """Spec one request into the registry. Returns (slug, status)."""
    spec = llm.parse_json(llm.chat(build_prompt(issue), model, rpm, max_tokens=max_tokens))
    slug = reg.unique_slug(slugify(spec.get("slug") or spec.get("title", "")))
    status = "spec" if spec.get("feasible", True) else "infeasible"
    reg.update(slug, status=status, arm="demand", atoms=[issue["url"]], request=issue["title"],
               request_url=issue["url"], request_thumbs_up=issue["thumbs_up"],
               spec=spec, spec_model=model, spec_prompt="demand")
    return slug, status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urls", nargs="+", help="GitHub issue URLs")
    p.add_argument("--out", default="factory_out/demand")
    p.add_argument("--model", default=DEFAULT_SPEC_MODEL, help="Spec writer (about $0.05 per spec with Sonnet 5.5)")
    p.add_argument("--reasoning", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--rpm", type=int, default=8)
    args = p.parse_args(argv)

    reasoning = args.reasoning
    if reasoning == "auto":
        reasoning = "on" if re.search(r"claude-(sonnet|opus)", args.model) else "off"
    os.environ["UNCHARTED_REASONING"] = reasoning

    reg = Registry(args.out)
    for url in args.urls:
        if reg.has_atoms([url]):
            print(f"  already specced: {url}")
            continue
        slug, status = write_spec(reg, fetch_issue(url), args.model, args.rpm)
        sp = reg.ideas[slug]["spec"]
        print(f"  {status}: {slug}: {sp.get('pitch') or sp.get('reason') or sp.get('why_unusual', '')}")
    print(f"Registry: {reg.path}")


if __name__ == "__main__":
    main()
