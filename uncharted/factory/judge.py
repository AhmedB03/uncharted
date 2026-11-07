"""Blind quality judge: are the alien builds any good, or just buildable?

Every built project from one or more benchmark registries is shown to a judge model with
nothing that reveals its arm: no arm label, no atoms, no "why unusual" pitch, no repo
support. The judge sees the title, pitch, features, usage and the source, and scores it.
Scores are cached per (registry, slug, judge model), so reruns only judge what is new.

    python uncharted.py judge --out factory_out/benchmark_haiku_n100 \
        --model openrouter/anthropic/claude-sonnet-5.5 --report results/judge_haiku_n100.md
"""

import argparse
import json
import os
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from math import erf, sqrt
from pathlib import Path

from uncharted import llm
from uncharted.factory.build import snapshot
from uncharted.factory.registry import Registry

DIMENSIONS = ("novelty", "usefulness", "substance", "fidelity", "integration")

SYSTEM = "You are a strict, experienced open source reviewer. You score honestly and use the whole scale."

PROMPT = """Review this small Python project. Score each dimension from 1 to 5 using the anchors.

novelty      1 = a well known tool already does exactly this; 3 = a recognisable idea with a twist;
             5 = you know of nothing that does this and the combination makes sense
usefulness   1 = nobody would run this; 3 = handy for a niche or for learning; 5 = many people would want it
substance    1 = a toy or stubs; 3 = a small but real implementation; 5 = a solid, non-trivial core
fidelity     1 = the claimed features are faked, simulated or missing; 3 = some are real, some are thin;
             5 = every claimed feature is really implemented
integration  1 = unrelated parts bolted side by side; 3 = parts interact in one place;
             5 = the parts genuinely depend on each other to make the whole work

Judge the code, not the pitch. Calling something "QUIC" or "TLS" while writing a dictionary is fidelity 1.

## Project
Title: {title}
Pitch: {pitch}
Kind: {kind}
Claimed features:
{features}
Usage:
{usage}

## Source
{source}

Reply with only a JSON object:
{{"closest_existing": "<the most similar real project you know, or none>",
  "novelty": n, "usefulness": n, "substance": n, "fidelity": n, "integration": n,
  "verdict": "<one sentence>"}}"""


SPEC_DIMENSIONS = ("novelty", "usefulness", "ambition", "integration", "realness")

SPEC_PROMPT = """Review this plan for a small Python project. No code exists yet: judge the plan itself.
Score each dimension from 1 to 5 using the anchors.

novelty      1 = a well known tool already does exactly this; 3 = a recognisable idea with a twist;
             5 = you know of nothing that does this and the combination makes sense
usefulness   1 = nobody would run this; 3 = handy for a niche or for learning; 5 = many people would want it
ambition     1 = a toy any tutorial covers; 3 = a small but real tool; 5 = a non-trivial core worth building
integration  1 = unrelated parts bolted side by side; 3 = parts interact in one place;
             5 = the parts genuinely depend on each other to make the whole work
realness     1 = the plan itself says its key features are simulated, mocked or stand-ins;
             3 = some key features are real, some are simulated; 5 = every feature is meant to be real

## Plan
Title: {title}
Pitch: {pitch}
Kind: {kind}
Features:
{features}
Usage:
{usage}
Tests planned:
{tests}

Reply with only a JSON object:
{{"closest_existing": "<the most similar real project you know, or none>",
  "novelty": n, "usefulness": n, "ambition": n, "integration": n, "realness": n,
  "verdict": "<one sentence>"}}"""


def prompt_for(idea, root, specs=False):
    s = idea["spec"]
    fields = dict(title=s.get("title", ""), pitch=s.get("pitch", ""), kind=s.get("kind", ""),
                  features="\n".join(f"- {f}" for f in s.get("features", [])), usage=s.get("usage", ""))
    if specs:
        return SPEC_PROMPT.format(**fields, tests="\n".join(f"- {t}" for t in s.get("tests", [])))
    return PROMPT.format(**fields, source=snapshot(root, limit=30000))


def judge_one(idea, root, args):
    dims = SPEC_DIMENSIONS if args.specs else DIMENSIONS
    last = None
    for _ in range(3):
        reply = llm.chat(prompt_for(idea, root, args.specs), model=args.model, rpm=args.rpm, max_tokens=8000,
                         temperature=0.0, system=SYSTEM)
        try:
            out = llm.parse_json(reply)
            scores = {d: int(out[d]) for d in dims}
            if all(1 <= v <= 5 for v in scores.values()):
                return {**scores, "closest_existing": str(out.get("closest_existing", "")),
                        "verdict": str(out.get("verdict", ""))}
            last = f"score out of range: {scores}"
        except Exception as e:  # malformed JSON, missing key
            last = e
    raise RuntimeError(f"judge reply unusable: {last}")


def mann_whitney_p(x, y):
    """Two-sided Mann-Whitney U p value, normal approximation with tie correction."""
    n1, n2 = len(x), len(y)
    if not n1 or not n2:
        return None
    pooled = sorted((v, g) for g, vals in ((0, x), (1, y)) for v in vals)
    ranks, i = [0.0] * len(pooled), 0
    ties = 0.0
    while i < len(pooled):
        j = i
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        ties += (j - i) ** 3 - (j - i)
        i = j
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u - n1 * n2 / 2) / sqrt(var)
    return round(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))), 4)


def report(rows, model, out_md, dims=DIMENSIONS, specs=False):
    arms = {a: [r for r in rows if r["arm"] == a] for a in ("common", "alien")}
    what = "specs (plan only, no code; every attempted idea)" if specs else "built projects (code)"
    lines = [f"# Blind quality judge: common vs alien, {what}", "", f"Judge: {model}", "",
             "The judge never saw the arm, the atoms or the spec's 'why unusual' text. Scores 1-5.", "",
             "| dimension | common mean | alien mean | alien - common | Mann-Whitney p (two-sided) |",
             "| --- | --- | --- | --- | --- |"]
    summary = {}
    for d in dims + ("total",):
        c = [r[d] for r in arms["common"]]
        a = [r[d] for r in arms["alien"]]
        if not c or not a:
            continue
        p = mann_whitney_p(a, c)
        summary[d] = {"common": round(statistics.mean(c), 2), "alien": round(statistics.mean(a), 2), "p": p}
        lines.append(f"| {d} | {summary[d]['common']} | {summary[d]['alien']} "
                     f"| {summary[d]['alien'] - summary[d]['common']:+.2f} | {p} |")
    lines += ["", f"Judged: common {len(arms['common'])}, alien {len(arms['alien'])} .",
              "Five tests on the same projects: treat p < 0.01 as the bar, not 0.05.", "",
              "## Projects", "", "| arm | slug | " + " | ".join(dims) + " | closest existing | verdict |",
              "| --- | --- | " + " | ".join("---" for _ in dims) + " | --- | --- |"]
    for r in sorted(rows, key=lambda r: (r["arm"], -r["total"], r["slug"])):
        clean = lambda t: t.replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['arm']} | {r['slug']} | " + " | ".join(str(r[d]) for d in dims)
                     + f" | {clean(r['closest_existing'])} | {clean(r['verdict'])} |")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(out_md).with_suffix(".json").write_text(json.dumps({"judge": model, "summary": summary,
                                                             "projects": rows}, indent=2), encoding="utf-8")
    print("\n".join(lines[:8 + len(summary) + 2]))
    print(f"Report: {out_md}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", nargs="+", default=["factory_out/benchmark"], help="Benchmark registry dir(s)")
    p.add_argument("--model", default="openrouter/anthropic/claude-sonnet-5.5")
    p.add_argument("--rpm", type=int, default=60)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Judge at most this many new projects (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default="results/judge.md")
    p.add_argument("--reasoning", default="on", choices=["on", "off"],
                   help="Sonnet 5.5 on OpenRouter refuses requests with reasoning off; thinking also helps judging")
    p.add_argument("--specs", action="store_true",
                   help="Judge the plan only (no code), for every attempted idea including failed builds")
    args = p.parse_args(argv)
    dims = SPEC_DIMENSIONS if args.specs else DIMENSIONS
    os.environ["UNCHARTED_REASONING"] = args.reasoning

    todo, rows, lock = [], [], threading.Lock()
    caches = {}
    for out in args.out:
        reg = Registry(out)
        cache_path = Path(out) / ("judge_specs.json" if args.specs else "judge.json")
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        caches[out] = (cache_path, cache)
        for slug, idea in reg.ideas.items():
            wanted = ("built", "failed") if args.specs else ("built",)
            if idea.get("status") not in wanted or idea.get("arm") not in ("common", "alien"):
                continue
            key = f"{args.model}::{slug}"
            if key in cache:
                rows.append({"out": out, "slug": slug, "arm": idea["arm"], "status": idea["status"], **cache[key]})
            else:
                todo.append((out, slug, idea, reg.project_dir(slug)))
    # Shuffle so arms interleave and a cut-off run is not biased toward one arm.
    random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} cached, {len(todo)} to judge with {args.model}")

    def one(item):
        out, slug, idea, root = item
        try:
            res = judge_one(idea, root, args)
        except Exception as e:
            print(f"  FAILED {slug}: {str(e)[:150]}", flush=True)
            return
        with lock:
            cache_path, cache = caches[out]
            cache[f"{args.model}::{slug}"] = res
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            rows.append({"out": out, "slug": slug, "arm": idea["arm"], "status": idea["status"], **res})
            print(f"  [{len(rows)}] {slug}: " + " ".join(f"{d[:3]}={res[d]}" for d in dims), flush=True)

    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        list(pool.map(one, todo))
    for r in rows:
        r["total"] = sum(r[d] for d in dims)
    report(rows, args.model, args.report, dims, args.specs)


if __name__ == "__main__":
    main()
