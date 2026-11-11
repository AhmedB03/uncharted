"""Novelty search with stepping stones, after Stanley and Lehman's "Why Greatness Cannot Be Planned".

`spec` ranks combinations toward a target (coherent, unbuilt, later "useful") and every idea starts
from nothing. This command has no target. It keeps an archive, and each step:

  stones      every project in the archive that built is a stepping stone
  children    a stone plus ONE technique it does not have yet, extending the stone's real code
  criterion   the only bar is minimal: the new technique goes with at least one of the stone's
              techniques more often than chance in the corpus (pair PMI > 0), it can be done in
              the standard library, and the child then has to build
  selection   children are picked for novelty alone: the mean distance from their technique set
              to the k nearest sets already in the archive, failures included, so ground that
              was tried is not tried again. No stone gets more than `--per-stone` children per
              step, since nobody can tell in advance which stone leads somewhere.

    python uncharted.py explore --seed factory_out/fidelity_ab_new factory_out/spec_exp_v2_sonnet --dry-run
    python uncharted.py explore --n 5                     # spec 5 children
    python uncharted.py build --out factory_out/explore   # the ones that build become stones
    python uncharted.py explore --n 5                     # next step, and so on
    python uncharted.py explore --tree                    # who descends from whom
"""

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from uncharted import llm
from uncharted.factory.registry import Registry
from uncharted.factory.spec import DEFAULT_SPEC_MODEL, slugify
from uncharted.rank import pair_stats, pmi
from uncharted.show import load_names, load_overrides

STONE = ("built", "published")

PROMPT = """You extend a working software project by one technique, to see what becomes possible.

THE PROJECT AS IT IS (built, tests passing):
{parent}

TECHNIQUE TO ADD (it is not in the project yet):
- {technique}   (e.g. {examples})

Design the next version: the existing project plus this technique. Do not ask whether the result
is useful or marketable, and do not aim it at a goal. Ask what the project can do once the
technique is in it that it could not do before, and pick the most interesting such thing, the one
that would make somebody say "I did not know you could do that".

Rules:
- It stands on the existing project: its core mechanisms stay and keep doing real work. You may
  drop a feature that the new direction does not need, to keep the size down.
- The new technique must be implemented for real and be load-bearing in the new capability. Never
  a renamed dict, counter, sleep or print standing in for it. Never "simulated", "mock" or "toy".
  If the full technique is out of reach in pure Python, implement one real, central piece of it.
- If adding it honestly leads nowhere new, set "feasible" to false and say why.

Hard constraints (the project is built and tested automatically):
- Python 3.11, standard library ONLY. No pip packages besides pytest for tests.
- No network access, no GUI, no API keys, deterministic behavior.
- About 600 to 1200 lines of code in total. A CLI tool or an importable library.

Return only JSON:
{{
  "feasible": true,
  "slug": "short-kebab-case-name, different from the existing project's",
  "title": "Readable Name",
  "pitch": "one sentence: what it does",
  "new_capability": "one sentence: what it can do that the existing project could not",
  "opens": "one sentence: what this would make possible to build next",
  "why_unusual": "one sentence: why nobody would normally build this",
  "kind": "cli",
  "features": ["4 to 7 concrete features, each naming the real mechanism it uses"],
  "usage": "example command or code a user would run",
  "tests": ["6 to 12 specific behaviors the test suite must check"]
}}

Each entry in "tests" must be a concrete input -> expected output check a unit test can assert
exactly. At least three tests must exercise the new capability. Never list performance, "handles
large inputs", real-time updates, logging, "no network access", or "uses only the standard
library": those cannot be asserted."""


def distance(a, b):
    """Jaccard distance between two technique sets."""
    return 1 - len(a & b) / len(a | b)


def novelty(techniques, archive, k=5):
    """Mean distance to the k nearest technique sets in the archive (1.0 for an empty archive)."""
    nearest = sorted(distance(techniques, other) for other in archive)[:k]
    return sum(nearest) / len(nearest) if nearest else 1.0


def seen_sets(reg):
    return [frozenset(i["cluster_ids"]) for i in reg.ideas.values() if i.get("cluster_ids")]


def stones(reg):
    return {s: i for s, i in reg.ideas.items() if i.get("status") in STONE and i.get("cluster_ids")}


def candidates(reg, support, exclude=(), k=5, max_techniques=5, min_repos=40, min_pmi=0.0):
    """Every (stone, new technique) that meets the minimal criterion, most novel first."""
    freq, pairs, n = pair_stats(support["repo_clusters"])
    usable = sorted(c for c in freq if freq[c] >= min_repos and c not in set(exclude))
    archive = seen_sets(reg)
    seen = set(archive)
    used = Counter(c for s in archive for c in s)
    kids = Counter(i.get("parent") for i in reg.ideas.values())
    out = []
    for slug, idea in stones(reg).items():
        have = frozenset(idea["cluster_ids"])
        if len(have) >= max_techniques:
            continue
        for t in usable:
            child = have | {t}
            if t in have or child in seen:
                continue
            link = max(pmi(t, h, freq, pairs, n) for h in have)
            if link <= min_pmi:
                continue
            out.append({"parent": slug, "add": t, "cluster_ids": sorted(child), "link_pmi": round(link, 3),
                        "novelty": round(novelty(child, archive, k), 4)})
    # ties (there are many: the distance is coarse) go to the stone with fewer children, then to the
    # technique the archive has used least
    out.sort(key=lambda c: (-c["novelty"], kids[c["parent"]], used[c["add"]], c["parent"], c["add"]))
    return out


def pick(cands, n, per_stone=1, rng=None):
    """Top n by novelty, at most per_stone children per stone and one use of each new technique."""
    if rng:  # shuffle within equal novelty so repeated steps do not always favour the same stones
        cands = sorted(cands, key=lambda c: (-c["novelty"], rng.random()))
    stone_n, added, out = Counter(), set(), []
    for c in cands:
        if len(out) >= n:
            break
        if stone_n[c["parent"]] >= per_stone or c["add"] in added:
            continue
        stone_n[c["parent"]] += 1
        added.add(c["add"])
        out.append(c)
    return out


def seed(reg, sources):
    """Copy built projects from other factory registries into the archive as generation 0."""
    n = 0
    for src in sources:
        other = Registry(src)
        for slug, idea in stones(other).items():
            if frozenset(idea["cluster_ids"]) in set(seen_sets(reg)) or not other.project_dir(slug).is_dir():
                continue
            new = reg.unique_slug(slug)
            shutil.copytree(other.project_dir(slug), reg.project_dir(new), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", ".git", ".pytest_cache"))
            kept = {k: v for k, v in idea.items() if k != "slug"}
            reg.update(new, **{**kept, "status": "built", "arm": "explore", "generation": 0, "parent": None,
                               "seeded_from": str(src)})
            n += 1
    return n


def describe(idea, names):
    sp = idea["spec"]
    lines = [f"{sp.get('title', idea['slug'])}: {sp.get('pitch', '')}",
             "Techniques in it: " + "; ".join(names.get(c, f"cluster {c}") for c in idea["cluster_ids"]),
             "Features:"] + [f"- {f}" for f in sp.get("features", [])]
    return "\n".join(lines + [f"Usage: {sp.get('usage', '')}"])


def write_spec(reg, cand, names, repos, model, rpm=8, max_tokens=16000):
    """Spec one child into the archive. Returns (slug, status)."""
    parent = reg.ideas[cand["parent"]]
    prompt = PROMPT.format(parent=describe(parent, names), technique=names.get(cand["add"], f"cluster {cand['add']}"),
                           examples=", ".join(sorted(repos[cand["add"]])[:3]) or "n/a")
    spec = llm.parse_json(llm.chat(prompt, model, rpm, max_tokens=max_tokens))
    slug = reg.unique_slug(slugify(spec.get("slug") or spec.get("title", "")))
    reg.update(slug, status="spec" if spec.get("feasible", True) else "infeasible", arm="explore",
               atoms=[names.get(c, f"cluster {c}") for c in cand["cluster_ids"]], cluster_ids=cand["cluster_ids"],
               parent=cand["parent"], added=cand["add"], generation=parent.get("generation", 0) + 1,
               novelty=cand["novelty"], link_pmi=cand["link_pmi"], spec=spec, spec_model=model, spec_prompt="explore")
    return slug, reg.ideas[slug]["status"]


def tree(reg, names):
    """The archive as an indented family tree, one line per idea."""
    children = {}
    for slug, idea in reg.ideas.items():
        children.setdefault(idea.get("parent"), []).append(slug)
    lines = []

    def walk(slug, depth):
        idea = reg.ideas[slug]
        added = f"  + {names.get(idea['added'], idea['added'])}" if idea.get("added") is not None else ""
        lines.append(f"{'    ' * depth}{slug} [{idea.get('status')}]{added}")
        for child in children.get(slug, []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="factory_out/explore", help="The archive (a factory registry)")
    p.add_argument("--seed", nargs="*", default=[], help="Factory registries whose built projects become the first stones")
    p.add_argument("--n", type=int, default=5, help="Children to spec this step")
    p.add_argument("--per-stone", type=int, default=1, help="Most children one stone gets per step")
    p.add_argument("--k", type=int, default=5, help="Nearest neighbours in the novelty score")
    p.add_argument("--max-techniques", type=int, default=5, help="Stones with this many techniques are not extended")
    p.add_argument("--min-repos", type=int, default=40, help="Leave out techniques seen in fewer corpus repos")
    p.add_argument("--support", default="data/support_big.json")
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--dry-run", action="store_true", help="Show the picks without calling a model")
    p.add_argument("--tree", action="store_true", help="Print the archive's family tree and stop")
    p.add_argument("--rng", type=int, default=None, help="Seed for breaking novelty ties at random")
    p.add_argument("--model", default=DEFAULT_SPEC_MODEL, help="Spec writer (about $0.05 per child with Sonnet 5.5)")
    p.add_argument("--reasoning", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--rpm", type=int, default=8)
    args = p.parse_args(argv)

    reg = Registry(args.out)
    names, repos = load_names(args.clusters)
    if args.seed:
        print(f"Seeded {seed(reg, args.seed)} stones into {reg.root}")
    if args.tree:
        print("\n".join(tree(reg, names)))
        return
    if not stones(reg):
        raise SystemExit("The archive has no built project to stand on: pass --seed <factory registry>.")

    support = json.loads(Path(args.support).read_text())
    exclude = [c for c, o in load_overrides(args.clusters).items() if o.get("stdlib") is False]
    cands = candidates(reg, support, exclude, args.k, args.max_techniques, args.min_repos)
    picks = pick(cands, args.n, args.per_stone, random.Random(args.rng) if args.rng is not None else None)
    gens = Counter(i.get("generation", 0) for i in stones(reg).values())
    print(f"Archive: {len(stones(reg))} stones of {len(reg.ideas)} ideas (by generation: {dict(sorted(gens.items()))}); "
          f"{len(cands)} possible children")

    reasoning = args.reasoning
    if reasoning == "auto":
        reasoning = "on" if re.search(r"claude-(sonnet|opus)", args.model) else "off"
    os.environ["UNCHARTED_REASONING"] = reasoning
    for c in picks:
        line = f"{c['parent']} + {names.get(c['add'], c['add'])}  (novelty {c['novelty']:.2f}, link PMI {c['link_pmi']:+.2f})"
        if args.dry_run:
            print(f"  would spec: {line}")
            continue
        try:
            slug, status = write_spec(reg, c, names, repos, args.model, args.rpm)
        except Exception as e:
            print(f"  spec failed: {line}: {e}")
            continue
        sp = reg.ideas[slug]["spec"]
        print(f"  {status}: {slug} <- {line}\n      {sp.get('new_capability') or sp.get('reason') or sp.get('pitch', '')}")
    if not args.dry_run:
        print(f"Next: python uncharted.py build --out {args.out}   then run explore again.")


if __name__ == "__main__":
    main()
