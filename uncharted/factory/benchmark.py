"""Common vs alien: can a coding model build what nobody is positioned to build?

Two arms, identical pipeline (same spec prompt, same build loop, same model):

  alien   top combinations from the Alien sampler (coherent, low availability)
  common  three-technique combinations that some real repo in the corpus already has,
          so by construction they sit in the dense part of GitHub

Everything goes into its own registry (default factory_out/benchmark) with an `arm` field,
so it never mixes with the publishable factory output. Every step is resumable.

    python uncharted.py benchmark --n 8 --model ollama/qwen2.5:7b
    python uncharted.py benchmark --report-only
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

from uncharted import llm
from uncharted.factory import build, spec
from uncharted.factory.registry import Registry
from uncharted.show import load_names


def common_triplets(clusters_path, n, seed=0, exclude=()):
    """Sample n distinct 3-cluster combinations that co-occur inside a single real repo."""
    data = json.loads(Path(clusters_path).read_text())
    by_repo = defaultdict(set)
    for a in data["atom_assignments"]:
        if a["cluster"] != -1:
            by_repo[a["paper_id"]].add(int(a["cluster"]))
    pool = set()
    for clusters in by_repo.values():
        if len(clusters) >= 3:
            pool.update(tuple(sorted(t)) for t in itertools.combinations(sorted(clusters), 3))
    pool -= {tuple(sorted(e)) for e in exclude}
    support = defaultdict(int)  # how many repos already contain each triple
    for clusters in by_repo.values():
        for t in itertools.combinations(sorted(clusters), 3):
            if t in pool:
                support[t] += 1
    # Prefer triples several repos share: the densest, most "already built" end.
    ranked = sorted(pool, key=lambda t: (-support[t], t))
    head = ranked[: max(n * 5, n)]
    random.Random(seed).shuffle(head)
    return [(list(t), support[t]) for t in head[:n]]


def supported_triplets(support_path, n, seed=0, exclude=()):
    """The n*3 triples the most repos of a bigger corpus share (see `uncharted.py support`)."""
    support = json.loads(Path(support_path).read_text())["triple_support"]
    skip = {tuple(sorted(e)) for e in exclude}
    ranked = [(list(map(int, k.split(","))), v) for k, v in support.items()]
    ranked = [(t, v) for t, v in sorted(ranked, key=lambda x: (-x[1], x[0])) if tuple(t) not in skip]
    head = ranked[: n * 3]
    random.Random(seed).shuffle(head)
    return head


def alien_triplets(results_path, n):
    seqs = json.loads(Path(results_path).read_text())["top_k_sequences"]
    return [s for s in seqs if len(s["cluster_ids"]) == 3][:n]


def fill_arm(reg, arm, combos, n, names, repos, args):
    have = [i for i in reg.ideas.values() if i.get("arm") == arm]
    seen = {tuple(sorted(i["cluster_ids"])) for i in have}
    for cluster_ids, extra in combos:
        if len(have) >= n:
            break
        if tuple(sorted(cluster_ids)) in seen:
            continue
        slug, status = spec.write_spec(reg, cluster_ids, names, repos, args.spec_model or args.model, args.rpm, arm=arm, **extra)
        if slug is None:
            print(f"  [{arm}] spec failed for {cluster_ids}: {status[:120]}")
            continue
        seen.add(tuple(sorted(cluster_ids)))
        have.append(reg.ideas[slug])
        print(f"  [{arm} {len(have)}/{n}] {slug} ({status}): {reg.ideas[slug]['spec'].get('pitch', '')[:100]}")


def fisher_one_sided(a, b, c, d):
    """P(common arm does at least this much better) for the 2x2 table [[a, b], [c, d]]
    with rows common/alien and columns built/not built. Exact, no scipy needed."""
    r1, c1, n = a + b, a + c, a + b + c + d
    denom = comb(n, c1)
    return sum(comb(r1, x) * comb(n - r1, c1 - x) for x in range(a, min(r1, c1) + 1)) / denom


def summarize(reg):
    arms = {}
    for arm in ("common", "alien"):
        ideas = [i for i in reg.ideas.values() if i.get("arm") == arm]
        tried = [i for i in ideas if i["status"] in ("built", "failed")]
        built = [i for i in tried if i["status"] == "built"]
        arms[arm] = {
            "specced": len(ideas),
            "infeasible": sum(i["status"] == "infeasible" for i in ideas),
            "attempted": len(tried),
            "built": len(built),
            "build_rate": round(len(built) / len(tried), 3) if tried else None,
            "mean_iters_when_built": round(sum(i["iterations"] for i in built) / len(built), 2) if built else None,
            "mean_tests_when_built": round(sum(i.get("tests_passed", 0) for i in built) / len(built), 1)
            if built else None,
            "first_draft_failures": sum(str(i.get("fail_reason", "")).startswith("first draft") for i in tried),
        }
    c, a = arms["common"], arms["alien"]
    p = None
    if c["attempted"] and a["attempted"]:
        p = round(fisher_one_sided(c["built"], c["attempted"] - c["built"],
                                   a["built"], a["attempted"] - a["built"]), 4)
    return arms, p


def report(reg, out_md):
    arms, p = summarize(reg)
    models = sorted({i.get("build_model") for i in reg.ideas.values() if i.get("build_model")})
    lines = ["# Common vs alien build benchmark", "",
             f"Build model(s): {', '.join(models) or 'n/a'}", "",
             "| arm | specced | infeasible | attempted | built | build rate | iters (built) | tests (built) "
             "| first-draft failures |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, s in arms.items():
        rate = f"{s['build_rate']:.0%}" if s["build_rate"] is not None else "n/a"
        lines.append(f"| {arm} | {s['specced']} | {s['infeasible']} | {s['attempted']} | {s['built']} | {rate} "
                     f"| {s['mean_iters_when_built'] or 'n/a'} | {s['mean_tests_when_built'] or 'n/a'} "
                     f"| {s['first_draft_failures']} |")
    lines += ["", f"One-sided Fisher exact p (common builds more often than alien): {p if p is not None else 'n/a'}",
              "", "With a handful of ideas per arm this is a pilot, not a result: read the table, not the p value.",
              "", "## Ideas", "", "| arm | slug | status | iters | atoms |", "| --- | --- | --- | --- | --- |"]
    for i in sorted(reg.ideas.values(), key=lambda i: (i.get("arm", ""), i["slug"])):
        lines.append(f"| {i.get('arm', '')} | {i['slug']} | {i['status']} | {i.get('iterations', '')} "
                     f"| {' + '.join(i['atoms'])} |")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (Path(out_md).with_suffix(".json")).write_text(json.dumps({"arms": arms, "fisher_p": p}, indent=2))
    print("\n".join(lines[:8 + len(arms)]))
    print(f"Report: {out_md}")
    return arms, p


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/alien/k3/beta_0.7/results.json")
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--out", default="factory_out/benchmark")
    p.add_argument("--n", type=int, default=8, help="Ideas per arm")
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--spec-model", default=None,
                   help="Model that writes specs (default: --model). Keep it fixed when extending a run.")
    p.add_argument("--support", default=None,
                   help="Pick common triples by repo support in a bigger corpus (output of `uncharted.py support`)")
    p.add_argument("--alien-max-support", type=int, default=None,
                   help="Keep only alien triples that at most this many repos of --support already combine")
    p.add_argument("--rpm", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--workers", type=int, default=1, help="Ideas built at once (paid APIs: 4 is fine)")
    p.add_argument("--sandbox", default="auto", choices=["auto", "docker", "local"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default="results/benchmark.md")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)

    reg = Registry(args.out)
    if not args.report_only:
        names, repos = load_names(args.clusters)
        alien = alien_triplets(args.results, args.n * 3)
        if args.alien_max_support is not None:
            if not args.support:
                raise SystemExit("--alien-max-support needs --support")
            support = json.loads(Path(args.support).read_text())["triple_support"]
            alien = [s for s in alien_triplets(args.results, 10 ** 6)
                     if support.get(",".join(map(str, sorted(s["cluster_ids"]))), 0) <= args.alien_max_support]
            print(f"{len(alien)} alien triples with support <= {args.alien_max_support} in {args.support}")
        alien_combos = [(s["cluster_ids"], {"rank": s["rank_fused"], "coherence_z": round(s["z_coherence"], 3),
                                            "unavailability_z": round(s["z_availability"], 3)}) for s in alien]
        taken = [s["cluster_ids"] for s in alien]
        common = (supported_triplets(args.support, args.n, args.seed, exclude=taken) if args.support
                  else common_triplets(args.clusters, args.n * 3, args.seed, exclude=taken))
        common_combos = [(ids, {"repo_support": sup}) for ids, sup in common]
        # Interleave arms so a quota cut-off leaves them balanced.
        for k in range(1, args.n + 1):
            fill_arm(reg, "common", common_combos, k, names, repos, args)
            fill_arm(reg, "alien", alien_combos, k, names, repos, args)
        reg = Registry(args.out)
        queue = [s for s, i in reg.ideas.items() if i["status"] == "spec"]
        # Alternate arms in the build queue for the same reason.
        by_arm = {a: [s for s in queue if reg.ideas[s].get("arm") == a] for a in ("common", "alien")}
        order = [s for pair in itertools.zip_longest(by_arm["common"], by_arm["alien"]) for s in pair if s]
        build_all(reg, order, args)
    report(Registry(args.out), args.report)


def build_all(reg, order, args):
    """Build in parallel worker processes; the time is almost all waiting on model replies."""
    arms = {s: reg.ideas[s]["arm"] for s in order}
    if args.workers <= 1:
        for slug in order:
            print(f"  building [{arms[slug]}] {slug}")
            build.main(["--out", args.out, "--slug", slug, "--model", args.model, "--rpm", str(args.rpm),
                        "--max-iters", str(args.max_iters), "--sandbox", args.sandbox])
        return
    if args.sandbox == "auto" and not build.docker_ready():
        raise SystemExit("Parallel builds need Docker (or pass --sandbox local explicitly).")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    print_lock = threading.Lock()

    def one(slug):
        cmd = [sys.executable, str(root / "uncharted.py"), "build", "--out", str(Path(args.out).resolve()),
               "--slug", slug, "--model", args.model, "--rpm", str(args.rpm),
               "--max-iters", str(args.max_iters), "--sandbox", args.sandbox]
        with print_lock:
            print(f"  start [{arms[slug]}] {slug}", flush=True)
        r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        with print_lock:  # print each build's log as one block so parallel output stays readable
            print(f"  ===== [{arms[slug]}] {slug} (exit {r.returncode}) =====")
            print((r.stdout + r.stderr).rstrip(), flush=True)
        return r.returncode

    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(one, order))
    if any(codes):
        # A non-zero exit is usually the daily-limit SystemExit; finished builds are kept.
        print(f"{sum(1 for c in codes if c)} build process(es) exited with an error; rerun to resume.")


if __name__ == "__main__":
    main()
