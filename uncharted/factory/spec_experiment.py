"""Does a stricter spec prompt (or a stronger spec model) keep the novelty that alien combos have?

The big benchmark showed verified-alien combos are judged clearly more novel than common ones
(+0.66), but most of that edge is gone once DeepSeek writes the spec (+0.17). This re-specs the
N verified-alien combos the combo judge rated most novel, once per variant, builds them with the
same builder, judges spec and code blind with the same judge, and compares each idea with its
own original spec and build (paired, so combo difficulty cancels out).

    python uncharted.py spec-experiment --n 20
"""

import argparse
import json
import os
import statistics
from argparse import Namespace
from pathlib import Path

from uncharted.factory import benchmark, judge, spec
from uncharted.factory.registry import Registry
from uncharted.show import load_names

VARIANTS = {  # name -> (spec model, prompt version, max_tokens, reasoning)
    "v2_deepseek": ("openrouter/deepseek/deepseek-v4-flash", "v2", 6000, "off"),
    "v2_sonnet": ("openrouter/anthropic/claude-sonnet-5.5", "v2", 16000, "on"),
}


def load_cache(out, name):
    f = Path(out) / name
    if not f.exists():
        return {}
    return {k.split("::", 1)[1]: v for k, v in json.loads(f.read_text(encoding="utf-8")).items()
            if not v.get("filtered")}


def pick(base, n):
    reg = Registry(base)
    combos = load_cache(base, "judge_combos.json")
    cand = [(s, i) for s, i in reg.ideas.items() if i.get("arm") == "alien" and i["status"] == "built" and s in combos]
    cand.sort(key=lambda si: (-combos[si[0]]["novelty"], -combos[si[0]]["potential"], si[0]))
    return cand[:n]


def sign_test_p(diffs):
    """Two-sided sign test on the non-zero paired differences (exact binomial)."""
    from math import comb
    nz = [d for d in diffs if d]
    if not nz:
        return 1.0
    k, n = sum(d > 0 for d in nz), len(nz)
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="factory_out/benchmark_haiku_big")
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--build-model", default="openrouter/anthropic/claude-haiku-5.5")
    p.add_argument("--max-iters", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--report", default="results/spec_experiment.md")
    args = p.parse_args(argv)

    picked = pick(args.base, args.n)
    names, repos = load_names(args.clusters)
    print(f"{len(picked)} verified-alien ideas picked from {args.base}")

    for v in args.variants:
        model, version, max_tokens, reasoning = VARIANTS[v]
        out = f"factory_out/spec_exp_{v}"
        reg = Registry(out)
        done = {tuple(i["cluster_ids"]) for i in reg.ideas.values()}
        os.environ["UNCHARTED_REASONING"] = reasoning
        for slug, idea in picked:
            if tuple(idea["cluster_ids"]) in done:
                continue
            new, status = spec.write_spec(reg, idea["cluster_ids"], names, repos, model, 60,
                                          prompt_version=version, max_tokens=max_tokens,
                                          arm="alien", base_slug=slug)
            print(f"  [{v}] {slug} -> {new} ({status if new else 'FAILED: ' + status[:100]})", flush=True)
        os.environ["UNCHARTED_REASONING"] = "off"  # builds must not inherit the spec model's setting
        reg = Registry(out)
        queue = [s for s, i in reg.ideas.items() if i["status"] == "spec"]
        benchmark.build_all(reg, queue, Namespace(out=out, model=args.build_model, rpm=60,
                                                  max_iters=args.max_iters, sandbox="auto",
                                                  workers=args.workers))
        judge.main(["--specs", "--out", out, "--workers", "6", "--report", f"results/spec_exp_{v}_specs.md"])
        judge.main(["--out", out, "--workers", "6", "--report", f"results/spec_exp_{v}_code.md"])
        os.environ["UNCHARTED_REASONING"] = "off"

    report(args, picked)


def report(args, picked):
    base_specs, base_code = load_cache(args.base, "judge_specs.json"), load_cache(args.base, "judge.json")
    combos = load_cache(args.base, "judge_combos.json")
    lines = ["# Spec experiment: can the spec stage keep alien novelty?", "",
             f"{len(picked)} verified-alien combos (zero support in ~3,000 repos) that the blind combo judge "
             "rated most novel. Each is compared with its own original v1/DeepSeek spec and build. "
             "Same builder (Haiku 5.5, 4 iterations) and same blind judge (Sonnet 5.5) throughout.", "",
             f"Combo novelty of the picked ideas (before any spec): "
             f"{statistics.mean(combos[s]['novelty'] for s, _ in picked):.2f}", "",
             "| variant | stage | dimension | original | new | paired diff | ideas | sign test p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    builds = []
    for v in args.variants:
        out = f"factory_out/spec_exp_{v}"
        reg = Registry(out)
        by_base = {i["base_slug"]: (s, i) for s, i in reg.ideas.items() if i.get("base_slug")}
        new_specs, new_code = load_cache(out, "judge_specs.json"), load_cache(out, "judge.json")
        built = sum(i["status"] == "built" for _, i in by_base.values())
        infeasible = sum(i["status"] == "infeasible" for _, i in by_base.values())
        builds.append(f"| {v} | {len(by_base)} | {infeasible} | {built} | "
                      f"{statistics.mean([i.get('loc', 0) for _, i in by_base.values() if i['status'] == 'built'] or [0]):.0f} |")
        for stage, old, new, dims in (("spec", base_specs, new_specs, judge.SPEC_DIMENSIONS),
                                      ("code", base_code, new_code, judge.DIMENSIONS)):
            for d in dims:
                pairs = [(old[b][d], new[s][d]) for b, (s, _) in by_base.items() if b in old and s in new]
                if not pairs:
                    continue
                diffs = [n - o for o, n in pairs]
                lines.append(f"| {v} | {stage} | {d} | {statistics.mean(o for o, _ in pairs):.2f} "
                             f"| {statistics.mean(n for _, n in pairs):.2f} | {statistics.mean(diffs):+.2f} "
                             f"| {len(pairs)} | {sign_test_p(diffs)} |")
    lines += ["", "| variant | specced | infeasible | built | mean LOC (built) |", "| --- | --- | --- | --- | --- |"]
    lines += builds
    base_loc = [i.get("loc", 0) for s, i in picked]
    lines += [f"| original v1/deepseek | {len(picked)} | 0 | {len(picked)} | {statistics.mean(base_loc):.0f} |", "",
              "Code rows only cover ideas built in both versions. Many tests on 20 ideas: read effect sizes, "
              "treat p < 0.01 as the bar."]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
