"""Rank every three-technique combination from corpus counts alone, with no trained model.

The Alien sampler scores only the ~850 combinations it happens to generate, and on a few hundred
repos its coherence score is close to a pair co-occurrence count. With `uncharted.py support` on a
bigger corpus, the counts can be used directly and over the whole space:

  coherence   the weakest of the three pair associations, as smoothed PMI:
              log((repos with both + 1) / (repos expected by chance + 1)).
              All three pairs must go together more often than chance for a high score.
  unbuilt     no corpus repo contains all three (`--max-support`).

Techniques marked `"stdlib": false` in `<clusters>.names.json` are left out (see `show.load_overrides`),
each technique appears in at most `--max-per-technique` picks so the list stays varied, and the list
stops where the weakest pair no longer beats chance (`--min-pmi`).

The output has the sampler's results.json shape, so `show`, `spec` and `benchmark` read it as is:

    python uncharted.py rank --out results/ranked/results.json
    python uncharted.py spec --results results/ranked/results.json --n 20
"""

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from uncharted.show import load_overrides


def pair_stats(repo_clusters):
    freq = Counter(c for cs in repo_clusters.values() for c in cs)
    pairs = Counter(p for cs in repo_clusters.values() for p in itertools.combinations(sorted(cs), 2))
    return freq, pairs, len(repo_clusters)


def pmi(a, b, freq, pairs, n):
    a, b = sorted((a, b))
    return math.log((pairs[(a, b)] + 1) / (freq[a] * freq[b] / n + 1))


def rank(support, top=300, max_support=0, min_repos=40, exclude=(), max_per_technique=12, min_pmi=0.0):
    """Best-first list of (triple, weakest pair PMI, mean pair PMI, repo support), and how many were scored."""
    freq, pairs, n = pair_stats(support["repo_clusters"])
    known = support["triple_support"]
    skip = set(exclude)
    ids = sorted(c for c in freq if freq[c] >= min_repos and c not in skip)
    edge = {p: pmi(*p, freq, pairs, n) for p in itertools.combinations(ids, 2)}
    scored = []
    for t in itertools.combinations(ids, 3):
        sup = known.get(",".join(map(str, t)), 0)
        if max_support >= 0 and sup > max_support:
            continue
        e = [edge[p] for p in itertools.combinations(t, 2)]
        scored.append((min(e), sum(e) / 3, t, sup))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    used, out = Counter(), []
    for weakest, mean, t, sup in scored:
        if len(out) >= top or weakest < min_pmi:
            break
        if max_per_technique and any(used[c] >= max_per_technique for c in t):
            continue
        used.update(t)
        out.append((t, weakest, mean, sup))
    return out, len(scored)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--support", default="data/support_big.json")
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--out", default="results/ranked/results.json")
    p.add_argument("--top", type=int, default=300)
    p.add_argument("--max-support", type=int, default=0,
                   help="Keep triples that at most this many repos contain (default 0; -1 keeps all)")
    p.add_argument("--min-repos", type=int, default=40,
                   help="Leave out techniques seen in fewer repos: their pair counts are too small to trust")
    p.add_argument("--max-per-technique", type=int, default=12, help="0 turns the variety cap off")
    p.add_argument("--min-pmi", type=float, default=0.0,
                   help="Stop once the weakest pair is below this (default 0: every pair beats chance)")
    p.add_argument("--all-techniques", action="store_true",
                   help="Keep techniques marked as not buildable in standard library Python")
    args = p.parse_args(argv)

    support = json.loads(Path(args.support).read_text())
    overrides = load_overrides(args.clusters)
    exclude = [] if args.all_techniques else [c for c, o in overrides.items() if o.get("stdlib") is False]
    picks, n_scored = rank(support, args.top, args.max_support, args.min_repos, exclude, args.max_per_technique,
                            args.min_pmi)
    if not picks:
        raise SystemExit("No combination passed the filters; lower --min-repos or raise --max-support.")

    weakest = [w for _, w, _, _ in picks]
    mu = sum(weakest) / len(weakest)
    sd = (sum((w - mu) ** 2 for w in weakest) / len(weakest)) ** 0.5 or 1.0
    seqs = [{"cluster_ids": list(t), "length": 3, "rank_fused": i, "fusion_score": round(w, 4),
             "min_pair_pmi": round(w, 4), "mean_pair_pmi": round(m, 4), "repo_support": sup,
             # the fields `spec` and `benchmark` record; unavailability is not modelled here
             "z_coherence": round((w - mu) / sd, 4), "z_availability": 0.0}
            for i, (t, w, m, sup) in enumerate(picks, 1)]
    out = {"config": {"method": "weakest pair PMI over all triples", "support": args.support,
                      "corpus_repos": len(support["repo_clusters"]), "max_support": args.max_support,
                      "min_repos": args.min_repos, "max_per_technique": args.max_per_technique, "min_pmi": args.min_pmi,
                      "excluded_techniques": sorted(exclude), "n_candidate_sets": n_scored},
           "top_k_sequences": seqs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"Scored {n_scored} triples ({len(exclude)} techniques left out), kept the top {len(seqs)}: "
          f"weakest pair PMI {weakest[0]:.2f} to {weakest[-1]:.2f}. Wrote {args.out}")


if __name__ == "__main__":
    main()
