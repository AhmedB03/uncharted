"""Print sampled alien combinations with human readable atom names and example repos.

    python uncharted.py show --results results/alien/k3/beta_0.7/results.json --top 15
    python uncharted.py show --results ... --json results/alien_named.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_names(clusters_path):
    data = json.loads(Path(clusters_path).read_text())
    repos = defaultdict(set)
    for a in data["atom_assignments"]:
        if a["cluster"] != -1:
            repos[a["cluster"]].add(a["paper_id"].replace("__", "/"))
    return {int(k): v["name"] for k, v in data["clusters"].items()}, repos


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--json", default=None, help="Also write the named combinations here")
    args = p.parse_args(argv)

    names, repos = load_names(args.clusters)
    seqs = json.loads(Path(args.results).read_text())["top_k_sequences"][: args.top]
    named = []
    for s in seqs:
        atoms = [names.get(c, f"cluster {c}") for c in s["cluster_ids"]]
        named.append({
            "rank": s["rank_fused"], "atoms": atoms,
            "coherence_z": round(s["z_coherence"], 2), "unavailability_z": round(s["z_availability"], 2),
            "example_repos": {a: sorted(repos[c])[:3] for a, c in zip(atoms, s["cluster_ids"])},
        })
        print(f"#{s['rank_fused']:<3} coherence z={s['z_coherence']:+.2f}  "
              f"unavailability z={s['z_availability']:+.2f}")
        for a, c in zip(atoms, s["cluster_ids"]):
            print(f"      + {a}   (e.g. {', '.join(sorted(repos[c])[:2])})")
    if args.json:
        Path(args.json).write_text(json.dumps(named, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
