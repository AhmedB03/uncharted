"""How many real repos already combine each set of techniques, measured on a bigger corpus.

The training corpus is small (about 300 repos), so almost every three-technique combination
shows up in at most one repo and "common" is not really common. This maps the atoms of a
bigger corpus onto the existing clusters (nearest centroid in the same BGE space, no
re-clustering, no retraining) and counts, for every triple, how many repos contain it.

The similarity cut-off is chosen so the same share of atoms gets a cluster as in the
original clustering (80% by default); agreement with the original labels is reported so
you can see how faithful the mapping is.

    python uncharted.py support --repos repos_big --out data/support_big.json

When the clusters were themselves made from a few thousand repos, there is nothing to map: `--self`
counts on that same corpus with the labels the clustering gave it.

    python uncharted.py support --self --clusters data/fun/clusters_80.json --out data/fun/support.json
"""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from uncharted.cluster import embed, load_atoms


def centroids(clusters_path, emb_path):
    data = json.loads(Path(clusters_path).read_text())
    emb = np.load(emb_path)
    labels = np.array([a["cluster"] for a in data["atom_assignments"]])
    assert len(labels) == len(emb), "embeddings and atom_assignments are out of step"
    ids = sorted(set(labels.tolist()) - {-1})
    cents = normalize(np.stack([emb[labels == c].mean(0) for c in ids]))
    return data, emb, labels, ids, cents


def nearest(emb, ids, cents):
    sims = emb @ cents.T
    return np.array(ids)[sims.argmax(1)], sims.max(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos", default="repos_big")
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--embeddings", default=None, help="Default: the clusters file with .npy")
    p.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--coverage", type=float, default=0.80)
    p.add_argument("--out", default="data/support_big.json")
    p.add_argument("--self", dest="own", action="store_true",
                   help="Count on the corpus the clusters were made from, using its own labels (no embedding)")
    args = p.parse_args(argv)

    if args.own:
        atoms = json.loads(Path(args.clusters).read_text())["atom_assignments"]
        labels, cut, agreement = [a["cluster"] for a in atoms], None, 1.0
    else:
        emb_path = args.embeddings or str(Path(args.clusters).with_suffix(".npy"))
        data, old_emb, old_labels, ids, cents = centroids(args.clusters, emb_path)

        # Faithfulness check on the original atoms: does nearest centroid reproduce their label?
        pred, _ = nearest(old_emb, ids, cents)
        kept = old_labels != -1
        agreement = float((pred[kept] == old_labels[kept]).mean())

        atoms = load_atoms(args.repos)
        print(f"{len(atoms)} atoms from {len({a['paper_id'] for a in atoms})} repos; embedding...")
        emb = embed([a["text"] for a in atoms], "bge", args.embedding_model)
        best, conf = nearest(np.asarray(emb), ids, cents)
        cut = round(float(np.quantile(conf, 1 - args.coverage)), 4)
        labels = np.where(conf >= cut, best, -1)

    by_repo = defaultdict(set)
    for a, c in zip(atoms, labels):
        if c != -1:
            by_repo[a["paper_id"]].add(int(c))
    triples = Counter(t for cs in by_repo.values() for t in itertools.combinations(sorted(cs), 3))
    out = {
        "summary": {"repos": len(by_repo), "atoms": len(atoms), "coverage": args.coverage,
                    "similarity_cutoff": cut,
                    "label_agreement_on_original_atoms": round(agreement, 4),
                    "distinct_triples": len(triples),
                    "triple_support_histogram": dict(sorted(Counter(triples.values()).items()))},
        "repo_clusters": {r: sorted(cs) for r, cs in sorted(by_repo.items())},
        "triple_support": {",".join(map(str, t)): n for t, n in triples.most_common()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(json.dumps({k: v for k, v in s.items() if k != "triple_support_histogram"}, indent=1))
    hist = s["triple_support_histogram"]
    print("triples by support:", {k: hist[k] for k in list(hist)[:6]}, "... max", max(hist) if hist else 0)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
