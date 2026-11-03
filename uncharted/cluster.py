"""Embed atoms, cluster them into a shared vocabulary, and write clusters_80.json.

Output matches the original Alien format, so its make-datasets and
make-availability commands consume it unchanged:

    {"summary": {...},
     "clusters": {"0": {"name", "description", "size", "examples"}, ...},
     "atom_assignments": [{"text", "paper_id", "cluster"}, ...]}

Pipeline: embed (BGE or TF-IDF) -> UMAP (or PCA) -> HDBSCAN -> reassign the
most confident noise atoms to their nearest centroid until coverage hits the
target (0.80 by default, as in the paper) -> name each cluster.

    python uncharted.py cluster --repos repos --out data/clusters_80.json
    python uncharted.py cluster --embedder tfidf --namer medoid   # fully offline
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from uncharted.llm import DEFAULT_MODEL

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize


def load_atoms(repos_dir):
    atoms = []
    for d in sorted(Path(repos_dir).iterdir()):
        f = d / "refined_ideas.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text()).get("ratings", []):
            text = r.get("revised_idea") or r.get("idea")
            if text and r.get("quality") != "weak":
                atoms.append({"text": text, "paper_id": d.name})
    return atoms


def embed(texts, embedder, model_name):
    if embedder == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
        k = max(2, min(128, X.shape[1] - 1, X.shape[0] - 1))
        return normalize(TruncatedSVD(k, random_state=0).fit_transform(X))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)


def reduce(emb, n_components, n_neighbors):
    if len(emb) <= n_components + 2:
        return emb
    try:
        from umap import UMAP
        return UMAP(n_neighbors=min(n_neighbors, len(emb) - 1), n_components=n_components,
                    min_dist=0.0, metric="cosine", random_state=42).fit_transform(emb)
    except ImportError:
        from sklearn.decomposition import PCA
        print("  umap-learn not installed, falling back to PCA")
        return PCA(min(n_components, emb.shape[1]), random_state=42).fit_transform(emb)


def reassign_noise(emb, labels, coverage):
    """Assign the noise atoms closest to a centroid first, until coverage is reached."""
    labels = labels.copy()
    ids = sorted(set(labels) - {-1})
    if not ids:
        return labels
    centroids = normalize(np.stack([emb[labels == c].mean(0) for c in ids]))
    noise = np.where(labels == -1)[0]
    need = int(np.ceil(coverage * len(labels))) - int((labels != -1).sum())
    if need <= 0 or len(noise) == 0:
        return labels
    sims = emb[noise] @ centroids.T
    best, conf = sims.argmax(1), sims.max(1)
    for i in np.argsort(-conf)[:need]:
        labels[noise[i]] = ids[best[i]]
    return labels


NAMING_PROMPT = """These phrases all describe the same software technique, pulled from
different GitHub repos. Write one canonical name for it: a noun phrase of 3 to 8 words
that is specific enough to tell it apart from neighboring techniques.

{examples}

Return only JSON: {{"name": "...", "description": "one sentence"}}"""


def name_clusters(emb, labels, atoms, namer, model, n_examples=12):
    clusters = {}
    for c in sorted(set(labels) - {-1}):
        idx = np.where(labels == c)[0]
        centroid = normalize(emb[idx].mean(0, keepdims=True))[0]
        order = idx[np.argsort(-(emb[idx] @ centroid))]
        examples = []
        for i in order:
            if atoms[i]["text"] not in examples:
                examples.append(atoms[i]["text"])
            if len(examples) >= n_examples:
                break
        clusters[str(int(c))] = {
            "name": examples[0],  # medoid by default
            "description": "",
            "size": int(len(idx)),
            "n_repos": len({atoms[i]["paper_id"] for i in idx}),
            "examples": examples,
        }
    if namer == "llm":
        _llm_names(clusters, model)
    return clusters


def _llm_names(clusters, model):
    import asyncio
    import re
    from litellm import acompletion

    sem = asyncio.Semaphore(8)

    async def one(info):
        async with sem:
            prompt = NAMING_PROMPT.format(examples="\n".join(f"- {e}" for e in info["examples"]))
            for attempt in range(3):
                try:
                    r = await acompletion(model=model, messages=[{"role": "user", "content": prompt}],
                                          temperature=0.0, max_tokens=200)
                    data = json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
                    info["name"] = data["name"].strip()
                    info["description"] = data.get("description", "").strip()
                    return
                except Exception:
                    await asyncio.sleep(2 ** attempt)

    async def run():
        await asyncio.gather(*(one(i) for i in clusters.values()))

    asyncio.run(run())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos", default="repos")
    p.add_argument("--out", default="data/clusters_80.json")
    p.add_argument("--embedder", choices=["bge", "tfidf"], default="bge")
    p.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--umap-components", type=int, default=10)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--coverage", type=float, default=0.80)
    p.add_argument("--namer", choices=["llm", "medoid"], default="llm")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)

    atoms = load_atoms(args.repos)
    if len(atoms) < 10:
        raise SystemExit(f"Only {len(atoms)} atoms found; run atomize first.")
    n_repos = len({a["paper_id"] for a in atoms})
    print(f"Loaded {len(atoms)} atoms from {n_repos} repos")

    emb = np.asarray(embed([a["text"] for a in atoms], args.embedder, args.embedding_model), dtype=np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), emb)

    reduced = reduce(emb, args.umap_components, args.umap_neighbors)
    labels = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, copy=True).fit_predict(reduced)
    raw_cov = float((labels != -1).mean())
    labels = reassign_noise(emb, labels, args.coverage)
    clusters = name_clusters(emb, labels, atoms, args.namer, args.model)

    sizes = Counter(int(l) for l in labels if l != -1)
    summary = {
        "n_atoms": len(atoms),
        "n_repos": n_repos,
        "n_clusters": len(clusters),
        "coverage_before_reassign": round(raw_cov, 4),
        "coverage": round(float((labels != -1).mean()), 4),
        "median_cluster_size": float(np.median(list(sizes.values()))) if sizes else 0,
        "clustering_params": {k: v for k, v in vars(args).items() if k not in ("repos", "out")},
    }
    data = {
        "summary": summary,
        "clusters": clusters,
        "atom_assignments": [
            {"text": a["text"], "paper_id": a["paper_id"], "cluster": int(l)}
            for a, l in zip(atoms, labels)
        ],
    }
    out.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "clustering_params"}, indent=2))
    top = sorted(clusters.values(), key=lambda c: -c["n_repos"])[:10]
    print("Most widespread atoms:")
    for c in top:
        print(f"  {c['n_repos']:>5} repos  {c['name']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
