"""`rank` (corpus-count ranking of every triple) and hand-reviewed cluster names."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uncharted import rank, show  # noqa: E402


def corpus():
    # 1, 2, 3 go together in pairs but never all three; 4 is everywhere (so it is tied to nothing);
    # 1, 2, 5 is already built.
    repos = ([[1, 2]] * 6 + [[1, 3]] * 6 + [[2, 3]] * 6 + [[1, 2, 5]] * 2
             + [[4]] * 40 + [[1, 4], [2, 4], [3, 4], [5, 4]] + [[6]] * 30)
    rc = {f"r{i}": cs for i, cs in enumerate(repos)}
    return {"repo_clusters": rc, "triple_support": {"1,2,5": 2}}


def test_ranks_unbuilt_triple_of_related_pairs_first():
    picks, scored = rank.rank(corpus(), min_repos=1)
    assert picks[0][0] == (1, 2, 3)
    assert picks[0][1] > 0 and picks[0][3] == 0
    assert all(t != (1, 2, 5) for t, *_ in picks)  # already built
    assert all(4 not in t for t, *_ in picks)  # 4 pairs with nothing above chance
    assert scored > len(picks)


def test_filters():
    c = corpus()
    assert any(t == (1, 2, 5) for t, *_ in rank.rank(c, min_repos=1, max_support=-1)[0])
    assert all(3 not in t for t, *_ in rank.rank(c, min_repos=1, exclude=[3], min_pmi=-9)[0])
    assert all(5 not in t for t, *_ in rank.rank(c, min_repos=4, min_pmi=-9)[0])  # 5 is in 3 repos
    capped = rank.rank(c, min_repos=1, min_pmi=-9, max_per_technique=1)[0]
    assert max(sum(1 in t for t, *_ in capped), sum(2 in t for t, *_ in capped)) == 1


def test_cli_output_reads_like_sampler_results(tmp_path):
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps({
        "clusters": {str(i): {"name": f"auto {i}"} for i in range(1, 7)},
        "atom_assignments": [{"text": "x", "paper_id": "a__b", "cluster": 1}]}))
    (tmp_path / "clusters.names.json").write_text(json.dumps({
        "1": {"name": "reviewed one", "stdlib": True}, "3": {"name": "needs a GPU", "stdlib": False}}))
    (tmp_path / "support.json").write_text(json.dumps(corpus()))
    out = tmp_path / "ranked" / "results.json"

    names, repos = show.load_names(clusters)
    assert names[1] == "reviewed one" and names[2] == "auto 2" and repos[1] == {"a/b"}

    args = ["--support", str(tmp_path / "support.json"), "--clusters", str(clusters), "--out", str(out),
            "--min-repos", "1", "--min-pmi", "-9"]
    rank.main(args)
    data = json.loads(out.read_text())
    assert data["config"]["excluded_techniques"] == [3]
    assert all(3 not in s["cluster_ids"] for s in data["top_k_sequences"])
    first = data["top_k_sequences"][0]
    assert first["rank_fused"] == 1 and {"z_coherence", "z_availability", "cluster_ids"} <= set(first)

    rank.main(args + ["--all-techniques"])
    assert json.loads(out.read_text())["top_k_sequences"][0]["cluster_ids"] == [1, 2, 3]


def test_buildable_false_is_left_out_like_stdlib_false(tmp_path):
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps({"clusters": {str(i): {"name": f"auto {i}"} for i in range(1, 7)},
                                    "atom_assignments": []}))
    (tmp_path / "clusters.names.json").write_text(json.dumps({"2": {"name": "native engine", "buildable": False}}))
    (tmp_path / "support.json").write_text(json.dumps(corpus()))
    out = tmp_path / "results.json"
    rank.main(["--support", str(tmp_path / "support.json"), "--clusters", str(clusters), "--out", str(out),
               "--min-repos", "1", "--min-pmi", "-9"])
    assert json.loads(out.read_text())["config"]["excluded_techniques"] == [2]


def test_support_self_counts_on_the_clustered_corpus(tmp_path):
    from uncharted import support
    clusters = tmp_path / "clusters.json"
    rows = [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("b", 3), ("b", -1), ("c", 1), ("c", 2)]
    clusters.write_text(json.dumps({"clusters": {}, "atom_assignments": [
        {"text": "x", "paper_id": r, "cluster": c} for r, c in rows]}))
    out = tmp_path / "support.json"
    support.main(["--self", "--clusters", str(clusters), "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["triple_support"] == {"1,2,3": 2}
    assert data["repo_clusters"] == {"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2]}


def test_bridge_keeps_only_open_triangles():
    # 1-2 and 2-3 go together, 1 and 3 have never met; 4-5-6 is a closed triangle of pairs.
    repos = [[1, 2]] * 5 + [[2, 3]] * 5 + [[4, 5]] * 5 + [[5, 6]] * 5 + [[4, 6]] * 5 + [[7]] * 30
    support = {"repo_clusters": {f"r{i}": cs for i, cs in enumerate(repos)}, "triple_support": {}}
    assert rank.rank(support, min_repos=1)[0][0][0] == (4, 5, 6)
    picks, _ = rank.rank(support, min_repos=1, bridge=True)
    assert [t for t, *_ in picks] == [(1, 2, 3)]
    assert picks[0][1] > 0  # scored on the two links that exist, not on the missing one
