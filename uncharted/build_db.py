"""Build the SQLite metadata DB and vocab.json that the original Alien code expects.

Mapping from the paper world to the code world:
    papers        -> repos         (paper_id = owner__name, venue_year = year created)
    authors       -> contributors  (author_id = GitHub user id, never email)
    paper_authors -> who committed to which repo

    python uncharted.py build-db --repos repos --clusters data/clusters_80.json
"""

import argparse
import json
import sqlite3
from pathlib import Path

# Same schema as alien-space-of-science/crawlers/db/schema.py
SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT, keywords TEXT,
    pdf_url TEXT NOT NULL, conference TEXT, venue_year INTEGER NOT NULL, venue_track TEXT,
    openreview_url TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS authors (author_id TEXT PRIMARY KEY, display_name TEXT, email TEXT);
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id TEXT NOT NULL, author_id TEXT NOT NULL, author_position INTEGER,
    PRIMARY KEY (paper_id, author_id));
CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id);
CREATE INDEX IF NOT EXISTS idx_paper_authors_paper ON paper_authors(paper_id);
"""


def build_db(repos_dir, db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    n_repos = n_links = 0
    authors = {}
    for d in sorted(Path(repos_dir).iterdir()):
        f = d / "meta.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        year = int((m.get("created_at") or "2000")[:4])
        conn.execute(
            "INSERT INTO papers (paper_id, title, abstract, keywords, pdf_url, conference, venue_year, venue_track)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (m["id"], m["full_name"], m.get("description"), ",".join(m.get("topics", [])),
             m["html_url"], "github", year, m.get("language")),
        )
        n_repos += 1
        for pos, c in enumerate(m.get("contributors", [])):
            aid = f"gh{c['id']}"
            authors[aid] = c["login"]
            conn.execute("INSERT OR IGNORE INTO paper_authors VALUES (?,?,?)", (m["id"], aid, pos))
            n_links += 1
    conn.executemany("INSERT INTO authors (author_id, display_name) VALUES (?,?)", authors.items())
    conn.commit()
    multi = conn.execute(
        "SELECT COUNT(*) FROM (SELECT author_id FROM paper_authors GROUP BY author_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    conn.close()
    print(f"DB: {n_repos} repos, {len(authors)} contributors, {n_links} links, "
          f"{multi} contributors on 2+ repos -> {db_path}")


def build_vocab(clusters_path, vocab_path, margin=64):
    """The original code maps each cluster to a token from vocab.json, picking the
    lowest ids first, and reserves id 50256 for BOS/EOS. Any id -> string table works."""
    n = len(json.loads(Path(clusters_path).read_text())["clusters"]) + margin
    if n >= 50256:
        raise SystemExit("Too many clusters for the reserved BOS id; raise min-cluster-size.")
    vocab = {f"<atom_{i:05d}>": i for i in range(n)}
    Path(vocab_path).write_text(json.dumps(vocab))
    print(f"Vocab: {n} tokens -> {vocab_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos", default="repos")
    p.add_argument("--db", default="data/uncharted.db")
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--vocab", default="data/vocab.json")
    args = p.parse_args(argv)
    build_db(args.repos, args.db)
    build_vocab(args.clusters, args.vocab)


if __name__ == "__main__":
    main()
