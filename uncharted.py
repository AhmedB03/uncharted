"""Uncharted: the Alien Space of Science, applied to GitHub.

Usage: python uncharted.py <command> [args...]

Step 1 (this repo):
    collect     Pull popular repos, READMEs, manifests, contributors from GitHub
    atomize     Break each repo into technique atoms with an LLM
    cluster     Cluster atoms into a shared vocabulary (clusters_80.json)
    build-db    Write the SQLite metadata DB and vocab.json the Alien code expects
    show        Print sampled combinations with readable atom names and example repos

The alien factory:
    spec        Turn alien combinations into small buildable project specs
    build       Model writes the code, pytest runs, failures go back, repeat
    publish     Ship passing projects to GitHub, labeled as AI made (dry run by default)
    gallery     Static page of everything made, ranked by stars
    benchmark   Common vs alien: same model, same loop, compare build success
    support     Count how many repos of a bigger corpus share each technique triple
    judge       Blind quality scores for benchmark builds (novelty, substance, ...)

Steps 2 and 3 reuse the original Alien training and sampling code unchanged;
see scripts/train_and_sample.sh.
"""

import importlib
import os
import sys
from pathlib import Path


def load_dotenv(path=Path(__file__).with_name(".env")):
    """Read KEY=value lines from .env into the environment; real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

COMMANDS = {
    "collect": "uncharted.collect",
    "atomize": "uncharted.atomize",
    "cluster": "uncharted.cluster",
    "build-db": "uncharted.build_db",
    "show": "uncharted.show",
    "support": "uncharted.support",
    "spec": "uncharted.factory.spec",
    "build": "uncharted.factory.build",
    "publish": "uncharted.factory.publish",
    "gallery": "uncharted.factory.gallery",
    "benchmark": "uncharted.factory.benchmark",
    "judge": "uncharted.factory.judge",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)
    load_dotenv()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles are cp1252; model text often is not
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    importlib.import_module(COMMANDS[sys.argv[1]]).main(sys.argv[2:])


if __name__ == "__main__":
    main()
