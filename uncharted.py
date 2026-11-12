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
    play        Web projects: record a short session and ask a model whether it plays like the spec
    publish     Ship passing projects to GitHub, labeled as AI made (dry run by default)
    gallery     Static page of everything made, ranked by stars
    benchmark   Common vs alien: same model, same loop, compare build success
    support     Count how many repos of a bigger corpus share each technique triple
    rank        Rank every technique triple from those counts (no trained model needed)
    demand      Most upvoted open requests across the corpus repos (what people ask for)
    demand-spec Spec a standalone tool from one of those requests (then `build` as usual)
    explore     Novelty search: extend built projects by one technique, with no target
    spec-experiment  Re-spec top verified-alien ideas with a stricter prompt and compare
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
    "rank": "uncharted.rank",
    "demand": "uncharted.demand",
    "demand-spec": "uncharted.factory.demand_spec",
    "explore": "uncharted.factory.explore",
    "spec": "uncharted.factory.spec",
    "build": "uncharted.factory.build",
    "play": "uncharted.factory.play",
    "publish": "uncharted.factory.publish",
    "gallery": "uncharted.factory.gallery",
    "benchmark": "uncharted.factory.benchmark",
    "judge": "uncharted.factory.judge",
    "spec-experiment": "uncharted.factory.spec_experiment",
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
