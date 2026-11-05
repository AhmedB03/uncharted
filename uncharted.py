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

Steps 2 and 3 reuse the original Alien training and sampling code unchanged;
see scripts/train_and_sample.sh.
"""

import importlib
import sys

COMMANDS = {
    "collect": "uncharted.collect",
    "atomize": "uncharted.atomize",
    "cluster": "uncharted.cluster",
    "build-db": "uncharted.build_db",
    "show": "uncharted.show",
    "spec": "uncharted.factory.spec",
    "build": "uncharted.factory.build",
    "publish": "uncharted.factory.publish",
    "gallery": "uncharted.factory.gallery",
    "benchmark": "uncharted.factory.benchmark",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)
    importlib.import_module(COMMANDS[sys.argv[1]]).main(sys.argv[2:])


if __name__ == "__main__":
    main()
