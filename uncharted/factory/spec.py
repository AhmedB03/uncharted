"""Turn sampled alien combinations into small, buildable project specs.

    python uncharted.py spec --results results/alien/k3/beta_0.7/results.json --n 20
"""

import argparse
import json
import re
from pathlib import Path

from uncharted import llm
from uncharted.factory.registry import Registry
from uncharted.show import load_names

PROMPT = """You invent small software projects from unusual combinations of techniques.

A model trained on 2,000+ GitHub repos found that these techniques fit together well,
but that no existing developer community combines them. Your job: design ONE small,
genuinely useful project that uses ALL of them together, so each technique does real work.

TECHNIQUES (with repos where each normally shows up):
{atoms}

Hard constraints (the project is built and tested automatically):
- Python 3.11, standard library ONLY. No pip packages besides pytest for tests.
- No network access, no GUI, no API keys, deterministic behavior.
- Under about 500 lines of code. A CLI tool or an importable library.
- If a technique normally needs heavy infrastructure (a GPU, Kubernetes, a database
  server), implement a faithful miniature of its core idea in pure Python.
- It must do something a person would actually want, not a demo of the techniques.

If no sensible project can use all of them, set "feasible" to false and explain why.

Return only JSON:
{{
  "feasible": true,
  "slug": "short-kebab-case-name",
  "title": "Readable Name",
  "pitch": "one sentence: what it does and who it is for",
  "why_unusual": "one sentence: why nobody would normally build this",
  "kind": "cli",
  "features": ["3 to 6 concrete features"],
  "usage": "example command or code a user would run",
  "tests": ["5 to 10 specific behaviors the test suite must check"]
}}

Each entry in "tests" must be a concrete input -> expected output check a unit test can assert
exactly (e.g. "merging {{a:1}} and {{a:2}} gives {{a:3}}"). Never list performance, "handles large
inputs", real-time updates, logging, "no network access", or "uses only the standard library":
those cannot be asserted."""


PROMPT_V2 = """You design software projects from combinations of techniques that no existing project combines.

These three techniques were checked against ~3,000 popular GitHub repos: none of them uses all three.
Design ONE project where the three genuinely depend on each other, so the result can do something
no existing tool does.

TECHNIQUES (with repos where each normally shows up):
{atoms}

Work in this order:
1. Name the closest existing tool or library to what you are about to build.
2. Decide what your project can do that that tool cannot. That capability is the project's point,
   and it must come from combining the three techniques, not from any one of them alone.
3. Design the smallest version that really delivers that capability.

Realness rules (a reviewer will read the code):
- Every technique must be implemented for real. If the full technique is out of reach in pure
  Python, pick one real, central piece of it and implement that piece properly (e.g. for QUIC:
  its packet numbering, ack ranges and loss detection over in-memory datagrams; for TLS: actual
  authenticated encryption built from hashlib/hmac). Never a renamed dict, counter, sleep or
  print standing in for the technique. Never call anything "simulated" or "mock".
- If no honest version fits these limits, set "feasible" to false and explain why.

Hard constraints (the project is built and tested automatically):
- Python 3.11, standard library ONLY. No pip packages besides pytest for tests.
- No network access, no GUI, no API keys, deterministic behavior.
- About 400 to 900 lines of code. A CLI tool or an importable library.

Return only JSON:
{{
  "feasible": true,
  "closest_existing": "the tool from step 1",
  "slug": "short-kebab-case-name",
  "title": "Readable Name",
  "pitch": "one sentence: what it does that nothing else does, and who it is for",
  "why_unusual": "one sentence: why nobody would normally build this",
  "kind": "cli",
  "features": ["3 to 6 concrete features, each naming the real mechanism it uses"],
  "usage": "example command or code a user would run",
  "tests": ["6 to 12 specific behaviors the test suite must check"]
}}

Each entry in "tests" must be a concrete input -> expected output check a unit test can assert
exactly. At least two tests must exercise the techniques working together. Never list performance,
"handles large inputs", real-time updates, logging, "no network access", or "uses only the
standard library": those cannot be asserted."""

PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2}


def slugify(s):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:40].strip("-") or "alien-project"


def write_spec(reg, cluster_ids, names, repos, model, rpm, prompt_version="v1", max_tokens=6000, **extra):
    """Spec one combination into the registry. Returns (slug, status) or (None, error message)."""
    atoms = [names.get(c, f"cluster {c}") for c in cluster_ids]
    listing = "\n".join(
        f"- {a}   (e.g. {', '.join(sorted(repos[c])[:3]) or 'n/a'})" for a, c in zip(atoms, cluster_ids))
    prompt = PROMPTS[prompt_version].format(atoms=listing)
    try:
        spec = llm.parse_json(llm.chat(prompt, model, rpm, max_tokens=max_tokens))
    except ValueError as e:  # unparseable reply: ask once more with the error attached
        try:
            spec = llm.parse_json(llm.chat(prompt + f"\n\nYour previous reply was not valid JSON ({e}). Reply "
                                           "with ONLY the JSON object, double-quoted keys, no comments.",
                                           model, rpm, max_tokens=max_tokens))
        except Exception as e2:
            return None, str(e2)
    except Exception as e:
        return None, str(e)
    slug = reg.unique_slug(slugify(spec.get("slug") or spec.get("title", "")))
    status = "spec" if spec.get("feasible", True) else "infeasible"
    reg.update(slug, status=status, atoms=atoms, cluster_ids=list(cluster_ids),
               example_repos={a: sorted(repos[c])[:3] for a, c in zip(atoms, cluster_ids)},
               spec=spec, spec_model=model, spec_prompt=prompt_version, **extra)
    return slug, status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="results.json from the Alien sampler")
    p.add_argument("--clusters", default="data/clusters_80.json")
    p.add_argument("--n", type=int, default=20, help="How many new specs to write")
    p.add_argument("--out", default="factory_out")
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--rpm", type=int, default=8)
    args = p.parse_args(argv)

    reg = Registry(args.out)
    names, repos = load_names(args.clusters)
    seqs = json.loads(Path(args.results).read_text())["top_k_sequences"]

    made = skipped = 0
    for s in seqs:
        if made >= args.n:
            break
        atoms = [names.get(c, f"cluster {c}") for c in s["cluster_ids"]]
        if reg.has_atoms(atoms):
            continue
        slug, status = write_spec(reg, s["cluster_ids"], names, repos, args.model, args.rpm,
                                  rank=s["rank_fused"], coherence_z=round(s["z_coherence"], 3),
                                  unavailability_z=round(s["z_availability"], 3))
        if slug is None:
            print(f"  spec failed for {atoms}: {status}")
        elif status == "infeasible":
            skipped += 1
            sp = reg.ideas[slug]["spec"]
            print(f"  infeasible: {' + '.join(atoms)}  ({sp.get('why_unusual') or sp.get('reason', '')})")
        else:
            made += 1
            print(f"  [{made}/{args.n}] {slug}: {reg.ideas[slug]['spec'].get('pitch', '')}")
    print(f"Done: {made} specs, {skipped} infeasible. Registry: {reg.path}")


if __name__ == "__main__":
    main()
