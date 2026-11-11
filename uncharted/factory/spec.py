"""Turn sampled alien combinations into small, buildable project specs.

    python uncharted.py spec --results results/alien/k3/beta_0.7/results.json --n 20

Defaults to Sonnet 5.5 with the v2 prompt, and only specs combinations that no repo in the big
corpus (data/support_big.json) already contains.
"""

import argparse
import json
import os
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

# v2 specs build and stay novel, but the judges keep calling them "a learning toy" (usefulness ~2/5
# in 18 of 22 builds). v3 starts from a person and a job they already do, and only then asks how the
# combination makes that job better. v3lib also allows a fixed set of real libraries, so a project
# can use real QUIC or real crypto instead of a pure-Python miniature.
PROMPT_V3 = """You design software projects from combinations of techniques that no existing project combines.

These three techniques were checked against ~3,000 popular GitHub repos: none of them uses all three.

TECHNIQUES (with repos where each normally shows up):
{atoms}

Work in this order, and be honest at each step:
1. Pick a real kind of person and a concrete job they already do every week, where these
   techniques could matter (e.g. "an SRE working out why last night's deploy got slower", "a
   maintainer triaging which of 40 dependency bumps is safe"). Be specific about the person.
2. Say what they use for that job today, and exactly where it falls short for them.
3. Show how the three techniques TOGETHER close that gap, so the combination is the reason the
   tool works, not decoration. Name the closest existing tool and what yours does that it cannot.
4. Design the smallest version that person would actually run on day one, on their own real
   inputs (their files, logs, repos, configs, data exports), producing output they act on.
If no such person exists for this combination, set "feasible" to false and say why. A smaller
tool that someone really needs beats a clever demo of the techniques.

Realness rules (a reviewer will read the code):
- Every technique must be implemented for real. Never a renamed dict, counter, sleep or print
  standing in for it. Never call anything "simulated", "mock" or "toy".
{libs}
Hard constraints (the project is built and tested automatically):
- No network access during tests (local sockets on 127.0.0.1 are fine), no GUI, no API keys.
- About 400 to 1200 lines of code. A CLI tool or an importable library.
- Tests must be deterministic: seed anything random.

Return only JSON:
{{
  "feasible": true,
  "user": "the person from step 1 and the job they do",
  "today": "what they use today and where it falls short",
  "closest_existing": "the closest existing tool",
  "slug": "short-kebab-case-name",
  "title": "Readable Name",
  "pitch": "one sentence: who it is for, and what it lets them do that they cannot do today",
  "why_unusual": "one sentence: why nobody would normally build this",
  "kind": "cli",
  "features": ["3 to 6 concrete features, each naming the real mechanism it uses"],
  "usage": "an example command a user would run on their own inputs",
  "tests": ["6 to 12 specific behaviors the test suite must check"]
}}

Each entry in "tests" must be a concrete input -> expected output check a unit test can assert
exactly. At least two tests must exercise the techniques working together. Never list performance,
"handles large inputs", real-time updates, logging, "no network access", or "uses only the
standard library": those cannot be asserted."""

STDLIB_ONLY = """- Python 3.11 standard library ONLY (plus pytest for tests). If a technique is out of reach in
  pure Python, implement one real, central piece of it properly (e.g. for QUIC: packet numbering,
  ack ranges and loss detection over in-memory datagrams)."""

# Installed in the build sandbox image (runner/Dockerfile) when a v3lib spec is built.
SANDBOX_LIBS = ["numpy", "cryptography", "aioquic", "networkx", "lark", "pillow", "msgpack", "zstandard",
                "pyyaml", "tomli-w"]

WITH_LIBS = ("- Python 3.11, its standard library, and ONLY these installed packages: " + ", ".join(SANDBOX_LIBS)
             + """.
  When one of them already implements a technique properly (e.g. aioquic for QUIC, cryptography for
  TLS-grade crypto), use it rather than re-implementing it, and spend the code on what is new.""")

# The Python prompts above turn every combination into developer plumbing, because a standard
# library CLI is all they are allowed to ask for. `web` asks for something a stranger would open
# and play with: one static page, built by web.WebTarget and tested in headless Chrome. Use it with
# clusters made from a corpus picked by subject (`collect --topics game generative-art music ...`).
PROMPT_WEB = """You design small things that people open in a browser tab and do not want to close: toys,
games, instruments, simulations, generators.

These three techniques come from popular open source games, creative coding, music and
visualization projects. They were checked against about 3,000 such repos: none uses all three.

TECHNIQUES (with repos where each normally shows up):
{atoms}

Work in this order, and be honest at each step:
1. Picture a stranger who opens a link to this page with no instructions. Say what they see in
   the first second, and what they are doing ten seconds later.
2. Say why they are still there after five minutes, and why they send the link to a friend. That
   hook has to come from the three techniques working on each other, not from one of them alone.
3. Name the closest existing thing (a game, a toy, a site) and what yours does that it does not.
4. Design the smallest version that really delivers the hook.
If the combination only makes a tech demo, or one of the techniques cannot be done for real inside
one offline web page, set "feasible" to false and say why. A small thing that is fun beats a
clever tour of the techniques.

Realness rules (a reviewer will read the code and play the page):
- Every technique must be implemented for real, in miniature if need be (a real rollback buffer,
  a real wave function collapse solver, a real FM synthesis voice). Never a renamed counter or a
  random number standing in for it. Never call anything "simulated", "mock" or "demo".

Hard constraints (the page is built and tested automatically in headless Chrome):
- One static web page: index.html plus vanilla JavaScript modules. No framework, library, build
  step, server, network, account or asset file. Graphics are drawn in code (Canvas 2D, WebGL or
  SVG) and sound is synthesized with Web Audio.
- One player on one device. Mouse or touch must be enough to play; keys are a bonus.
- It works the moment it opens: no menu, no setup, no tutorial wall. The first click, drag or
  key press does something visible.
- About 500 to 1200 lines of JavaScript.
- The rules, generation and simulation are plain functions of a seed and the inputs, kept apart
  from the drawing, so they can be tested exactly without a screen.

Return only JSON:
{{
  "feasible": true,
  "first_ten_seconds": "what a stranger sees and does, from step 1",
  "hook": "why they stay and why they share it, from step 2",
  "closest_existing": "the closest existing thing, and what yours adds",
  "slug": "short-kebab-case-name",
  "title": "Readable Name",
  "pitch": "one sentence that would make a stranger click the link",
  "why_unusual": "one sentence: why nobody would normally build this",
  "kind": "web",
  "controls": "how to play: mouse or touch first, then keys",
  "features": ["3 to 6 concrete features, each naming the real mechanism it uses"],
  "tests": ["6 to 12 specific behaviors the test suite must check"]
}}

Each entry in "tests" must be a concrete input -> expected output check on the rules, generation
or simulation that a unit test can assert exactly (e.g. "one step turns the column [[0,1,0],
[0,1,0],[0,1,0]] into the row [[0,0,0],[1,1,1],[0,0,0]]", "seed 42 gives the same level twice and
seed 43 gives a different one"). At least two tests must exercise the techniques working
together. Never list how it looks, sounds or feels, frame rate, "responsive", or anything that
needs a screen or a speaker: the page itself is checked separately."""

PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2,
           "v3": PROMPT_V3.replace("{libs}", STDLIB_ONLY),
           "v3lib": PROMPT_V3.replace("{libs}", WITH_LIBS),
           "web": PROMPT_WEB}

# The spec experiment (results/spec_experiment.md) found the spec model is the lever: Sonnet 5.5 with
# the v2 prompt kept alien novelty into the code (+1.07 code novelty, paired), DeepSeek with v2 did not.
DEFAULT_SPEC_MODEL = os.environ.get("UNCHARTED_SPEC_MODEL", "openrouter/anthropic/claude-sonnet-5.5")
DEFAULT_SUPPORT = "data/support_big.json"


def support_key(cluster_ids):
    return ",".join(map(str, sorted(cluster_ids)))


def load_support(path, explicit):
    """triple -> repo count from `uncharted.py support`, or None (with a warning) if the default is missing."""
    if not Path(path).exists():
        if explicit:
            raise SystemExit(f"--support file not found: {path}")
        print(f"  note: {path} not found, so combinations are not checked against a big corpus. Many sampler "
              "picks already exist in real repos; run `uncharted.py support` to build it.")
        return None
    return json.loads(Path(path).read_text())["triple_support"]


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
    p.add_argument("--model", default=DEFAULT_SPEC_MODEL,
                   help="Spec writer (default Sonnet 5.5 on OpenRouter, about $0.05 per spec; "
                        "set UNCHARTED_SPEC_MODEL or pass e.g. gemini/gemini-3.8-flash for a free tier)")
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="v2")
    p.add_argument("--max-tokens", type=int, default=None, help="Default 16000 for v2, 6000 for v1")
    p.add_argument("--reasoning", choices=["auto", "on", "off"], default="auto",
                   help="auto: on for Claude Sonnet/Opus (OpenRouter refuses them without it), off otherwise")
    p.add_argument("--support", default=None,
                   help=f"Repo support per triple from `uncharted.py support` (default {DEFAULT_SUPPORT} if present)")
    p.add_argument("--max-support", type=int, default=0,
                   help="Skip combinations that more than this many corpus repos already contain (default 0: "
                        "only verified-unbuilt). Use -1 to turn the check off")
    p.add_argument("--rpm", type=int, default=8)
    args = p.parse_args(argv)

    reasoning = args.reasoning
    if reasoning == "auto":
        reasoning = "on" if re.search(r"claude-(sonnet|opus)", args.model) else "off"
    os.environ["UNCHARTED_REASONING"] = reasoning
    max_tokens = args.max_tokens or (6000 if args.prompt == "v1" else 16000)
    support = None if args.max_support < 0 else load_support(args.support or DEFAULT_SUPPORT, bool(args.support))

    reg = Registry(args.out)
    names, repos = load_names(args.clusters)
    seqs = json.loads(Path(args.results).read_text())["top_k_sequences"]

    made = skipped = known = 0
    for s in seqs:
        if made >= args.n:
            break
        atoms = [names.get(c, f"cluster {c}") for c in s["cluster_ids"]]
        if reg.has_atoms(atoms):
            continue
        extra = {}
        if support is not None:
            n_repos = support.get(support_key(s["cluster_ids"]), 0)
            if n_repos > args.max_support:
                known += 1
                continue
            extra["repo_support"] = n_repos
        slug, status = write_spec(reg, s["cluster_ids"], names, repos, args.model, args.rpm,
                                  prompt_version=args.prompt, max_tokens=max_tokens,
                                  rank=s["rank_fused"], coherence_z=round(s["z_coherence"], 3),
                                  unavailability_z=round(s["z_availability"], 3), **extra)
        if slug is None:
            print(f"  spec failed for {atoms}: {status}")
        elif status == "infeasible":
            skipped += 1
            sp = reg.ideas[slug]["spec"]
            print(f"  infeasible: {' + '.join(atoms)}  ({sp.get('why_unusual') or sp.get('reason', '')})")
        else:
            made += 1
            print(f"  [{made}/{args.n}] {slug}: {reg.ideas[slug]['spec'].get('pitch', '')}")
    gate = f", {known} skipped as already built in the corpus" if support is not None else ""
    print(f"Done: {made} specs, {skipped} infeasible{gate}. Registry: {reg.path}")
    if made < args.n and support is not None:
        print("  ran out of unbuilt combinations: re-run the sampler with a larger --top-k")


if __name__ == "__main__":
    main()
