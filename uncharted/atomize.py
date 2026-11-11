"""Break each repo into technique atoms with an LLM.

Writes repos/{id}/refined_ideas.json in the same format the original Alien
clustering stage reads ({"ratings": [{"idea", "revised_idea", "quality"}]}),
so either our cluster step or theirs can consume it.

    export GEMINI_API_KEY=...           # free from Google AI Studio (or any LiteLLM provider key)
    python uncharted.py atomize --repos repos          # free gemini/gemini-3.8-flash, paced at --rpm
    python uncharted.py atomize --repos repos --mock   # offline, no API calls
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from uncharted.llm import DEFAULT_MODEL, reasoning_off

PROMPT = """You are building a vocabulary of reusable software techniques.

Read this GitHub repository and list the {min_atoms} to {max_atoms} distinct technical
techniques, architectural patterns, algorithms, or capabilities that define how it works.

Rules for each atom:
- A short noun phrase, 3 to 10 words, phrased so it could apply to other projects.
  Good: "CRDT based offline sync", "WebGPU accelerated tensor inference",
  "event sourcing with append only log", "retrieval augmented generation over code".
- Name a concrete technique, not a product. Mention a specific library only when the
  library IS the technique (for example "eBPF kernel tracing").
- Skip generic things every repo has: the language itself, "open source", "unit tests",
  "CLI interface", "Docker support", "documentation", "MIT license".
- No two atoms should say the same thing.
- Mark each atom "core" if the project is built around it, "supporting" otherwise.

Return only JSON: {{"atoms": [{{"atom": "...", "role": "core"}}]}}

REPOSITORY: {full_name}
LANGUAGE: {language}
TOPICS: {topics}
DESCRIPTION: {description}

DEPENDENCY MANIFESTS:
{deps}

README:
{readme}
"""


# For a corpus picked by subject (`collect --topics game generative-art music ...`). The prompt
# above turns a roguelike into "entity component system" and "Lua scripting bindings"; this one
# asks what it is like to play with and what produces that, so combinations of its atoms read as
# things to play rather than engine plumbing.
PLAY_PROMPT = """You are building a vocabulary of the ideas that make software fun to use: game mechanics,
generative and simulation techniques, ways of making pictures and sound, ways of interacting.

Read this GitHub repository and list the {min_atoms} to {max_atoms} distinct ideas that define what it
is like to use and how it produces that.

Rules for each atom:
- A short noun phrase, 3 to 10 words, phrased so it could be reused in a completely different
  project. Good: "wave function collapse level generation", "rhythm timing windows scored against
  a beat map", "falling sand cellular automaton", "FM synthesis voice with envelopes", "deck
  building with permanent death", "force directed graph layout", "replay by re-simulating
  recorded inputs".
- Name the mechanic or the technique, never the product, a genre label or a franchise ("turn
  based tactics on a hex grid", not "a Civilization clone").
- Prefer what a player sees, hears or does, and the algorithm behind it, over engineering
  plumbing. Skip: the language, the engine or framework it uses, build systems, cross platform
  support, plugin and modding systems, asset pipelines, file formats, server infrastructure,
  "open source", tests, documentation, licenses.
- If the repository is only plumbing (a list of links, a build tool, a binding), return fewer
  atoms, or none.
- No two atoms should say the same thing.
- Mark each atom "core" if the project is built around it, "supporting" otherwise.

Return only JSON: {{"atoms": [{{"atom": "...", "role": "core"}}]}}

REPOSITORY: {full_name}
LANGUAGE: {language}
TOPICS: {topics}
DESCRIPTION: {description}

DEPENDENCY MANIFESTS:
{deps}

README:
{readme}
"""

# (prompt, how many characters of dependency manifests it is shown)
PROMPTS = {"tech": (PROMPT, 6000), "play": (PLAY_PROMPT, 1200)}


def build_prompt(d, min_atoms, max_atoms, style="tech"):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    readme = (d / "readme.md").read_text(encoding="utf-8") if (d / "readme.md").exists() else ""
    deps = (d / "deps.txt").read_text(encoding="utf-8") if (d / "deps.txt").exists() else ""
    prompt, deps_chars = PROMPTS[style]
    return prompt.format(
        min_atoms=min_atoms, max_atoms=max_atoms,
        full_name=meta["full_name"], language=meta.get("language") or "unknown",
        topics=", ".join(meta.get("topics", [])) or "none",
        description=meta.get("description") or "none",
        deps=deps[:deps_chars] or "none", readme=readme[:10000] or "none",
    )


def parse_atoms(text):
    """Pull the JSON object out of an LLM reply, tolerating code fences and chatter."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    data = json.loads(m.group(0))
    atoms, seen = [], set()
    for a in data.get("atoms", []):
        text_ = (a.get("atom") or "").strip().rstrip(".")
        key = text_.lower()
        if 2 <= len(text_.split()) <= 14 and key not in seen:
            seen.add(key)
            atoms.append({"atom": text_, "role": a.get("role", "supporting")})
    return atoms


def to_refined(atoms, model):
    return {
        "source": "uncharted",
        "model": model,
        "ratings": [
            {"idea": a["atom"], "revised_idea": a["atom"],
             "quality": "strong" if a["role"] == "core" else "medium"}
            for a in atoms
        ],
    }


# Offline stand in so the pipeline can be tested end to end with no API key.
MOCK_RULES = [
    (r"torch|tensorflow|jax|onnx", "neural network model training"),
    (r"transformers|llama|openai|anthropic|langchain|vllm", "large language model inference"),
    (r"faiss|chroma|qdrant|pgvector|embedding", "vector similarity search"),
    (r"react|vue|svelte|solid-js|preact", "component based reactive UI"),
    (r"next|nuxt|remix|astro", "server side rendered web frontend"),
    (r"express|fastapi|flask|django|actix|axum|gin|fiber|spring", "HTTP API server"),
    (r"grpc|protobuf|tonic", "RPC with protocol buffers"),
    (r"postgres|sqlx|prisma|sqlalchemy|gorm|diesel", "relational database access layer"),
    (r"redis", "in memory key value caching"),
    (r"kafka|nats|rabbitmq|pulsar", "message queue event streaming"),
    (r"tokio|asyncio|async-std|uvloop", "asynchronous event loop concurrency"),
    (r"wasm|wasm-bindgen|webassembly", "WebAssembly compilation target"),
    (r"webgpu|wgpu|vulkan|metal|opengl", "GPU graphics rendering pipeline"),
    (r"cuda|triton", "custom GPU compute kernels"),
    (r"yjs|automerge|crdt", "CRDT based collaborative sync"),
    (r"electron|tauri", "desktop app with web technologies"),
    (r"kubernetes|k8s|helm", "container orchestration on Kubernetes"),
    (r"ebpf|bpf", "eBPF kernel instrumentation"),
    (r"opentelemetry|prometheus", "metrics and distributed tracing"),
    (r"numpy|pandas|polars|arrow", "columnar dataframe processing"),
    (r"llvm|parser|lexer|tree-sitter|compiler", "parsing and compiler toolchain"),
    (r"raft|paxos|consensus", "distributed consensus replication"),
    (r"zstd|lz4|compression", "data compression codecs"),
    (r"crypto|tls|rustls|openssl|jwt", "cryptographic authentication"),
    (r"ffmpeg|opencv|image|video", "image and video processing"),
    (r"game|bevy|godot|unity|sdl", "real time game engine loop"),
    (r"sqlite", "embedded SQLite storage"),
    (r"cli|clap|click|cobra|typer", "command line tooling"),
]


def mock_atoms(d):
    meta = json.loads((d / "meta.json").read_text())
    blob = " ".join([
        meta.get("description") or "", " ".join(meta.get("topics", [])),
        (d / "deps.txt").read_text(encoding="utf-8")[:6000] if (d / "deps.txt").exists() else "",
    ]).lower()
    atoms = [{"atom": label, "role": "core" if i < 2 else "supporting"}
             for i, (pat, label) in enumerate(r for r in MOCK_RULES if re.search(r[0], blob))]
    return atoms[:8]


_next_slot = [0.0]


async def pace(rpm):
    """Space requests evenly so a free tier key stays under its per minute limit."""
    loop = asyncio.get_running_loop()
    slot = max(_next_slot[0], loop.time())
    _next_slot[0] = slot + 60.0 / max(rpm, 1)
    await asyncio.sleep(max(0.0, slot - loop.time()))


async def atomize_one(d, args, sem, completion):
    out = d / "refined_ideas.json"
    if out.exists() and not args.force:
        return "skip"
    if args.mock:
        atoms = mock_atoms(d)
    else:
        prompt = build_prompt(d, args.min_atoms, args.max_atoms, args.prompt)
        extra = reasoning_off(args.model)  # hidden reasoning would use up the 800 tokens
        if args.model.startswith("ollama"):
            extra["format"] = "json"  # local models: force valid JSON output
        async with sem:
            last = None
            for attempt in range(4):
                await pace(args.rpm)
                try:
                    resp = await completion(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2, max_tokens=800, timeout=120, **extra,
                    )
                    atoms = parse_atoms(resp.choices[0].message.content)
                    break
                except Exception as e:  # retry rate limits, timeouts, bad JSON
                    last = e
                    await asyncio.sleep(min(120, 10 * 2 ** attempt))
            else:
                return f"error: {last}"
    if not atoms:
        return "empty"
    out.write_text(json.dumps(to_refined(atoms, "mock" if args.mock else args.model), indent=2))
    return "ok"


async def run(args):
    dirs = sorted(p for p in Path(args.repos).iterdir() if (p / "meta.json").exists())
    if args.limit:
        dirs = dirs[: args.limit]
    completion = None
    if not args.mock:
        from litellm import acompletion
        completion = acompletion
    if args.model.startswith("ollama") and args.concurrency > 2 and not args.mock:
        print("Local model detected: lowering concurrency to 2 so your laptop does not choke.")
        args.concurrency = 2
    sem = asyncio.Semaphore(args.concurrency)
    counts = {}
    tasks = [atomize_one(d, args, sem, completion) for d in dirs]
    for n, coro in enumerate(asyncio.as_completed(tasks), 1):
        status = await coro
        key = "error" if status.startswith("error") else status
        counts[key] = counts.get(key, 0) + 1
        if key == "error" and counts[key] <= 5:
            print(f"  {status[:200]}")
        if n % 100 == 0:
            print(f"  {n}/{len(dirs)} {counts}")
    print(f"Done: {counts}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repos", default="repos")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Any LiteLLM model string, e.g. ollama/qwen2.5:7b or anthropic/claude-haiku-4-5")
    p.add_argument("--rpm", type=int, default=8,
                   help="Max requests per minute; keeps free tiers from throttling you")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--min-atoms", type=int, default=3)
    p.add_argument("--max-atoms", type=int, default=8)
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="tech",
                   help="tech: engineering techniques (default). play: mechanics and generative techniques, "
                        "for a corpus of games, creative coding and music")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N repos")
    p.add_argument("--force", action="store_true", help="Redo repos that already have atoms")
    p.add_argument("--mock", action="store_true", help="Heuristic atoms, no API calls")
    args = p.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
