# Uncharted

**Finding software projects that make sense but nobody is positioned to build, then testing whether AI coding agents can build them.**

Uncharted adapts [The Alien Space of Science](https://arxiv.org/abs/2603.01092) (Artiles et al., 2026) from research papers to GitHub. The original work breaks papers into "idea atoms" and trains two models: a coherence transformer that scores whether a combination of atoms is a viable direction, and an author dual encoder that scores whether any existing research community is likely to produce it. Ranking for high coherence and low availability surfaces plausible ideas that sit outside where people are looking.

Uncharted does the same for code:

| Paper world | Uncharted |
| --- | --- |
| paper | GitHub repository |
| idea atom | technique atom ("CRDT based offline sync", "WebGPU tensor inference") |
| author community | contributors who have committed to repos using those techniques |
| alien research direction | coherent technique combination no developer community is near |

The question it ends with: **coding agents are trained on the dense parts of GitHub. Do they get worse as a project spec moves into the alien space?**

## Pipeline

```
collect    GitHub search + README + dependency manifests + contributors   -> repos/
atomize    LLM extracts 3 to 8 technique atoms per repo                    -> repos/*/refined_ideas.json
cluster    BGE embeddings -> UMAP -> HDBSCAN -> 80% coverage -> naming      -> data/clusters_80.json
build-db   repos as papers, contributors as authors (original schema)      -> data/uncharted.db, data/vocab.json
-------------------------------------------------------------------------------------------------------------
train      original Alien code, unchanged: coherence transformer + contributor dual encoder
sample     rank triplets by (1 - beta) z(coherence) + beta z(unavailability)
show       readable combinations with example repos per atom
benchmark  (planned) spec writing + Claude Code / Copilot agent build trials, common vs alien
```

Step 1 writes exactly the file formats the original repo reads, so steps 2 and 3 run on the authors' code with no modifications. That was verified end to end: `make-datasets`, `make-availability`, `train-coherence`, `train-availability` and `sample` all run on Uncharted output.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GITHUB_TOKEN=...          # a classic token with no scopes is enough
export GEMINI_API_KEY=...        # free from Google AI Studio; any LiteLLM model works via --model

python uncharted.py collect  --target 2000
python uncharted.py atomize                # free gemini/gemini-3.8-flash by default
python uncharted.py cluster
python uncharted.py build-db

bash scripts/train_and_sample.sh      # clones the original repo into third_party/ and trains
python uncharted.py show --results results/alien/k3/beta_0.7/results.json
```

### Running it for free

You do not need a paid API key. Install [Ollama](https://ollama.com), pull a small model, and point Uncharted at it:

```bash
ollama pull qwen2.5:7b
python uncharted.py atomize --model ollama/qwen2.5:7b
python uncharted.py cluster --namer llm --model ollama/qwen2.5:7b    # or --namer medoid, no LLM at all
```

On a laptop this runs a few hundred repos per hour, so start with `collect --target 500` and let the full run go overnight. Google's Gemini API also has a free tier with no card required (`--model gemini/<model>` with `GEMINI_API_KEY` from AI Studio); note that free tier prompts may be used to improve Google products.

No model at all? `atomize --mock` and `cluster --embedder tfidf --namer medoid` run the whole pipeline offline with heuristic atoms, which is handy for checking that everything is wired up.

## The alien factory

Once you have sampled combinations, the factory turns them into real, public software:

```bash
export GEMINI_API_KEY=...        # free key from Google AI Studio
python uncharted.py spec    --results results/alien/k3/beta_0.7/results.json --n 20
python uncharted.py build   --max 5
python uncharted.py publish --org your-github-org            # dry run first
python uncharted.py publish --org your-github-org --yes      # ships up to 3 per day
python uncharted.py gallery --refresh-stars                  # factory_out/gallery/index.html
```

* **spec** asks the model to design one small, useful, standard library Python project that uses every technique in the combination, or to say honestly that none fits.
* **build** has the model write the code and tests, runs pytest, sends failures back, and repeats up to 6 times. Weak tests (no real assertions, `assert True`) are rejected and sent back too.
* **publish** pushes passing projects to GitHub with a README that says plainly they were invented and built by AI, tagged `ai-generated`. It is capped per day so the org never looks like spam.
* **gallery** is a static page of everything, failures included, ranked by stars. Host it on GitHub Pages.

Everything lands in `factory_out/registry.json`, which becomes the dataset: which kinds of alien combinations get built, which fail, and which people actually star.

**Safety:** generated code runs inside Docker with no network when Docker is installed (`uncharted-runner` image, built automatically). Without Docker it runs as a plain subprocess after a warning, so install Docker Desktop before running the factory on your main machine.

**Free tier pacing:** every LLM call goes through `uncharted/llm.py`, which spaces calls at `--rpm` (default 8) and backs off on rate limits. The default model is `gemini/gemini-3.8-flash`; set `UNCHARTED_MODEL` to change it everywhere. If the daily quota runs out, rerun the next day and finished work is skipped.

## Cost and scale

| Step | 2,000 repos | 10,000 repos |
| --- | --- | --- |
| collect | about 6,000 REST calls, roughly 1.5 hours at 5,000/hour | about 6 hours, resumable |
| atomize | a few dollars on a small model | tens of dollars |
| cluster | minutes on CPU | under an hour on CPU, faster on GPU |
| train | CPU works for a smoke run; one rented GPU for a real run | one rented GPU, a few hours |

Every step is resumable. Rerunning `collect` or `atomize` skips repos that are already done.

## Design notes

* **Contributors, not stargazers.** Availability should reflect who can actually build a combination. Bots and anyone under 3 commits are dropped so drive by typo fixes do not count as repertoire.
* **Corpus balance.** Search quota is split across 20 languages so the vocabulary is not dominated by Python and JavaScript.
* **Atom quality.** The prompt rejects atoms every repo has (the language, "unit tests", "Docker support") and asks for techniques phrased so they transfer between projects.
* **Manifests via raw URLs.** Dependency files come from `raw.githubusercontent.com`, which keeps the REST budget to three calls per repo.
* **No emails.** Contributor ids are GitHub numeric ids; nothing else about a person is stored.

## Tests

```bash
python -m pytest tests -q
```

Covers bot and drive by filtering, resumability, LLM reply parsing, and an offline run from repos through clustering and the database.

## Credit

Built on [alejandrohdez00/alien-space-of-science](https://github.com/alejandrohdez00/alien-space-of-science) (MIT). The coherence and availability models, dataset builders and sampler are the authors' work; Uncharted contributes the GitHub corpus, atomization, clustering adapter and the coding agent benchmark.
