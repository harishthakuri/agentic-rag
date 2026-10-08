# Reranking: the LLM reranker vs a cross-encoder (sentence-transformers)

This document explains what reranking is, how it works in Agentic RAG today, and how a dedicated reranker model would run inside the app with the **sentence-transformers** library. It also answers the questions that tend to come up along the way.

> **Status:** both rerankers are implemented. The LLM reranker (`gpt-oss`) is the default. The cross-encoder reranker (`bge-reranker-v2-m3`) is **opt-in** (`RERANKER=cross_encoder`). It becomes the default only if the evaluation on a harder test set shows it reviews at least as well (see [section 7](#7-how-we-will-decide-planned), _Planned_).

---

## 1. What is a reranker?

Search happens in **two stages**:

```
all chunks ──(stage 1: fast, approximate)──► ~10 candidates ──(stage 2: slow, precise)──► the best few
             hybrid search: vectors + keywords                 the reranker
```

- **Stage 1, retrieval:** vector search and keyword search must scan _everything_, so they are built for speed. They are good at finding _related_ passages, but rough at ordering them.
- **Stage 2, reranking:** a reranker reads the **question and each candidate together** and judges whether the passage actually **answers** the question. It is too slow to run on every chunk, but affordable for ~10 candidates.

A real example from our sample documents, for the question _"What does a 304 response mean?"_:

| Rank | Hybrid search alone                                 | After reranking                                   |
| ---- | --------------------------------------------------- | ------------------------------------------------- |
| 1    | HTTP Caching › Freshness ❌ (says "response" a lot) | **HTTP Caching › Validation › ETag ✅** (grade 3) |
| 2    | HTTP Caching › Validation › ETag                    | HTTP Caching › Freshness (grade 0)                |

Retrieval found the right passage, but ranked it second. The reranker put it first.

### Is the reranker only used in the evaluation?

**No. It is part of the application** and runs on real user questions:

| Where                                                 | Reranker used?                                                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| `POST /collections/{id}/ask` (and Ask mode in the UI) | **Yes**, every question (step "Review the passages")                     |
| `POST /collections/{id}/search`                       | Yes, unless the request sets `"rerank": false`                           |
| UI search lab, "Hybrid + rerank" column               | Yes                                                                      |
| `/agent/ask` (Agent mode)                             | No, by default (`AGENT_RERANK=false`): the agent judges relevance itself |
| Uploading documents                                   | Never                                                                    |
| Evaluation (`make eval`)                              | Measures search **with and without** reranking, to see if it helps       |

---

## 2. How many AI models? One model, several roles

Today the app uses **two AI models**, and the chat model plays several roles:

| Model                            | Role                                                             | Where                        |
| -------------------------------- | ---------------------------------------------------------------- | ---------------------------- |
| `qwen3-embedding:8b` (embedding) | Turns passages and questions into meaning fingerprints (vectors) | App: upload and every search |
| `gpt-oss:20b` (chat)             | **Reranker:** grades the top 10 passages 0–3                     | App: `/ask`, `/search`       |
|                                  | **Writer:** writes the cited answer                              | App: `/ask`, `/agent/ask`    |
|                                  | **Planner:** decides what to search for                          | App: `/agent/ask`            |
|                                  | **Judge:** grades finished answers                               | Evaluation only              |

| Setup                                                       | AI models                          | Who reranks                                  |
| ----------------------------------------------------------- | ---------------------------------- | -------------------------------------------- |
| **Today** (`RERANKER=llm`)                                  | **2**: embedding + chat            | The chat model (~4–5 s)                      |
| **With a cross-encoder** (`RERANKER=cross_encoder`, opt-in) | **3**: embedding + chat + reranker | A small, dedicated model (~0.85 s, measured) |
| No reranking (`RERANKER=none`)                              | **2**                              | Nobody: the search order is used as-is       |

The third model is **optional**: a speed upgrade, adopted only if the evaluation shows it reviews at least as well. The minimum for RAG is always two models: one to _find_ (embedding) and one to _write_ (chat).

---

## 3. Today: the LLM reranker (`gpt-oss`)

The chat model is asked to grade passages, like a human reviewer would.

**How it works** ([`app/infrastructure/reranking/llm_reranker.py`](../app/infrastructure/reranking/llm_reranker.py)):

1. The top 10 candidates (`RERANK_DEPTH`) are sent in **one batch** (`RERANKER_BATCH_SIZE=10`), numbered `[1]`…`[10]`. The model never sees or repeats database IDs.
2. The system prompt defines the scale: **3** answers it fully, **2** partially, **1** same topic but doesn't help, **0** unrelated. It also marks passages as untrusted data, so instructions inside documents are ignored.
3. The model must reply with **JSON that matches a schema** (structured output) containing **exactly one grade per passage**.
4. Results are re-sorted by grade. Ties keep the original search order.
5. **If anything fails** (timeout, invalid JSON, a missing grade), search falls back to the original order and reports `rerank_error`. A reranker failure never fails the request.

**Lesson learned:** with a generic "array of grades" schema, `gpt-oss` answered `{"grades": []}` **every time**: valid JSON, zero grades. Constrained decoding only enforces what the schema says. Requiring **exactly n items** (`minItems = maxItems = n`) and passage numbers `1..n` fixed it completely.

**Measured cost** (M1 Max, `reasoning_effort=low`): **~3.4 s for 10 candidates** and ~7.8 s for 20. Ollama processes requests one after another, so larger batches don't parallelise locally. That is why `RERANK_DEPTH` defaults to 10.

**Pros:** no extra model or dependency; explainable grades; works with any OpenAI-compatible chat model.
**Cons:** slow; uses tokens (a cost with paid APIs); coarse 0–3 grades produce ties.

---

## 4. The cross-encoder reranker

### Bi-encoders vs cross-encoders

|             | **Bi-encoder** (our embedding model)                     | **Cross-encoder** (a reranker)                                      |
| ----------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| Input       | Question and passage **separately**                      | Question and passage **together**, as one input                     |
| Output      | A vector per text; similarity = distance between vectors | **One relevance score** for the pair                                |
| Precompute? | **Yes**: passage vectors are computed once at upload     | **No**: must run for every (question, passage) pair at query time   |
| Speed       | Very fast, over millions of chunks                       | Slower per pair, fine for ~10–50                                    |
| Precision   | Good: the whole passage is squeezed into one vector      | Better: it sees how each question word relates to each passage word |

Example: for _"How do I **disable** TLS verification?"_, a bi-encoder may rank a passage about _enabling_ TLS highly (same topic). A cross-encoder reads both texts together and notices the negation.

### Should the bigger model not be faster?

No. A bigger model is usually slower. The cross-encoder is faster because it is **smaller and does less work**:

|               | `gpt-oss` (LLM reranker)                                                    | Cross-encoder (e.g. `bge-reranker-v2-m3`)                                                |
| ------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Size          | ~20 billion parameters                                                      | ~0.6 billion (about 35× smaller)                                                         |
| How it scores | **Writes** its answer token by token: it reasons first, then types out JSON | Reads the pair and outputs **one number** in a single forward pass; nothing is generated |
| Built for     | Everything: chat, writing, reasoning                                        | Only this: "how relevant is this passage?"                                               |

An analogy: asking a senior expert to grade ten passages works, but they read, think and write a report. A specialist checker, trained for exactly this job, stamps a score on each. The expert is still the right choice for _writing the answer_, which is why `gpt-oss` keeps that role.

### Candidate models

All are on Hugging Face. Sizes and licences were checked in October 2026.

| Model                                           | Size  | Licence          | Notes                                                                                       |
| ----------------------------------------------- | ----- | ---------------- | ------------------------------------------------------------------------------------------- |
| `cross-encoder/ms-marco-MiniLM-L6-v2`           | 23 M  | Apache 2.0       | Tiny and very fast; English only. **Speed baseline**                                        |
| `Alibaba-NLP/gte-reranker-modernbert-base`      | 150 M | Apache 2.0       | Fast and strong; English                                                                    |
| **`BAAI/bge-reranker-v2-m3`**                   | 568 M | Apache 2.0       | **Most widely used open reranker**; multilingual; a classic cross-encoder. **First choice** |
| **`Qwen/Qwen3-Reranker-0.6B`**                  | 596 M | Apache 2.0       | Same family as our embedding model; newer; also 4B and 8B. **Challenger**                   |
| `mixedbread-ai/mxbai-rerank-base-v2`            | 494 M | Apache 2.0       | Good, but less widely adopted                                                               |
| ~~`jinaai/jina-reranker-v2-base-multilingual`~~ | 278 M | **CC BY-NC 4.0** | ❌ Non-commercial licence: avoid                                                            |

**Implemented with `bge-reranker-v2-m3`** (the default `CROSS_ENCODER_MODEL`). Any classic cross-encoder from the table works by changing that one setting. **Next:** compare it with `Qwen3-Reranker-0.6B` and MiniLM (the "how fast can it get" baseline) on the harder test set, and keep **one** winner. The others are only used for the comparison.

> `Qwen3-Reranker` is technically a small LLM that scores relevance through the probability of answering "yes". Its official repository now ships the sentence-transformers configuration for this, so it **works with our `CrossEncoder` code unchanged**: set `CROSS_ENCODER_MODEL=Qwen/Qwen3-Reranker-0.6B`. Its scores run lower than `bge-reranker-v2-m3`'s (on our 304 example, 0.23 for the right passage versus 0.59 with bge, in the same order), so the `/ask` threshold must be checked per model.

### Why not run it in Ollama?

- **Qwen3-Reranker is not in the official Ollama library** (`ollama.com/library/qwen3-reranker` returns 404). Only community uploads exist, made by unknown people with no guarantee they are correct.
- **Ollama has no reranking endpoint.** On our Ollama 0.31, both `/api/rerank` and `/v1/rerank` return **404**. Ollama serves chat/generate and embeddings only.
- A reranker must return a **score per passage**. Through Ollama it could only be run as a chatbot that writes text, which is slow and defeats the purpose.

So **the reranker runs inside our Python app**, and Ollama keeps doing embeddings and chat.

---

## 5. sentence-transformers, explained

[sentence-transformers](https://www.sbert.net) is a Python library, built on PyTorch and Hugging Face `transformers`, for working with embedding and reranking models. Its `CrossEncoder` class is exactly what a reranker needs:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # 1. load (downloads once)

pairs = [(question, passage) for passage in candidates]  # 2. build pairs
scores = reranker.predict(pairs, batch_size=16)  # 3. score them all
# scores -> e.g. [0.97, 0.02, 0.41, ...], one relevance score per passage

# Or let the library sort for you:
ranked = reranker.rank(question, candidates)  # [{"corpus_id": 0, "score": 0.97}, ...]
```

### What happens inside `predict()`

```
("What does a 304 response mean?", "If the resource has not changed, the server replies 304 …")
        │
        ▼  1. Tokenise the pair as ONE input:   [CLS] question tokens [SEP] passage tokens [SEP]
        ▼  2. One forward pass through the model (attention sees question and passage together)
        ▼  3. A classification head outputs one raw number (a "logit")
        ▼  4. Optionally, a sigmoid turns it into a score between 0 and 1
   0.97  → very relevant
```

Whether step 4 happens by default depends on the model. Some (e.g. the MS MARCO family) return raw logits unless you load them with
`activation_fn=torch.nn.Sigmoid()`. **The ranking is the same either way**, because a sigmoid never changes the order, so for reranking
it doesn't matter. It only matters if you want to compare scores to a fixed threshold.

- **Batching:** many pairs go through the model at once, on the GPU, which is much faster than scoring them one by one.
- **Device:** set with `device=`: **`mps`** (Apple-silicon GPU, as on our M1 Max), `cuda` (NVIDIA) or `cpu`. If not set, an available GPU is used.
- **Truncation:** long passages are cut to the model's maximum length (`max_length`, e.g. 512 tokens). Our chunks are about 500 tokens, so this rarely matters.
- **Scores are relative:** compare them _within one question_ to rank passages. A score's absolute value means little across models.

### Where the model comes from

There is nothing to pull manually. On first use, the model is **downloaded automatically from Hugging Face** (the official source) and cached in `~/.cache/huggingface/`. After that it loads from disk, with no network needed.

Approximate download sizes: MiniLM ~90 MB; `bge-reranker-v2-m3` ~2.3 GB (float32 weights); `Qwen3-Reranker-0.6B` ~1.2 GB. On top of that, PyTorch itself is about 1 GB.

### How our implementation handles it

- **Load once, at startup.** Loading takes seconds, so the API loads the model when it starts (`Container.warm_up()`), not per request. The ingestion worker and the admin CLI never rerank, so they never load it.
- **Don't block the server.** `predict()` is synchronous, CPU/GPU-bound work. It runs in a worker thread (`asyncio.to_thread`), so other requests keep being served. One prediction runs at a time: batching already keeps the GPU busy.
- **Failures fall back.** If the model fails (for example, out of GPU memory), search uses the original order and reports `rerank_error`, exactly as with the LLM reranker.
- **Scores are always 0 to 1.** We always apply the sigmoid (step 4), whatever the model's default. This matters for `/ask`, which drops passages below a relevance threshold. The LLM's threshold is a grade (`ANSWER_MIN_RERANK_GRADE=1`, on a 0–3 scale), which would drop _every_ cross-encoder passage, so the cross-encoder has its own: `ANSWER_MIN_CROSS_ENCODER_SCORE=0.02`. In a first test, unrelated passages scored 0.001–0.003 and the right one 0.59. The evaluation will calibrate this value.
- **Memory:** the model stays in RAM/GPU memory (~1–2.5 GB) for as long as the app runs.
- **Optional dependency:** PyTorch is large, so `sentence-transformers` is an **optional install** (`uv sync --extra rerank`). The default install and the Docker image stay small. Choosing `RERANKER=cross_encoder` without it stops the app at startup with a message saying what to install.

**Measured speed** (M1 Max, `mps`): about **0.85 s for 10 passages**, versus ~3.4 s for the LLM reranker, so roughly **4× faster**. That is slower than the "~100 ms" often quoted for cross-encoders: `bge-reranker-v2-m3` is one of the larger ones (568 M parameters), and those figures usually come from NVIDIA GPUs. This was a quick measurement; the evaluation will measure latency properly, and MiniLM shows how fast a tiny model can go.

---

## 6. Where the code is

The design uses a **port and adapters**: search depends on a `Reranker` interface, not on a specific model.

| File                                                                                                                  | What it does                                                                      | Changed for the cross-encoder?                                     |
| --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [`app/application/ports/reranking.py`](../app/application/ports/reranking.py)                                         | The `Reranker` interface: `rerank(question, candidates) → scores`                 | No                                                                 |
| [`app/infrastructure/reranking/llm_reranker.py`](../app/infrastructure/reranking/llm_reranker.py)                     | The `gpt-oss` implementation                                                      | No: **stays available**                                            |
| [`app/infrastructure/reranking/cross_encoder_reranker.py`](../app/infrastructure/reranking/cross_encoder_reranker.py) | The sentence-transformers implementation of the same interface                    | **New file**                                                       |
| [`app/bootstrap/container.py`](../app/bootstrap/container.py)                                                         | `_build_reranker()` picks the reranker from settings; `warm_up()` loads the model | Yes: builds the cross-encoder and picks the right `/ask` threshold |
| [`app/core/config.py`](../app/core/config.py)                                                                         | All settings                                                                      | Yes: adds `CROSS_ENCODER_*` and `ANSWER_MIN_CROSS_ENCODER_SCORE`   |
| [`app/main.py`](../app/main.py)                                                                                       | Starts the API                                                                    | Yes: one line to warm up the model at startup                      |
| [`app/application/use_cases/retrieval/__init__.py`](../app/application/use_cases/retrieval/__init__.py)               | Search calls whichever reranker it is given                                       | No                                                                 |
| API, UI, `/ask`, `/agent/ask`                                                                                         | —                                                                                 | No                                                                 |

The search logic, the API and the UI did not change at all. That is the point of the port-and-adapter design.

Choosing a reranker is a configuration switch in `.env`:

```bash
RERANKER=llm             # gpt-oss grades the passages (default)
RERANKER=cross_encoder   # the small, fast reranker model
RERANKER=none            # no reranking

# Cross-encoder only (all optional):
CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3   # any Hugging Face cross-encoder
CROSS_ENCODER_DEVICE=                         # mps | cuda | cpu; empty: best available
ANSWER_MIN_CROSS_ENCODER_SCORE=0.02           # /ask drops passages scored below this
```

To try it:

```bash
uv sync --extra rerank                        # once: installs sentence-transformers + PyTorch
RERANKER=cross_encoder make run               # first start downloads the model (~2.3 GB)
uv run pytest -m live -k cross_encoder        # checks the real model ranks a known example correctly
```

The LLM reranker is **not removed**. Keeping both makes a fair comparison possible, and you can switch back at any time. The default changes only after the evaluation shows the cross-encoder is at least as good.

---

## 7. How we will decide (_Planned_)

1. **First, a harder test set** (details below). On today's sample corpus _every_ reranker scores 1.00 (see [EVALUATION.md](EVALUATION.md)), so they can't be compared.
2. **Then compare:** `none`, `llm` (gpt-oss), `bge-reranker-v2-m3`, `Qwen3-Reranker-0.6B` and MiniLM, on recall@5, MRR, nDCG **and latency**.
3. **Keep the winner** as the default, if it is at least as good as `gpt-oss` and clearly faster.

### The harder test set

Our four sample documents are short and cover different topics, so finding the right passage is easy and every method scores perfectly. A useful test set needs **many passages that sound alike but answer different questions**.

The plan uses official documentation sets that are freely licensed and **overlap heavily with each other and with our sample documents**. That is exactly what makes search hard:

| Source                                   | Licence                         | Pages to use (~45 in total)                                                                                                                                                                                                                                     | Why it's hard                                                                                               |
| ---------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Kubernetes docs** (kubernetes/website) | CC BY 4.0                       | ~19: Service, Ingress, Ingress controllers, **Gateway API**, EndpointSlices, Network policies, DNS, traffic policy, topology routing; Deployment vs StatefulSet vs DaemonSet vs ReplicaSet vs Job; ConfigMap vs Secret; liveness vs readiness vs startup probes | Many near-identical concepts ("which object exposes my app?"); also overlaps our sample Kubernetes document |
| **PostgreSQL docs**                      | PostgreSQL License (permissive) | ~17: index types (**B-tree, GIN, GiST, BRIN, Hash**), partial, multicolumn and expression indexes, index-only scans; full-text search chapter; VACUUM / autovacuum; MVCC; EXPLAIN; memory settings (`work_mem`, `maintenance_work_mem`)                         | GIN vs GiST for full text, `work_mem` vs `maintenance_work_mem`: similar words, different answers           |
| **MDN HTTP docs** (mdn/content)          | CC BY-SA 2.5                    | ~9: HTTP caching guide, Cache-Control, ETag, Last-Modified, Vary, If-None-Match, If-Modified-Since, 304, conditional requests                                                                                                                                   | Overlaps our sample HTTP caching document almost section by section                                         |
| **pgvector README**                      | PostgreSQL License              | 1                                                                                                                                                                                                                                                               | Overlaps our sample pgvector document                                                                       |

The licences were checked in the source repositories, and the Kubernetes pages listed exist under `content/en/docs/concepts/…` (October 2026).

**How it will be set up:**

- **The documents are committed as a fixed test corpus**, in [`data/benchmark/docs/`](../data/benchmark/docs/), so results can be reproduced without downloading anything. [`scripts/download_benchmark_docs.py`](../scripts/download_benchmark_docs.py) (`make benchmark-docs`) regenerates them from pinned versions. They are **not** under this repository's MIT licence: each folder keeps its original licence, and [`ATTRIBUTION.md`](../data/benchmark/docs/ATTRIBUTION.md) lists every source URL, licence and change (the PostgreSQL License text sits next to those files; the converted MDN pages stay CC BY-SA 2.5). _Done: 46 pages._
- **Formats are converted on the way in.** Kubernetes and MDN pages are Markdown with template tags, which the script strips. PostgreSQL docs are **HTML**, which the app can't ingest yet, so the script converts them to Markdown and adds the chapter to each title (four pages are otherwise just called "Introduction"). An HTML parser would still be a useful app feature in its own right. _Done._
- **A new `benchmark` collection** with about 30 new hand-written, reviewed questions. Many deliberately target look-alike sections (GIN vs GiST, readiness vs liveness), plus exact-term questions such as "what does `maintenance_work_mem` affect?".

---

## Run make and delete old and ingest new docs

**Steps (with `make run` running):**

```bash
# 1. Restart the worker so it uses the parser fix: Ctrl+C in its terminal, then
make worker

# 2. In another terminal, with your API key:
export RAG_API_KEY=arag_...

# 3. Delete the polluted 'samples' collection (this also deletes its documents)
curl -s -H "Authorization: Bearer $RAG_API_KEY" localhost:8000/api/v1/collections   # copy the samples "id"
curl -X DELETE -H "Authorization: Bearer $RAG_API_KEY" localhost:8000/api/v1/collections/<samples-id>

# 4. Rebuild both collections
make ingest-samples      # the 4 sample docs, back into 'samples'
make ingest-benchmark    # the 46 docs, into a new 'benchmark' collection
```

Each `make` command waits and prints every document's status and chunk count when it's done. `ingest-benchmark` takes a few minutes for embedding.

If you'd rather not delete `samples`, you can remove the 46 extra documents one by one with the ✕ button in the UI, then run only `make ingest-benchmark`.

## Run evaluation

Next is **evaluation**. The cross-encoder is already built; what's missing is the measurement that decides whether it should replace `gpt-oss` as the default.

Run the search-only evaluation on the benchmark once per reranker:

```bash
RERANKER=none          uv run python -m evals --collection benchmark --dataset evals/datasets/benchmark.jsonl --skip-answers
RERANKER=llm           uv run python -m evals --collection benchmark --dataset evals/datasets/benchmark.jsonl --skip-answers
RERANKER=cross_encoder uv run python -m evals --collection benchmark --dataset evals/datasets/benchmark.jsonl --skip-answers
RERANKER=cross_encoder CROSS_ENCODER_MODEL=Qwen/Qwen3-Reranker-0.6B uv run python -m evals --collection benchmark --dataset evals/datasets/benchmark.jsonl --skip-answers
```

Each run reports recall@5, MRR and nDCG for vector, keyword, hybrid and hybrid + rerank, plus timings. Comparing the four runs answers:

1. **Is the benchmark hard enough?** If plain hybrid search already scores near 1.00, the test still can't tell rerankers apart.
2. **Which reranker ranks best:** `gpt-oss`, `bge` or `Qwen`?
3. **How fast is each one per question?**

After that, depending on the results: pick the winner and change the default, adjust `ANSWER_MIN_CROSS_ENCODER_SCORE` for it, and write up the numbers in EVALUATION.md and RERANKING.md.

The runs don't change anything; they only write results under `results/`. All four take a few minutes in total. The LLM run is the slowest, at about 3–4 s per question.

## benchmark result

**The benchmark works**: it now shows real differences between search methods, where the old sample set gave everything 1.00. Qwen3-Reranker matches the LLM reranker on finding the right section and is about 4× faster.

**Results** (31 answerable questions, each search method followed by reranking of the top 10 hybrid results):

| Config                   | Recall@5 | MRR@10   | nDCG@10  | Time per question (p50) |
| ------------------------ | -------- | -------- | -------- | ----------------------- |
| hybrid, no rerank        | 0.77     | 0.68     | 0.70     | 0.3 s                   |
| + `gpt-oss` (LLM)        | **0.88** | **0.91** | **0.88** | 8.5 s                   |
| + `Qwen3-Reranker-0.6B`  | **0.88** | 0.87     | 0.84     | **2.2 s**               |
| + `bge-reranker-v2-m3`   | 0.83     | 0.84     | 0.83     | **1.2 s**               |
| _vector only, no rerank_ | _0.84_   | _0.81_   | _0.80_   | _0.3 s_                 |

(Recall@5: share of the right sections found in the top 5. MRR and nDCG: how close to the top the right section sits; 1.00 is perfect.)

## What are these column?

> **Short answer: for Recall@5, MRR@10 and nDCG@10, higher is better.** All three run from 0 to 1. **1.0 is perfect**: every right section is found, at the top. **0** means the right section wasn't found in the top 5 (Recall) or top 10 (MRR, nDCG). Only the time column is better when lower.
>
> - **MRR** = **Mean Reciprocal Rank**: how high the _first_ right answer is ranked. 1.0 = always at rank 1, 0.5 = typically at rank 2.
> - **nDCG** = **normalised Discounted Cumulative Gain**: how well _all_ the right answers are ordered, compared with the perfect order.
> - **Recall@5** = the share of the right answers that appear in the top 5 results.
> - The **@5 / @10** suffix is how many top results the score looks at.

For every question, the evaluation runs a search and gets back a **ranked list of sections** (rank 1 = shown first). Each question is labelled with the section(s) that actually answer it. The scores compare the two: _did the right section come back, and how high up?_ Each score is calculated per question and then averaged over the 31 answerable questions. The 3 "not in the documents" questions are left out, because there is no right section to find.

| Column                      | Question it answers                                         | Range   | Good means |
| --------------------------- | ----------------------------------------------------------- | ------- | ---------- |
| **Config**                  | Which search setup produced the ranking                     | —       | —          |
| **Recall@5**                | Of the right sections, how many are in the **top 5**?       | 0–1     | Higher     |
| **MRR@10**                  | How high is the **first** right section, within the top 10? | 0–1     | Higher     |
| **nDCG@10**                 | How high are **all** the right sections, within the top 10? | 0–1     | Higher     |
| **Time per question (p50)** | How long one search takes, typically                        | seconds | Lower      |

**Config (the rows)**

- **hybrid, no rerank:** vector search and keyword search combined, without a reranker. This is the baseline the rerankers start from.
- **+ `gpt-oss` / + `Qwen3-Reranker-0.6B` / + `bge-reranker-v2-m3`:** the same hybrid search, then that reranker re-orders its top 10 results (`RERANK_DEPTH=10`).
- **vector only, no rerank** (in italics): plain meaning-based search, shown for comparison.

**Recall@5: "is the answer in the top 5 at all?"**
The share of a question's right sections that appear in the first 5 results. One right section in the top 5 scores 1.0, and missing scores 0. With two right sections and only one of them in the top 5, the question scores 0.5. The top 5 matters because `/ask` gives the writing model only the best few passages, so an answer at rank 8 is effectively lost.

**MRR@10 (Mean Reciprocal Rank): "how far down is the first right answer?"**
The score is 1 divided by the rank of the first right section: rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, rank 5 → 0.2. If no right section is in the top 10, the score is 0. It rewards putting a right answer **first**, which matters most because [1] is what the model and the reader trust most. An MRR of 0.91 means the right section is almost always at rank 1.

**nDCG@10 (normalised Discounted Cumulative Gain): "how good is the whole order?"**
Like MRR, but it counts **every** right section, not only the first. Each right section earns points that shrink with its rank (rank 1 = 1, rank 2 = 0.63, rank 3 = 0.5, rank 7 = 0.33, …). The total is divided by the best possible total, so a perfect order scores 1.0. It matters for questions with several right sections, such as the multi-part ones.

**Time per question (p50)**
The **median** time for one search: half of the questions were faster, half slower. It includes the first-stage search (~0.3 s) plus the reranker. The full report also has **p95**, the time 95% of questions stay under, which shows the slow cases.

**A worked example.** The right section comes back at rank 3:

|          | Score | Why                                          |
| -------- | ----- | -------------------------------------------- |
| Recall@5 | 1.0   | it is in the top 5                           |
| MRR@10   | 0.33  | 1 ÷ 3                                        |
| nDCG@10  | 0.5   | rank 3 earns 0.5 points, out of a possible 1 |

A reranker that moves it to rank 1 raises all three scores to 1.0. Recall barely changes, but MRR and nDCG jump. That's why the rerankers' gains show up mostly in **MRR and nDCG**: they rarely find _new_ sections, they move the right ones to the top.

**Extra columns in the full report** (`evals/results/*.md`): **Recall@5 paraphrase / exact / multi-part** split recall by question type (reworded questions, exact names and numbers, questions needing two or more sections). Two more rows, **vector** and **keyword**, show each first-stage search on its own.

**Reading the differences.** With 31 questions, a single question moves MRR by up to about 0.03. Differences of 0.03–0.04 (e.g. 0.87 vs 0.91) are within noise; gaps of 0.15 or more (e.g. 0.68 → 0.84) are real.

**What this shows**

- **Every reranker helps a lot.** MRR rises from 0.68 to between 0.84 and 0.91. In several questions the right section moved from rank 5 or lower up to rank 1.
- **Qwen is as good as `gpt-oss` on recall, and close on MRR.** The gap in MRR (0.87 vs 0.91) is one or two questions out of 31, which is within noise for a test set this size. It's about 4× faster.
- **bge is the fastest but clearly weaker.** It ranked badly on the BRIN question (rank 6), the 304 headers question and the transaction-wraparound question.
- **The LLM reranker took 8.5 s here,** not the 3.4 s we measured before. The benchmark passages are longer, so it has more to read.

**Two findings beyond the rerankers**

1. **Hybrid search is worse than vector search alone on this corpus** (0.77 vs 0.84). Keyword search is weak here (0.50) and pulls hybrid down. That's worth looking into separately.
2. **One question can't be fixed by any reranker.** For k8s-02 (readiness probes), the right section isn't among the 10 passages hybrid search hands the reranker. Reranking more candidates (`RERANK_DEPTH=20`) costs little with a cross-encoder and could fix cases like this.

**My recommendation:** make `Qwen/Qwen3-Reranker-0.6B` the default model for `RERANKER=cross_encoder`, and consider switching the app's default from `llm` to it

---

## Quick answers

| Question                                                        | Answer                                                                                                                                      |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Is the reranker only for the evaluation?                        | No. It runs on every `/ask` question and on `/search`                                                                                       |
| How many AI models do we need?                                  | **2** by default (embedding + chat, which also reranks). **3** with the optional cross-encoder                                              |
| How do I turn the cross-encoder on?                             | `uv sync --extra rerank`, then `RERANKER=cross_encoder`. The model downloads on first start                                                 |
| How fast is it?                                                 | ~0.85 s for 10 passages on an M1 Max, about 4× faster than the LLM reranker (~3.4 s)                                                        |
| Is it the default now?                                          | No. `llm` stays the default until the harder test set shows the cross-encoder is at least as good                                           |
| Do we need both `bge-reranker-v2-m3` and `Qwen3-Reranker-0.6B`? | No. We test both and keep one                                                                                                               |
| Does it have to be a `gpt-oss` model?                           | No. A cross-encoder is a separate, small model built only for relevance scoring                                                             |
| Shouldn't the big model be faster?                              | No. Bigger is usually slower; the cross-encoder is ~35× smaller and outputs one number instead of writing text                              |
| Why does the cross-encoder have its own `/ask` threshold?       | Its scores are 0 to 1, while the LLM's grades are 0 to 3. One threshold can't fit both scales                                               |
| Can I `ollama pull` the reranker?                               | No: it's not in Ollama's official library, and Ollama has no rerank API. sentence-transformers downloads it from Hugging Face automatically |
| What happens to the gpt-oss reranker?                           | It stays, selectable with `RERANKER=llm`                                                                                                    |

## Sources

- [sentence-transformers documentation: CrossEncoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [Best Rerankers for RAG in 2026 (FutureAGI)](https://futureagi.com/blog/best-rerankers-for-rag-2026/)
- [Reranker Benchmark: Top 8 Models Compared (AIMultiple)](https://aimultiple.com/rerankers)
- [Ollama: reranking support discussion (issue #4510)](https://github.com/ollama/ollama/issues/4510)
- Model pages: [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) · [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) · [gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) · [mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) · [jina-reranker-v2](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
