# Evaluation

How well does each retrieval strategy find the right passages, and how good are the answers built on them? This page describes the method, the latest results and what they mean.

```bash
make eval-retrieval      # retrieval only, ~2 min
make eval                # retrieval + answers + LLM judge, ~20 min locally
uv run python -m evals --help
```

Each run writes `evals/results/<timestamp>.json` (every row, with the models, prompt versions and settings used) and a `.md` summary.

## Method

### Dataset

[`evals/datasets/samples.jsonl`](../evals/datasets/samples.jsonl) contains 27 hand-written questions over the `sample_data/` corpus (4 documents, 32 chunks). Each question is labelled with the **sections** that answer it, as a `(document title, heading path)` pair, and with a short reference answer. Sections, unlike chunk IDs, stay valid when documents are re-chunked.

| Type | Count | What it tests |
|---|---|---|
| paraphrase | 13 | Wording that differs from the source ("newest version of my JavaScript" → *Cache busting*) |
| exact | 8 | Identifiers, numbers and defaults (`ef_search`, `304`, NodePort range) |
| multi-part | 3 | Answers that need two sections, sometimes from two documents |
| unanswerable | 3 | Not in the documents: the right answer is to decline |

### Retrieval metrics

These are computed per section: a section split into several chunks counts once, at its best rank.

- **Recall@5:** the share of expected sections found in the top 5, which is what fits in an LLM's context.
- **MRR@10:** 1 / rank of the first relevant result. This rewards getting the *top* result right.
- **nDCG@10:** rewards relevant results more the higher they rank, scaled so that 1.0 is a perfect ranking.

### Answer metrics

`/ask` (one search, hybrid + rerank) and `/agent/ask` (the model decides the searches) answer every question. An **LLM judge** then grades each answer:

- **Correctness** compares the answer with the reference: correct = 1, partially correct = 0.5, incorrect = 0. For unanswerable questions, "correct" means the system declined.
- **Faithfulness** splits the answer into claims and checks each one against the sources the system actually had. A claim the sources don't support counts as unsupported **even if it is true**: it came from the model's memory, not the documents.
- **Without a judge:** whether the answer cited an expected section, invalid citations, declines, latency and tokens.

**Judge caveats.** The judge is a model too, and it makes mistakes, especially strict "unsupported" calls on paraphrased claims. Here it is the same model that wrote the answers (`gpt-oss:20b`), which can bias it towards leniency. Use `--judge-model` with a stronger, different model for more trustworthy numbers, and spot-check the `unsupported_claims` and `sources` saved in the JSON.

## Results

Run on 2026-10-07 on an M1 Max (64 GB), everything local: embeddings `qwen3-embedding:8b` (1024 dims), chat and judge `gpt-oss:20b` (reasoning effort low), reranker = the same LLM grading the top 10 candidates. Prompts `answer-v2` and `agent-v2`.

### Retrieval (24 answerable questions)

| Config | Recall@5 | MRR@10 | nDCG@10 | Recall@5 paraphrase | Recall@5 exact | Recall@5 multi-part | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| vector | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 259 | 300 |
| keyword | 0.98 | 0.89 | 0.89 | 1.00 | 1.00 | 0.83 | 30 | 42 |
| hybrid | 0.98 | 0.98 | 0.98 | 1.00 | 1.00 | 0.83 | 264 | 290 |
| hybrid+rerank | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4868 | 7748 |

### Answers (27 questions)

| System | Correctness | Faithfulness | Answers w/ unsupported claims | Answers with citations | Cited expected section | Declined unanswerable | Wrongly declined | Avg tool calls | p50 s | p95 s | Avg tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `/ask` | 1.00 | 0.97 | 8% | 79% | 79% | 3/3 | 0 | 0.0 | 7.6 | 15.5 | 575 |
| `/agent/ask` | 1.00 | 0.95 | 12% | 88% | 88% | 3/3 | 0 | 1.3 | 6.3 | 12.4 | 2,481 |

## What the numbers say

**1. This benchmark is too easy to rank retrieval strategies.** Plain vector search scores a perfect 1.00. The corpus is small (32 chunks) and its four documents cover clearly separate topics, so the right section is almost always the nearest one. The only differences are on multi-part questions, where keyword search misses a second section (0.83) and pulls hybrid down with it. Reranking restores 1.00, at about 5 s per query. These numbers **don't show** that hybrid or reranking are useless. They show this test set can't tell them apart. A meaningful comparison needs a larger corpus with overlapping topics and near-duplicate distractors (see *Next steps*).

**2. The answers are correct; the open problem is citation discipline.** Both systems answer every answerable question correctly and decline all three unanswerable ones. The weak spot is citations: 21% of `/ask` answers and 12% of agent answers cite nothing at all. Whenever an answer *does* cite, the cited source is an expected one ("cited expected section" equals "answers with citations"). So retrieval is fine; the model sometimes just doesn't cite. This is what to improve next, for example by requiring citations in a structured output, or by retrying uncited answers.

**3. Unsupported claims come from the model's own knowledge.** The judge flagged 8% of `/ask` answers and 12% of agent answers. Reading them, the flagged claims are background knowledge that isn't in the sources: BM25 details, MetalLB setup steps, "DNS resolves the domain to an external IP". They are mostly *true* but not grounded, which is exactly what faithfulness is meant to catch. The agent adds slightly more of them; more context and more freedom mean more room to improvise.

**4. The agent is not slower here, but it costs 4× more tokens.** It averaged 1.3 tool calls, and its median latency is lower than `/ask`'s, because `/ask` pays for an LLM rerank on every question, while the agent's searches skip reranking and the model judges relevance itself. That costs prompt tokens (2,481 vs 575 per question), which matters with paid APIs.

**5. Correctness at 1.00 is a ceiling, not proof of perfection.** The questions are answerable from one or two sections, and the judge is the same model that wrote the answers. Expect lower scores with harder questions and a stronger, independent judge (`--judge-model`).

## The evaluation found bugs before it produced numbers

The first full run produced numbers that looked plausible but were wrong. Auditing the per-answer JSON (sources, unsupported claims, raw answers) found three bugs:

| Bug | Effect | Fix |
|---|---|---|
| The citation regex ignored citations attached to a word (`mode[1]`) | Real citations went uncounted, in production responses too | Count attached citations; remove code blocks before scanning |
| Context merging joined adjacent chunks from *different* sections | Cache-busting text was shown to the model under the heading "The Vary header" | Merge only chunks from the same section |
| Decline detection missed `couldn’t` (curly apostrophe) | The agent showed 1/3 correct declines (really 3/3), and its refusals counted as unsupported claims | Normalise apostrophes |

**The lesson: never trust an evaluation you haven't audited.** Spot-check the rows behind every surprising number.

## Next steps

- **A harder corpus:** dozens of documents with overlapping topics (e.g. a slice of the PostgreSQL and Kubernetes documentation), so retrieval strategies can actually be told apart.
- **An independent judge:** `--judge-model` with a stronger hosted model, then compare it with the local judge on the same answers.
- **Citation enforcement:** measure whether a structured answer format or a retry of uncited answers raises "answers with citations" to about 100% without hurting faithfulness.
- **Cross-encoder reranker** (phase 9): the same precision as the LLM reranker for about 0.2 s instead of 5 s?
