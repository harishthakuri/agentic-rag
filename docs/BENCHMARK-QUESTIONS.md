# Benchmark question

The first six are ones where hybrid search alone ranked the right section 3rd to 8th and Qwen moved it to 1st, so they show the reranker at work. Comparing "Hybrid" with "Hybrid + rerank" in the UI's search lab makes the difference visible.

**Where reranking helps most**

1. How large can a single Kubernetes Secret be? *(1 MiB)*
2. My response has no Cache-Control header at all. Can browsers still cache it, and for how long?
3. My database replicas each need a stable network name and their own persistent volume. Should I use a Deployment?
4. What is the default value of autovacuum_vacuum_scale_factor?
5. What is the default maxSurge for a Deployment's rolling update?
6. I edited a ConfigMap. Will my running pods see the new values, or do I have to restart them?

**Look-alike topics (similar wording, different answers)**

7. Compare work_mem and maintenance_work_mem: what does each limit, and what are their defaults?
8. If a pod fails its readiness check, does Kubernetes restart the container? *(No. This is the one question every reranker missed, so expect a weaker answer.)*
9. Which index should I use on a tsvector column for full-text search, GIN or GiST?
10. Does Cache-Control: no-cache mean the browser won't store the response?

**Should say "not found"**

11. How do I make a Kubernetes CronJob run in a specific time zone?
12. Which status code should my API return when a client hits a rate limit?

All twelve are in the benchmark set, so the right answers are listed in [benchmark.jsonl](vscode-webview://1h4nfntt6k0pfscehv19lqe4cf0o7a9r4mevpg5p9u0s1vh2oas5/evals/datasets/benchmark.jsonl) if you want to check the replies.
