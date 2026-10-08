# Benchmark corpus: sources and licences

These pages are third-party documentation, kept here as a fixed test corpus for
the evaluation. **They are not covered by this repository's MIT licence**: each
folder stays under its original licence, listed below. The converted MDN pages
are shared under the same licence as the originals (CC BY-SA 2.5).

**Changes:** downloaded by `scripts/download_benchmark_docs.py` and converted to
Markdown. Navigation, templates, website-only markup and embedded code-sample
references were removed, and PostgreSQL titles were prefixed with their chapter
name. The text is otherwise unchanged.

- Kubernetes: kubernetes/website at commit `aac74ad6c5291fe88f2d7d0c4c13ccd37203984c`
- MDN: mdn/content at commit `259ae6f55fa8009dd492e010e497c8a18737523c`
- PostgreSQL: version 16 documentation
- pgvector: tag `v0.8.2`

## Kubernetes documentation

© the respective authors, [https://github.com/kubernetes/website](https://github.com/kubernetes/website). Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| File | Source |
| --- | --- |
| `kubernetes/service.md` | https://kubernetes.io/docs/concepts/services-networking/service/ |
| `kubernetes/ingress.md` | https://kubernetes.io/docs/concepts/services-networking/ingress/ |
| `kubernetes/ingress-controllers.md` | https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/ |
| `kubernetes/gateway.md` | https://kubernetes.io/docs/concepts/services-networking/gateway/ |
| `kubernetes/endpoint-slices.md` | https://kubernetes.io/docs/concepts/services-networking/endpoint-slices/ |
| `kubernetes/network-policies.md` | https://kubernetes.io/docs/concepts/services-networking/network-policies/ |
| `kubernetes/dns-pod-service.md` | https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/ |
| `kubernetes/service-traffic-policy.md` | https://kubernetes.io/docs/concepts/services-networking/service-traffic-policy/ |
| `kubernetes/topology-aware-routing.md` | https://kubernetes.io/docs/concepts/services-networking/topology-aware-routing/ |
| `kubernetes/deployment.md` | https://kubernetes.io/docs/concepts/workloads/controllers/deployment/ |
| `kubernetes/statefulset.md` | https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/ |
| `kubernetes/daemonset.md` | https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/ |
| `kubernetes/replicaset.md` | https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/ |
| `kubernetes/job.md` | https://kubernetes.io/docs/concepts/workloads/controllers/job/ |
| `kubernetes/configmap.md` | https://kubernetes.io/docs/concepts/configuration/configmap/ |
| `kubernetes/secret.md` | https://kubernetes.io/docs/concepts/configuration/secret/ |
| `kubernetes/probes.md` | https://kubernetes.io/docs/concepts/workloads/pods/probes/ |
| `kubernetes/pod-lifecycle.md` | https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/ |
| `kubernetes/configure-liveness-readiness-startup-probes.md` | https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/ |

## PostgreSQL documentation

© the respective authors, [https://www.postgresql.org/docs/](https://www.postgresql.org/docs/). Licence: [PostgreSQL License](https://www.postgresql.org/about/licence/). The licence text is in `postgresql/LICENSE`.

| File | Source |
| --- | --- |
| `postgresql/indexes-intro.md` | https://www.postgresql.org/docs/16/indexes-intro.html |
| `postgresql/indexes-types.md` | https://www.postgresql.org/docs/16/indexes-types.html |
| `postgresql/indexes-multicolumn.md` | https://www.postgresql.org/docs/16/indexes-multicolumn.html |
| `postgresql/indexes-partial.md` | https://www.postgresql.org/docs/16/indexes-partial.html |
| `postgresql/indexes-expressional.md` | https://www.postgresql.org/docs/16/indexes-expressional.html |
| `postgresql/indexes-index-only-scans.md` | https://www.postgresql.org/docs/16/indexes-index-only-scans.html |
| `postgresql/brin-intro.md` | https://www.postgresql.org/docs/16/brin-intro.html |
| `postgresql/textsearch-intro.md` | https://www.postgresql.org/docs/16/textsearch-intro.html |
| `postgresql/textsearch-controls.md` | https://www.postgresql.org/docs/16/textsearch-controls.html |
| `postgresql/textsearch-tables.md` | https://www.postgresql.org/docs/16/textsearch-tables.html |
| `postgresql/textsearch-indexes.md` | https://www.postgresql.org/docs/16/textsearch-indexes.html |
| `postgresql/routine-vacuuming.md` | https://www.postgresql.org/docs/16/routine-vacuuming.html |
| `postgresql/runtime-config-autovacuum.md` | https://www.postgresql.org/docs/16/runtime-config-autovacuum.html |
| `postgresql/mvcc-intro.md` | https://www.postgresql.org/docs/16/mvcc-intro.html |
| `postgresql/transaction-iso.md` | https://www.postgresql.org/docs/16/transaction-iso.html |
| `postgresql/using-explain.md` | https://www.postgresql.org/docs/16/using-explain.html |
| `postgresql/runtime-config-resource.md` | https://www.postgresql.org/docs/16/runtime-config-resource.html |

## MDN Web Docs

© the respective authors, [https://github.com/mdn/content](https://github.com/mdn/content). Licence: [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/).

| File | Source |
| --- | --- |
| `mdn-http/caching.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/guides/caching |
| `mdn-http/conditional_requests.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/guides/conditional_requests |
| `mdn-http/cache-control.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/reference/headers/cache-control |
| `mdn-http/etag.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/reference/headers/etag |
| `mdn-http/last-modified.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/reference/headers/last-modified |
| `mdn-http/vary.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/reference/headers/vary |
| `mdn-http/if-none-match.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/reference/headers/if-none-match |
| `mdn-http/if-modified-since.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/reference/headers/if-modified-since |
| `mdn-http/304.md` | https://developer.mozilla.org/en-US/docs/Web/HTTP/reference/status/304 |

## pgvector README

© the respective authors, [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector). Licence: [PostgreSQL License](https://github.com/pgvector/pgvector/blob/master/LICENSE). The licence text is in `pgvector/LICENSE`.

| File | Source |
| --- | --- |
| `pgvector/pgvector-readme.md` | https://github.com/pgvector/pgvector/blob/v0.8.2/README.md |
