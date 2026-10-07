# Vector Indexing with pgvector

pgvector adds a `vector` data type and similarity search to PostgreSQL. Without an index, a similarity query compares the query vector with every row: an exact but slow sequential scan. Approximate nearest neighbour (ANN) indexes trade a little recall for much faster queries.

## Distance operators

pgvector supports several distance functions, each with its own operator:

- `<->` Euclidean (L2) distance
- `<=>` cosine distance (1 minus cosine similarity)
- `<#>` negative inner product
- `<+>` L1 (taxicab) distance

An index only speeds up queries that use the operator matching its operator class. A `vector_cosine_ops` index is used by `ORDER BY embedding <=> $1`, but not by a query ordering on `<->`.

If vectors are normalised to unit length, cosine distance, inner product and Euclidean distance produce the same ranking. Inner product is then the cheapest to compute.

## HNSW indexes

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph in which each vector is linked to its close neighbours. A search starts at the sparse top layer and greedily walks towards the query, descending layer by layer.

```sql
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Build parameters

- `m` is the maximum number of connections per node per layer. The default is 16. Higher values improve recall on high-dimensional data but use more memory.
- `ef_construction` is the size of the candidate list while building the graph. The default is 64. Higher values build a better graph but make index creation slower.

Building an HNSW index is much faster when the graph fits in `maintenance_work_mem`. Raise it for the session before building a large index.

### Query parameter

`hnsw.ef_search` controls the size of the candidate list at query time. The default is 40. Raising it improves recall and slows queries down. A query cannot return more rows than `ef_search`, so a `LIMIT` larger than `ef_search` needs a higher setting.

```sql
SET hnsw.ef_search = 100;
```

### Dimension limits

The `vector` type can be indexed with HNSW up to 2,000 dimensions. The `halfvec` type stores 16-bit floats and can be indexed up to 4,000 dimensions, which also halves storage. Larger embeddings must be shortened, for example with Matryoshka truncation, or quantised before they can be indexed.

## IVFFlat indexes

IVFFlat divides the vectors into lists (clusters) using k-means. A query searches only the lists closest to the query vector.

- The index must be created **after** the table contains data, because the clusters are learned from existing rows.
- `lists` sets the number of clusters. A common starting point is rows / 1000 for up to one million rows.
- `ivfflat.probes` sets how many lists are searched at query time. More probes give better recall and slower queries.

Compared with IVFFlat, HNSW has a better speed/recall trade-off and needs no training step, but it builds more slowly and uses more memory.

## Filtering and iterative scans

Approximate indexes apply `WHERE` filters *after* the index scan. With a selective filter, for example one tenant out of hundreds, the index may return 40 candidates of which none match, so the query returns fewer rows than requested.

pgvector 0.8 added iterative index scans. When enabled, the scan continues automatically until enough rows pass the filter:

```sql
SET hnsw.iterative_scan = relaxed_order;
```

`strict_order` keeps results exactly ordered by distance. `relaxed_order` allows slightly out-of-order results and is faster. `hnsw.max_scan_tuples` limits how far an iterative scan may go.
