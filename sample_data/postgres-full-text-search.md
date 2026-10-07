# Full-Text Search in PostgreSQL

PostgreSQL has built-in full-text search. It finds documents that contain the *words* of a query, after normalising them, and ranks the matches. It complements vector search: keywords catch exact terms such as error codes, product names and acronyms, which embeddings often blur.

## Documents and queries

A `tsvector` is a sorted list of normalised words (lexemes) with their positions:

```sql
SELECT to_tsvector('english', 'The quick brown foxes jumped');
-- 'brown':3 'fox':4 'jump':5 'quick':2
```

Normalisation is done by a *text search configuration* such as `english`. It removes stop words like "the" and stems words, so "foxes" becomes `fox` and "jumped" becomes `jump`.

A `tsquery` is a search condition with boolean operators. The match operator `@@` tests a document against a query:

```sql
SELECT to_tsvector('english', 'Ingress routes HTTPS traffic') @@ to_tsquery('english', 'route & traffic');
-- true
```

## Building queries from user input

`to_tsquery` expects operator syntax and raises an error on malformed input. For text typed by users, use one of these instead:

- `plainto_tsquery` ANDs all the words together.
- `phraseto_tsquery` requires the words to appear in order, next to each other.
- `websearch_to_tsquery` understands search-engine syntax: `"quoted phrases"`, `or`, and `-excluded` words. It never raises a syntax error.

## Ranking

Matching only filters rows. To order them by relevance, PostgreSQL offers two ranking functions:

- `ts_rank` scores documents by how often the query terms appear.
- `ts_rank_cd` uses *cover density*: it also rewards query terms that appear close together.

Both functions accept a normalisation flag that penalises long documents. Without it, a long document mentioning a term many times can outrank a short, focused one.

Neither function uses corpus-wide statistics such as inverse document frequency. That is the main difference from BM25, the ranking function used by search engines like Elasticsearch. Extensions such as ParadeDB's `pg_search` add true BM25 ranking to PostgreSQL.

## Indexing

Computing `to_tsvector` for every row at query time is slow. The usual pattern is to store the `tsvector` in a generated column and index it with GIN:

```sql
ALTER TABLE chunks
  ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED;
CREATE INDEX ON chunks USING gin (tsv);
```

A generated column is recomputed automatically whenever `body` changes, so it can never go out of sync.

GIN (Generalized Inverted Index) maps each lexeme to the rows that contain it, the same idea as the index at the back of a book. GIN indexes are fast to search but slower to update than B-tree indexes.

## Hybrid search

Combining full-text and vector results is called hybrid search. Because `ts_rank` scores and cosine distances use different scales, they should not be added together directly. Reciprocal Rank Fusion (RRF) combines the result lists using only the *rank* of each document in each list:

```
score(d) = sum over lists of 1 / (k + rank(d))
```

The constant `k` is usually 60. Documents ranked highly by both methods rise to the top.
