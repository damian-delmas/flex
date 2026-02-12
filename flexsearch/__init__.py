"""
Flexsearch — SQL-first agentic knowledge engine.

The AI writes SQL. The schema speaks for itself.
Flexsearch provides: SQLite cells, VectorCache (matrix multiply),
meditate (graph intelligence), ONNX embeddings, FTS5 keyword search.

Protocol: 6 rules.
1. Content → _raw_chunks (id, content, embedding, timestamp)
2. Relationships → _edges_* tables (chunk_id FK)
3. Scores → _enrich_* tables (chunk_id FK, safe to wipe)
4. Views compose them into queryable surface
5. SQL is the query interface
6. _meta describes the cell
"""
