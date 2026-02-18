-- @name: keyword-hybrid
-- @description: BM25 keyword match — use as SQL pre-filter for vec_ops or standalone
-- @params: query (required)

SELECT m.id, m.content, m.source_id, m.timestamp
FROM chunks_fts
JOIN messages m ON chunks_fts.rowid = m.rowid
WHERE chunks_fts MATCH :query
ORDER BY bm25(chunks_fts)
LIMIT 200
