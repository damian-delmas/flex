-- @name: doc-chunks
-- @description: All chunks from a specific source, ordered by position
-- @params: source_id (required)

SELECT c.id, e.source_id, e.position,
    substr(c.content, 1, 500) as content
FROM _raw_chunks c
JOIN _edges_source e ON c.id = e.chunk_id
WHERE e.source_id LIKE '%' || :source_id || '%'
ORDER BY e.position;
