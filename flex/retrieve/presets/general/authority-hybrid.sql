-- @name: authority-hybrid
-- @description: High-centrality content — compose with vec_ops for semantic+authority
-- @params: query (required)

SELECT s.id, s.content, s.source_id, s.timestamp
FROM sections s
WHERE s.centrality IS NOT NULL
ORDER BY s.centrality DESC
LIMIT 200
