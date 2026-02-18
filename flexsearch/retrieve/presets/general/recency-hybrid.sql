-- @name: recency-hybrid
-- @description: Recent content — compose with vec_ops recent:N for semantic+temporal
-- @params: query (required)

SELECT id, content, source_id, timestamp
FROM messages
WHERE content IS NOT NULL
ORDER BY timestamp DESC
LIMIT 200
