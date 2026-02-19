-- @name: random-hybrid
-- @description: Random exploration — serendipitous discovery baseline
-- @params: query (required)

SELECT id, content, source_id, timestamp
FROM messages
WHERE content IS NOT NULL
ORDER BY RANDOM()
LIMIT 200
