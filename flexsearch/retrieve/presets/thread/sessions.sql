-- @name: sessions
-- @description: List recent sessions
-- @params: limit (default: 15)

SELECT
    substr(source_id, 1, 8) as session,
    title,
    datetime(start_time, 'unixepoch', 'localtime') as started,
    message_count as ops,
    CASE WHEN is_hub = 1 THEN '*' ELSE '' END as hub
FROM sessions
ORDER BY start_time DESC
LIMIT :limit
