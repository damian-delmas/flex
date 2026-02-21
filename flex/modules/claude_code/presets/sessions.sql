-- @name: sessions
-- @description: Recent sessions (excludes warmups and micro-sessions)
-- @params: limit (default: 15)

SELECT
    substr(session_id, 1, 8) as session,
    substr(COALESCE(title, ''), 1, 80) as title,
    started_at as started,
    message_count as ops,
    CASE WHEN is_hub = 1 THEN '*' ELSE '' END as hub
FROM sessions
WHERE message_count >= 5
  AND session_id NOT LIKE 'agent-%'
  AND is_warmup_only = 0
ORDER BY started_at DESC
LIMIT :limit
