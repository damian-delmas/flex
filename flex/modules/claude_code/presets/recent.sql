-- @name: recent
-- @description: Recent file operations
-- @params: limit (default: 20)

SELECT
    tool_name,
    substr(session_id, 1, 8) as session,
    target_file,
    datetime(timestamp, 'unixepoch', 'localtime') as ts
FROM messages
WHERE tool_name IN ('Write', 'Edit', 'Read', 'MultiEdit', 'Glob', 'Grep')
  AND target_file IS NOT NULL
ORDER BY timestamp DESC
LIMIT :limit
