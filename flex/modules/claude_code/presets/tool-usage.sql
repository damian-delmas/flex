-- @name: tool-usage
-- @description: Tool usage breakdown across sessions

SELECT
    tool_name as tool,
    COUNT(*) as total,
    COUNT(DISTINCT session_id) as sessions,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT session_id), 1) as avg_per_session
FROM messages
WHERE tool_name IS NOT NULL
GROUP BY tool_name
ORDER BY total DESC;
