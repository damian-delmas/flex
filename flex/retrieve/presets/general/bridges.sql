-- @name: bridges
-- @description: Cross-community connector sessions

SELECT
    substr(session_id, 1, 8) as session,
    title,
    community_id,
    ROUND(centrality, 4) as centrality
FROM sessions
WHERE is_bridge = 1
ORDER BY centrality DESC
LIMIT 20;
