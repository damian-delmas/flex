-- @name: file-hotspots
-- @description: Most-touched files across all sessions (SOMA-aware dedup)
-- @params: limit (default: 30)

SELECT
    COALESCE(fi.file_uuid, t.target_file) as file_key,
    t.target_file,
    COUNT(*) as total_ops,
    COUNT(DISTINCT es.source_id) as session_count,
    GROUP_CONCAT(DISTINCT t.tool_name) as tools
FROM _edges_tool_ops t
JOIN _edges_source es ON t.chunk_id = es.chunk_id
LEFT JOIN _edges_file_identity fi ON t.chunk_id = fi.chunk_id
WHERE t.target_file IS NOT NULL
GROUP BY COALESCE(fi.file_uuid, t.target_file)
ORDER BY total_ops DESC
LIMIT :limit;
