-- @name: session-files
-- @description: All files touched in a session with operations (SOMA-aware dedup)
-- @params: session (required)

SELECT
    COALESCE(fi.file_uuid, t.target_file) as file_key,
    t.target_file,
    t.tool_name,
    COUNT(*) as ops,
    MIN(tm.chunk_number) as first_touch,
    MAX(tm.chunk_number) as last_touch
FROM _edges_tool_ops t
JOIN _edges_source es ON t.chunk_id = es.chunk_id
JOIN _types_message tm ON t.chunk_id = tm.chunk_id
LEFT JOIN _edges_file_identity fi ON t.chunk_id = fi.chunk_id
WHERE es.source_id LIKE '%' || :session || '%'
  AND t.target_file IS NOT NULL
GROUP BY COALESCE(fi.file_uuid, t.target_file), t.tool_name
ORDER BY first_touch;
