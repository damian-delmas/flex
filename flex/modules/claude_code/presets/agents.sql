-- @name: agents
-- @description: Find spawned agents and their parents
-- @params: session (required)

SELECT
    substr(COALESCE(d.parent_source_id, substr(d.chunk_id, 1, 36)), 1, 8) as parent,
    d.agent_type,
    substr(d.child_doc_id, 1, 12) as child,
    datetime(d.created_at, 'unixepoch', 'localtime') as spawned_at
FROM _edges_delegations d
WHERE COALESCE(d.parent_source_id, substr(d.chunk_id, 1, 36)) LIKE '%' || :session || '%'
   OR d.child_doc_id LIKE '%' || :session || '%'
ORDER BY d.created_at
