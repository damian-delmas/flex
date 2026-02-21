-- @name: delegation-tree
-- @description: Recursive delegation tree from a parent session
-- @params: session (required)

WITH RECURSIVE tree AS (
    SELECT
        d.child_doc_id,
        d.agent_type,
        1 as depth
    FROM _edges_delegations d
    WHERE COALESCE(d.parent_source_id, substr(d.chunk_id, 1, 36)) LIKE '%' || :session || '%'

    UNION ALL

    SELECT
        d2.child_doc_id,
        d2.agent_type,
        t.depth + 1
    FROM _edges_delegations d2
    JOIN tree t ON COALESCE(d2.parent_source_id, substr(d2.chunk_id, 1, 36)) = t.child_doc_id
    WHERE t.depth < 5
)
SELECT child_doc_id as session, agent_type, depth
FROM tree
ORDER BY depth, child_doc_id;
