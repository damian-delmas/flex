-- @name: reading-list
-- @description: Prioritized reading order for a topic — hubs first, then bridges
-- @params: concept (required)

SELECT src.source_id, src.title,
    CASE WHEN g.is_hub = 1 THEN 'hub'
         WHEN g.is_bridge = 1 THEN 'bridge'
         ELSE 'related' END as category,
    ROUND(g.centrality, 4) as centrality,
    g.community_id
FROM _raw_sources src
JOIN _enrich_source_graph g ON src.source_id = g.source_id
WHERE src.source_id IN (
    SELECT DISTINCT e.source_id
    FROM _raw_chunks c
    JOIN _edges_source e ON c.id = e.chunk_id
    WHERE c.content LIKE '%' || :concept || '%'
)
ORDER BY
    CASE WHEN g.is_hub = 1 THEN 0
         WHEN g.is_bridge = 1 THEN 1
         ELSE 2 END,
    g.centrality DESC
LIMIT 15;
