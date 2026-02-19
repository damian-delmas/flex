-- @name: bridges
-- @description: Cross-community connector sources

SELECT src.source_id, src.title,
    g.community_id,
    ROUND(g.centrality, 4) as centrality
FROM _enrich_source_graph g
JOIN _raw_sources src ON g.source_id = src.source_id
WHERE g.is_bridge = 1
ORDER BY g.centrality DESC
LIMIT 20;
