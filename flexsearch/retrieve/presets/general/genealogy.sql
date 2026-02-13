-- @name: genealogy
-- @description: Trace a concept's lineage — timeline, hubs, key excerpts
-- @params: concept (required)
-- @multi: true

-- @query: timeline
SELECT DISTINCT src.source_id, src.file_date, src.title
FROM _raw_chunks c
JOIN _edges_source e ON c.id = e.chunk_id
JOIN _raw_sources src ON e.source_id = src.source_id
WHERE c.content LIKE '%' || :concept || '%'
ORDER BY src.file_date;

-- @query: hub_sources
SELECT src.source_id, src.title,
    ROUND(g.centrality, 4) as centrality
FROM _raw_sources src
JOIN _enrich_source_graph g ON src.source_id = g.source_id
WHERE g.is_hub = 1
  AND src.source_id IN (
    SELECT DISTINCT e2.source_id
    FROM _raw_chunks c2
    JOIN _edges_source e2 ON c2.id = e2.chunk_id
    WHERE c2.content LIKE '%' || :concept || '%'
  )
ORDER BY g.centrality DESC LIMIT 5;

-- @query: excerpts
SELECT substr(c.content, 1, 300) as excerpt, e.source_id
FROM _raw_chunks c
JOIN _edges_source e ON c.id = e.chunk_id
WHERE c.content LIKE '%' || :concept || '%'
ORDER BY c.timestamp DESC
LIMIT 5;
