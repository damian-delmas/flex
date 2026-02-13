-- @name: introspect
-- @description: Full cell orientation — shape, schema, graph intelligence, presets, samples
-- @multi: true

-- @query: about
SELECT value as description FROM _meta WHERE key = 'description';

-- @query: shape
SELECT 'chunks' as what, COUNT(*) as n FROM _raw_chunks
UNION ALL
SELECT 'sources', COUNT(*) FROM _raw_sources;

-- @query: tables
SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%fts%' AND name NOT LIKE '_qmem%' ORDER BY name;

-- @query: views
SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;

-- @query: pipeline
SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '_types_%';

-- @query: edge_tables
SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '_edges_%' ORDER BY name;

-- @query: enrich_tables
SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '_enrich_%' ORDER BY name;

-- @query: hubs
SELECT g.source_id, src.title,
    ROUND(g.centrality, 4) as centrality, g.community_id
FROM _enrich_source_graph g
JOIN _raw_sources src ON g.source_id = src.source_id
WHERE g.is_hub = 1
ORDER BY g.centrality DESC LIMIT 10;

-- @query: communities
SELECT g.community_id, COUNT(*) as sources
FROM _enrich_source_graph g
GROUP BY g.community_id ORDER BY sources DESC LIMIT 8;

-- @query: presets
SELECT name, description FROM _presets ORDER BY name;

-- @query: sample
SELECT substr(content, 1, 150) as preview FROM _raw_chunks
WHERE length(content) > 100 ORDER BY RANDOM() LIMIT 3;
