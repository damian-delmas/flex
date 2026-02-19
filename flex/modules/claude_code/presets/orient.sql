-- @name: orient
-- @description: Full cell orientation — shape, schema, graph intelligence, presets, samples
-- @multi: true
-- NOTE: claude_code override — enhances hubs with topic_summary from _enrich_session_summary

-- @query: about
SELECT value as description FROM _meta WHERE key = 'description';

-- @query: shape
SELECT 'chunks' as what, COUNT(*) as n FROM _raw_chunks
UNION ALL
SELECT 'sources', COUNT(*) FROM _raw_sources;

-- @query: schema
SELECT name,
    CASE
        WHEN name LIKE '_raw_%' THEN 'raw (immutable, COMPILE)'
        WHEN name LIKE '_edges_%' THEN 'edges (relationships)'
        WHEN name LIKE '_types_%' THEN 'types (classification)'
        WHEN name LIKE '_enrich_%' THEN 'enrich (mutable, meditate)'
        WHEN name LIKE '_meta' OR name LIKE '_presets' OR name LIKE '_views' OR name LIKE '_ops' THEN 'infrastructure'
        ELSE 'other'
    END as lifecycle
FROM sqlite_master
WHERE type='table' AND name NOT LIKE '%fts%' AND name NOT LIKE '_qmem%'
ORDER BY lifecycle, name;

-- @query: views
SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;

-- @query: view_schemas
SELECT m.name as view_name, GROUP_CONCAT(p.name, ', ') as columns
FROM sqlite_master m, pragma_table_info(m.name) p
WHERE m.type = 'view'
GROUP BY m.name
ORDER BY m.name;

-- @query: hubs
SELECT g.source_id,
    COALESCE(ess.topic_summary, src.title) as label,
    ROUND(g.centrality, 4) as centrality, g.community_id
FROM _enrich_source_graph g
JOIN _raw_sources src ON g.source_id = src.source_id
LEFT JOIN _enrich_session_summary ess ON g.source_id = ess.source_id
WHERE g.is_hub = 1
ORDER BY g.centrality DESC LIMIT 10;

-- @query: communities
SELECT g.community_id, COUNT(*) as sources
FROM _enrich_source_graph g
GROUP BY g.community_id ORDER BY sources DESC LIMIT 8;

-- @query: presets
SELECT name, description, params FROM _presets ORDER BY name;

-- @query: retrieval
SELECT key, value FROM _meta WHERE key LIKE 'retrieval:%' ORDER BY key;

-- @query: sample
SELECT substr(content, 1, 150) as preview FROM _raw_chunks
WHERE length(content) > 100 ORDER BY RANDOM() LIMIT 3;
