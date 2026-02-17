-- @name: sessions
-- @description: Source-level surface for claude_code cells. Session metadata with graph intelligence.

DROP VIEW IF EXISTS sessions;
CREATE VIEW sessions AS
SELECT
    src.source_id,
    src.project,
    src.title,
    src.message_count,
    src.start_time,
    src.end_time,
    src.duration_minutes,
    COUNT(DISTINCT s.chunk_id) as chunk_count,
    g.centrality,
    g.is_hub,
    g.is_bridge,
    g.community_id
FROM _raw_sources src
LEFT JOIN _edges_source s ON src.source_id = s.source_id
LEFT JOIN _enrich_source_graph g ON src.source_id = g.source_id
GROUP BY src.source_id;
