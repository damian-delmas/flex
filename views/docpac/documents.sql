-- @name: documents
-- @description: Source-level surface for doc-pac cells. Document metadata with graph intelligence.

DROP VIEW IF EXISTS documents;
CREATE VIEW documents AS
SELECT
    src.source_id,
    src.title,
    src.file_date,
    src.temporal,
    src.doc_type,
    COUNT(DISTINCT s.chunk_id) as chunk_count,
    g.centrality,
    g.is_hub,
    g.is_bridge,
    g.community_id
FROM _raw_sources src
LEFT JOIN _edges_source s ON src.source_id = s.source_id
LEFT JOIN _enrich_source_graph g ON src.source_id = g.source_id
GROUP BY src.source_id;
