-- @name: sections
-- @description: Chunk-level surface for doc-pac cells. Document sections with graph intelligence.

DROP VIEW IF EXISTS sections;
CREATE VIEW sections AS
SELECT
    r.id,
    r.content,
    r.timestamp,
    s.source_id AS doc_id,
    s.position,
    src.title AS doc_title,
    tp.doc_type,
    tp.temporal,
    tp.facet,
    tp.section_title,
    g.centrality,
    g.is_hub,
    g.is_bridge,
    g.community_id
FROM _raw_chunks r
LEFT JOIN _edges_source s ON r.id = s.chunk_id
LEFT JOIN _raw_sources src ON s.source_id = src.source_id
LEFT JOIN _types_docpac tp ON r.id = tp.chunk_id
LEFT JOIN _enrich_source_graph g ON s.source_id = g.source_id;
