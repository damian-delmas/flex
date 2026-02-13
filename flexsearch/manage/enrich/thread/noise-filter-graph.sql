-- @name: noise-filter-graph
-- @description: WHERE clause for thread graph rebuild
-- @target: _enrich_source_graph
-- Use as: build_similarity_graph(db, where=<this content>)
source_id IN (
    SELECT source_id FROM _edges_source
    GROUP BY source_id HAVING COUNT(*) >= 20
) AND source_id NOT IN (
    SELECT source_id FROM _raw_sources WHERE title = 'Warmup'
)
