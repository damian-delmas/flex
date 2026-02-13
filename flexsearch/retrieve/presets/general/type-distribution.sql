-- @name: type-distribution
-- @description: Distribution of chunk classifications
-- @multi: true

-- @query: by_kind
SELECT semantic_role as kind, COUNT(*) as n
FROM _enrich_types
GROUP BY semantic_role
ORDER BY n DESC;

-- @query: by_pipeline
SELECT name as pipeline_table,
    (SELECT COUNT(*) FROM sqlite_master sm2 WHERE sm2.name = name) as exists_flag
FROM sqlite_master WHERE type='table' AND name LIKE '_types_%';
