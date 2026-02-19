-- @name: concept-diff
-- @description: Compare a concept between two time periods
-- @params: concept (required), before (required), after (required)
-- @multi: true

-- @query: old_period
SELECT COUNT(DISTINCT e.source_id) as sources
FROM _raw_chunks c
JOIN _edges_source e ON c.id = e.chunk_id
JOIN _raw_sources src ON e.source_id = src.source_id
WHERE c.content LIKE '%' || :concept || '%'
  AND src.file_date <= :before;

-- @query: new_period
SELECT COUNT(DISTINCT e.source_id) as sources
FROM _raw_chunks c
JOIN _edges_source e ON c.id = e.chunk_id
JOIN _raw_sources src ON e.source_id = src.source_id
WHERE c.content LIKE '%' || :concept || '%'
  AND src.file_date >= :after;

-- @query: new_sources
SELECT DISTINCT src.source_id, src.file_date, src.title
FROM _raw_chunks c
JOIN _edges_source e ON c.id = e.chunk_id
JOIN _raw_sources src ON e.source_id = src.source_id
WHERE c.content LIKE '%' || :concept || '%'
  AND src.file_date >= :after
ORDER BY src.file_date
LIMIT 10;
