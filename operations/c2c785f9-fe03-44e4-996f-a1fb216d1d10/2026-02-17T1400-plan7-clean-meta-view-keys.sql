-- Plan 7: Clean _meta view keys
-- View config moved from _meta to code params (regenerate_views views= arg)
-- Renames moved to curated .sql views (_views table)

DELETE FROM _meta WHERE key LIKE 'view:%';

-- Verify: must return 0
SELECT COUNT(*) as remaining_view_keys FROM _meta WHERE key LIKE 'view:%';
