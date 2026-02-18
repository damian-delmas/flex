-- @target: _enrich_session_profile
-- @description: Drop zombie table. Plan 9 specified DROP; only column pruning landed.
-- No Python write path exists. No curated view or preset references it.

DROP TABLE IF EXISTS _enrich_session_profile;
