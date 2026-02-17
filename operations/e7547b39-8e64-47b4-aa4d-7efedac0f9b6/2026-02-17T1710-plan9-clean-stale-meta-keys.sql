-- @name: plan9-clean-stale-meta-keys
-- @description: Plan 9 Task 4 — Replace stale retrieval:phase keys referencing dead tokens.
-- @target: _meta
-- Applied to all 7 cells. Old keys referenced community:N, kind:TYPE, detect_communities.

DELETE FROM _meta WHERE key LIKE 'retrieval:phase%';
INSERT OR REPLACE INTO _meta (key, value) VALUES ('retrieval:phase1', 'SQL PRE-FILTER (4th arg to vec_ops): Any SQL returning chunk_ids. Restricts which chunks enter the landscape.');
INSERT OR REPLACE INTO _meta (key, value) VALUES ('retrieval:phase2', 'LANDSCAPE (numpy on filtered N): diverse, recent[:N], unlike:TEXT, like:id1,id2, from:TEXT to:TEXT');
INSERT OR REPLACE INTO _meta (key, value) VALUES ('retrieval:phase3', 'ENRICH (query-time topology on K candidates): local_communities → _community column (per-query Louvain)');
