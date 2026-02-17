-- cell: inventory
-- date: 2026-02-17T1540
-- reason: _enrich_types was drained (Plan 9) and write path stopped. Table kept as
--         "reserved slot" but empty table pollutes sqlite_master — schema-reading agents
--         discover semantic_role/confidence columns on a permanently empty table.
--         Curated views never referenced it. Drop clean; recreate when real semantic
--         classification lands.

DROP TABLE IF EXISTS _enrich_types;
