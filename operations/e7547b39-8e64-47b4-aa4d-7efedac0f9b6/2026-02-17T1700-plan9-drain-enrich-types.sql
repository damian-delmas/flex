-- @name: plan9-drain-enrich-types
-- @description: Plan 9 Task 1 — Drain stale heuristic _enrich_types rows.
-- @target: _enrich_types
-- Worker stopped writing (classify_chunk removed). Table kept as reserved slot.

DELETE FROM _enrich_types;
