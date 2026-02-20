-- Unified Write Path: add is_sidechain + entry_uuid to _types_message
-- These columns capture JSONL fields previously dropped during indexing.
-- is_sidechain: marks branch conversations (agent sidechains)
-- entry_uuid: Claude Code's per-entry unique ID for cross-referencing

ALTER TABLE _types_message ADD COLUMN is_sidechain INTEGER;
ALTER TABLE _types_message ADD COLUMN entry_uuid TEXT;
