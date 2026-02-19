-- SOMA module: add old_blob_hash column to _edges_content_identity
-- Enables "what was this file before the edit?" queries
-- Safe: SQLite ALTER TABLE ADD COLUMN is non-destructive

ALTER TABLE _edges_content_identity ADD COLUMN old_blob_hash TEXT;
