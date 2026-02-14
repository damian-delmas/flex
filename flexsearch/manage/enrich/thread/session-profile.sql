-- @name: session-profile
-- @description: Structured session columns + keyword exhaust for signal-dense search
-- @target: _enrich_session_profile
-- @depends: _enrich_session_summary (must run enrich_session_summary.py first)
CREATE TABLE IF NOT EXISTS _enrich_session_profile (
    source_id TEXT PRIMARY KEY,
    session_shape TEXT,        -- planning | implementation | debugging | exploration | delegation | conversation
    dominant_tool TEXT,        -- most frequent tool_name
    tool_signature TEXT,       -- top 3 tools: "Read>Edit>Bash"
    file_count INTEGER,        -- distinct files touched
    files_touched TEXT,        -- "vec_search.py core.py mcp_server.py" (top 10 basenames, space-separated)
    keyword_exhaust TEXT       -- signal-dense text blob (embeddable, LIKE-searchable)
);
-- keyword_exhaust format (pipe-delimited):
--   files | centroid_keywords | tool_signature | shape | first_prompt | community
--
-- Retrieval layers:
--   1. SQL WHERE on columns: session_shape, dominant_tool, file_count
--   2. LIKE on keyword_exhaust / files_touched for narrowing
--   3. vec_search on _raw_sources (after re-embedding from keyword_exhaust)
