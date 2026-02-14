-- @name: session-summary
-- @description: Embedding-relative session summaries via HDBSCAN clustering
-- @target: _enrich_session_summary
CREATE TABLE IF NOT EXISTS _enrich_session_summary (
    source_id TEXT PRIMARY KEY,
    topic_clusters TEXT,      -- JSON: [{"label": "auth.py + router.ts", "pct": 65.2, "count": 45}, ...]
    community_label TEXT,     -- from titled hub sessions in same community
    topic_summary TEXT        -- composed one-liner (named topic_summary to avoid _raw_sources.summary collision)
);
