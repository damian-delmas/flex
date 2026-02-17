"""Claude Code noise filtering — session eligibility and graph filters.

These are specific to claude-code cells where:
- Warmup sessions exist (title = 'Warmup')
- Agent children are short-lived (source_id LIKE 'agent-%')
- 2,404 of 5,774 sessions have <= 2 chunks (aborts, /mcp reconnects)
- Sessions with < 20 chunks carry 4.7% of content

Doc-pac cells do NOT use these filters — 3-8 chunk sources are normal there.
"""

# Minimum chunk count for a session to be considered "real work"
MIN_CHUNKS = 20

# Minimum message_count for session summary enrichment
MIN_MESSAGES = 5

# Orchestrator detection threshold (agents spawned)
ORCHESTRATOR_THRESHOLD = 5

# File graph: skip files touched by too many sessions (noise like .gitignore)
MAX_SESSIONS_PER_FILE = 200


def session_filter_sql():
    """WHERE clause for eligible sessions (summary, profile enrichments).

    Returns SQL that selects source_ids from _raw_sources.
    Filters: min messages, no agent children, no warmups.
    """
    return """
        SELECT source_id FROM _raw_sources
        WHERE message_count >= {min_messages}
          AND source_id NOT LIKE 'agent-%'
          AND (title IS NULL OR title != 'Warmup')
    """.format(min_messages=MIN_MESSAGES)


def graph_filter_sql():
    """WHERE fragment for build_similarity_graph().

    Pass as: build_similarity_graph(db, where=graph_filter_sql())
    Filters: min chunks, no warmups, no agent children (Plan 9).
    Unified with session_filter_sql() — same exclusion policy.
    """
    return """source_id IN (
        SELECT source_id FROM _edges_source
        GROUP BY source_id HAVING COUNT(*) >= {min_chunks}
    ) AND source_id NOT LIKE 'agent-%'
    AND source_id NOT IN (
        SELECT source_id FROM _raw_sources WHERE title = 'Warmup'
    )""".format(min_chunks=MIN_CHUNKS)
