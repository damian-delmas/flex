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
# Lowered from 5 → 2 to match Claude's own session filtering (user_message_count >= 2).
# message_count is raw JSONL count (includes tool calls), so >= 2 still excludes
# empty/abort sessions (0-1 messages) while capturing short but real sessions.
MIN_MESSAGES = 2

# Orchestrator detection threshold (agents spawned)
ORCHESTRATOR_THRESHOLD = 5

# File graph: skip files touched by too many sessions (noise like .gitignore)
MAX_SESSIONS_PER_FILE = 200

# Infrastructure paths that appear across virtually all sessions.
# These carry no project signal — they pollute project attribution votes,
# file co-edit graphs, and source embedding pooling.
INFRA_PATH_PATTERNS = [
    '/.nexus/',        # nexus knowledge injection (reads every session)
    '/.claude/hooks/', # Claude Code hook scripts
]

# Repo-level equivalents for _enrich_repo_identity.repo_path comparisons.
# No trailing slash — repo paths are directory roots.
INFRA_REPO_PATH_PATTERNS = [
    '/.nexus',
    '/.claude',
]


def infra_repo_exclude_sql(col='eri.repo_path'):
    """SQL AND-fragment to exclude infrastructure repo paths from attribution queries.

    Usage — append to WHERE clauses joining _enrich_repo_identity:
        WHERE eri.project IS NOT NULL
          AND {infra_repo_exclude_sql()}
    """
    clauses = [f"{col} NOT LIKE '%{p}%'" for p in INFRA_REPO_PATH_PATTERNS]
    return ' AND '.join(clauses)


def infra_file_exclude_sql(col='t.target_file'):
    """SQL AND-fragment to exclude infrastructure file paths from tool_op queries.

    Usage — append to any WHERE clause joining _edges_tool_ops:
        WHERE rs.git_root IS NULL
          AND {infra_file_exclude_sql()}
    """
    clauses = [f"{col} NOT LIKE '%{p}%'" for p in INFRA_PATH_PATTERNS]
    return ' AND '.join(clauses)


def session_filter_sql():
    """WHERE clause for eligible sessions (summary, profile enrichments).

    Returns SQL that selects source_ids from _raw_sources.
    Filters: min messages, no agent children, no warmups.
    """
    return """
        SELECT source_id FROM _raw_sources
        WHERE message_count >= {min_messages}
          AND source_id NOT LIKE 'agent-%'
          AND source_id NOT IN (
              SELECT source_id FROM _types_source_warmup WHERE is_warmup_only = 1
          )
    """.format(min_messages=MIN_MESSAGES)


def graph_filter_sql():
    """WHERE fragment for build_similarity_graph().

    Pass as: build_similarity_graph(db, where=graph_filter_sql())
    Filters: min chunks, no warmups (_types_source_warmup), no agent children.
    Unified with session_filter_sql() — same exclusion policy.
    """
    return """source_id IN (
        SELECT source_id FROM _edges_source
        GROUP BY source_id HAVING COUNT(*) >= {min_chunks}
    ) AND source_id NOT LIKE 'agent-%'
    AND source_id NOT IN (
        SELECT source_id FROM _types_source_warmup WHERE is_warmup_only = 1
    )""".format(min_chunks=MIN_CHUNKS)
