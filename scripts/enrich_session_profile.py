"""Session Profile Enrichment — structured columns + keyword exhaust.

MUST run after enrich_session_summary.py (reads topic_clusters JSON).

SQL recipe: flexsearch/manage/enrich/thread/session-profile.sql

Pipeline per session:
  1. Bulk SQL load: tool histogram, file list, kind counts
  2. Session shape heuristic from tool/kind distribution
  3. Keyword exhaust composition from topics + files + tools + shape + prompt
  4. INSERT into _enrich_session_profile
  5. regenerate_views()

Output: _enrich_session_profile table (source_id PK -> auto-JOINs sessions view)
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FLEX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLEX_ROOT))

from flexsearch.core import open_cell
from flexsearch.views import regenerate_views

CELLS_ROOT = Path.home() / '.qmem' / 'cells' / 'projects'
THREAD_DB = CELLS_ROOT / 'thread' / 'main.db'

SESSION_FILTER = """
    SELECT source_id FROM _raw_sources
    WHERE message_count >= 5
      AND source_id NOT LIKE 'agent-%'
      AND (title IS NULL OR title != 'Warmup')
"""

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS _enrich_session_profile (
    source_id TEXT PRIMARY KEY,
    session_shape TEXT,
    dominant_tool TEXT,
    tool_signature TEXT,
    file_count INTEGER,
    files_touched TEXT,
    keyword_exhaust TEXT
)
"""

# Prefixes that indicate a non-meaningful first prompt (commands, system, etc.)
_GARBAGE_PROMPT_PREFIXES = ('/', '<', '[', '```', 'SYSTEM', 'system', 'http')


# ---------------------------------------------------------------------------
# Bulk data loading (all sessions at once)
# ---------------------------------------------------------------------------

def load_tool_histograms(db):
    """Tool usage histogram per session. Returns {source_id: Counter(tool_name -> count)}."""
    rows = db.execute("""
        SELECT e.source_id, t.tool_name, COUNT(*) as cnt
        FROM _edges_source e
        JOIN _edges_tool_ops t ON e.chunk_id = t.chunk_id
        WHERE t.tool_name IS NOT NULL
        GROUP BY e.source_id, t.tool_name
    """).fetchall()
    result = defaultdict(Counter)
    for r in rows:
        result[r['source_id']][r['tool_name']] = r['cnt']
    return dict(result)


def load_file_lists(db):
    """Distinct files touched per session. Returns {source_id: [basenames]}."""
    rows = db.execute("""
        SELECT e.source_id, t.target_file
        FROM _edges_source e
        JOIN _edges_tool_ops t ON e.chunk_id = t.chunk_id
        WHERE t.target_file IS NOT NULL
        GROUP BY e.source_id, t.target_file
    """).fetchall()
    result = defaultdict(list)
    for r in rows:
        basename = os.path.basename(r['target_file'])
        if basename:
            result[r['source_id']].append(basename)
    # Deduplicate per session
    return {sid: list(dict.fromkeys(files)) for sid, files in result.items()}


def load_kind_counts(db):
    """Kind distribution per session. Returns {source_id: Counter(kind -> count)}."""
    rows = db.execute("""
        SELECT e.source_id, et.semantic_role, COUNT(*) as cnt
        FROM _edges_source e
        JOIN _enrich_types et ON e.chunk_id = et.chunk_id
        WHERE et.semantic_role IS NOT NULL
        GROUP BY e.source_id, et.semantic_role
    """).fetchall()
    result = defaultdict(Counter)
    for r in rows:
        result[r['source_id']][r['semantic_role']] = r['cnt']
    return dict(result)


def load_first_prompts(db):
    """First meaningful user prompt per session. Returns {source_id: str}."""
    rows = db.execute("""
        SELECT e.source_id, c.content
        FROM _edges_source e
        JOIN _raw_chunks c ON e.chunk_id = c.id
        JOIN _enrich_types et ON c.id = et.chunk_id
        WHERE et.semantic_role = 'prompt'
        ORDER BY e.source_id, e.position
    """).fetchall()
    result = {}
    for r in rows:
        sid = r['source_id']
        if sid in result:
            continue  # already have first prompt
        content = (r['content'] or '').strip()
        if not content:
            continue
        # Skip garbage prompts
        if any(content.startswith(p) for p in _GARBAGE_PROMPT_PREFIXES):
            continue
        # Truncate to 80 chars at word boundary
        if len(content) > 80:
            content = content[:77].rsplit(' ', 1)[0] or content[:77]
        result[sid] = content
    return result


def load_topic_data(db):
    """Load topic_clusters and community_label from _enrich_session_summary.
    Returns {source_id: {'topic_clusters': [...], 'community_label': str}}.
    """
    rows = db.execute("""
        SELECT source_id, topic_clusters, community_label
        FROM _enrich_session_summary
    """).fetchall()
    result = {}
    for r in rows:
        tc = json.loads(r['topic_clusters']) if r['topic_clusters'] else []
        result[r['source_id']] = {
            'topic_clusters': tc,
            'community_label': r['community_label'],
        }
    return result


# ---------------------------------------------------------------------------
# Session shape heuristic
# ---------------------------------------------------------------------------

def classify_session_shape(tool_hist, kind_counts):
    """Heuristic session classification from tool/kind distribution.

    Returns one of: planning | implementation | debugging | exploration |
                    delegation | conversation
    """
    total_tools = sum(tool_hist.values()) if tool_hist else 0
    total_kinds = sum(kind_counts.values()) if kind_counts else 0

    if not total_tools and not total_kinds:
        return 'exploration'

    # Kind ratios
    prompt_count = kind_counts.get('prompt', 0) + kind_counts.get('response', 0)
    delegation_count = kind_counts.get('delegation', 0)

    if total_kinds > 0:
        prompt_ratio = prompt_count / total_kinds
        delegation_ratio = delegation_count / total_kinds
    else:
        prompt_ratio = 0
        delegation_ratio = 0

    # Conversation-heavy
    if prompt_ratio > 0.4:
        return 'conversation'

    # Delegation-heavy
    if delegation_ratio > 0.1:
        return 'delegation'

    # Tool-based classification
    if total_tools > 0:
        write_edit = tool_hist.get('Write', 0) + tool_hist.get('Edit', 0)
        read_count = tool_hist.get('Read', 0)
        grep_glob = tool_hist.get('Grep', 0) + tool_hist.get('Glob', 0)
        bash_count = tool_hist.get('Bash', 0)
        planning = (tool_hist.get('TodoWrite', 0) + tool_hist.get('TaskCreate', 0)
                     + tool_hist.get('ExitPlanMode', 0))

        bash_ratio = bash_count / total_tools

        # Implementation: more writing than reading
        if write_edit > read_count:
            return 'implementation'

        # Debugging: bash-heavy
        if bash_ratio > 0.4:
            return 'debugging'

        # Planning: explicit planning tools
        if planning > 0 and planning / total_tools > 0.1:
            return 'planning'

        # Exploration: read/search dominant
        if (read_count + grep_glob) > total_tools * 0.5:
            return 'exploration'

    return 'exploration'


# ---------------------------------------------------------------------------
# Tool signature
# ---------------------------------------------------------------------------

def make_tool_signature(tool_hist):
    """Top 3 tools by frequency, formatted as 'Read>Edit>Bash'."""
    if not tool_hist:
        return None
    top3 = tool_hist.most_common(3)
    return '>'.join(t for t, _ in top3)


# ---------------------------------------------------------------------------
# Keyword exhaust composition
# ---------------------------------------------------------------------------

def compose_keyword_exhaust(files_touched, topic_data, tool_sig, shape,
                            first_prompt, community_label):
    """Build signal-dense text blob from structured components.

    Format (pipe-delimited):
      files | centroid_keywords | tool_signature | shape | first_prompt | community
    """
    parts = []

    # 1. Files touched (top 10 basenames)
    if files_touched:
        parts.append(' '.join(files_touched[:10]))
    else:
        parts.append('')

    # 2. Centroid keywords from topic_clusters
    keywords = []
    if topic_data and topic_data.get('topic_clusters'):
        for topic in topic_data['topic_clusters']:
            kw = topic.get('keywords', [])
            keywords.extend(kw)
    # Deduplicate preserving order
    seen = set()
    unique_kw = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique_kw.append(k)
    parts.append(' '.join(unique_kw[:15]))

    # 3. Tool signature
    parts.append(tool_sig or '')

    # 4. Session shape
    parts.append(shape or '')

    # 5. First meaningful prompt
    parts.append(first_prompt or '')

    # 6. Community label
    parts.append(community_label or '')

    exhaust = ' | '.join(parts)
    return exhaust.strip() if exhaust.strip(' |') else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Session Profile Enrichment")
    print("=" * 60)

    t_start = time.time()

    db = open_cell(str(THREAD_DB))
    print(f"\nOpened: {THREAD_DB}")

    # Check dependency: _enrich_session_summary must exist
    dep_check = db.execute("""
        SELECT COUNT(*) FROM sqlite_master
        WHERE type='table' AND name='_enrich_session_summary'
    """).fetchone()[0]
    if not dep_check:
        print("ERROR: _enrich_session_summary table not found.")
        print("Run enrich_session_summary.py first.")
        sys.exit(1)

    # Eligible sessions
    eligible = db.execute(SESSION_FILTER).fetchall()
    eligible_ids = set(r['source_id'] for r in eligible)
    print(f"Eligible sessions: {len(eligible_ids)}")

    # Create table
    db.execute("DROP TABLE IF EXISTS _enrich_session_profile")
    db.execute(CREATE_TABLE)
    db.commit()
    print("Created _enrich_session_profile table")

    # Bulk data loading
    print("\nLoading bulk data...")
    t_load = time.time()
    tool_hists = load_tool_histograms(db)
    file_lists = load_file_lists(db)
    kind_counts = load_kind_counts(db)
    first_prompts = load_first_prompts(db)
    topic_data = load_topic_data(db)
    print(f"  Loaded in {time.time() - t_load:.1f}s")
    print(f"  Tool histograms: {len(tool_hists)} sessions")
    print(f"  File lists: {len(file_lists)} sessions")
    print(f"  Kind counts: {len(kind_counts)} sessions")
    print(f"  First prompts: {len(first_prompts)} sessions")
    print(f"  Topic data: {len(topic_data)} sessions")

    # Process sessions
    print("\nProcessing sessions...")
    processed = 0
    shape_dist = Counter()

    for sid in eligible_ids:
        tool_hist = tool_hists.get(sid, Counter())
        files = file_lists.get(sid, [])
        kinds = kind_counts.get(sid, Counter())
        prompt = first_prompts.get(sid)
        topics = topic_data.get(sid, {})
        comm_label = topics.get('community_label') if topics else None

        # Session shape
        shape = classify_session_shape(tool_hist, kinds)
        shape_dist[shape] += 1

        # Tool signature
        tool_sig = make_tool_signature(tool_hist)

        # Dominant tool
        dominant = tool_hist.most_common(1)[0][0] if tool_hist else None

        # File count and files_touched
        file_count = len(files)
        files_str = ' '.join(files[:10]) if files else None

        # Keyword exhaust
        exhaust = compose_keyword_exhaust(
            files, topics, tool_sig, shape, prompt, comm_label
        )

        db.execute("""
            INSERT OR REPLACE INTO _enrich_session_profile
            (source_id, session_shape, dominant_tool, tool_signature,
             file_count, files_touched, keyword_exhaust)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (sid, shape, dominant, tool_sig, file_count, files_str, exhaust))
        processed += 1

    db.commit()
    print(f"  Processed: {processed}")

    # Regenerate views
    print("\nRegenerating views...")
    regenerate_views(db)
    print("  Done")

    # Verification
    print("\nVerification:")
    total = db.execute(
        "SELECT COUNT(*) FROM _enrich_session_profile"
    ).fetchone()[0]
    print(f"  Total rows: {total}")

    print(f"\n  Shape distribution:")
    for shape, cnt in shape_dist.most_common():
        print(f"    {shape}: {cnt}")

    with_exhaust = db.execute(
        "SELECT COUNT(*) FROM _enrich_session_profile WHERE keyword_exhaust IS NOT NULL"
    ).fetchone()[0]
    print(f"\n  With keyword_exhaust: {with_exhaust}/{total}")

    # Sample keyword exhaust
    print("\n  Sample keyword_exhaust (hub sessions):")
    samples = db.execute("""
        SELECT p.source_id, p.keyword_exhaust, p.session_shape
        FROM _enrich_session_profile p
        JOIN _enrich_source_graph g ON p.source_id = g.source_id
        WHERE g.is_hub = 1
        LIMIT 5
    """).fetchall()
    for s in samples:
        sid_short = s['source_id'][:8]
        exhaust = (s['keyword_exhaust'] or '')[:120]
        print(f"    [{sid_short}] ({s['session_shape']}) {exhaust}")

    # Check sessions view columns
    print("\n  Sessions view columns:")
    cols = db.execute("PRAGMA table_info(sessions)").fetchall()
    col_names = [c[1] for c in cols]
    new_cols = [c for c in col_names if c in (
        'session_shape', 'dominant_tool', 'tool_signature',
        'file_count', 'files_touched', 'keyword_exhaust'
    )]
    print(f"    New columns present: {new_cols}")

    db.close()
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print("Restart flexsearch-mcp to pick up changes.")


if __name__ == '__main__':
    main()
