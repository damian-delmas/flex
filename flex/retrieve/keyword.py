"""FTS5 keyword materializer — peer primitive to vec_ops.

AI writes:  FROM keyword('search term') k
Becomes:    FROM _kw_results_xxxx k  (temp table with id, rank, snippet)
"""

import json
import re
import sqlite3
import uuid


def materialize_keyword(db, sql: str) -> str:
    """Transparently materialize keyword() as a temp table.

    Returns original SQL unchanged if no keyword() table source found.
    Returns JSON error string on failure.
    """
    # Find keyword(...) call
    start = re.search(r'keyword\s*\(', sql)
    if not start:
        return sql

    # Only materialize when used as a table source (FROM/JOIN position)
    before = sql[:start.start()].rstrip().upper()
    if not (before.endswith('FROM') or before.endswith('JOIN') or before.endswith(',')):
        return sql

    # Balanced-paren extraction (handles quoted strings with escaped '' quotes)
    paren_start = start.end() - 1
    depth = 0
    in_quote = False
    end_pos = None
    i = paren_start
    while i < len(sql):
        c = sql[i]
        if in_quote:
            if c == "'":
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2
                    continue
                else:
                    in_quote = False
        else:
            if c == "'":
                in_quote = True
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break
        i += 1
    if end_pos is None:
        return sql

    # Extract args from keyword('term', 'modifiers')
    inner = sql[paren_start + 1:end_pos - 1].strip()
    args = _split_args(inner)
    if not args:
        return json.dumps({"error": "keyword() requires a non-empty search term"})

    term = args[0].strip()
    # Strip surrounding quotes
    if len(term) >= 2 and term[0] == "'" and term[-1] == "'":
        term = term[1:-1].replace("''", "'")

    if not term or not term.strip():
        return json.dumps({"error": "keyword() requires a non-empty search term"})

    # Parse modifiers (2nd arg)
    limit = 200
    if len(args) > 1:
        mod_str = args[1].strip()
        if len(mod_str) >= 2 and mod_str[0] == "'" and mod_str[-1] == "'":
            mod_str = mod_str[1:-1]
        m = re.search(r'limit:(\d+)', mod_str)
        if m:
            limit = int(m.group(1)) or 200

    # Execute FTS5 query — raw-first, quote-on-error fallback
    fts_sql = (
        "SELECT c.id, "
        "  -bm25(chunks_fts) as rank, "
        "  snippet(chunks_fts, 0, '>>>', '<<<', '...', 30) as snippet "
        "FROM chunks_fts "
        "JOIN _raw_chunks c ON chunks_fts.rowid = c.rowid "
        "WHERE chunks_fts MATCH ? "
        "ORDER BY bm25(chunks_fts) "
        "LIMIT ?"
    )

    try:
        try:
            rows = db.execute(fts_sql, (term, limit)).fetchall()
        except sqlite3.OperationalError:
            # Fallback: double-quote for literal matching
            escaped = '"' + term.replace('"', '""') + '"'
            rows = db.execute(fts_sql, (escaped, limit)).fetchall()
    except Exception as e:
        return json.dumps({"error": f"keyword() search failed: {e}"})

    # Create temp table (always — even on empty results)
    tmp_name = f"_kw_results_{uuid.uuid4().hex[:8]}"
    db.execute(f"CREATE TEMP TABLE [{tmp_name}] (id TEXT PRIMARY KEY, rank REAL, snippet TEXT)")
    if rows:
        db.executemany(
            f"INSERT INTO [{tmp_name}] VALUES (?, ?, ?)",
            [(r[0], r[1], r[2]) for r in rows]
        )

    # Rewrite: replace keyword(...) with temp table name
    return sql[:start.start()] + tmp_name + sql[end_pos:]


def _split_args(inner: str) -> list[str]:
    """Split comma-separated args respecting single-quoted strings."""
    args = []
    current = []
    in_quote = False
    i = 0
    while i < len(inner):
        c = inner[i]
        if in_quote:
            current.append(c)
            if c == "'":
                if i + 1 < len(inner) and inner[i + 1] == "'":
                    current.append("'")
                    i += 2
                    continue
                else:
                    in_quote = False
        else:
            if c == "'":
                in_quote = True
                current.append(c)
            elif c == ',':
                args.append(''.join(current))
                current = []
            else:
                current.append(c)
        i += 1
    if current:
        args.append(''.join(current))
    return args
