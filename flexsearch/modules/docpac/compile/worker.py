"""
Docpac incremental index worker — single-file upsert into chunk-atom cells.

Drains the `pending` table in ~/.flex/queue.db.
Each row is a file path written by the flex-index.sh PostToolUse hook.

Pipeline per file:
  resolve_cell_for_path → parse_docpac_file → frontmatter → normalize →
  split_sections → embed → upsert (delete old + re-insert) → mean-pool source

No graph rebuild — that stays expensive and explicit.
"""

import hashlib
import sqlite3
import sys
import time

import numpy as np
from pathlib import Path

from flexsearch.registry import resolve_cell_for_path, FLEX_HOME
from flexsearch.modules.docpac.compile.docpac import parse_docpac_file
from flexsearch.compile.markdown import normalize_headers, extract_frontmatter, split_sections

QUEUE_DB = FLEX_HOME / "queue.db"


def make_source_id(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def make_chunk_id(source_id: str, position: int) -> str:
    return f"{source_id}:{position}"


def _find_context_root(file_path: str) -> Path | None:
    """Walk up from file to find the context/ root directory."""
    p = Path(file_path)
    for parent in p.parents:
        if parent.name == 'context':
            return parent
    return None


def _embed_texts(texts: list[str], embed_fn) -> list[bytes | None]:
    """Embed texts using the shared ONNX embedder. Returns list of blobs."""
    if not texts:
        return []
    try:
        vecs = embed_fn(texts)
        if hasattr(vecs, 'shape') and len(vecs.shape) == 2:
            return [v.astype(np.float32).tobytes() for v in vecs]
        return [vecs.astype(np.float32).tobytes()]
    except Exception as e:
        print(f"[docpac-worker] embed error: {e}", file=sys.stderr)
        return [None] * len(texts)


def index_file(conn: sqlite3.Connection, file_path: str, embed_fn) -> bool:
    """Index a single markdown file into its docpac cell.

    Upsert semantics: delete old chunks for this source, re-insert.
    """
    p = Path(file_path)
    if not p.exists():
        return False

    context_root = _find_context_root(file_path)
    if not context_root:
        return False

    entry = parse_docpac_file(file_path, str(context_root))
    if entry.skip:
        return False

    try:
        content = p.read_text(encoding='utf-8')
    except (UnicodeDecodeError, FileNotFoundError):
        return False

    source_id = make_source_id(file_path)
    frontmatter, body = extract_frontmatter(content)
    normalized = normalize_headers(body)
    sections = split_sections(normalized, level=2)
    if not sections:
        sections = [('', body.strip(), 0)]

    # Embed all section texts
    section_texts = [s[1] for s in sections]
    embeddings = _embed_texts(section_texts, embed_fn)

    # --- Upsert: delete old data for this source ---
    old_chunk_ids = [r[0] for r in conn.execute(
        "SELECT chunk_id FROM _edges_source WHERE source_id = ?",
        (source_id,)
    ).fetchall()]

    if old_chunk_ids:
        ph = ','.join('?' * len(old_chunk_ids))
        conn.execute(f"DELETE FROM _raw_chunks WHERE id IN ({ph})", old_chunk_ids)
        conn.execute(f"DELETE FROM _types_docpac WHERE chunk_id IN ({ph})", old_chunk_ids)
        # _enrich_types may not exist in all cells
        try:
            conn.execute(f"DELETE FROM _enrich_types WHERE chunk_id IN ({ph})", old_chunk_ids)
        except sqlite3.OperationalError:
            pass
        conn.execute("DELETE FROM _edges_source WHERE source_id = ?", (source_id,))

    # --- Insert source ---
    conn.execute("""
        INSERT OR REPLACE INTO _raw_sources
        (source_id, file_date, temporal, doc_type, title, source_path,
         type, status, keywords)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        source_id,
        entry.file_date,
        entry.temporal,
        entry.doc_type,
        entry.title,
        file_path,
        frontmatter.get('type'),
        frontmatter.get('status'),
        ','.join(frontmatter.get('keywords', [])) if isinstance(frontmatter.get('keywords'), list)
            else frontmatter.get('keywords'),
    ))

    # --- Insert chunks + edges + types ---
    for section_title, section_content, position in sections:
        chunk_id = make_chunk_id(source_id, position)
        emb = embeddings[position] if position < len(embeddings) else None

        conn.execute("""
            INSERT OR REPLACE INTO _raw_chunks (id, content, embedding, timestamp)
            VALUES (?, ?, ?, ?)
        """, (chunk_id, section_content, emb, None))

        conn.execute("""
            INSERT OR REPLACE INTO _edges_source
            (chunk_id, source_id, source_type, position)
            VALUES (?, ?, 'markdown', ?)
        """, (chunk_id, source_id, position))

        conn.execute("""
            INSERT OR REPLACE INTO _types_docpac
            (chunk_id, temporal, doc_type, facet, section_title,
             yaml_type, yaml_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            entry.temporal,
            entry.doc_type,
            None,
            section_title or None,
            frontmatter.get('type'),
            frontmatter.get('status'),
        ))

    # --- Mean-pool source embedding ---
    valid = [e for e in embeddings if e is not None]
    if valid:
        vecs = [np.frombuffer(e, dtype=np.float32) for e in valid]
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        conn.execute(
            "UPDATE _raw_sources SET embedding = ? WHERE source_id = ?",
            (mean_vec.tobytes(), source_id))

    return True


def process_queue(embed_fn) -> dict:
    """Drain the pending table. Returns stats dict."""
    stats = {'processed': 0, 'indexed': 0, 'skipped': 0}

    if not QUEUE_DB.exists():
        return stats

    qconn = sqlite3.connect(str(QUEUE_DB), timeout=5)
    qconn.execute("PRAGMA journal_mode=WAL")

    try:
        rows = qconn.execute("SELECT path, ts FROM pending ORDER BY ts").fetchall()
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        qconn.close()
        return stats

    if not rows:
        qconn.close()
        return stats

    # Group by cell via registry
    by_cell: dict[str, dict] = {}
    no_cell: list[str] = []

    for path, ts in rows:
        result = resolve_cell_for_path(path)
        if result is None:
            no_cell.append(path)
            continue
        cell_name, cell_path = result
        if cell_name not in by_cell:
            by_cell[cell_name] = {'db': str(cell_path), 'files': []}
        by_cell[cell_name]['files'].append(path)

    # Index files per cell
    processed_paths = list(no_cell)  # clear unknowns from queue too

    for cell_name, data in by_cell.items():
        conn = sqlite3.connect(data['db'], timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")

        for file_path in data['files']:
            try:
                if index_file(conn, file_path, embed_fn):
                    stats['indexed'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                print(f"[docpac-worker] error on {Path(file_path).name}: {e}",
                      file=sys.stderr)
                stats['skipped'] += 1
            processed_paths.append(file_path)

        conn.commit()
        conn.close()
        stats['processed'] += len(data['files'])

    # Clear processed from queue
    if processed_paths:
        ph = ','.join('?' * len(processed_paths))
        qconn.execute(f"DELETE FROM pending WHERE path IN ({ph})", processed_paths)
        qconn.commit()

    qconn.close()
    return stats
