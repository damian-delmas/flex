"""Claude Code session summary — HDBSCAN config and labeling maps.

Translates claude-code tool names and semantic roles into human-readable
labels for session topic summaries. These maps are meaningless for doc-pac
cells which have no tool_name or semantic_role columns.
"""

import os
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# HDBSCAN parameters — tuned for claude-code session sizes
# ---------------------------------------------------------------------------

# Minimum chunks for HDBSCAN to attempt clustering
HDBSCAN_MIN_CHUNKS = 20

# HDBSCAN algorithm parameters
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_METRIC = 'euclidean'


# ---------------------------------------------------------------------------
# Tool name -> human label (claude-code specific)
# ---------------------------------------------------------------------------

ACTION_MAP = {
    'Bash': 'shell',
    'Read': 'reading',
    'Write': 'writing',
    'Edit': 'editing',
    'Grep': 'search',
    'Glob': 'search',
    'Task': 'delegation',
    'TodoWrite': 'planning',
    'TaskCreate': 'planning',
    'TaskUpdate': 'planning',
    'TaskOutput': 'delegation',
    'BashOutput': 'shell',
    'WebFetch': 'web research',
    'WebSearch': 'web research',
    'Skill': 'skill invocation',
    'ExitPlanMode': 'planning',
    'UserPrompt': 'conversation',
}

# Semantic role -> human label
KIND_MAP = {
    'prompt': 'conversation',
    'response': 'conversation',
    'command': 'shell',
    'read': 'reading',
    'file_operation': 'file ops',
    'search': 'search',
    'delegation': 'delegation',
    'message': 'conversation',
}


# ---------------------------------------------------------------------------
# Cluster labeling — priority cascade for naming HDBSCAN clusters
# ---------------------------------------------------------------------------

def label_cluster(chunks, cluster_indices):
    """Label a cluster from its centroid-adjacent chunks.

    Priority: file basenames > action pattern > kind pattern > content snippet.
    """
    if not cluster_indices:
        return "mixed"

    # Compute centroid
    vecs = np.array([chunks[i]['embedding'] for i in cluster_indices])
    centroid = vecs.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-10)

    # Find 5 nearest to centroid
    sims = cosine_similarity(centroid.reshape(1, -1), vecs)[0]
    top_k = min(5, len(sims))
    nearest_idx = np.argsort(sims)[-top_k:][::-1]
    nearest_chunks = [chunks[cluster_indices[i]] for i in nearest_idx]

    # Strategy 1: file basenames (deduplicated)
    files = []
    for ch in nearest_chunks:
        if ch['target_file']:
            basename = os.path.basename(ch['target_file'])
            if basename and basename not in files:
                files.append(basename)
    if files:
        return ' + '.join(files[:3])

    # Strategy 2: dominant action (mapped to human label)
    actions = [ch['action'] for ch in nearest_chunks if ch['action']]
    if actions:
        counts = Counter(actions)
        top_action = counts.most_common(1)[0][0]
        if top_action.startswith('mcp__'):
            parts = top_action.split('__')
            return parts[-1] if len(parts) > 2 else 'MCP tool'
        return ACTION_MAP.get(top_action, top_action)

    # Strategy 3: dominant kind from ALL cluster chunks
    all_cluster_chunks = [chunks[i] for i in cluster_indices]
    kinds = [ch['kind'] for ch in all_cluster_chunks if ch['kind']]
    if kinds:
        counts = Counter(kinds)
        top_kind = counts.most_common(1)[0][0]
        return KIND_MAP.get(top_kind, top_kind)

    return "mixed"


def short_session_label(chunks):
    """For sessions with <HDBSCAN_MIN_CHUNKS, build simple label."""
    files = []
    actions = []
    kinds = []
    for ch in chunks:
        if ch['target_file']:
            basename = os.path.basename(ch['target_file'])
            if basename:
                files.append(basename)
        if ch['action']:
            actions.append(ch['action'])
        if ch['kind']:
            kinds.append(ch['kind'])

    if files:
        counts = Counter(files)
        top_files = [f for f, _ in counts.most_common(3)]
        label = ' + '.join(top_files)
    elif actions:
        counts = Counter(actions)
        top_action = counts.most_common(1)[0][0]
        if top_action.startswith('mcp__'):
            parts = top_action.split('__')
            label = parts[-1] if len(parts) > 2 else 'MCP tool'
        else:
            label = ACTION_MAP.get(top_action, top_action)
    elif kinds:
        counts = Counter(kinds)
        top_kind = counts.most_common(1)[0][0]
        label = KIND_MAP.get(top_kind, top_kind)
    else:
        label = "mixed"

    return [{'label': label, 'pct': 100.0, 'count': len(chunks)}]
