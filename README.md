# Flex

you've built more than you remember.

```
pip install getflex
flex init
```

Every session you've had with Claude Code, compiled into one SQLite file. Wires Claude Code automatically. Gives you an MCP endpoint for your agents including claude.ai.

---

## What you get

```
Indexing 1,540 sessions

Scanning sessions    ✓
Building vectors     ✓  62,172 / 62,172 chunks
Building graph       ✓

1,780 sessions · 62,172 chunks

╭──────────────────────────────────────────────────────────────────╮
│  Claude Code    ready  (reopen to activate)                      │
│  MCP Server     https://fb84d659.getflex.dev/sse                 │
╰──────────────────────────────────────────────────────────────────╯
```

**Claude Code** — wired automatically. Reopen Claude Code to activate.

**claude.ai** — paste the SSE URL into Settings → MCP. Done.

---

## How it works

```
Claude Code tool use → hooks → queue.db → worker → cell.db → MCP → Claude
```

- **Hooks** capture every tool use and user prompt in real time
- **Worker daemon** indexes them into a local SQLite cell with 128-dim embeddings
- **MCP server** exposes the cell to Claude as a SQL endpoint — no SDK, no API client
- **Enrichment** runs every 30 minutes: graph, fingerprints, file lineage

All data lives at `~/.flex/`. Nothing leaves your machine.

---

## Asking Claude

Once wired, just ask:

> "why did we drop the Redis cache layer?"

> "tell me the lineage of chain.py"

> "what did I decide about auth last month?"

Claude reads the schema, writes the queries, and answers from your actual sessions.

---

## CLI

```bash
flex search "@orient"        # explore the cell schema
flex search "@health"        # check pipeline freshness
flex search "SELECT COUNT(*) FROM sessions"
```

---

## License

MIT
