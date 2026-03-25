# Client-Side Entity Cache with JSON Patch Diffs

## Goal

Instant glossary restore on browser start. Client caches all entity/glossary state in IndexedDB, seeds the server on connect, receives RFC 6902 JSON Patch diffs for incremental updates. No loading spinner — glossary panel renders immediately from local data, then silently merges server deltas.

## Data Format

ID-keyed JSON object (not array — enables O(n) server-side diffing). Ultra-short keys to minimize payload (~30-40% smaller than verbose keys).

```json
{
  "e": {
    "abc123": {"a": "norway", "b": "Norway", "c": "concept", "d": "GPE", "e": "hr-reporting"},
    "def456": {"a": "order", "b": "Order", "c": "concept", "d": "SCHEMA", "e": "sales-analytics"}
  },
  "g": {
    "ghi789": {"a": "order", "b": "Order", "c": "A customer purchase", "d": "reviewed", "e": null, "f": ["purchase"]}
  },
  "r": {
    "rel001": {"a": "customer", "b": "places", "c": "order", "d": 0.95}
  },
  "k": {
    "norway": ["sweden", "denmark", "finland"],
    "order": ["invoice", "purchase", "transaction"]
  }
}
```

Key maps (shared constant in both client and server):

| Top-level | Meaning |
|-----------|---------|
| `e` | entities |
| `g` | glossary |
| `r` | relationships |
| `k` | clusters |

| Entity key | Field |
|------------|-------|
| `a` | name |
| `b` | display_name |
| `c` | semantic_type |
| `d` | ner_type |
| `e` | domain_id |

| Glossary key | Field |
|--------------|-------|
| `a` | name |
| `b` | display_name |
| `c` | definition |
| `d` | status |
| `e` | parent_id |
| `f` | aliases |

| Relationship key | Field |
|------------------|-------|
| `a` | subject |
| `b` | verb |
| `c` | object |
| `d` | confidence |

## Sizing

| Scale | Entities | JSON size (short keys) | Patch size (typical heartbeat) |
|-------|----------|------------------------|-------------------------------|
| Small | 2,500 | ~300KB | ~1KB |
| Medium | 10,000 | ~1.2MB | ~3KB |
| Large | 50,000 | ~6MB | ~10KB |

IndexedDB has no practical size limit. localStorage caps at 5-10MB per origin (use IndexedDB for large sessions).

## Diff Format

RFC 6902 JSON Patch. Interoperable between Python and JS.

- Python: `python-jsonpatch` (`pip install jsonpatch`)
- JS: `fast-json-patch` (`npm install fast-json-patch`, ~3KB gzipped, 1.5M weekly downloads)

```python
# Server: compute patch
import jsonpatch
patch = jsonpatch.make_patch(client_state, server_state)
# [{"op": "add", "path": "/entities/xyz", "value": {...}}, {"op": "remove", "path": "/entities/old123"}]
```

```typescript
// Client: apply patch
import { applyPatch } from 'fast-json-patch'
const { newDocument } = applyPatch(localState, serverPatch)
```

## Flow

```
Browser start (same browser — warm cache)
  -> Load entity JSON from IndexedDB (<10ms)
  -> Render glossary panel immediately from local data
  -> Open WebSocket
  -> Send cached state to server as WS "seed" message
  -> Server diffs current state vs client seed -> JSON Patch
  -> Send patch to client
  -> Client applies patch -> glossary silently updates if anything changed
  -> Client persists updated state to IndexedDB
  -> Subsequent heartbeats: server sends patches against last ack'd state

Browser start (new device — cold cache)
  -> No IndexedDB data, glossary panel shows skeleton loader
  -> Open WebSocket
  -> Send empty seed to server
  -> Server sends full state as replace-root patch: [{"op": "replace", "path": "", "value": {...}}]
  -> Client applies, renders, persists to IndexedDB
  -> Subsequent heartbeats: normal delta patches
```

## WS Protocol

New WS message types:

```
Client -> Server: seed on connect
{"action": "entity_seed", "data": {"state": { ... }, "version": 42}}

Server -> Client: patch response
{"type": "event", "payload": {"event_type": "entity_patch", "data": {"patch": [...], "version": 43}}}
```

`version` is an incrementing counter to detect missed patches. If client version doesn't match server's expected base, server falls back to full-state replace.

## What This Replaces

- Server-side `entity_rebuild_complete` event with full entity list -> replaced by delta patches
- Server-side in-memory diff baseline in `ManagedSession` -> client seed is the baseline
- Glossary loading spinner on reconnect -> instant render from local cache

Server-side NER scope cache (`ner_cached_entities`, etc.) is retained for NER skip optimization (fingerprint matching) but no longer used as a diff baseline.

## File Changes

| File | Change |
|------|--------|
| `constat-ui/src/store/entityCache.ts` | **New** — IndexedDB read/write, JSON Patch apply, version tracking |
| `constat-ui/src/api/websocket.ts` | Send `entity_seed` on connect, handle `entity_patch` events |
| `constat-ui/src/store/glossaryStore.ts` | Load from local cache on init before WS connect |
| `constat/server/diff_generators.py` | `EntityDiffGenerator`: accept client seed, compute JSON Patch, track version per connection |
| `constat/server/routes/queries.py` | WS handler for `entity_seed` command |
| `constat/server/models.py` | `ENTITY_PATCH` event type |
| `package.json` | Add `fast-json-patch` dependency |
| `pyproject.toml` | Add `jsonpatch` dependency |

## Future

Once proven reliable, the server-side NER scope cache tables (`ner_cached_entities`, `ner_cached_chunk_entities`, `ner_cached_clusters`) can be removed entirely. The client seed becomes the sole diff baseline.

## Status

Ready to implement.
