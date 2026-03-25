# User Source Encryption at Rest

## Goal

Optional per-source encryption for user data sources. Chunk content is encrypted at rest with a user passkey. Derived data (entities, glossary, clusters) is regenerated from decrypted content each session — never persisted to disk.

Depends on: [local-cache.md](local-cache.md) (client-side entity cache provides instant glossary restore while NER re-runs on decrypted content).

## Eligible Source Types

- Email (IMAP/OAuth)
- SharePoint
- Cloud drives (Google Drive, OneDrive, etc.)
- Attached storage (uploaded files)

Schema sources (databases, APIs) are not encrypted — they contain structural metadata, not personal content.

## Encryption Scope

**Encrypted at rest (AES-256-GCM)**:

- `embeddings.content` — chunk text stored as ciphertext
- Loaded document text (in-memory only while session active)
- Attachment blobs

**NOT encrypted**:

- Embedding vectors (float arrays, not reversible to text) — vector search works without the passkey
- Chunk metadata (`document_name`, `section`, `chunk_index`) — needed for navigation
- Entity names, glossary terms, relationships, clusters — derived data, regenerated each session from decrypted chunks, deleted on teardown

## Key Management

User passkey with PBKDF2 key derivation. Browser password manager stores the passkey.

```
User enables encryption for a source
  -> Prompted for passkey
  -> Browser prompts to save (autocomplete="current-password")
  -> Server derives AES-256-GCM key via PBKDF2 + per-user salt (stored in DB)
  -> Key lives in ManagedSession memory only — never on disk
  -> On session eviction or restart -> key gone
  -> On reconnect, browser auto-fills passkey (or user re-enters)
```

- Salt: random, per-user, stored in DB
- Key: never written to disk or database
- Browser password manager provides device-side storage via standard web platform

## Session Lifecycle

**At rest (disk)**: chunk content is ciphertext. Vectors and metadata in the clear.

**In memory (active session)**: content decrypted. All pipelines work normally — NER, FTS, BM25, clustering, preview.

**On session teardown**: delete derived data for encrypted sources. Only ciphertext persists.

## NER for Encrypted Sources

- NER scope cache is **not stored** for encrypted sources (no `ner_cached_entities` rows)
- On reconnect: decrypt chunks -> run NER -> generate entities
- Client-side cache (local-cache.md) provides instant glossary restore — NER runs in background, patches arrive via JSON Patch diff mechanism
- Slightly slower background sync vs unencrypted sources (NER re-run) — not user-visible thanks to client cache

## Threat Model

**Protects against**:

- Database file stolen from disk: chunk content unreadable
- Server admin reading stored data: ciphertext only
- Other users on same server: their sessions don't have the key

**Does NOT protect against**:

- Server process memory dump while session is active
- Compromised server code (could exfiltrate key in RAM)

## UX Flow

1. User adds email/document source. Toggle "Encrypt content".
2. Prompted for passkey. Browser offers to save it.
3. On reconnect, if encrypted sources exist, passkey field shown. Browser auto-fills if saved.
4. Without passkey: vector search still works (embeddings unencrypted), but previews show `[encrypted]` and FTS disabled for encrypted sources.

## File Changes

| File | Change |
|------|--------|
| `constat/core/config.py` | `DocumentConfig.encrypted: bool` flag |
| `constat/server/encryption.py` | **New** — PBKDF2 key derivation, AES-256-GCM encrypt/decrypt |
| `constat/server/session_manager.py` | Derived key in `ManagedSession` memory, passkey prompt on reconnect |
| `constat/storage/duckdb_backend.py` | Encrypt/decrypt `embeddings.content` for encrypted sources |
| `constat/storage/relational.py` | Skip NER scope cache for encrypted-source entities |
| `constat/discovery/doc_tools/_access.py` | Decrypt content for preview |
| `constat-ui/src/components/` | Encryption toggle on source add, passkey prompt on reconnect |

## Future

- Client-side chunk cache: cache decrypted chunks in IndexedDB for offline preview and client-side FTS
- Deprecate server-side NER scope cache once client-side caching (local-cache.md) is proven

## Status

Design complete. Depends on local-cache.md being implemented first.

## Non-Goals

- Key escrow or recovery (passkey lost = data lost)
- Encrypting embedding vectors
- Encrypting derived data (entities, glossary, clusters)
- Protection against active server compromise
