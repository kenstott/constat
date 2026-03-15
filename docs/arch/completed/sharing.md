# Deep Links & Sharing Architecture

> **Status:** Planned. Not yet implemented.

## Problem

Constat has no way to share work. Sessions and domains are siloed per user. Two pain points:

1. **No shareable URLs** — A user looking at a domain or session cannot send a link to a colleague. The existing deep link system handles in-app navigation (`/db/{name}/{table}`, `/glossary/{term}`) but there are no addressable URLs for domains or sessions themselves.
2. **No access grants** — Even if a URL existed, the recipient has no permission to view the sender's session or (in the case of user-tier domains) the sender's domain. There is no mechanism to grant access to a specific person or to make something public.

## Core Idea

Two features, layered:

```
Layer 1: Deep Links        — addressable URLs for domains and sessions
Layer 2: Share Grants       — per-resource ACL: grant access to a user or to everyone
```

Layer 1 is useful alone (bookmarkable URLs). Layer 2 depends on Layer 1 for link-based sharing.

```
┌──────────────────────────────────────────────────────────┐
│                      Deep Link URL                       │
│  /domain/{filename}                                      │
│  /session/{session_id}                                   │
├──────────────────────────────────────────────────────────┤
│                    Share Grant Check                      │
│  Owner? ──────────────────────────────────── ✅ allow    │
│  Explicit grant for user_id? ─────────────── ✅ allow    │
│  Public grant exists? ────────────────────── ✅ allow    │
│  Has domain permission (existing system)? ── ✅ allow    │
│  Otherwise ───────────────────────────────── ❌ 403      │
├──────────────────────────────────────────────────────────┤
│                    Existing Auth                          │
│  Firebase JWT / Admin Token / Auth Disabled               │
└──────────────────────────────────────────────────────────┘
```

## Layer 1: Deep Links

### URL Schema

| Resource | URL Pattern | Example |
|----------|------------|---------|
| Domain | `/domain/{filename}` | `/domain/sales-analytics.yaml` |
| Session | `/session/{session_id}` | `/session/a1b2c3d4-...` |

These extend the existing deep link patterns (`/db/...`, `/doc/...`, `/glossary/...`, `/apis/...`).

### Frontend Changes

**`uiStore.ts`** — Extend `DeepLink` type:

```typescript
export interface DeepLink {
  // ... existing types ...
  type: 'table' | 'document' | 'api' | 'glossary_term' | 'domain' | 'session'
  domainFilename?: string   // For domain
  sessionId?: string        // For session
}
```

**`deepLinkToPath`** — Add cases:

```typescript
case 'domain':
  return `/domain/${encodeURIComponent(link.domainFilename!)}`
case 'session':
  return `/session/${encodeURIComponent(link.sessionId!)}`
```

**`pathToDeepLink`** — Add parsing:

```typescript
case 'domain':
  if (parts.length >= 2) return { type: 'domain', domainFilename: parts[1] }
  break
case 'session':
  if (parts.length >= 2) return { type: 'session', sessionId: parts[1] }
  break
```

**`NavigationSync`** — No change needed; existing logic already parses location and calls `applyDeepLink`.

**`applyDeepLink`** — Add handlers:

- **Domain**: Expand domain panel, scroll to and highlight the target domain node in the tree. If the domain is not in the user's active set, show a prompt: "Activate domain {name}?"
- **Session**: If `sessionId` differs from current session, call `setSession()` to switch. If the session belongs to another user (shared), load it read-only (see Layer 2).

### Backend Changes

No new endpoints needed for Layer 1. The frontend resolves deep links client-side using existing APIs:

- `GET /api/domains/{filename}` — already exists
- `GET /api/sessions/{session_id}` — already exists (currently restricted to owner)

## Layer 2: Share Grants

### Data Model

A share grant is a row in a `share_grants` table. Grants are additive — they never reduce access, only expand it.

```sql
CREATE TABLE share_grants (
    id          VARCHAR PRIMARY KEY,     -- UUID
    resource_type VARCHAR NOT NULL,      -- 'domain' | 'session'
    resource_id   VARCHAR NOT NULL,      -- domain filename or session_id
    granted_by    VARCHAR NOT NULL,      -- user_id of the sharer
    granted_to    VARCHAR,               -- user_id of recipient, NULL = public
    permission    VARCHAR NOT NULL,      -- 'view' | 'edit'
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Unique constraint: one grant per (resource, grantee, permission)
CREATE UNIQUE INDEX uq_share_grant
  ON share_grants(resource_type, resource_id, COALESCE(granted_to, '__public__'), permission);
```

**`granted_to = NULL`** means public (everyone with the link).

**Permission levels:**

| Level | Domain | Session |
|-------|--------|---------|
| `view` | See domain config, databases, glossary, skills. Cannot modify. | See session history, tables, artifacts. Cannot submit queries. |
| `edit` | Full domain access (same as owner). | Submit queries, modify facts, add resources. |

### Storage Location

Share grants live in **DataStore** (`constat/storage/datastore.py`) — the SQLAlchemy-backed store that already handles analytics tables and supports SQLite, DuckDB, and PostgreSQL. This is the right place because grants are cross-session metadata, not session-scoped data.

Add table definition to `DataStore.metadata`:

```python
share_grants = Table(
    "share_grants",
    metadata,
    Column("id", String, primary_key=True),
    Column("resource_type", String, nullable=False),
    Column("resource_id", String, nullable=False),
    Column("granted_by", String, nullable=False),
    Column("granted_to", String, nullable=True),
    Column("permission", String, nullable=False, default="view"),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("resource_type", "resource_id",
                     # COALESCE for unique constraint handled at app level
                     "granted_to", "permission", name="uq_share_grant"),
)
```

### API Endpoints

All endpoints require authentication. Only the resource owner (or platform admin) can create/revoke grants.

```
POST   /api/shares                      — Create a share grant
GET    /api/shares?resource_type=&resource_id=  — List grants for a resource
DELETE /api/shares/{grant_id}           — Revoke a grant
GET    /api/shares/mine                 — List resources shared with current user
```

#### Request/Response Models

```python
class ShareGrantCreate(BaseModel):
    resource_type: Literal["domain", "session"]
    resource_id: str                          # domain filename or session_id
    granted_to: str | None = None             # user_id or None for public
    permission: Literal["view", "edit"] = "view"

class ShareGrantResponse(BaseModel):
    id: str
    resource_type: str
    resource_id: str
    granted_by: str
    granted_to: str | None                    # None = public
    permission: str
    created_at: datetime
    # Enriched fields (resolved server-side)
    resource_name: str                        # domain name or session summary
    granted_to_email: str | None              # looked up from user_id if available
```

### Authorization Flow

Modify the existing access check points to consult share grants:

#### Sessions

**`session_manager.py`** — `get_session()` currently returns session only if `user_id` matches. Change to:

```python
async def get_session(self, session_id: str, user_id: str) -> ManagedSession | None:
    ms = self._sessions.get(session_id)
    if not ms:
        return None
    if ms.user_id == user_id:
        return ms                              # Owner — full access
    grant = await self.datastore.get_share_grant(
        resource_type="session",
        resource_id=session_id,
        user_id=user_id
    )
    if grant:
        ms._shared_permission = grant.permission  # Attach for downstream checks
        return ms
    return None                                # No access
```

For `view` grants on sessions, the API layer blocks mutation endpoints (query submission, fact creation, resource addition).

#### Domains

**`permissions.py`** — `get_user_permissions()` already resolves domain access. Extend to merge in share grants:

```python
def get_user_permissions(server_config, user_id, email) -> UserPermissions:
    perms = _resolve_config_permissions(server_config, user_id, email)
    # Merge in shared domains
    shared_domains = datastore.list_share_grants(
        resource_type="domain", user_id=user_id
    )
    for grant in shared_domains:
        if grant.resource_id not in perms.domains:
            perms.domains.append(grant.resource_id)
    return perms
```

This means shared domains appear in the user's domain list naturally — no special UI path needed.

### Frontend: Share Dialog

A `ShareDialog` component, triggered from:

1. **Domain panel** — context menu or share icon on domain tree nodes
2. **Session list** — share icon on session rows

```
┌─────────────────────────────────────────┐
│  Share "Sales Analytics"                │
│                                         │
│  🔗 Copy link                          │
│                                         │
│  ─── People with access ────────────── │
│  you (owner)                    Owner   │
│  jane@company.com               View  ✕ │
│  bob@company.com                Edit  ✕ │
│                                         │
│  ─── Add people ─────────────────────── │
│  [ email or user ID          ] [Add]    │
│  Permission: [View ▾]                   │
│                                         │
│  ─── General access ────────────────── │
│  ○ Restricted — only people above       │
│  ● Anyone with the link — can view      │
│                                         │
│  [Done]                                 │
└─────────────────────────────────────────┘
```

**Copy link** generates the deep link URL (Layer 1) and copies to clipboard.

**Add people** requires an email or user ID. The backend resolves email → user_id via Firebase Admin SDK (`auth.get_user_by_email()`).

**General access** toggle creates/removes the public grant (`granted_to = NULL`).

### Frontend: Shared-With-Me View

Add a "Shared with me" section to the session list and domain panel. Query `GET /api/shares/mine` on load.

For sessions: show shared sessions in a separate group in the session picker, badged with the owner's name and "View only" / "Can edit" indicator.

For domains: shared domains appear in the domain tree automatically (permissions merge). Add a "shared" badge on domain nodes the user doesn't own.

## Implementation Phases

### Phase 1: Deep Links (Layer 1)

1. Extend `DeepLink` type with `domain` and `session` variants
2. Update `deepLinkToPath` / `pathToDeepLink`
3. Update `applyDeepLink` to handle domain and session types
4. Add "Copy link" button to domain panel context menu and session list

**No backend changes.** Frontend-only.

### Phase 2: Share Grants — Backend

1. Add `share_grants` table to DataStore
2. Add `ShareGrantCreate` / `ShareGrantResponse` models
3. Add `/api/shares` CRUD routes
4. Add `datastore.get_share_grant()` / `datastore.list_share_grants()` methods
5. Modify `session_manager.get_session()` to check grants
6. Modify `permissions.get_user_permissions()` to merge shared domains
7. Add read-only enforcement for `view` grants (block mutation endpoints)

### Phase 3: Share Grants — Frontend

1. Build `ShareDialog` component
2. Wire share button into DomainPanel and session list
3. Add "Shared with me" sections
4. Add "View only" badges and mutation blocking in UI
5. Add "Copy link" functionality (clipboard API)

### Phase 4: Email Resolution & Notifications (optional)

1. Resolve email → user_id via Firebase Admin SDK
2. Send email notification when a resource is shared (via SendGrid, SES, or similar)
3. In-app notification banner: "Jane shared Sales Analytics with you"

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Share a session that expires (timeout) | Grant persists. Session can be rehydrated if data is still available. If not, show "Session expired" message. |
| Revoke grant while user has session open | Next API call returns 403. Frontend shows "Access revoked" and disconnects. |
| Share a user-tier domain then delete it | Grant becomes dangling. `GET /api/shares/mine` filters out grants for deleted resources. |
| User has both config-based and grant-based domain access | Additive. Config permissions are not reduced by grants. |
| Auth disabled mode | All resources accessible. Share UI hidden (no user identity to grant to). |
| Owner shares with `edit`, then owner deletes the session | Cascade delete grants for that resource. |
| Public session link opened by unauthenticated user | Requires login first (existing auth gate). After login, grant check passes. |

## Security

- **Grants are additive only** — a grant never reduces existing permissions.
- **Owner-only grant management** — only the resource owner or platform admin can create/revoke grants.
- **No token-based anonymous access** — public grants still require authentication. True anonymous access (no login) is out of scope.
- **Audit trail** — `granted_by` and `created_at` fields provide basic audit. Extend with a `share_audit_log` table if needed.
- **Rate limiting** — share creation should be rate-limited to prevent abuse (e.g., 100 grants per hour per user).

## Files to Modify

| File | Change |
|------|--------|
| `constat-ui/src/store/uiStore.ts` | Extend `DeepLink`, `deepLinkToPath`, `pathToDeepLink`, `applyDeepLink` |
| `constat-ui/src/components/artifacts/DomainPanel.tsx` | Share button, shared badge |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Handle `domain` and `session` deep links |
| `constat-ui/src/components/ShareDialog.tsx` | New component |
| `constat-ui/src/api/sessions.ts` | `createShareGrant`, `listShareGrants`, `revokeShareGrant`, `listSharedWithMe` |
| `constat/storage/datastore.py` | `share_grants` table, CRUD methods |
| `constat/server/routes/shares.py` | New router: `/api/shares` CRUD |
| `constat/server/app.py` | Register shares router |
| `constat/server/models.py` | `ShareGrantCreate`, `ShareGrantResponse` |
| `constat/server/session_manager.py` | Grant check in `get_session()` |
| `constat/server/permissions.py` | Merge shared domains into user permissions |