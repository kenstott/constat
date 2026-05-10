# Personal Resources (`+` Button)

## Goal

Add a `+` button in the Sources section of the artifact panel that lets users connect personal resources — email accounts, cloud drives, calendars, and SharePoint sites — directly from the UI. Connected accounts persist at the user level (`.constat/{user_id}/accounts.yaml`) and auto-load into every new session, functioning as a user-tier alternative to admin-configured document sources in domain YAML files.

## Current State

Today, adding personal resources requires one of:
1. **Admin config**: Add to domain YAML under `documents:` — requires server restart
2. **Email modal**: Click "Email" `+` button in Sources → complete OAuth flow → tokens live in browser memory only (not persisted across sessions)
3. **Document modal**: Click "Document" `+` button → add URI or upload file — session-scoped, not persisted

**Problems**:
- OAuth tokens are not persisted — user must re-authenticate every session
- No way to add calendar, drive, or SharePoint from the UI
- No concept of "my accounts" that follow the user across sessions

## UX Design

### `+` Button Location

In the Sources section header (alongside existing `+` buttons for Database, API, Document, Email):

```
Sources
├─ Databases [+]
├─ APIs [+]
├─ Documents [+]
├─ Personal [+]          ← NEW: opens resource picker
└─ Facts [+]
```

Or: a single unified `+` button at the Sources section level that opens a resource type picker.

### Resource Picker Flow

```
[+] → Resource Type Picker
      ├─ Email Inbox        → OAuth flow (existing, enhanced)
      ├─ Google Drive        → OAuth flow → folder picker
      ├─ OneDrive            → OAuth flow → folder picker
      ├─ SharePoint Site     → OAuth flow → site URL input
      ├─ Google Calendar     → OAuth flow → calendar picker
      └─ Outlook Calendar    → OAuth flow → calendar picker
```

Each flow:
1. **Select resource type** → shows provider-specific card
2. **Authenticate** → OAuth2 browser popup (reuses existing `oauth_email.py` pattern)
3. **Configure** → provider-specific options (folder, mailbox, calendar, etc.)
4. **Name & Save** → user gives it a friendly name → persisted to `accounts.yaml`
5. **Load into session** → immediately adds as document source to current session

### Account Manager (Settings)

A "My Accounts" panel accessible from the user menu or Sources section:

```
My Accounts
├─ Gmail (ken@gmail.com)           [✓ active] [⟳ re-auth] [✕ remove]
│   └─ INBOX, since 2026-01-01
├─ Google Drive (Shared/Analytics) [✓ active] [⟳ re-auth] [✕ remove]
│   └─ 127 files indexed
├─ Outlook Calendar                [✓ active] [⟳ re-auth] [✕ remove]
│   └─ 45 events indexed
└─ SharePoint (Analytics Site)     [✓ active] [⟳ re-auth] [✕ remove]
    └─ 3 libraries, 2 lists
```

Each account shows: provider icon, display name, email/path, status, last synced, actions.

## Data Model

### User accounts file: `.constat/{user_id}/accounts.yaml`

```yaml
accounts:
  my-gmail:
    type: imap
    provider: google
    display_name: "Gmail (ken@gmail.com)"
    email: ken@gmail.com
    auth_type: oauth2_refresh
    oauth2_client_id: ""              # uses server-configured client ID
    refresh_token: "<encrypted>"
    created_at: "2026-03-24T10:00:00Z"
    active: true
    options:
      mailbox: INBOX
      since: "2026-01-01"
      max_messages: 500
      extract_attachments: true

  analytics-drive:
    type: drive
    provider: google
    display_name: "Analytics Shared Drive"
    email: ken@gmail.com
    auth_type: oauth2_refresh
    refresh_token: "<encrypted>"
    created_at: "2026-03-24T10:05:00Z"
    active: true
    options:
      folder_id: "1aBcDeFgHiJk"
      folder_path: "/Shared Drives/Analytics"
      recursive: true
      max_files: 500
      include_types: [".pdf", ".docx", ".xlsx", ".pptx"]

  work-calendar:
    type: calendar
    provider: microsoft
    display_name: "Outlook Calendar"
    email: ken@company.com
    auth_type: oauth2_refresh
    oauth2_tenant_id: "abc-123"
    refresh_token: "<encrypted>"
    created_at: "2026-03-24T10:10:00Z"
    active: true
    options:
      calendar_id: primary
      since: "2026-01-01"
      expand_recurring: true

  analytics-sp:
    type: sharepoint
    provider: microsoft
    display_name: "Analytics SharePoint"
    email: ken@company.com
    auth_type: oauth2_refresh
    oauth2_tenant_id: "abc-123"
    refresh_token: "<encrypted>"
    created_at: "2026-03-24T10:15:00Z"
    active: true
    options:
      site_url: https://contoso.sharepoint.com/sites/analytics
      discover_libraries: true
      discover_lists: true
      discover_calendars: true
      library_names: ["Shared Documents", "Reports"]
```

### Python model

```python
@dataclass
class PersonalAccount:
    name: str                            # config key
    type: str                            # imap, drive, calendar, sharepoint
    provider: str                        # google, microsoft
    display_name: str
    email: str
    auth_type: str                       # oauth2_refresh
    refresh_token: str                   # encrypted at rest
    created_at: str
    active: bool = True
    oauth2_tenant_id: str | None = None  # Microsoft only
    options: dict = field(default_factory=dict)
```

## OAuth2 Flow (Generalized)

### Current: Email-only OAuth

```
oauth_email.py:
  /oauth/email/providers   → { google: bool, microsoft: bool }
  /oauth/email/authorize   → redirect to provider
  /oauth/email/callback    → exchange code → return refresh_token
```

### New: Multi-resource OAuth

Generalize the existing email OAuth routes to support multiple resource types with different scopes:

```
oauth.py (replaces oauth_email.py):
  /oauth/providers                     → { google: [...types], microsoft: [...types] }
  /oauth/authorize?provider=X&type=Y   → redirect with type-specific scopes
  /oauth/callback                      → exchange code → return refresh_token + email
```

### Scope Matrix

| Provider | Resource Type | OAuth2 Scopes |
|----------|--------------|---------------|
| Google | email | `https://mail.google.com/ email` |
| Google | drive | `https://www.googleapis.com/auth/drive.readonly email` |
| Google | calendar | `https://www.googleapis.com/auth/calendar.readonly email` |
| Microsoft | email | `https://outlook.office365.com/IMAP.AccessAsUser.All offline_access` |
| Microsoft | drive | `Files.Read.All offline_access` |
| Microsoft | calendar | `Calendars.Read offline_access` |
| Microsoft | sharepoint | `Sites.Read.All Files.Read.All offline_access` |

All flows include `offline_access` (Microsoft) or `access_type=offline` (Google) to get a refresh token.

### Combined scopes for single sign-on

When a user connects multiple resource types from the same provider, request combined scopes in a single OAuth flow:

```python
# User connects Gmail + Google Drive + Google Calendar in one go
combined_scopes = "https://mail.google.com/ https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/calendar.readonly email"
```

One refresh token grants access to all selected resources. Stored once per provider-email pair.

### Token Encryption

Refresh tokens stored in `accounts.yaml` are encrypted at rest:

```python
from cryptography.fernet import Fernet

# Key derived from server secret + user ID
def _get_encryption_key(server_secret: str, user_id: str) -> bytes:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=user_id.encode(), iterations=100_000)
    return base64.urlsafe_b64encode(kdf.derive(server_secret.encode()))

def encrypt_token(token: str, key: bytes) -> str:
    return Fernet(key).encrypt(token.encode()).decode()

def decrypt_token(encrypted: str, key: bytes) -> str:
    return Fernet(key).decrypt(encrypted.encode()).decode()
```

## Backend API

### New routes: `constat/server/routes/accounts.py`

```python
router = APIRouter(prefix="/api/accounts")

@router.get("/")
async def list_accounts(request: Request, user_id: str) -> list[AccountSummary]:
    """List user's connected accounts (no tokens in response)."""

@router.post("/")
async def add_account(request: Request, user_id: str, body: AddAccountRequest) -> AccountSummary:
    """Add a new personal account from completed OAuth flow."""

@router.put("/{account_name}")
async def update_account(request: Request, user_id: str, account_name: str, body: UpdateAccountRequest) -> AccountSummary:
    """Update account options (active toggle, rename, change folder, etc.)."""

@router.delete("/{account_name}")
async def delete_account(request: Request, user_id: str, account_name: str) -> dict:
    """Remove an account and its cached tokens."""

@router.post("/{account_name}/refresh-auth")
async def refresh_auth(request: Request, user_id: str, account_name: str) -> dict:
    """Re-authenticate an account (opens new OAuth flow)."""

@router.post("/{account_name}/test")
async def test_account(request: Request, user_id: str, account_name: str) -> dict:
    """Test account connectivity (verify token, list a few resources)."""
```

### Request/Response models

```python
class AddAccountRequest(BaseModel):
    name: str                            # user-chosen name
    type: str                            # imap, drive, calendar, sharepoint
    provider: str                        # google, microsoft
    email: str                           # from OAuth flow
    refresh_token: str                   # from OAuth flow
    oauth2_tenant_id: str | None = None
    options: dict = {}

class UpdateAccountRequest(BaseModel):
    display_name: str | None = None
    active: bool | None = None
    options: dict | None = None

class AccountSummary(BaseModel):
    name: str
    type: str
    provider: str
    display_name: str
    email: str
    active: bool
    created_at: str
    last_synced: str | None = None
    resource_count: int | None = None    # files, events, messages indexed
```

### Generalized OAuth routes: `constat/server/routes/oauth.py`

Replaces `oauth_email.py`. Same pattern, parameterized by resource type:

```python
SCOPE_MAP = {
    ("google", "email"): "https://mail.google.com/ email",
    ("google", "drive"): "https://www.googleapis.com/auth/drive.readonly email",
    ("google", "calendar"): "https://www.googleapis.com/auth/calendar.readonly email",
    ("microsoft", "email"): "https://outlook.office365.com/IMAP.AccessAsUser.All offline_access",
    ("microsoft", "drive"): "Files.Read.All offline_access",
    ("microsoft", "calendar"): "Calendars.Read offline_access",
    ("microsoft", "sharepoint"): "Sites.Read.All Files.Read.All offline_access",
}

@router.get("/authorize")
async def authorize(request: Request, provider: str, resource_type: str, session_id: str) -> RedirectResponse:
    """Redirect user to OAuth2 provider with type-specific scopes."""
    scopes = SCOPE_MAP.get((provider, resource_type))
    # ... same flow as oauth_email.py, with dynamic scopes ...
```

## Session Integration

### Loading personal accounts into sessions

On session creation (`sessions.py:_load_domains_into_session`), after loading domain configs:

```python
# Load personal accounts as user-tier document sources
accounts = load_user_accounts(user_id)
for name, account in accounts.items():
    if not account.active:
        continue
    doc_config = account_to_document_config(account)
    session.add_document_from_config(name, doc_config, source="personal")
```

### `account_to_document_config()` conversion

```python
def account_to_document_config(account: PersonalAccount) -> DocumentConfig:
    """Convert a personal account to a DocumentConfig for session loading."""
    base = {
        "type": account.type,
        "description": account.display_name,
        "auth_type": account.auth_type,
        "oauth2_client_id": "",  # filled from server config at runtime
        "oauth2_client_secret": decrypt_token(account.refresh_token, key),
        **account.options,
    }
    if account.provider == "google":
        if account.type == "imap":
            base["url"] = "imaps://imap.gmail.com:993"
            base["username"] = account.email
        elif account.type == "drive":
            base["provider"] = "google"
        elif account.type == "calendar":
            base["provider"] = "google"
    elif account.provider == "microsoft":
        base["oauth2_tenant_id"] = account.oauth2_tenant_id
        if account.type == "imap":
            base["url"] = "imaps://outlook.office365.com:993"
            base["username"] = account.email
        elif account.type == "drive":
            base["provider"] = "microsoft"
        elif account.type == "calendar":
            base["provider"] = "microsoft"
        elif account.type == "sharepoint":
            base["site_url"] = account.options.get("site_url")
    return DocumentConfig(**base)
```

### Source tagging

Personal accounts are tagged with `source="personal"` so the UI can distinguish them from domain-configured sources and display the user badge:

```
Sources > Documents
├─ business_rules (Sales Analytics)        ← domain source
├─ Gmail (ken@gmail.com) [personal]       ← personal account
└─ Analytics Drive [personal]             ← personal account
```

## Frontend

### New store: `accountStore.ts`

```typescript
interface PersonalAccount {
  name: string
  type: 'imap' | 'drive' | 'calendar' | 'sharepoint'
  provider: 'google' | 'microsoft'
  displayName: string
  email: string
  active: boolean
  createdAt: string
  lastSynced: string | null
  resourceCount: number | null
}

interface AccountStore {
  accounts: PersonalAccount[]
  loading: boolean
  fetchAccounts: (userId: string) => Promise<void>
  addAccount: (userId: string, account: AddAccountRequest) => Promise<void>
  updateAccount: (userId: string, name: string, updates: Partial<PersonalAccount>) => Promise<void>
  removeAccount: (userId: string, name: string) => Promise<void>
  testAccount: (userId: string, name: string) => Promise<{ ok: boolean; message: string }>
}
```

### New component: `PersonalResourcePicker.tsx`

Resource type selection grid:

```tsx
const RESOURCE_TYPES = [
  { type: 'email', provider: 'google', label: 'Gmail', icon: EnvelopeIcon, color: 'red' },
  { type: 'email', provider: 'microsoft', label: 'Outlook', icon: EnvelopeIcon, color: 'blue' },
  { type: 'drive', provider: 'google', label: 'Google Drive', icon: FolderIcon, color: 'green' },
  { type: 'drive', provider: 'microsoft', label: 'OneDrive', icon: FolderIcon, color: 'blue' },
  { type: 'calendar', provider: 'google', label: 'Google Calendar', icon: CalendarIcon, color: 'blue' },
  { type: 'calendar', provider: 'microsoft', label: 'Outlook Calendar', icon: CalendarIcon, color: 'blue' },
  { type: 'sharepoint', provider: 'microsoft', label: 'SharePoint', icon: GlobeAltIcon, color: 'purple' },
]
```

Each card shows the provider logo, resource type name, and a "Connect" button that initiates the OAuth flow.

### New component: `AccountManager.tsx`

List of connected accounts with status, sync info, and actions. Accessible from:
- Sources section header (gear icon)
- User menu dropdown

### Modified: `ArtifactPanel.tsx`

Add `personal` to `ModalType`:

```typescript
type ModalType = 'database' | 'api' | 'document' | 'email' | 'fact' | 'rule' | 'personal' | null
```

Add `[+]` button in Sources section that opens `PersonalResourcePicker`.

### Modified: `oauth_email.py` → generalized OAuth

The existing email OAuth UI flow (popup window, `postMessage` callback) is reused for all resource types. The `resource_type` parameter determines which scopes to request.

## Backward Compatibility

### Existing email modal

The existing "Email" `+` button and modal (`showModal === 'email'`) continue to work for session-scoped email sources. The new "Personal" `+` button offers the same email connection flow but persists the account.

### Migration path

Existing session-scoped email sources (added via old modal) are not auto-migrated. Users can re-connect via the Personal flow to get persistence.

### `oauth_email.py` routes

Keep existing routes at `/api/oauth/email/*` for backward compatibility. New generalized routes at `/api/oauth/*`. Once the frontend is fully migrated, deprecate the email-specific routes.

## Server Configuration

OAuth2 client credentials are configured at the server level (not per-user):

```python
class ServerConfig(BaseModel):
    # Existing (email only)
    google_email_client_id: str | None
    google_email_client_secret: str | None
    microsoft_email_client_id: str | None
    microsoft_email_client_secret: str | None
    microsoft_email_tenant_id: str | None

    # New (shared across resource types)
    google_oauth_client_id: str | None         # single Google client for all resource types
    google_oauth_client_secret: str | None
    microsoft_oauth_client_id: str | None      # single Azure app for all resource types
    microsoft_oauth_client_secret: str | None
    microsoft_oauth_tenant_id: str | None

    # Token encryption
    account_encryption_secret: str | None      # for encrypting refresh tokens at rest
```

Google and Microsoft both support requesting multiple scopes from a single OAuth app registration. One Azure AD app with `Mail.Read`, `Files.Read.All`, `Calendars.Read`, `Sites.Read.All` covers all resource types.

## New Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `cryptography` | Token encryption at rest | `pip install cryptography` |
| `httpx` | OAuth token exchange, API calls | Already installed |

## File Changes Summary

| File | Change |
|------|--------|
| `constat/server/routes/oauth.py` | **New**: generalized OAuth routes (authorize, callback, providers) |
| `constat/server/routes/accounts.py` | **New**: account CRUD API (list, add, update, delete, test) |
| `constat/server/accounts.py` | **New**: `PersonalAccount`, `load_user_accounts()`, `save_user_accounts()`, `account_to_document_config()`, encryption helpers |
| `constat/server/config.py` | Add `google_oauth_*`, `microsoft_oauth_*`, `account_encryption_secret` fields |
| `constat/server/routes/sessions.py` | Load active personal accounts into new sessions |
| `constat/server/routes/oauth_email.py` | Deprecate; redirect to generalized routes |
| `constat-ui/src/store/accountStore.ts` | **New**: Zustand store for personal accounts |
| `constat-ui/src/api/sessionsApi.ts` | Add account CRUD API calls |
| `constat-ui/src/components/artifacts/PersonalResourcePicker.tsx` | **New**: resource type grid + OAuth flow |
| `constat-ui/src/components/artifacts/AccountManager.tsx` | **New**: connected accounts list with actions |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Add `personal` modal type, `[+]` button in Sources |
| `constat-ui/src/types/index.ts` | Add `PersonalAccount`, `AddAccountRequest` types |
| `tests/test_accounts.py` | Unit tests for account CRUD, encryption, config conversion |
| `tests/test_oauth_generalized.py` | Unit tests for multi-resource OAuth flow |

## Testing Strategy

1. **Unit tests** (`test_accounts.py`):
   - `load_user_accounts()` / `save_user_accounts()` round-trip
   - `account_to_document_config()` for each type × provider combination
   - Token encryption round-trip: encrypt → decrypt → original
   - Account validation: reject missing required fields
   - Active/inactive filtering on session load

2. **Unit tests** (`test_oauth_generalized.py`):
   - Scope selection by (provider, resource_type) pair
   - Combined scopes for multi-resource single sign-on
   - Authorization URL generation with correct scopes
   - Token exchange + email extraction (Google userinfo, Microsoft JWT)
   - State token lifecycle: create → validate → expire → cleanup

3. **Integration tests**:
   - Full flow: OAuth authorize → callback → add account → load into session → verify document source active
   - Account persistence: add account → restart server → new session → account auto-loads
   - Account removal: delete account → new session → account not loaded
   - Re-authentication: refresh token expired → re-auth flow → token updated

4. **Frontend tests** (Vitest):
   - `accountStore`: fetch, add, update, remove operations
   - `PersonalResourcePicker`: renders all resource types, filters by available providers
   - `AccountManager`: displays accounts, handles toggle/remove actions
   - OAuth popup flow: message listener receives token correctly

## Edge Cases

| Case | Handling |
|------|----------|
| Refresh token expired / revoked | Mark account as `needs_reauth`; show warning badge in UI; skip on session load |
| Server OAuth credentials not configured | Hide unavailable resource types in picker; show "Contact admin" message |
| User connects same email as both personal + domain source | Both load; domain source takes precedence in dedup by name |
| Multiple users on same server | Each user's `accounts.yaml` is isolated under their `user_id` directory |
| Encryption key changes (server secret rotated) | Tokens become undecryptable → mark all accounts as `needs_reauth` |
| Account YAML corruption | Load what's parseable; log warning; skip corrupted entries |
| OAuth popup blocked by browser | Show fallback with manual copy-paste of result URL (existing pattern) |
| Rate limiting during initial sync of large drive | Respect `Retry-After`; show progress in UI; allow partial sync |
| User removes all personal accounts | Session loads with only domain-configured sources (existing behavior) |

## Non-Goals (v1)

- Sharing personal accounts with other users or teams
- Admin management of user accounts (users manage their own)
- OAuth device flow (headless/CLI) — browser popup only
- Personal database connections (only document-type sources)
- Selective per-session account activation (all active accounts load into every session)
- Real-time sync push notifications
- Google Workspace admin (domain-wide delegation)
- Personal fact or rule persistence (separate feature)
