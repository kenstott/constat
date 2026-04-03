# Authentication Architecture

## Current State

### Backend (`constat/server/`)

| Component | File | Purpose |
|-----------|------|---------|
| Auth middleware | `auth.py` | `get_current_user_id()` — validates tokens, returns user ID |
| Local auth | `local_auth.py` | `hash_password()`, `verify_password()` (scrypt), opaque token store |
| Auth routes | `routes/auth_routes.py` | `POST /api/auth/login` (local), `POST /api/auth/firebase-login` |
| Passkey routes | `routes/passkey.py` | WebAuthn registration/authentication endpoints |
| Config | `config.py` | `ServerConfig.auth_disabled`, `firebase_project_id`, `local_users`, `admin_token` |

Token validation order in `get_current_user_id()`:
1. `auth_disabled` → return `"default"`
2. `admin_token` match → return `"admin"`
3. Local opaque token (`validate_local_token`) → return username
4. Firebase JWT (`verify_firebase_token`) → return Firebase UID

### Frontend (`constat-ui/src/`)

| Component | File | Purpose |
|-----------|------|---------|
| Firebase config | `config/firebase.ts` | Firebase init, Google/email/link sign-in, `isAuthDisabled` flag |
| Auth store | `store/authStore.ts` | Zustand store — `loginWithGoogle`, `loginWithEmail`, `signupWithEmail` |
| Login page | `components/auth/LoginPage.tsx` | Google button, email/password form, signup, reset, email link |
| Passkey UI | `components/auth/PasskeySetup.tsx`, `PasskeyUnlock.tsx` | WebAuthn registration/unlock |

### Health endpoint auth discovery

`GET /health` returns `auth.auth_methods: string[]` — currently `["local"]` and/or `["firebase"]` based on config.

### Config (YAML)

```yaml
server:
  auth_disabled: false
  firebase_project_id: my-project
  firebase_api_key: AIza...
  local_users:
    demo:
      password_hash: "scrypt:..."
      email: demo@localhost
```

## What's Missing

### 1. Local Login in Frontend

The backend has `POST /api/auth/login` (username/password → opaque token) but the frontend has **no UI to call it**. `LoginPage.tsx` only calls Firebase `signInWithEmailAndPassword`.

**Required changes:**

**Backend** — `routes/auth_routes.py`:
- `POST /api/auth/register` — create local user (self-registration), hash password with `local_auth.hash_password()`, persist to user store. Consider email verification requirement.

**Frontend** — `config/firebase.ts`:
- Add `localLogin(username: string, password: string): Promise<{token, user_id, email}>` — calls `POST /api/auth/login`
- Add `localRegister(username: string, password: string, email: string)` — calls `POST /api/auth/register`

**Frontend** — `store/authStore.ts`:
- Add `loginWithLocal(username: string, password: string)` action
- Add `signupLocal(username: string, password: string, email: string)` action
- Store the opaque token from local login (currently only Firebase tokens are handled via `getIdToken()`)
- `getToken()` must return the local opaque token when logged in locally

**Frontend** — `components/auth/LoginPage.tsx`:
- When `auth_methods` includes `"local"`, show username/password form that calls `loginWithLocal`
- "Create account" mode calls `signupLocal`
- Label distinction: local login uses **username**, Firebase uses **email**

### 2. Microsoft / Azure AD Login

Support Microsoft Entra ID (Azure AD) as an OAuth2/OIDC provider for enterprise SSO.

**Required changes:**

**Backend** — `config.py`:
- Add `microsoft_auth_client_id: Optional[str]` — Azure AD app registration client ID
- Add `microsoft_auth_client_secret: Optional[str]` — client secret
- Add `microsoft_auth_tenant_id: str = "common"` — tenant (single-tenant or `common` for multi-tenant)
- Env var overrides: `MICROSOFT_AUTH_CLIENT_ID`, `MICROSOFT_AUTH_CLIENT_SECRET`, `MICROSOFT_AUTH_TENANT_ID`

Note: `microsoft_email_*` fields already exist for IMAP OAuth2; auth fields are separate.

**Backend** — `routes/auth_routes.py`:
- `POST /api/auth/microsoft-login` — exchange authorization code for tokens via Microsoft identity platform
  - Endpoint: `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`
  - Validate the ID token, extract `oid` (user ID) and `preferred_username` (email)
  - Create a local opaque token (same as local auth) and return it
- `GET /api/auth/microsoft-config` — return `client_id` and `tenant_id` for frontend MSAL init (no secrets)

**Backend** — `auth.py`:
- Add Microsoft JWT validation path: verify token signed by Microsoft, audience matches `client_id`
- Or: rely on the code-exchange approach where backend validates via token endpoint and issues opaque token

**Backend** — `app.py` health endpoint:
- Add `"microsoft"` to `auth_methods` when `microsoft_auth_client_id` is configured

**Frontend** — new `config/microsoft.ts`:
- Initialize MSAL.js (`@azure/msal-browser`) with client ID and tenant from `/api/auth/microsoft-config`
- `signInWithMicrosoft()` — trigger popup/redirect login, get auth code
- Send auth code to `POST /api/auth/microsoft-login`, receive opaque token

**Frontend** — `store/authStore.ts`:
- Add `loginWithMicrosoft()` action
- Uses `signInWithMicrosoft()` from `config/microsoft.ts`

**Frontend** — `components/auth/LoginPage.tsx`:
- When `auth_methods` includes `"microsoft"`, show "Sign in with Microsoft" button (Microsoft branding guidelines)

**Frontend** — `package.json`:
- Add `@azure/msal-browser` dependency

### 3. Dynamic Login Page Based on Available Methods

The login page currently always shows Google + email/password (Firebase). It should adapt based on what the server reports.

**Required changes:**

**Frontend** — new hook `hooks/useAuthMethods.ts`:
```typescript
// Fetch /health on mount, extract auth.auth_methods
// Returns { methods: string[], loading: boolean }
// Cached for session lifetime
```

**Frontend** — `components/auth/LoginPage.tsx`:
- Fetch available methods via `useAuthMethods()`
- Conditionally render:
  - `"firebase"` → Google button + Firebase email/password form + signup + email link + reset
  - `"local"` → Username/password form + "Create account" link
  - `"microsoft"` → "Sign in with Microsoft" button
- If multiple methods available, show all with dividers ("or continue with...")
- If only one method, show it directly without dividers

### Config Examples

**Local-only deployment (no external IdP):**
```yaml
server:
  local_users:
    admin:
      password_hash: "scrypt:..."
      email: admin@company.com
```

**Firebase + local:**
```yaml
server:
  firebase_project_id: my-project
  firebase_api_key: AIza...
  local_users:
    service-account:
      password_hash: "scrypt:..."
      email: svc@localhost
```

**Microsoft SSO + local:**
```yaml
server:
  microsoft_auth_client_id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
  microsoft_auth_client_secret: "secret~..."
  microsoft_auth_tenant_id: "contoso.onmicrosoft.com"
  local_users:
    fallback-admin:
      password_hash: "scrypt:..."
      email: admin@localhost
```

**All methods:**
```yaml
server:
  firebase_project_id: my-project
  firebase_api_key: AIza...
  microsoft_auth_client_id: "aaaaaaaa-..."
  microsoft_auth_client_secret: "secret~..."
  microsoft_auth_tenant_id: "common"
  local_users:
    demo:
      password_hash: "scrypt:..."
      email: demo@localhost
```

## Implementation Order

1. **Local login frontend** — wire `POST /api/auth/login` to LoginPage, add local signup route
2. **Dynamic login page** — fetch `/health` auth_methods, conditionally render login options
3. **Microsoft login** — backend code-exchange route, MSAL.js frontend, Microsoft button
4. **Local user self-registration** — `POST /api/auth/register` with optional email verification
