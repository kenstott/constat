# Email Inbox (IMAP) Document Source

## Goal

Add IMAP email inboxes as a document data source. Each message becomes a document. Attachments and embedded images are extracted as child documents and routed through the existing vectorization pipeline (including the image pipeline from `images.md`).

## Addressing Scheme

All document names use colon-separated hierarchical addresses:

```
<data_source>:<message_id>                          # email body
<data_source>:<message_id>:<attachment_filename>     # attachment
<data_source>:<message_id>:<image_name>              # embedded image
```

Examples:
```
support-inbox:msg_20260301_143022_abc:report.pdf
support-inbox:msg_20260301_143022_abc:screenshot.png
support-inbox:msg_20260301_143022_abc:inline_image_1.jpg
```

This follows the existing `<config_key>:<child>` pattern used by glob expansion and link crawling.

## Configuration

### In domain YAML or root config:

```yaml
documents:
  support-inbox:
    type: imap
    url: imaps://imap.example.com:993
    username: ${IMAP_USER}
    password: ${IMAP_PASS}
    description: "Customer support inbox"

    # IMAP-specific options
    mailbox: INBOX              # IMAP folder (default: INBOX)
    search_criteria: UNSEEN     # IMAP search string (default: ALL)
    since: "2026-01-01"         # only messages after this date
    max_messages: 500           # cap per sync
    extract_attachments: true   # default: true
    extract_images: true        # extract inline/embedded images (default: true)
    include_headers: true       # index From/To/CC/Subject/Date (default: true)
    attachment_types:            # allowlist; omit = all supported types
      - ".pdf"
      - ".docx"
      - ".xlsx"
      - ".pptx"
      - ".png"
      - ".jpg"
      - ".md"
      - ".txt"
      - ".html"
```

### `DocumentConfig` additions

```python
class DocumentConfig(BaseModel):
    ...
    # IMAP fields
    mailbox: str = "INBOX"
    search_criteria: str = "ALL"
    since: Optional[str] = None
    max_messages: int = 500
    extract_attachments: bool = True
    extract_images: bool = True          # repurpose existing field
    include_headers: bool = True
    attachment_types: Optional[list[str]] = None
```

## Pipeline Overview

```
IMAP connection
  → fetch message list (UIDs)
  → for each message:
      ├─ parse MIME structure (email.message_from_bytes)
      ├─ extract body (text/plain or text/html → markdown)
      ├─ extract metadata (From, To, CC, Subject, Date, Message-ID)
      ├─ render as document text
      ├─ _chunk_document → encode → add_chunks    (existing pipeline)
      │
      ├─ for each attachment:
      │     ├─ decode payload bytes
      │     ├─ detect type via _mime.detect_type_from_source()
      │     ├─ if document type → _extract_content() → _chunk_document → encode → add_chunks
      │     ├─ if image type → _extract_image() → (LLM vision if needed) → _chunk_document → encode → add_chunks
      │     └─ address: <source>:<msg_id>:<filename>
      │
      └─ for each inline/embedded image (Content-Disposition: inline, or CID reference):
            ├─ decode payload bytes
            ├─ _extract_image() → image pipeline
            └─ address: <source>:<msg_id>:<cid_or_name>
```

## Implementation

### New file: `constat/discovery/doc_tools/_imap.py`

```python
import email
import imaplib
from email.message import Message
from dataclasses import dataclass

@dataclass
class EmailMessage:
    uid: str
    message_id: str
    subject: str
    sender: str
    recipients: list[str]
    cc: list[str]
    date: str
    body_text: str              # plain text or html→markdown
    body_html: str | None
    attachments: list[EmailAttachment]
    inline_images: list[EmailAttachment]

@dataclass
class EmailAttachment:
    filename: str
    content_type: str
    data: bytes
    content_id: str | None      # CID for inline images
    is_inline: bool


class IMAPFetcher:
    """Connects to IMAP server, fetches messages, parses MIME."""

    def __init__(self, config: DocumentConfig, config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir

    def connect(self) -> imaplib.IMAP4_SSL:
        parsed = urlparse(config.url)
        host = parsed.hostname
        port = parsed.port or 993
        conn = imaplib.IMAP4_SSL(host, port)
        conn.login(config.username, config.password)
        return conn

    def fetch_messages(self) -> list[EmailMessage]:
        conn = self.connect()
        try:
            conn.select(self._config.mailbox, readonly=True)
            criteria = self._build_search_criteria()
            _, uid_data = conn.uid("SEARCH", None, *criteria)
            uids = uid_data[0].split()[:self._config.max_messages]

            messages = []
            for uid in uids:
                _, msg_data = conn.uid("FETCH", uid, "(RFC822)")
                raw = msg_data[0][1]
                parsed = self._parse_message(uid.decode(), raw)
                messages.append(parsed)
            return messages
        finally:
            conn.logout()

    def _build_search_criteria(self) -> list[str]:
        parts = [self._config.search_criteria]
        if self._config.since:
            parts.extend(["SINCE", self._config.since])
        return parts

    def _parse_message(self, uid: str, raw: bytes) -> EmailMessage:
        msg = email.message_from_bytes(raw)
        body_text, body_html = self._extract_body(msg)
        attachments, inline_images = self._extract_parts(msg)
        msg_id = self._make_msg_id(uid, msg)
        return EmailMessage(
            uid=uid,
            message_id=msg_id,
            subject=msg.get("Subject", ""),
            sender=msg.get("From", ""),
            recipients=msg.get_all("To", []),
            cc=msg.get_all("Cc", []),
            date=msg.get("Date", ""),
            body_text=body_text,
            body_html=body_html,
            attachments=attachments,
            inline_images=inline_images,
        )

    def _extract_body(self, msg: Message) -> tuple[str, str | None]:
        """Walk MIME parts, prefer text/plain, fallback to text/html → markdown."""
        ...

    def _extract_parts(self, msg: Message) -> tuple[list[EmailAttachment], list[EmailAttachment]]:
        """Walk MIME tree, collect attachments and inline images."""
        attachments = []
        inline_images = []
        for part in msg.walk():
            disposition = part.get_content_disposition()
            content_type = part.get_content_type()
            if disposition == "attachment" or (disposition == "inline" and content_type.startswith("image/")):
                data = part.get_payload(decode=True)
                filename = part.get_filename() or f"part_{len(attachments)}"
                cid = part.get("Content-ID", "").strip("<>") or None
                att = EmailAttachment(
                    filename=filename,
                    content_type=content_type,
                    data=data,
                    content_id=cid,
                    is_inline=(disposition == "inline"),
                )
                if content_type.startswith("image/"):
                    inline_images.append(att)
                else:
                    attachments.append(att)
        return attachments, inline_images

    def _make_msg_id(self, uid: str, msg: Message) -> str:
        """Deterministic short ID from UID + date."""
        date = email.utils.parsedate_to_datetime(msg.get("Date", ""))
        return f"msg_{date.strftime('%Y%m%d_%H%M%S')}_{uid}"
```

### Rendering Email Body as Document Text

```python
def _render_email(msg: EmailMessage, source_name: str) -> str:
    parts = [f"# {msg.subject}"]
    if msg.include_headers:
        parts.append(f"From: {msg.sender}")
        parts.append(f"To: {', '.join(msg.recipients)}")
        if msg.cc:
            parts.append(f"CC: {', '.join(msg.cc)}")
        parts.append(f"Date: {msg.date}")
        parts.append("")
    parts.append(msg.body_text)
    if msg.attachments:
        parts.append("\n## Attachments")
        for att in msg.attachments:
            parts.append(f"- {att.filename} ({att.content_type})")
    return "\n".join(parts)
```

### Integration with `_core.py`

In `_load_document(name)`:

```python
if doc_config.type == "imap":
    fetcher = IMAPFetcher(doc_config, config_dir=self._config_dir)
    messages = fetcher.fetch_messages()
    for msg in messages:
        msg_name = f"{name}:{msg.message_id}"

        # Index email body
        body_text = _render_email(msg, name)
        self._loaded_documents[msg_name] = LoadedDocument(
            name=msg_name, content=body_text, doc_type="markdown",
        )

        # Index attachments (documents)
        for att in msg.attachments:
            att_name = f"{msg_name}:{att.filename}"
            att_type = detect_type_from_source(att.content_type, att.filename)
            if att_type.startswith("image/"):
                # Route to image pipeline (images.md)
                image_result = _extract_image(path=None, data=att.data)
                if image_result.category == "image-primary":
                    desc = await self._describe_image(self._provider, att.data, att_type)
                    image_result.description = desc.description
                    image_result.subcategory = desc.subcategory
                    image_result.labels = desc.labels
                content = _render_image_result(image_result, att.filename)
            else:
                content = _extract_content_from_bytes(att.data, att_type)
            self._loaded_documents[att_name] = LoadedDocument(
                name=att_name, content=content, doc_type=att_type,
            )

        # Index inline images
        for img in msg.inline_images:
            img_name = f"{msg_name}:{img.content_id or img.filename}"
            image_result = _extract_image(path=None, data=img.data)
            if image_result.category == "image-primary":
                desc = await self._describe_image(self._provider, img.data, img.content_type)
                image_result.description = desc.description
                image_result.subcategory = desc.subcategory
                image_result.labels = desc.labels
            content = _render_image_result(image_result, img.filename)
            self._loaded_documents[img_name] = LoadedDocument(
                name=img_name, content=content, doc_type="markdown",
            )
```

After all messages are loaded, the existing `_index_loaded_documents()` flow handles chunking, embedding, and storage for every entry in `_loaded_documents`.

### Transport Layer (`_transport.py`)

Add `imap` transport recognition in `infer_transport()`:

```python
if url_scheme in ("imap", "imaps"):
    return "imap"
```

However, IMAP fetching is handled entirely by `IMAPFetcher`, not the generic `fetch_document()` path, because IMAP requires multi-message iteration. The transport inference is used only for type detection/validation.

### MIME Type Detection (`_mime.py`)

No changes needed. Attachment MIME types come from the email MIME headers (`Content-Type`), which are more reliable than file extension detection.

## Incremental Sync

Track which messages have been indexed to avoid re-processing on subsequent loads.

```python
# In vector store: check if document_name already has chunks
def _is_message_indexed(self, msg_name: str) -> bool:
    return self._vector_store.has_document(msg_name)
```

The existing `add_chunks()` already skips documents with existing chunks (idempotent). But for IMAP efficiency, check before fetching full RFC822 bodies:

1. First pass: `FETCH (ENVELOPE)` to get lightweight metadata
2. Build `msg_name` from envelope data
3. Skip `FETCH (RFC822)` if `_is_message_indexed(msg_name)` returns True

This avoids downloading large messages with attachments that are already indexed.

## New Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| None | `imaplib` and `email` are stdlib | — |

Zero new dependencies. Python stdlib provides full IMAP and MIME parsing.

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/doc_tools/_imap.py` | **New**: `IMAPFetcher`, `EmailMessage`, `EmailAttachment`, `_render_email()` |
| `constat/discovery/doc_tools/_core.py` | Add IMAP branch in `_load_document()` |
| `constat/discovery/doc_tools/_transport.py` | Recognize `imap`/`imaps` URL scheme |
| `constat/core/config.py` | Add IMAP fields to `DocumentConfig` |
| `constat/server/routes/files.py` | Add IMAP URI support in `add_document_uri` route |
| `tests/test_imap_ingestion.py` | Unit tests with mocked IMAP server |

## Testing Strategy

1. **Unit tests** (`test_imap_ingestion.py`):
   - Mock `imaplib.IMAP4_SSL` with canned RFC822 bytes
   - `_parse_message()` with multipart/mixed (text + PDF attachment + inline image)
   - `_extract_body()` with text/plain only, text/html only, multipart/alternative
   - `_extract_parts()` with nested MIME structures
   - `_render_email()` output format
   - Address generation: verify `<source>:<msg_id>:<attachment>` naming

2. **Integration tests**:
   - Full pipeline mock: IMAP fetch → parse → index body + attachments + images
   - Verify chunks in vector store with correct `document_name` addresses
   - Incremental sync: second run skips already-indexed messages

3. **Fixtures**: Construct test emails programmatically using `email.mime` stdlib:
   - Plain text email (no attachments)
   - HTML email with inline images (CID references)
   - Email with PDF + DOCX + PNG attachments

## Edge Cases

| Case | Handling |
|------|----------|
| Duplicate filenames in attachments | Append `_2`, `_3` suffix: `report.pdf`, `report_2.pdf` |
| Missing filename | Use `part_<index>.<ext>` based on MIME type |
| Charset encoding issues | `email.policy.default` handles charset decoding; fallback to latin-1 |
| Very large attachments (>50MB) | Skip with warning log; configurable `max_attachment_size` field |
| Nested email (message/rfc822) | Recursively parse inner message, address as `<source>:<outer_id>:fwd_<inner_id>` |
| IMAP connection timeout | Raise on connection failure; no silent retry |
| Self-signed TLS certificates | Not supported in v1; require valid certs |

## OAuth2 / XOAUTH2 Authentication

Required for Microsoft 365 / Exchange Online (basic auth disabled since 2022). Also supported by Gmail as an alternative to app passwords.

### Config

```yaml
documents:
  m365-inbox:
    type: imap
    url: imaps://outlook.office365.com:993
    auth_type: oauth2           # new field: "basic" (default) | "oauth2"
    username: user@company.com
    description: "M365 mailbox"

    # OAuth2 fields
    oauth2_client_id: ${AZURE_CLIENT_ID}
    oauth2_client_secret: ${AZURE_CLIENT_SECRET}    # omit for public client / device flow
    oauth2_tenant_id: ${AZURE_TENANT_ID}            # Azure AD tenant
    oauth2_scopes:                                   # default below
      - "https://outlook.office365.com/.default"

  gmail-oauth:
    type: imap
    url: imaps://imap.gmail.com:993
    auth_type: oauth2
    username: user@gmail.com

    oauth2_client_id: ${GOOGLE_CLIENT_ID}
    oauth2_client_secret: ${GOOGLE_CLIENT_SECRET}
    oauth2_scopes:
      - "https://mail.google.com/"
```

### `DocumentConfig` additions

```python
class DocumentConfig(BaseModel):
    ...
    # OAuth2 fields
    auth_type: str = "basic"                        # "basic" | "oauth2"
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_tenant_id: Optional[str] = None          # Azure AD
    oauth2_scopes: list[str] = []
    oauth2_token_cache: Optional[str] = None        # path to token cache file
```

### Token Acquisition

#### Microsoft 365 (Azure AD)

Uses `msal` (Microsoft Authentication Library):

```python
import msal

class AzureOAuth2Provider:
    def __init__(self, config: DocumentConfig):
        self._app = msal.ConfidentialClientApplication(
            client_id=config.oauth2_client_id,
            client_credential=config.oauth2_client_secret,
            authority=f"https://login.microsoftonline.com/{config.oauth2_tenant_id}",
            token_cache=self._load_cache(config.oauth2_token_cache),
        )
        self._scopes = config.oauth2_scopes or ["https://outlook.office365.com/.default"]

    def get_access_token(self) -> str:
        # Try silent (cached) first
        result = self._app.acquire_token_silent(self._scopes, account=None)
        if not result:
            # Client credentials flow (daemon/service — no user interaction)
            result = self._app.acquire_token_for_client(scopes=self._scopes)
        if "access_token" not in result:
            raise AuthenticationError(f"OAuth2 failed: {result.get('error_description', 'unknown')}")
        self._save_cache()
        return result["access_token"]
```

For interactive/device-code flow (user-delegated):

```python
    def get_access_token_interactive(self) -> str:
        result = self._app.acquire_token_silent(self._scopes, account=None)
        if not result:
            flow = self._app.initiate_device_flow(scopes=self._scopes)
            print(flow["message"])  # "Go to https://microsoft.com/devicelogin and enter code XXXXXX"
            result = self._app.acquire_token_by_device_flow(flow)
        return result["access_token"]
```

#### Gmail (Google)

Uses `google-auth-oauthlib`:

```python
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

class GoogleOAuth2Provider:
    def __init__(self, config: DocumentConfig):
        self._config = config
        self._scopes = config.oauth2_scopes or ["https://mail.google.com/"]

    def get_access_token(self) -> str:
        creds = self._load_cached_creds()
        if creds and creds.valid:
            return creds.token
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
            self._save_creds(creds)
            return creds.token
        # First-time: interactive browser flow
        flow = InstalledAppFlow.from_client_config(
            {"installed": {
                "client_id": self._config.oauth2_client_id,
                "client_secret": self._config.oauth2_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }},
            scopes=self._scopes,
        )
        creds = flow.run_local_server(port=0)
        self._save_creds(creds)
        return creds.token
```

### IMAP XOAUTH2 Login

Replace `conn.login()` with XOAUTH2 SASL authentication:

```python
import base64

def _imap_oauth2_login(conn: imaplib.IMAP4_SSL, username: str, access_token: str):
    """Authenticate via XOAUTH2 SASL mechanism."""
    auth_string = f"user={username}\x01auth=Bearer {access_token}\x01\x01"
    conn.authenticate("XOAUTH2", lambda _: auth_string.encode())
```

### Updated `IMAPFetcher.connect()`

```python
def connect(self) -> imaplib.IMAP4_SSL:
    parsed = urlparse(self._config.url)
    host = parsed.hostname
    port = parsed.port or 993
    conn = imaplib.IMAP4_SSL(host, port)

    if self._config.auth_type == "oauth2":
        provider = self._get_oauth2_provider()
        token = provider.get_access_token()
        _imap_oauth2_login(conn, self._config.username, token)
    else:
        conn.login(self._config.username, self._config.password)

    return conn

def _get_oauth2_provider(self):
    if self._config.oauth2_tenant_id:
        return AzureOAuth2Provider(self._config)
    else:
        return GoogleOAuth2Provider(self._config)
```

### Azure AD App Registration Setup

To use OAuth2 with M365, the user must register an app in Azure AD:

1. Go to Azure Portal → Azure Active Directory → App registrations → New registration
2. Set redirect URI to `http://localhost` (for device/interactive flow)
3. API permissions → Add → Microsoft Graph → `IMAP.AccessAsUser.All` (delegated) or `Mail.Read` (application)
4. For client credentials (daemon): Certificates & secrets → New client secret
5. Record: Application (client) ID, Directory (tenant) ID, Client secret value

### Token Caching

Tokens are cached to avoid repeated interactive auth:

```python
# Default cache path
~/.constat/oauth2_tokens/<source_name>.json
```

- `msal` has built-in `SerializableTokenCache` — serialize to JSON file
- `google-auth` credentials serialize via `creds.to_json()` / `Credentials.from_authorized_user_info()`
- Cache path configurable via `oauth2_token_cache` field

### New Dependencies (OAuth2 only)

| Package | Purpose | Install |
|---------|---------|---------|
| `msal` | Microsoft OAuth2 token acquisition | `pip install msal` |
| `google-auth-oauthlib` | Google OAuth2 token acquisition | `pip install google-auth-oauthlib` |

Both are optional — only required when `auth_type: oauth2` is configured. Import guarded:

```python
def _get_oauth2_provider(self):
    if self._config.oauth2_tenant_id:
        try:
            import msal
        except ImportError:
            raise ImportError("pip install msal — required for M365 OAuth2")
        return AzureOAuth2Provider(self._config)
    else:
        try:
            import google.oauth2  # noqa: F401
        except ImportError:
            raise ImportError("pip install google-auth-oauthlib — required for Google OAuth2")
        return GoogleOAuth2Provider(self._config)
```

### Additional File Changes

| File | Change |
|------|--------|
| `constat/discovery/doc_tools/_imap.py` | Add `AzureOAuth2Provider`, `GoogleOAuth2Provider`, `_imap_oauth2_login()` |
| `constat/core/config.py` | Add `auth_type`, `oauth2_*` fields to `DocumentConfig` |
| `pyproject.toml` | Add `msal` and `google-auth-oauthlib` as optional deps (`[email-oauth2]`) |

### Testing (OAuth2)

- Mock `msal.ConfidentialClientApplication.acquire_token_for_client()` → return fake token
- Mock `imaplib.IMAP4_SSL.authenticate()` → verify XOAUTH2 auth string format
- Token cache round-trip: write → read → verify silent acquisition succeeds

## Non-Goals (v1)

- SMTP sending / reply composition
- Real-time push (IMAP IDLE) — sync is pull-based on session load
- Calendar invite (`.ics`) extraction
- Email thread reconstruction / conversation threading
- Deduplication across multiple mailboxes
