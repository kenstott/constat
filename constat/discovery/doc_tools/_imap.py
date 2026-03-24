# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""IMAP email ingestion — fetches messages and extracts attachments/inline images."""

from __future__ import annotations

import base64
import email
import email.policy
import imaplib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from constat.core.config import DocumentConfig

logger = logging.getLogger(__name__)

_MAX_ATTACHMENT_SIZE = 50 * 1024 * 1024  # 50 MB


@dataclass
class EmailAttachment:
    filename: str
    content_type: str
    data: bytes
    content_id: str | None = None  # CID for inline images
    is_inline: bool = False


@dataclass
class EmailMessage:
    uid: str
    message_id: str
    subject: str
    sender: str
    recipients: list[str]
    cc: list[str]
    date: str
    body_text: str
    body_html: str | None
    attachments: list[EmailAttachment] = field(default_factory=list)
    inline_images: list[EmailAttachment] = field(default_factory=list)


def _decode_header_value(raw: str | None) -> str:
    """Decode RFC 2047 encoded header value."""
    if not raw:
        return ""
    decoded_parts = email.header.decode_header(raw)
    parts = []
    for data, charset in decoded_parts:
        if isinstance(data, bytes):
            parts.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(data)
    return " ".join(parts)


def _make_msg_id(uid: str, msg: email.message.Message) -> str:
    """Deterministic message ID: msg_YYYYMMDD_HHMMSS_uid."""
    date_str = msg.get("Date", "")
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        ts = parsed.strftime("%Y%m%d_%H%M%S")
    except (ValueError, TypeError):
        ts = "00000000_000000"
    return f"msg_{ts}_{uid}"


def _extract_body(msg: email.message.Message) -> tuple[str, str | None]:
    """Extract body text from an email message.

    Returns (text, html) — prefers text/plain; falls back to text/html.
    """
    text_body = None
    html_body = None

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            if ct == "text/plain" and text_body is None:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    text_body = payload.decode(charset, errors="replace")
            elif ct == "text/html" and html_body is None:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html_body = payload.decode(charset, errors="replace")
    else:
        ct = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace")
            if ct == "text/html":
                html_body = decoded
            else:
                text_body = decoded

    # If only HTML, convert to markdown
    if text_body is None and html_body is not None:
        from ._file_extractors import _convert_html_to_markdown
        text_body = _convert_html_to_markdown(html_body)

    return text_body or "", html_body


def _extract_parts(
    msg: email.message.Message,
    attachment_types: list[str] | None = None,
) -> tuple[list[EmailAttachment], list[EmailAttachment]]:
    """Walk MIME tree and extract attachments + inline images.

    Returns (attachments, inline_images).
    """
    attachments: list[EmailAttachment] = []
    inline_images: list[EmailAttachment] = []
    seen_filenames: dict[str, int] = {}

    for i, part in enumerate(msg.walk()):
        ct = part.get_content_type()
        maintype = part.get_content_maintype()
        disp = str(part.get("Content-Disposition", ""))

        # Skip multipart containers and text bodies
        if maintype == "multipart":
            continue
        if ct in ("text/plain", "text/html") and "attachment" not in disp:
            continue

        # Handle nested message/rfc822
        if ct == "message/rfc822":
            payload = part.get_payload()
            if isinstance(payload, list) and payload:
                inner_msg = payload[0]
                inner_atts, inner_imgs = _extract_parts(inner_msg, attachment_types)
                for att in inner_atts:
                    att.filename = f"fwd_{att.filename}"
                    attachments.append(att)
                for img in inner_imgs:
                    img.filename = f"fwd_{img.filename}"
                    inline_images.append(img)
            continue

        data = part.get_payload(decode=True)
        if data is None:
            continue

        if len(data) > _MAX_ATTACHMENT_SIZE:
            logger.warning("Skipping attachment >50MB: %s (%d bytes)", ct, len(data))
            continue

        # Determine filename
        filename = part.get_filename()
        if filename:
            filename = _decode_header_value(filename)
        if not filename:
            import mimetypes
            ext = mimetypes.guess_extension(ct) or ".bin"
            filename = f"part_{i}{ext}"

        # Deduplicate filenames
        if filename in seen_filenames:
            seen_filenames[filename] += 1
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem}_{seen_filenames[filename]}{suffix}"
        else:
            seen_filenames[filename] = 1

        # Filter by allowed attachment types
        if attachment_types:
            suffix = Path(filename).suffix.lstrip(".").lower()
            if suffix not in [t.lower().lstrip(".") for t in attachment_types]:
                continue

        content_id = part.get("Content-ID")
        if content_id:
            content_id = content_id.strip("<>")

        is_inline = "inline" in disp or (content_id is not None and maintype == "image")

        att = EmailAttachment(
            filename=filename,
            content_type=ct,
            data=data,
            content_id=content_id,
            is_inline=is_inline,
        )

        if is_inline and maintype == "image":
            inline_images.append(att)
        else:
            attachments.append(att)

    return attachments, inline_images


def _parse_message(uid: str, raw_bytes: bytes) -> EmailMessage:
    """Parse raw RFC822 bytes into an EmailMessage."""
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)

    subject = _decode_header_value(msg.get("Subject"))
    sender = _decode_header_value(msg.get("From", ""))
    to_raw = msg.get_all("To", [])
    cc_raw = msg.get_all("Cc", [])

    recipients = [_decode_header_value(str(r)) for r in to_raw]
    cc = [_decode_header_value(str(r)) for r in cc_raw]
    date = _decode_header_value(msg.get("Date", ""))

    body_text, body_html = _extract_body(msg)
    attachments, inline_images = _extract_parts(msg)

    message_id = _make_msg_id(uid, msg)

    return EmailMessage(
        uid=uid,
        message_id=message_id,
        subject=subject,
        sender=sender,
        recipients=recipients,
        cc=cc,
        date=date,
        body_text=body_text,
        body_html=body_html,
        attachments=attachments,
        inline_images=inline_images,
    )


def _render_email(msg: EmailMessage, include_headers: bool = True) -> str:
    """Render an EmailMessage as markdown text."""
    lines = []
    lines.append(f"# {msg.subject}")
    lines.append("")

    if include_headers:
        lines.append(f"From: {msg.sender}")
        lines.append(f"To: {', '.join(msg.recipients)}")
        if msg.cc:
            lines.append(f"CC: {', '.join(msg.cc)}")
        lines.append(f"Date: {msg.date}")
        lines.append("")

    lines.append(msg.body_text)

    if msg.attachments:
        lines.append("")
        lines.append("## Attachments")
        for att in msg.attachments:
            lines.append(f"- {att.filename} ({att.content_type})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OAuth2 providers
# ---------------------------------------------------------------------------

def _imap_oauth2_login(conn: imaplib.IMAP4_SSL, username: str, access_token: str) -> None:
    """Authenticate via XOAUTH2 SASL mechanism."""
    auth_string = f"user={username}\x01auth=Bearer {access_token}\x01\x01"
    conn.authenticate("XOAUTH2", lambda _: auth_string.encode())


class RefreshTokenOAuth2Provider:
    """OAuth2 token provider using refresh tokens (browser flow)."""

    def __init__(self, config: "DocumentConfig"):
        self._config = config

    def get_access_token(self) -> str:
        import httpx

        refresh_token = self._config.oauth2_client_secret
        client_id = self._config.oauth2_client_id

        if self._config.oauth2_tenant_id:
            # Microsoft
            tenant = self._config.oauth2_tenant_id
            resp = httpx.post(
                f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "scope": "https://outlook.office365.com/IMAP.AccessAsUser.All offline_access",
                },
            )
        else:
            # Google — client_secret stored in password field for refresh flow
            data = {
                "client_id": client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
            if self._config.password:
                data["client_secret"] = self._config.password
            resp = httpx.post(
                "https://oauth2.googleapis.com/token",
                data=data,
            )

        resp.raise_for_status()
        data = resp.json()
        if "access_token" not in data:
            raise RuntimeError(f"Token refresh failed: {data}")
        return data["access_token"]


class AzureOAuth2Provider:
    """Azure AD / M365 OAuth2 token provider using MSAL client credentials."""

    def __init__(self, config: "DocumentConfig"):
        try:
            import msal
        except ImportError as e:
            raise ImportError(
                "msal is required for M365 OAuth2. Install with: pip install msal"
            ) from e

        self._config = config
        scopes = config.oauth2_scopes or ["https://outlook.office365.com/.default"]

        cache = msal.SerializableTokenCache()
        if config.oauth2_token_cache:
            cache_path = Path(config.oauth2_token_cache)
            if cache_path.exists():
                cache.deserialize(cache_path.read_text())

        authority = f"https://login.microsoftonline.com/{config.oauth2_tenant_id}"
        self._app = msal.ConfidentialClientApplication(
            config.oauth2_client_id,
            authority=authority,
            client_credential=config.oauth2_client_secret,
            token_cache=cache,
        )
        self._scopes = scopes
        self._cache = cache
        self._cache_path = Path(config.oauth2_token_cache) if config.oauth2_token_cache else None

    def get_access_token(self) -> str:
        result = self._app.acquire_token_silent(self._scopes, account=None)
        if not result:
            result = self._app.acquire_token_for_client(scopes=self._scopes)

        if "access_token" not in result:
            raise RuntimeError(f"Azure OAuth2 token acquisition failed: {result.get('error_description', result)}")

        # Persist cache
        if self._cache_path and self._cache.has_state_changed:
            self._cache_path.write_text(self._cache.serialize())

        return result["access_token"]


class GoogleOAuth2Provider:
    """Google OAuth2 token provider using google-auth."""

    def __init__(self, config: "DocumentConfig"):
        try:
            from google.oauth2.credentials import Credentials
        except ImportError as e:
            raise ImportError(
                "google-auth is required for Gmail OAuth2. Install with: pip install google-auth-oauthlib"
            ) from e

        scopes = config.oauth2_scopes or ["https://mail.google.com/"]
        self._credentials = Credentials(
            token=None,
            refresh_token=config.oauth2_client_secret,  # stored as "secret" in config
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.oauth2_client_id,
            client_secret=config.oauth2_client_secret,
            scopes=scopes,
        )

    def get_access_token(self) -> str:
        from google.auth.transport.requests import Request
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials.token


# ---------------------------------------------------------------------------
# IMAPFetcher
# ---------------------------------------------------------------------------

class IMAPFetcher:
    """Connects to an IMAP server and fetches messages."""

    def __init__(self, config: "DocumentConfig", config_dir: Path | None = None):
        self._config = config
        self._config_dir = config_dir

    def connect(self) -> imaplib.IMAP4_SSL:
        """Open an IMAP4_SSL connection, authenticate, and select mailbox."""
        parsed = urlparse(self._config.url)
        host = parsed.hostname
        port = self._config.port or parsed.port or 993
        username = self._config.username or parsed.username
        password = self._config.password or parsed.password

        logger.info("[IMAP] Connecting to %s:%d as %s", host, port, username)
        conn = imaplib.IMAP4_SSL(host, port)

        if self._config.auth_type in ("oauth2", "oauth2_refresh"):
            logger.info("[IMAP] Authenticating via %s", self._config.auth_type)
            token = self._get_oauth2_token()
            _imap_oauth2_login(conn, username, token)
        else:
            logger.info("[IMAP] Authenticating via basic login")
            conn.login(username, password)

        logger.info("[IMAP] Selecting mailbox: %s", self._config.mailbox)
        conn.select(self._config.mailbox)
        return conn

    def _get_oauth2_token(self) -> str:
        """Get an OAuth2 access token from the configured provider."""
        if self._config.auth_type == "oauth2_refresh":
            return RefreshTokenOAuth2Provider(self._config).get_access_token()
        if self._config.oauth2_tenant_id:
            return AzureOAuth2Provider(self._config).get_access_token()
        else:
            return GoogleOAuth2Provider(self._config).get_access_token()

    def _build_search_criteria(self) -> list[str]:
        """Build IMAP SEARCH criteria list."""
        parts = []
        criteria = self._config.search_criteria.strip()
        if criteria and criteria != "ALL":
            parts.append(criteria)

        if self._config.since:
            # IMAP SINCE format: DD-Mon-YYYY
            try:
                dt = datetime.strptime(self._config.since, "%Y-%m-%d")
                imap_date = dt.strftime("%d-%b-%Y")
                parts.append(f"SINCE {imap_date}")
            except ValueError:
                parts.append(f"SINCE {self._config.since}")

        if not parts:
            return ["ALL"]
        return parts

    def fetch_messages(self) -> list[EmailMessage]:
        """Connect, search, and fetch messages. Returns parsed EmailMessages."""
        conn = self.connect()
        try:
            criteria = self._build_search_criteria()
            search_str = " ".join(criteria) if len(criteria) > 1 else criteria[0]

            # Parenthesize compound criteria
            if len(criteria) > 1:
                search_str = f"({search_str})"

            logger.info("[IMAP] Searching: %s", search_str)
            status, data = conn.search(None, search_str)
            if status != "OK":
                raise RuntimeError(f"IMAP SEARCH failed: {status}")

            uids = data[0].split() if data[0] else []
            logger.info("[IMAP] Search returned %d messages (max_messages=%d)",
                        len(uids), self._config.max_messages)

            # Cap at max_messages (take most recent)
            if len(uids) > self._config.max_messages:
                uids = uids[-self._config.max_messages:]
                logger.info("[IMAP] Capped to %d most recent messages", len(uids))

            messages = []
            total = len(uids)
            for i, uid_bytes in enumerate(uids):
                uid = uid_bytes.decode()
                status, msg_data = conn.fetch(uid_bytes, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    logger.warning("[IMAP] Failed to fetch UID %s: status=%s", uid, status)
                    continue
                raw = msg_data[0][1]
                if isinstance(raw, bytes):
                    messages.append(_parse_message(uid, raw))
                if (i + 1) % 50 == 0:
                    logger.info("[IMAP] Fetched %d/%d messages", i + 1, total)

            logger.info("[IMAP] Fetched and parsed %d messages", len(messages))
            return messages
        finally:
            try:
                conn.close()
                conn.logout()
            except Exception:
                pass
