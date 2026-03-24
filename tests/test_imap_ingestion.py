# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for IMAP email ingestion pipeline."""

from __future__ import annotations

import email
import email.policy
import imaplib
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from constat.discovery.doc_tools._imap import (
    EmailAttachment,
    EmailMessage,
    IMAPFetcher,
    _decode_header_value,
    _extract_body,
    _extract_parts,
    _make_msg_id,
    _parse_message,
    _render_email,
    _imap_oauth2_login,
    AzureOAuth2Provider,
    GoogleOAuth2Provider,
)


# ---------------------------------------------------------------------------
# Fixtures: programmatic email construction
# ---------------------------------------------------------------------------

def _make_plain_email(
    subject: str = "Test Subject",
    sender: str = "alice@example.com",
    to: str = "bob@example.com",
    cc: str = "",
    date: str = "Mon, 01 Jan 2026 10:00:00 +0000",
    body: str = "Hello, this is a test email.",
) -> bytes:
    """Create a simple text/plain email as RFC822 bytes."""
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    if cc:
        msg["Cc"] = cc
    msg["Date"] = date
    msg["Message-ID"] = "<test-001@example.com>"
    return msg.as_bytes()


def _make_html_email(
    body_html: str = "<html><body><h1>Hello</h1><p>World</p></body></html>",
    inline_image: bytes | None = None,
) -> bytes:
    """Create an HTML email with optional inline CID image."""
    if inline_image:
        msg = MIMEMultipart("related")
        html_part = MIMEText(
            '<html><body><h1>Hello</h1><img src="cid:logo123"></body></html>',
            "html",
        )
        msg.attach(html_part)
        img = MIMEImage(inline_image, "png")
        img.add_header("Content-ID", "<logo123>")
        img.add_header("Content-Disposition", "inline", filename="logo.png")
        msg.attach(img)
    else:
        msg = MIMEText(body_html, "html")

    msg["Subject"] = "HTML Email"
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Date"] = "Mon, 01 Jan 2026 10:00:00 +0000"
    msg["Message-ID"] = "<test-html@example.com>"
    return msg.as_bytes()


def _make_multipart_email() -> bytes:
    """Create multipart/mixed: text body + PDF attachment + PNG attachment + inline JPEG."""
    msg = MIMEMultipart("mixed")
    msg["Subject"] = "Report Q4"
    msg["From"] = "reports@example.com"
    msg["To"] = "team@example.com"
    msg["Cc"] = "manager@example.com"
    msg["Date"] = "Tue, 15 Jan 2026 14:30:00 +0000"
    msg["Message-ID"] = "<test-multi@example.com>"

    # Text body
    msg.attach(MIMEText("Please find the Q4 report attached.", "plain"))

    # PDF attachment
    pdf_att = MIMEBase("application", "pdf")
    pdf_att.set_payload(b"%PDF-1.4 fake pdf content")
    encoders.encode_base64(pdf_att)
    pdf_att.add_header("Content-Disposition", "attachment", filename="report.pdf")
    msg.attach(pdf_att)

    # PNG attachment
    png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    png_att = MIMEImage(png_data, "png")
    png_att.add_header("Content-Disposition", "attachment", filename="chart.png")
    msg.attach(png_att)

    # Inline JPEG
    jpeg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 50
    jpeg_inline = MIMEImage(jpeg_data, "jpeg")
    jpeg_inline.add_header("Content-ID", "<chart-inline>")
    jpeg_inline.add_header("Content-Disposition", "inline", filename="inline-chart.jpg")
    msg.attach(jpeg_inline)

    return msg.as_bytes()


# ---------------------------------------------------------------------------
# Test: _parse_message
# ---------------------------------------------------------------------------

class TestParseMessage:
    def test_plain_email(self):
        raw = _make_plain_email()
        msg = _parse_message("1", raw)
        assert msg.uid == "1"
        assert msg.subject == "Test Subject"
        assert msg.sender == "alice@example.com"
        assert "bob@example.com" in msg.recipients[0]
        assert msg.body_text == "Hello, this is a test email."
        assert msg.attachments == []
        assert msg.inline_images == []

    def test_multipart_email(self):
        raw = _make_multipart_email()
        msg = _parse_message("42", raw)
        assert msg.subject == "Report Q4"
        assert msg.sender == "reports@example.com"
        assert msg.body_text == "Please find the Q4 report attached."
        assert len(msg.attachments) == 2  # PDF + PNG
        assert msg.attachments[0].filename == "report.pdf"
        assert msg.attachments[0].content_type == "application/pdf"
        assert msg.attachments[1].filename == "chart.png"
        assert len(msg.inline_images) == 1
        assert msg.inline_images[0].content_id == "chart-inline"

    def test_html_with_inline_image(self):
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        raw = _make_html_email(inline_image=png_data)
        msg = _parse_message("5", raw)
        assert "Hello" in msg.body_text  # HTML converted to markdown
        assert len(msg.inline_images) == 1
        assert msg.inline_images[0].content_id == "logo123"
        assert msg.inline_images[0].filename == "logo.png"


# ---------------------------------------------------------------------------
# Test: _extract_body
# ---------------------------------------------------------------------------

class TestExtractBody:
    def test_plain_only(self):
        msg = email.message_from_bytes(_make_plain_email(), policy=email.policy.default)
        text, html = _extract_body(msg)
        assert text == "Hello, this is a test email."
        assert html is None

    def test_html_only(self):
        msg = email.message_from_bytes(
            _make_html_email("<html><body><p>Test</p></body></html>"),
            policy=email.policy.default,
        )
        text, html = _extract_body(msg)
        assert "Test" in text  # HTML converted to markdown
        assert html is not None

    def test_multipart_alternative_prefers_plain(self):
        outer = MIMEMultipart("alternative")
        outer.attach(MIMEText("Plain version", "plain"))
        outer.attach(MIMEText("<p>HTML version</p>", "html"))
        outer["Subject"] = "Alt"
        outer["From"] = "a@b.com"
        outer["To"] = "c@d.com"
        outer["Date"] = "Mon, 01 Jan 2026 10:00:00 +0000"

        msg = email.message_from_bytes(outer.as_bytes(), policy=email.policy.default)
        text, html = _extract_body(msg)
        assert text == "Plain version"
        assert html is not None
        assert "HTML version" in html


# ---------------------------------------------------------------------------
# Test: _extract_parts
# ---------------------------------------------------------------------------

class TestExtractParts:
    def test_attachment_filtering(self):
        raw = _make_multipart_email()
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        # Filter to PDF only
        atts, imgs = _extract_parts(msg, attachment_types=["pdf"])
        assert len(atts) == 1
        assert atts[0].filename == "report.pdf"

    def test_duplicate_filenames(self):
        outer = MIMEMultipart("mixed")
        outer["Subject"] = "Dup"
        outer["From"] = "a@b.com"
        outer["To"] = "c@d.com"
        outer["Date"] = "Mon, 01 Jan 2026 10:00:00 +0000"
        outer.attach(MIMEText("body", "plain"))

        for _ in range(3):
            att = MIMEBase("application", "pdf")
            att.set_payload(b"pdf data")
            encoders.encode_base64(att)
            att.add_header("Content-Disposition", "attachment", filename="doc.pdf")
            outer.attach(att)

        msg = email.message_from_bytes(outer.as_bytes(), policy=email.policy.default)
        atts, _ = _extract_parts(msg)
        filenames = [a.filename for a in atts]
        assert filenames[0] == "doc.pdf"
        assert filenames[1] == "doc_2.pdf"
        assert filenames[2] == "doc_3.pdf"

    def test_inline_image_cid(self):
        raw = _make_multipart_email()
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        _, imgs = _extract_parts(msg)
        assert len(imgs) == 1
        assert imgs[0].content_id == "chart-inline"
        assert imgs[0].is_inline is True


# ---------------------------------------------------------------------------
# Test: _render_email
# ---------------------------------------------------------------------------

class TestRenderEmail:
    def test_with_headers(self):
        raw = _make_plain_email()
        parsed = _parse_message("1", raw)
        rendered = _render_email(parsed, include_headers=True)
        assert "# Test Subject" in rendered
        assert "From: alice@example.com" in rendered
        assert "Hello, this is a test email." in rendered

    def test_without_headers(self):
        raw = _make_plain_email()
        parsed = _parse_message("1", raw)
        rendered = _render_email(parsed, include_headers=False)
        assert "# Test Subject" in rendered
        assert "From:" not in rendered

    def test_attachments_listed(self):
        raw = _make_multipart_email()
        parsed = _parse_message("1", raw)
        rendered = _render_email(parsed, include_headers=True)
        assert "## Attachments" in rendered
        assert "report.pdf" in rendered


# ---------------------------------------------------------------------------
# Test: _make_msg_id
# ---------------------------------------------------------------------------

class TestMakeMessageId:
    def test_deterministic(self):
        raw = _make_plain_email(date="Mon, 01 Jan 2026 10:00:00 +0000")
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        mid1 = _make_msg_id("100", msg)
        mid2 = _make_msg_id("100", msg)
        assert mid1 == mid2
        assert mid1.startswith("msg_20260101_")
        assert mid1.endswith("_100")

    def test_different_uid(self):
        raw = _make_plain_email()
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        mid1 = _make_msg_id("1", msg)
        mid2 = _make_msg_id("2", msg)
        assert mid1 != mid2

    def test_missing_date(self):
        msg_obj = MIMEText("body", "plain")
        msg_obj["Subject"] = "No date"
        msg_obj["From"] = "a@b.com"
        msg_obj["To"] = "c@d.com"
        # No Date header
        parsed = email.message_from_bytes(msg_obj.as_bytes(), policy=email.policy.default)
        mid = _make_msg_id("1", parsed)
        assert "00000000_000000" in mid


# ---------------------------------------------------------------------------
# Test: _build_search_criteria
# ---------------------------------------------------------------------------

class TestBuildSearchCriteria:
    def _make_fetcher(self, **kwargs):
        from constat.core.config import DocumentConfig
        defaults = {"url": "imaps://mail.example.com", "username": "user", "password": "pass"}
        defaults.update(kwargs)
        config = DocumentConfig(**defaults)
        return IMAPFetcher(config)

    def test_default_all(self):
        fetcher = self._make_fetcher()
        assert fetcher._build_search_criteria() == ["ALL"]

    def test_custom_criteria(self):
        fetcher = self._make_fetcher(search_criteria="UNSEEN")
        criteria = fetcher._build_search_criteria()
        assert "UNSEEN" in criteria

    def test_since_date(self):
        fetcher = self._make_fetcher(since="2026-01-01")
        criteria = fetcher._build_search_criteria()
        assert any("SINCE" in c for c in criteria)
        assert any("01-Jan-2026" in c for c in criteria)

    def test_combined(self):
        fetcher = self._make_fetcher(search_criteria="UNSEEN", since="2026-01-01")
        criteria = fetcher._build_search_criteria()
        assert len(criteria) == 2


# ---------------------------------------------------------------------------
# Test: OAuth2
# ---------------------------------------------------------------------------

class TestOAuth2Login:
    def test_xoauth2_auth_string(self):
        conn = MagicMock(spec=imaplib.IMAP4_SSL)
        _imap_oauth2_login(conn, "user@example.com", "fake-token")
        conn.authenticate.assert_called_once()
        call_args = conn.authenticate.call_args
        assert call_args[0][0] == "XOAUTH2"
        # Call the lambda to verify auth string
        auth_fn = call_args[0][1]
        auth_bytes = auth_fn(None)
        assert b"user=user@example.com" in auth_bytes
        assert b"auth=Bearer fake-token" in auth_bytes


class TestAzureOAuth2Provider:
    def test_token_acquisition(self):
        import sys
        from constat.core.config import DocumentConfig

        mock_msal = MagicMock()
        mock_cache = MagicMock()
        mock_cache.has_state_changed = False
        mock_msal.SerializableTokenCache.return_value = mock_cache

        mock_app = MagicMock()
        mock_app.acquire_token_silent.return_value = None
        mock_app.acquire_token_for_client.return_value = {"access_token": "azure-token-123"}
        mock_msal.ConfidentialClientApplication.return_value = mock_app

        with patch.dict(sys.modules, {"msal": mock_msal}):
            # Re-import to pick up mocked msal
            from constat.discovery.doc_tools._imap import AzureOAuth2Provider as AzureProvider

            config = DocumentConfig(
                url="imaps://outlook.office365.com",
                username="user@corp.com",
                password="unused",
                auth_type="oauth2",
                oauth2_client_id="client-id",
                oauth2_client_secret="client-secret",
                oauth2_tenant_id="tenant-id",
            )

            provider = AzureProvider(config)
            token = provider.get_access_token()
            assert token == "azure-token-123"
            mock_app.acquire_token_for_client.assert_called_once()


class TestGoogleOAuth2Provider:
    def test_token_refresh(self):
        import sys
        from constat.core.config import DocumentConfig

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.token = "google-token-456"

        mock_google_oauth2_creds = MagicMock()
        mock_google_oauth2_creds.Credentials.return_value = mock_creds

        mock_google_auth_transport = MagicMock()

        with patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.oauth2": MagicMock(),
            "google.oauth2.credentials": mock_google_oauth2_creds,
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": mock_google_auth_transport,
        }):
            from constat.discovery.doc_tools._imap import GoogleOAuth2Provider as GoogleProvider

            config = DocumentConfig(
                url="imaps://imap.gmail.com",
                username="user@gmail.com",
                password="unused",
                auth_type="oauth2",
                oauth2_client_id="google-client-id",
                oauth2_client_secret="google-refresh-token",
            )

            provider = GoogleProvider(config)
            token = provider.get_access_token()
            assert token == "google-token-456"
            mock_creds.refresh.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Full pipeline integration (mocked IMAP)
# ---------------------------------------------------------------------------

class TestIMAPPipeline:
    """Integration test: mock imaplib, verify full fetch→parse→loaded_documents flow."""

    def test_full_pipeline(self):
        from constat.core.config import DocumentConfig

        # Build canned IMAP responses
        plain_bytes = _make_plain_email(subject="Meeting Notes", body="Discuss Q4 results")
        multi_bytes = _make_multipart_email()

        mock_conn = MagicMock(spec=imaplib.IMAP4_SSL)
        mock_conn.search.return_value = ("OK", [b"1 2"])
        mock_conn.fetch.side_effect = [
            ("OK", [(b"1", plain_bytes)]),
            ("OK", [(b"2", multi_bytes)]),
        ]

        config = DocumentConfig(
            url="imaps://mail.example.com",
            username="user",
            password="pass",
            include_headers=True,
            extract_attachments=True,
            extract_images=False,  # skip image pipeline for this test
        )

        fetcher = IMAPFetcher(config)
        with patch.object(fetcher, "connect", return_value=mock_conn):
            messages = fetcher.fetch_messages()

        assert len(messages) == 2

        # First message: plain email
        m1 = messages[0]
        assert m1.subject == "Meeting Notes"
        assert m1.body_text == "Discuss Q4 results"
        assert m1.attachments == []

        # Second message: multipart with attachments
        m2 = messages[1]
        assert m2.subject == "Report Q4"
        assert len(m2.attachments) == 2
        assert m2.attachments[0].filename == "report.pdf"
        assert len(m2.inline_images) == 1

        # Verify addressing scheme
        rendered = _render_email(m1, include_headers=True)
        assert "# Meeting Notes" in rendered
        assert "From: alice@example.com" in rendered

    def test_max_messages_cap(self):
        from constat.core.config import DocumentConfig

        mock_conn = MagicMock(spec=imaplib.IMAP4_SSL)
        mock_conn.search.return_value = ("OK", [b"1 2 3 4 5"])

        raw = _make_plain_email()
        mock_conn.fetch.return_value = ("OK", [(b"1", raw)])

        config = DocumentConfig(
            url="imaps://mail.example.com",
            username="user",
            password="pass",
            max_messages=2,
        )

        fetcher = IMAPFetcher(config)
        with patch.object(fetcher, "connect", return_value=mock_conn):
            messages = fetcher.fetch_messages()

        # Should only fetch last 2 UIDs (4 and 5)
        assert mock_conn.fetch.call_count == 2

    def test_addressing_scheme(self):
        """Verify colon-separated addressing: config_key:message_id:attachment."""
        raw = _make_multipart_email()
        msg = _parse_message("10", raw)

        config_key = "company_inbox"
        msg_name = f"{config_key}:{msg.message_id}"
        assert msg_name.startswith("company_inbox:msg_")

        for att in msg.attachments:
            att_name = f"{msg_name}:{att.filename}"
            assert att_name.count(":") == 2

        for img in msg.inline_images:
            img_name = f"{msg_name}:{img.content_id or img.filename}"
            assert img_name.count(":") == 2
