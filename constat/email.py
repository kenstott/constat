# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Email utility for sending results from generated code."""

import io
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable, Literal, Optional, Union

import pandas as pd

from constat.core.config import EmailConfig


class SensitiveDataError(Exception):
    """Raised when attempting to email sensitive data without explicit consent."""
    pass


# Type for checking if current context is sensitive
# Returns True if sensitive data is involved
SensitivityChecker = Callable[[], bool]


# Basic email-safe CSS for rendered Markdown
EMAIL_CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; color: #333; }
h1, h2, h3 { color: #2c3e50; margin-top: 1em; margin-bottom: 0.5em; }
p { margin: 0.5em 0; }
ul, ol { margin: 0.5em 0; padding-left: 1.5em; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #f5f5f5; }
code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
pre { background-color: #f4f4f4; padding: 1em; border-radius: 5px; overflow-x: auto; }
blockquote { border-left: 4px solid #ddd; margin: 1em 0; padding-left: 1em; color: #666; }
</style>
"""


def markdown_to_html(text: str) -> str:
    """Convert Markdown text to HTML with email-safe styling.

    Args:
        text: Markdown formatted text

    Returns:
        HTML string with inline-friendly CSS
    """
    try:
        import markdown
        html_body = markdown.markdown(
            text,
            extensions=['tables', 'fenced_code', 'nl2br']
        )
        return f"<!DOCTYPE html><html><head>{EMAIL_CSS}</head><body>{html_body}</body></html>"
    except ImportError:
        # Fall back to basic HTML wrapping if markdown not installed
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_body = f"<pre>{escaped}</pre>"
        return f"<!DOCTYPE html><html><head>{EMAIL_CSS}</head><body>{html_body}</body></html>"


class EmailSender:
    """Email sender configured from EmailConfig."""

    def __init__(self, config: EmailConfig):
        self.config = config

    def send(
        self,
        to: Union[str, list[str]],
        subject: str,
        body: str,
        html: bool = False,
        attachments: Optional[list[tuple[str, bytes, str]]] = None,
    ) -> bool:
        """
        Send an email.

        Args:
            to: Recipient email address(es)
            subject: Email subject
            body: Email body (plain text or HTML)
            html: If True, body is treated as HTML
            attachments: List of (filename, data, mime_type) tuples

        Returns:
            True if sent successfully, False otherwise
        """
        if isinstance(to, str):
            to = [to]

        msg = MIMEMultipart()
        msg["From"] = f"{self.config.from_name} <{self.config.from_address}>"
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject

        # Attach body
        if html:
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        # Attach files
        if attachments:
            for filename, data, mime_type in attachments:
                part = MIMEBase(*mime_type.split("/", 1))
                part.set_payload(data)
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{filename}"',
                )
                msg.attach(part)

        # Send
        try:
            if self.config.tls:
                server = smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout_seconds,
                )
                server.starttls()
            else:
                server = smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout_seconds,
                )

            if self.config.smtp_user and self.config.smtp_password:
                server.login(self.config.smtp_user, self.config.smtp_password)

            server.sendmail(self.config.from_address, to, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            print(f"Email error: {e}")
            return False


def create_send_email(
    config: Optional[EmailConfig],
    is_sensitive: Optional[SensitivityChecker] = None,
):
    """
    Create a send_email function for injection into execution globals.

    If email is not configured, returns a stub that prints an error message.

    Args:
        config: EmailConfig or None
        is_sensitive: Optional function that returns True if the current execution
            context involves sensitive data. Used to block emailing confidential
            data without explicit authorization.

    Returns:
        A send_email function for use in generated code
    """
    if config is None:
        def send_email_stub(
            to: str,
            subject: str,
            body: str,
            format: Literal["plain", "markdown", "html"] = "plain",
            html: bool = False,
            df: Optional[pd.DataFrame] = None,
            attachment_name: str = "data.csv",
            allow_sensitive: bool = False,
        ) -> bool:
            """Email not configured. Add email config to your constat.yaml."""
            # Still check sensitivity even if email not configured
            if is_sensitive is not None and not allow_sensitive:
                if is_sensitive():
                    raise SensitiveDataError(
                        "BLOCKED: This analysis involves sensitive/confidential data.\n\n"
                        "Sensitive data should not be emailed without explicit authorization."
                    )
            print("ERROR: Email not configured. Add email section to your config file.")
            print("Example:")
            print("  email:")
            print("    smtp_host: smtp.gmail.com")
            print("    smtp_port: 587")
            print("    smtp_user: ${EMAIL_USER}")
            print("    smtp_password: ${EMAIL_PASSWORD}")
            print("    from_address: noreply@company.com")
            print("    tls: true")
            return False
        return send_email_stub

    sender = EmailSender(config)

    def send_email(
        to: str,
        subject: str,
        body: str,
        format: Literal["plain", "markdown", "html"] = "plain",
        html: bool = False,  # Deprecated, use format="html" instead
        df: Optional[pd.DataFrame] = None,
        attachment_name: str = "data.csv",
        allow_sensitive: bool = False,
    ) -> bool:
        """
        Send an email with optional DataFrame attachment.

        Args:
            to: Recipient email address (can be comma-separated for multiple)
            subject: Email subject line
            body: Email body text (plain text, Markdown, or HTML)
            format: Body format - "plain" (default), "markdown", or "html"
                - "plain": Send as plain text
                - "markdown": Convert Markdown to styled HTML
                - "html": Send body as raw HTML
            html: DEPRECATED - use format="html" instead
            df: Optional DataFrame to attach as CSV
            attachment_name: Filename for the DataFrame attachment
            allow_sensitive: If True, bypasses sensitivity check.
                WARNING: Only set this to True if you have verified the recipient
                is authorized to receive this data.

        Returns:
            True if sent successfully, False otherwise

        Raises:
            SensitiveDataError: If sensitive data is detected and allow_sensitive=False

        Example:
            # Plain text email
            send_email(
                to="alice@example.com",
                subject="Monthly Report",
                body="Please find the report attached.",
                df=report_df,
            )

            # Markdown formatted email
            send_email(
                to="team@example.com",
                subject="Weekly Summary",
                body=\"\"\"# Weekly Summary

## Key Metrics
- Revenue: $1.2M
- Orders: 450

## Action Items
1. Review pending orders
2. Update forecasts
\"\"\",
                format="markdown",
            )
        """
        # Check if current context involves sensitive data
        if is_sensitive is not None and not allow_sensitive:
            if is_sensitive():
                raise SensitiveDataError(
                    "BLOCKED: This analysis involves sensitive/confidential data.\n\n"
                    "Sensitive data should not be emailed without explicit authorization.\n"
                    "If you need to send this data:\n"
                    "1. Verify the recipient is authorized to receive it\n"
                    "2. Consider whether email is the appropriate channel\n"
                    "3. Use allow_sensitive=True only after manual review"
                )

        recipients = [addr.strip() for addr in to.split(",")]
        attachments = None

        if df is not None:
            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode("utf-8")
            attachments = [(attachment_name, csv_data, "text/csv")]

        # Handle format conversion
        send_as_html = html  # Support deprecated html parameter
        final_body = body

        if format == "markdown":
            final_body = markdown_to_html(body)
            send_as_html = True
        elif format == "html":
            send_as_html = True

        return sender.send(
            to=recipients,
            subject=subject,
            body=final_body,
            html=send_as_html,
            attachments=attachments,
        )

    return send_email
