"""Email utility for sending results from generated code."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, Union
import pandas as pd
import io

from constat.core.config import EmailConfig


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


def create_send_email(config: Optional[EmailConfig]):
    """
    Create a send_email function for injection into execution globals.

    If email is not configured, returns a stub that prints an error message.

    Args:
        config: EmailConfig or None

    Returns:
        A send_email function for use in generated code
    """
    if config is None:
        def send_email_stub(
            to: str,
            subject: str,
            body: str,
            html: bool = False,
            df: Optional[pd.DataFrame] = None,
            attachment_name: str = "data.csv",
        ) -> bool:
            """Email not configured. Add email config to your constat.yaml."""
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
        html: bool = False,
        df: Optional[pd.DataFrame] = None,
        attachment_name: str = "data.csv",
    ) -> bool:
        """
        Send an email with optional DataFrame attachment.

        Args:
            to: Recipient email address (can be comma-separated for multiple)
            subject: Email subject line
            body: Email body text
            html: If True, body is treated as HTML
            df: Optional DataFrame to attach as CSV
            attachment_name: Filename for the DataFrame attachment

        Returns:
            True if sent successfully, False otherwise

        Example:
            send_email(
                to="alice@example.com",
                subject="Monthly Report",
                body="Please find the report attached.",
                df=report_df,
            )
        """
        recipients = [addr.strip() for addr in to.split(",")]
        attachments = None

        if df is not None:
            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode("utf-8")
            attachments = [(attachment_name, csv_data, "text/csv")]

        return sender.send(
            to=recipients,
            subject=subject,
            body=body,
            html=html,
            attachments=attachments,
        )

    return send_email
