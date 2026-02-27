# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Transport abstraction for document fetching.

Infers transport from URL scheme or field presence — never configured explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from constat.core.config import DocumentConfig

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of fetching a document via any transport."""

    data: bytes
    detected_mime: str | None = None
    source_path: str | None = None


def infer_transport(config: "DocumentConfig") -> str:
    """Infer transport from config fields.

    Returns: inline, file, http, s3, ftp, sftp
    """
    if config.content is not None:
        return "inline"
    if config.path is not None:
        return "file"
    if config.url:
        scheme = urlparse(config.url).scheme.lower()
        if scheme in ("http", "https"):
            return "http"
        if scheme == "s3":
            return "s3"
        if scheme == "ftp":
            return "ftp"
        if scheme == "sftp":
            return "sftp"
        return "http"  # default for URL
    raise ValueError("DocumentConfig has no content, path, or url")


def fetch_document(config: "DocumentConfig", config_dir: str | None = None) -> FetchResult:
    """Fetch raw bytes via appropriate transport."""
    transport = infer_transport(config)

    if transport == "inline":
        return _fetch_inline(config)
    elif transport == "file":
        return _fetch_file(config, config_dir)
    elif transport == "http":
        return _fetch_http(config)
    elif transport == "s3":
        return _fetch_s3(config)
    elif transport == "ftp":
        return _fetch_ftp(config)
    elif transport == "sftp":
        return _fetch_sftp(config)
    else:
        raise ValueError(f"Unknown transport: {transport}")


def _fetch_inline(config: "DocumentConfig") -> FetchResult:
    """Encode inline content to bytes."""
    content = config.content or ""
    return FetchResult(
        data=content.encode("utf-8"),
        detected_mime=None,
        source_path=None,
    )


def _fetch_file(config: "DocumentConfig", config_dir: str | None) -> FetchResult:
    """Read file from local filesystem."""
    path = Path(config.path)
    if not path.is_absolute() and config_dir:
        path = (Path(config_dir) / config.path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {config.path}")

    return FetchResult(
        data=path.read_bytes(),
        detected_mime=None,
        source_path=str(path),
    )


def _fetch_http(config: "DocumentConfig") -> FetchResult:
    """Fetch via HTTP/HTTPS."""
    import requests

    headers = config.headers or {}
    response = requests.get(config.url, headers=headers, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type")

    return FetchResult(
        data=response.content,
        detected_mime=content_type,
        source_path=config.url,
    )


def _fetch_s3(config: "DocumentConfig") -> FetchResult:
    """Fetch from S3 via boto3."""
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "boto3 is required for S3 transport. Install with: pip install constat[s3]"
        ) from e

    parsed = urlparse(config.url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    session_kwargs = {}
    if config.aws_profile:
        session_kwargs["profile_name"] = config.aws_profile
    if config.aws_region:
        session_kwargs["region_name"] = config.aws_region

    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    return FetchResult(
        data=obj["Body"].read(),
        detected_mime=obj.get("ContentType"),
        source_path=config.url,
    )


def _fetch_ftp(config: "DocumentConfig") -> FetchResult:
    """Fetch via FTP."""
    from ftplib import FTP
    from io import BytesIO

    parsed = urlparse(config.url)
    host = parsed.hostname
    port = config.port or parsed.port or 21
    remote_path = parsed.path

    ftp = FTP()
    ftp.connect(host, port)
    ftp.login(config.username or "anonymous", config.password or "")

    buf = BytesIO()
    ftp.retrbinary(f"RETR {remote_path}", buf.write)
    ftp.quit()

    return FetchResult(
        data=buf.getvalue(),
        detected_mime=None,
        source_path=config.url,
    )


def _fetch_sftp(config: "DocumentConfig") -> FetchResult:
    """Fetch via SFTP using paramiko."""
    try:
        import paramiko
    except ImportError as e:
        raise ImportError(
            "paramiko is required for SFTP transport. Install with: pip install constat[sftp]"
        ) from e

    from io import BytesIO

    parsed = urlparse(config.url)
    host = parsed.hostname
    port = config.port or parsed.port or 22
    remote_path = parsed.path

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict = {
        "hostname": host,
        "port": port,
    }
    if config.username:
        connect_kwargs["username"] = config.username
    if config.password:
        connect_kwargs["password"] = config.password
    if config.key_path:
        connect_kwargs["key_filename"] = config.key_path

    client.connect(**connect_kwargs)
    sftp = client.open_sftp()

    buf = BytesIO()
    sftp.getfo(remote_path, buf)

    sftp.close()
    client.close()

    return FetchResult(
        data=buf.getvalue(),
        detected_mime=None,
        source_path=config.url,
    )
