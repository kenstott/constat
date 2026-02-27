# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document crawler — link extraction and recursive fetching."""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import TYPE_CHECKING, Callable
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from constat.core.config import DocumentConfig
    from ._transport import FetchResult

from ._mime import is_loadable_mime

logger = logging.getLogger(__name__)

# Regex patterns for link extraction
_HTML_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')

# File extensions that are never crawlable content
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".pdf", ".zip", ".gz", ".tar", ".mp3", ".mp4", ".wav", ".avi",
    ".woff", ".woff2", ".ttf", ".eot", ".css", ".js",
})

# URL path segments that indicate non-content pages (login, auth, special)
_SKIP_PATH_PREFIXES = (
    "/wiki/Special:", "/wiki/Wikipedia:", "/wiki/Talk:",
    "/wiki/User:", "/wiki/Help:", "/wiki/File:",
    "/w/", "/api/",
)


def extract_links(content: str, doc_type: str, base_url: str | None) -> list[str]:
    """Extract links from document content.

    Args:
        content: Document text content
        doc_type: Document type (html, markdown, text, etc.)
        base_url: Base URL for resolving relative links

    Returns:
        List of unique absolute URLs extracted from content (fragments stripped)
    """
    raw_links: list[str] = []

    if doc_type == "html":
        # Raw HTML: extract href attributes
        raw_links = [m.group(1) for m in _HTML_HREF_RE.finditer(content)]
    elif doc_type == "markdown":
        # Markdown: extract [text](url) links
        raw_links = [m.group(2) for m in _MARKDOWN_LINK_RE.finditer(content)]
    else:
        # Office docs / plain text: no crawlable links
        return []

    base_normalized = _normalize_url(base_url) if base_url else None

    # Resolve relative URLs and filter
    seen: set[str] = set()
    resolved = []
    for link in raw_links:
        # Skip anchors, javascript, mailto
        if link.startswith(("#", "javascript:", "mailto:")):
            continue
        if base_url:
            link = urljoin(base_url, link)
        # Strip fragment — we only care about the page, not the anchor
        parsed = urlparse(link)
        if parsed.scheme not in ("http", "https"):
            continue
        clean = parsed._replace(fragment="", query="").geturl()
        # Skip binary file URLs
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in _BINARY_EXTENSIONS):
            continue
        # Skip non-content paths (login, auth, special pages, etc.)
        if any(parsed.path.startswith(prefix) for prefix in _SKIP_PATH_PREFIXES):
            continue
        # Skip self-links (relative or absolute links back to the same page)
        normalized = _normalize_url(clean)
        if normalized == base_normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(clean)

    return resolved


def crawl_document(
    config: "DocumentConfig",
    config_dir: str | None,
    fetch_fn: Callable,
) -> list[tuple[str, "FetchResult"]]:
    """Recursively fetch linked documents via BFS.

    Args:
        config: Root document config (must have url and follow_links=True)
        config_dir: Config directory for relative path resolution
        fetch_fn: Function(config, config_dir) -> FetchResult

    Returns:
        List of (url, FetchResult) pairs including the root document
    """
    from copy import copy
    from ._mime import detect_type_from_source

    max_depth = config.max_depth
    max_documents = config.max_documents
    link_pattern = re.compile(config.link_pattern) if config.link_pattern else None
    same_domain_only = config.same_domain_only
    exclude_res = [re.compile(p) for p in (getattr(config, 'exclude_patterns', None) or [])]

    root_url = config.url
    root_domain = urlparse(root_url).netloc if root_url else None

    # BFS state
    visited: set[str] = set()
    results: list[tuple[str, "FetchResult"]] = []
    queue: deque[tuple[str, int]] = deque()  # (url, depth)

    # Fetch root document
    root_result = fetch_fn(config, config_dir)
    results.append((root_url or "", root_result))
    visited.add(_normalize_url(root_url or ""))

    if not root_url:
        return results

    # Detect type and extract links from root
    root_type = detect_type_from_source(root_result.source_path, root_result.detected_mime)
    try:
        root_content = root_result.data.decode("utf-8")
    except UnicodeDecodeError:
        return results  # Binary root doc, no links to follow

    root_links = extract_links(root_content, root_type, root_url)
    for link in root_links:
        normalized = _normalize_url(link)
        if normalized not in visited:
            queue.append((link, 1))
            visited.add(normalized)

    # BFS traversal
    while queue and len(results) < max_documents:
        url, depth = queue.popleft()

        # Apply filters
        if same_domain_only and root_domain:
            if urlparse(url).netloc != root_domain:
                continue
        if link_pattern and not link_pattern.search(url):
            continue
        if any(ep.search(url) for ep in exclude_res):
            logger.debug(f"Crawler: excluded by pattern: {url}")
            continue

        # Fetch linked document
        linked_config = copy(config)
        linked_config.url = url
        linked_config.content = None
        linked_config.path = None

        try:
            result = fetch_fn(linked_config, config_dir)
        except Exception as e:
            logger.warning(f"Crawler: failed to fetch {url}: {e}")
            continue

        # Skip responses with unloadable MIME types (images, fonts, etc.)
        if not is_loadable_mime(result.detected_mime):
            logger.debug(f"Crawler: skipping unloadable mime {result.detected_mime} for {url}")
            continue

        results.append((url, result))

        # Extract links for next depth level
        if depth < max_depth:
            link_type = detect_type_from_source(result.source_path, result.detected_mime)
            try:
                link_content = result.data.decode("utf-8")
            except UnicodeDecodeError:
                continue

            child_links = extract_links(link_content, link_type, url)
            for child_link in child_links:
                normalized = _normalize_url(child_link)
                if normalized not in visited:
                    queue.append((child_link, depth + 1))
                    visited.add(normalized)

    return results


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication (strip fragment, trailing slash)."""
    parsed = urlparse(url)
    # Strip fragment, normalize path
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"
