# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tiered incremental sync detection for MCP resources.

Detection tiers (in order of preference):
1. **Content hash** — exact change detection via stored hash vs. current hash.
2. **Last-modified / size** — cheap metadata comparison from resource listings.
3. **URI presence** — detect added/removed resources when no metadata available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceMeta:
    """Stored metadata for a single resource."""
    last_modified: Optional[str] = None
    size: Optional[int] = None
    content_hash: Optional[str] = None


@dataclass
class ProbeResult:
    """Result of comparing current resources against stored state."""
    added: list[dict] = field(default_factory=list)
    changed: list[dict] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


class ChangeProbe:
    """Detects added/changed/removed MCP resources using tiered comparison.

    Usage::

        probe = ChangeProbe()
        # After first fetch, store hashes
        probe.update_hash("file:///a.txt", "sha256:abc...")

        # On subsequent sync, compare
        result = probe.probe(current_resources)
        # result.added / .changed / .removed / .unchanged
    """

    def __init__(self) -> None:
        self._stored: dict[str, ResourceMeta] = {}

    def probe(self, current_resources: list[dict]) -> ProbeResult:
        """Compare *current_resources* against stored metadata.

        Each resource dict should have at minimum a ``uri`` key, and
        optionally ``lastModified``, ``size``, and ``contentHash``.

        Returns a :class:`ProbeResult` categorising every resource.
        """
        result = ProbeResult()
        seen_uris: set[str] = set()

        for resource in current_resources:
            uri = resource["uri"]
            seen_uris.add(uri)

            stored = self._stored.get(uri)
            if stored is None:
                # New resource
                result.added.append(resource)
                self._stored[uri] = self._extract_meta(resource)
                continue

            # Tier 1: content hash
            current_hash = resource.get("contentHash")
            if current_hash is not None and stored.content_hash is not None:
                if current_hash != stored.content_hash:
                    result.changed.append(resource)
                    self._stored[uri] = self._extract_meta(resource)
                else:
                    result.unchanged.append(uri)
                continue

            # Tier 2: last-modified + size
            current_modified = resource.get("lastModified")
            current_size = resource.get("size")
            if current_modified is not None or current_size is not None:
                changed = False
                if current_modified is not None and current_modified != stored.last_modified:
                    changed = True
                if current_size is not None and current_size != stored.size:
                    changed = True
                if changed:
                    result.changed.append(resource)
                    self._stored[uri] = self._extract_meta(resource)
                else:
                    result.unchanged.append(uri)
                continue

            # Tier 3: no metadata — assume unchanged (hash check on fetch)
            result.unchanged.append(uri)

        # Detect removed resources
        for uri in list(self._stored):
            if uri not in seen_uris:
                result.removed.append(uri)
                del self._stored[uri]

        logger.debug(
            "Probe: added=%d changed=%d removed=%d unchanged=%d",
            len(result.added), len(result.changed),
            len(result.removed), len(result.unchanged),
        )
        return result

    def update_hash(self, uri: str, content_hash: str) -> None:
        """Update the stored content hash for a resource after fetching."""
        if uri in self._stored:
            self._stored[uri].content_hash = content_hash
        else:
            self._stored[uri] = ResourceMeta(content_hash=content_hash)

    def load(self, data: dict) -> None:
        """Restore stored metadata from a serialized dict."""
        self._stored.clear()
        for uri, meta_dict in data.items():
            self._stored[uri] = ResourceMeta(
                last_modified=meta_dict.get("last_modified"),
                size=meta_dict.get("size"),
                content_hash=meta_dict.get("content_hash"),
            )

    def dump(self) -> dict:
        """Serialize stored metadata to a dict for persistence."""
        return {
            uri: {
                "last_modified": meta.last_modified,
                "size": meta.size,
                "content_hash": meta.content_hash,
            }
            for uri, meta in self._stored.items()
        }

    @staticmethod
    def _extract_meta(resource: dict) -> ResourceMeta:
        """Extract metadata fields from a resource dict."""
        return ResourceMeta(
            last_modified=resource.get("lastModified"),
            size=resource.get("size"),
            content_hash=resource.get("contentHash"),
        )
