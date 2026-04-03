# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for constat.mcp.change_probe — ChangeProbe incremental sync."""

from __future__ import annotations

from constat.mcp.change_probe import ChangeProbe, ProbeResult, ResourceMeta


class TestProbeAddedResources:
    def test_all_new(self):
        probe = ChangeProbe()
        resources = [
            {"uri": "file:///a.txt"},
            {"uri": "file:///b.txt"},
        ]
        result = probe.probe(resources)
        assert len(result.added) == 2
        assert len(result.changed) == 0
        assert len(result.removed) == 0
        assert len(result.unchanged) == 0

    def test_second_probe_shows_unchanged(self):
        probe = ChangeProbe()
        resources = [{"uri": "file:///a.txt", "lastModified": "2025-01-01", "size": 100}]
        probe.probe(resources)
        result = probe.probe(resources)
        assert len(result.added) == 0
        assert len(result.unchanged) == 1


class TestProbeChangedResources:
    def test_size_change_detected(self):
        probe = ChangeProbe()
        resources_v1 = [{"uri": "file:///a.txt", "size": 100}]
        probe.probe(resources_v1)

        resources_v2 = [{"uri": "file:///a.txt", "size": 200}]
        result = probe.probe(resources_v2)
        assert len(result.changed) == 1
        assert result.changed[0]["uri"] == "file:///a.txt"

    def test_last_modified_change_detected(self):
        probe = ChangeProbe()
        probe.probe([{"uri": "file:///a.txt", "lastModified": "2025-01-01"}])
        result = probe.probe([{"uri": "file:///a.txt", "lastModified": "2025-06-15"}])
        assert len(result.changed) == 1

    def test_hash_change_detected(self):
        probe = ChangeProbe()
        probe.probe([{"uri": "file:///a.txt", "contentHash": "sha256:aaa"}])
        result = probe.probe([{"uri": "file:///a.txt", "contentHash": "sha256:bbb"}])
        assert len(result.changed) == 1

    def test_hash_unchanged(self):
        probe = ChangeProbe()
        probe.probe([{"uri": "file:///a.txt", "contentHash": "sha256:aaa"}])
        result = probe.probe([{"uri": "file:///a.txt", "contentHash": "sha256:aaa"}])
        assert len(result.unchanged) == 1
        assert len(result.changed) == 0


class TestProbeRemovedResources:
    def test_removed_detected(self):
        probe = ChangeProbe()
        probe.probe([{"uri": "file:///a.txt"}, {"uri": "file:///b.txt"}])
        result = probe.probe([{"uri": "file:///a.txt"}])
        assert result.removed == ["file:///b.txt"]

    def test_all_removed(self):
        probe = ChangeProbe()
        probe.probe([{"uri": "file:///a.txt"}])
        result = probe.probe([])
        assert result.removed == ["file:///a.txt"]


class TestHashBasedDetection:
    def test_update_hash_after_fetch(self):
        probe = ChangeProbe()
        probe.probe([{"uri": "file:///a.txt"}])
        probe.update_hash("file:///a.txt", "sha256:abc")

        # Now probe with same hash — unchanged
        result = probe.probe([{"uri": "file:///a.txt", "contentHash": "sha256:abc"}])
        assert len(result.unchanged) == 1

        # Probe with different hash — changed
        result = probe.probe([{"uri": "file:///a.txt", "contentHash": "sha256:xyz"}])
        assert len(result.changed) == 1

    def test_update_hash_for_unknown_uri(self):
        probe = ChangeProbe()
        probe.update_hash("file:///new.txt", "sha256:new")
        assert "file:///new.txt" in probe._stored


class TestPersistence:
    def test_dump_and_load(self):
        probe = ChangeProbe()
        probe.probe([
            {"uri": "file:///a.txt", "lastModified": "2025-01-01", "size": 100},
            {"uri": "file:///b.txt", "contentHash": "sha256:bbb"},
        ])

        data = probe.dump()
        assert "file:///a.txt" in data
        assert data["file:///a.txt"]["last_modified"] == "2025-01-01"
        assert data["file:///a.txt"]["size"] == 100
        assert data["file:///b.txt"]["content_hash"] == "sha256:bbb"

        # Load into a fresh probe
        probe2 = ChangeProbe()
        probe2.load(data)

        # Should detect no changes
        result = probe2.probe([
            {"uri": "file:///a.txt", "lastModified": "2025-01-01", "size": 100},
            {"uri": "file:///b.txt", "contentHash": "sha256:bbb"},
        ])
        assert len(result.unchanged) == 2
        assert len(result.added) == 0
        assert len(result.changed) == 0

    def test_load_clears_existing(self):
        probe = ChangeProbe()
        probe.probe([{"uri": "file:///old.txt"}])
        probe.load({"file:///new.txt": {"last_modified": None, "size": None, "content_hash": None}})
        assert "file:///old.txt" not in probe._stored
        assert "file:///new.txt" in probe._stored
