"""Integration tests for Cassandra connector (requires Docker)."""

import pytest
from datetime import datetime, timedelta

from constat.catalog.nosql.cassandra import CassandraConnector
from constat.catalog.nosql.base import NoSQLType


# =============================================================================
# Basic Connection Tests
# =============================================================================

@pytest.mark.requires_cassandra
@pytest.mark.integration
class TestCassandraConnection:
    """Test Cassandra connection with real Docker container."""

    def test_connect_to_cassandra(self, cassandra_container):
        """Test connecting to Cassandra container."""
        connector = CassandraConnector(
            keyspace="system",
            hosts=["localhost"],
            port=cassandra_container["port"],
        )

        connector.connect()
        assert connector.is_connected is True

        connector.disconnect()
        assert connector.is_connected is False

    def test_connect_and_list_system_tables(self, cassandra_container):
        """Test listing tables from system keyspace."""
        connector = CassandraConnector(
            keyspace="system",
            hosts=["localhost"],
            port=cassandra_container["port"],
        )

        connector.connect()
        tables = connector.get_collections()

        # System keyspace should have some tables
        assert len(tables) > 0
        assert "local" in tables or "peers" in tables

        connector.disconnect()


# =============================================================================
# IoT Data Integration Tests
# =============================================================================

@pytest.mark.requires_cassandra
@pytest.mark.integration
class TestCassandraIoTData:
    """Test Cassandra with IoT sample data."""

    def test_list_iot_tables(self, cassandra_with_iot_data):
        """Test listing tables in IoT keyspace."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        tables = connector.get_collections()

        assert "sensor_readings" in tables
        assert "devices" in tables
        assert "alerts" in tables

        connector.disconnect()

    def test_get_devices_schema(self, cassandra_with_iot_data):
        """Test getting schema for devices table."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        schema = connector.get_collection_schema("devices")

        assert schema.name == "devices"
        assert schema.nosql_type == NoSQLType.WIDE_COLUMN
        assert schema.partition_key == "device_id"

        field_names = [f.name for f in schema.fields]
        assert "device_id" in field_names
        assert "name" in field_names
        assert "location" in field_names
        assert "status" in field_names

        connector.disconnect()

    def test_get_sensor_readings_schema(self, cassandra_with_iot_data):
        """Test getting schema for sensor_readings table with clustering key."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        schema = connector.get_collection_schema("sensor_readings")

        assert schema.name == "sensor_readings"
        assert schema.partition_key == "device_id"
        assert "timestamp" in schema.clustering_keys

        connector.disconnect()

    def test_query_all_devices(self, cassandra_with_iot_data):
        """Test querying all devices."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        devices = connector.query("devices", {}, limit=100)

        assert len(devices) == 5
        device_ids = [d["device_id"] for d in devices]
        assert "sensor-001" in device_ids
        assert "sensor-005" in device_ids

        connector.disconnect()

    def test_query_device_by_id(self, cassandra_with_iot_data):
        """Test querying a specific device."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        devices = connector.query("devices", {"device_id": "sensor-001"})

        assert len(devices) == 1
        assert devices[0]["device_id"] == "sensor-001"
        assert devices[0]["name"] == "Temperature Sensor A"
        assert devices[0]["status"] == "active"

        connector.disconnect()

    def test_query_sensor_readings_by_device(self, cassandra_with_iot_data):
        """Test querying sensor readings for a specific device."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        readings = connector.query(
            "sensor_readings",
            {"device_id": "sensor-001"},
            limit=10
        )

        assert len(readings) > 0
        for reading in readings:
            assert reading["device_id"] == "sensor-001"
            assert "value" in reading
            assert "timestamp" in reading

        connector.disconnect()

    def test_query_alerts_by_device(self, cassandra_with_iot_data):
        """Test querying alerts for a device."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        alerts = connector.query(
            "alerts",
            {"device_id": "sensor-001"},
            limit=10
        )

        assert len(alerts) >= 2
        for alert in alerts:
            assert alert["device_id"] == "sensor-001"
            assert "alert_type" in alert
            assert "severity" in alert

        connector.disconnect()

    def test_execute_cql_count(self, cassandra_with_iot_data):
        """Test executing raw CQL count query."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        result = connector.execute_cql(
            f"SELECT COUNT(*) as cnt FROM {cassandra_with_iot_data['keyspace']}.devices"
        )

        assert len(result) == 1
        assert result[0]["cnt"] == 5

        connector.disconnect()

    def test_execute_cql_with_where(self, cassandra_with_iot_data):
        """Test executing CQL with WHERE clause."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        result = connector.execute_cql(
            f"SELECT * FROM {cassandra_with_iot_data['keyspace']}.devices WHERE device_id = %s",
            ["sensor-002"]
        )

        assert len(result) == 1
        assert result[0]["name"] == "Humidity Sensor B"

        connector.disconnect()


# =============================================================================
# Insert and Delete Tests
# =============================================================================

@pytest.mark.requires_cassandra
@pytest.mark.integration
class TestCassandraInsertDelete:
    """Test insert and delete operations."""

    def test_insert_device(self, cassandra_with_iot_data):
        """Test inserting a new device."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        # Insert new device
        new_device = {
            "device_id": "sensor-test",
            "name": "Test Sensor",
            "location": "Test Location",
            "status": "testing",
        }
        result = connector.insert("devices", [new_device])
        assert result == 1

        # Verify insertion
        devices = connector.query("devices", {"device_id": "sensor-test"})
        assert len(devices) == 1
        assert devices[0]["name"] == "Test Sensor"

        # Cleanup
        connector.delete("devices", {"device_id": "sensor-test"})

        connector.disconnect()

    def test_delete_device(self, cassandra_with_iot_data):
        """Test deleting a device."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        # Insert device to delete
        new_device = {
            "device_id": "sensor-to-delete",
            "name": "Delete Me",
            "location": "Nowhere",
            "status": "inactive",
        }
        connector.insert("devices", [new_device])

        # Verify it exists
        devices = connector.query("devices", {"device_id": "sensor-to-delete"})
        assert len(devices) == 1

        # Delete it
        result = connector.delete("devices", {"device_id": "sensor-to-delete"})
        assert result == 1

        # Verify deletion
        devices = connector.query("devices", {"device_id": "sensor-to-delete"})
        assert len(devices) == 0

        connector.disconnect()


# =============================================================================
# Schema Discovery Tests
# =============================================================================

@pytest.mark.requires_cassandra
@pytest.mark.integration
class TestCassandraSchemaDiscovery:
    """Test schema discovery capabilities."""

    def test_get_all_metadata(self, cassandra_with_iot_data):
        """Test getting metadata for all tables."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        all_metadata = connector.get_all_metadata()

        assert len(all_metadata) >= 3
        table_names = [m.name for m in all_metadata]
        assert "sensor_readings" in table_names
        assert "devices" in table_names
        assert "alerts" in table_names

        connector.disconnect()

    def test_get_overview(self, cassandra_with_iot_data):
        """Test generating overview summary."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        overview = connector.get_overview()

        assert cassandra_with_iot_data["keyspace"] in overview
        assert "sensor_readings" in overview or "devices" in overview

        connector.disconnect()

    def test_sample_documents(self, cassandra_with_iot_data):
        """Test sampling documents from a table."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()
        samples = connector.sample_documents("devices", limit=3)

        assert len(samples) == 3
        for sample in samples:
            assert "device_id" in sample
            assert "name" in sample

        connector.disconnect()


# =============================================================================
# NLQ Integration Tests (Natural Language Query scenarios)
# =============================================================================

@pytest.mark.requires_cassandra
@pytest.mark.integration
class TestCassandraNLQScenarios:
    """Test scenarios that would be used in natural language queries.

    These tests verify that queries typical in NLQ use cases work correctly.
    """

    def test_nlq_count_active_devices(self, cassandra_with_iot_data):
        """NLQ: How many active devices are there?"""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        # Get all devices and filter (Cassandra needs ALLOW FILTERING or secondary index)
        devices = connector.query("devices", {}, limit=100)
        active_count = sum(1 for d in devices if d.get("status") == "active")

        assert active_count == 4  # sensor-001, 002, 004, 005 are active

        connector.disconnect()

    def test_nlq_list_devices_in_building(self, cassandra_with_iot_data):
        """NLQ: List all devices in Building 1."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        devices = connector.query("devices", {}, limit=100)
        building1_devices = [d for d in devices if "Building 1" in d.get("location", "")]

        assert len(building1_devices) == 3

        connector.disconnect()

    def test_nlq_recent_readings_for_device(self, cassandra_with_iot_data):
        """NLQ: Show me the 5 most recent readings from sensor-001."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        # Query with limit (already ordered by timestamp DESC due to clustering order)
        readings = connector.query(
            "sensor_readings",
            {"device_id": "sensor-001"},
            limit=5
        )

        assert len(readings) == 5
        # Verify readings are for the correct device
        for reading in readings:
            assert reading["device_id"] == "sensor-001"

        connector.disconnect()

    def test_nlq_unacknowledged_alerts(self, cassandra_with_iot_data):
        """NLQ: Show me all unacknowledged alerts."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        # Query all alerts (would need secondary index for filtering by acknowledged)
        result = connector.execute_cql(
            f"SELECT * FROM {cassandra_with_iot_data['keyspace']}.alerts"
        )
        unacknowledged = [a for a in result if a.get("acknowledged") is False]

        assert len(unacknowledged) >= 2

        connector.disconnect()

    def test_nlq_critical_alerts(self, cassandra_with_iot_data):
        """NLQ: List all critical severity alerts."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        result = connector.execute_cql(
            f"SELECT * FROM {cassandra_with_iot_data['keyspace']}.alerts"
        )
        critical_alerts = [a for a in result if a.get("severity") == "critical"]

        assert len(critical_alerts) >= 1
        assert any("device_offline" in a.get("alert_type", "") for a in critical_alerts)

        connector.disconnect()

    def test_nlq_temperature_readings_analysis(self, cassandra_with_iot_data):
        """NLQ: What's the average temperature from sensor-001?"""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        readings = connector.query(
            "sensor_readings",
            {"device_id": "sensor-001"},
            limit=100
        )

        temp_readings = [r for r in readings if r.get("sensor_type") == "temperature"]
        if temp_readings:
            avg_temp = sum(r["value"] for r in temp_readings) / len(temp_readings)
            # Verify we got a reasonable temperature value
            assert 15 <= avg_temp <= 40

        connector.disconnect()

    def test_nlq_device_status_summary(self, cassandra_with_iot_data):
        """NLQ: Give me a summary of device statuses."""
        connector = CassandraConnector(
            keyspace=cassandra_with_iot_data["keyspace"],
            hosts=["localhost"],
            port=cassandra_with_iot_data["port"],
        )

        connector.connect()

        devices = connector.query("devices", {}, limit=100)

        status_counts = {}
        for device in devices:
            status = device.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        assert "active" in status_counts
        assert status_counts["active"] == 4
        assert "inactive" in status_counts
        assert status_counts["inactive"] == 1

        connector.disconnect()
