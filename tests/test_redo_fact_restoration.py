"""Tests for redo fact restoration functionality.

This tests the scenario where:
1. User runs an auditable query that resolves facts from database
2. User does a redo operation
3. Cached facts should be found without re-resolution
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from constat.execution.fact_resolver import Fact, FactSource, FactResolver


class TestTableFactVerification:
    """Tests for table fact verification in get_fact()."""

    def test_get_fact_returns_none_when_table_missing(self):
        """Table facts should return None if their table doesn't exist in datastore."""
        resolver = FactResolver()

        fact = Fact(
            name="customers",
            value="12 rows",
            confidence=0.9,
            source=FactSource.DATABASE,
            table_name="customers",
            row_count=12,
        )
        resolver._cache["customers"] = fact

        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = []  # Table doesn't exist
        resolver._datastore = mock_datastore

        result = resolver.get_fact("customers")
        assert result is None  # Should return None because table doesn't exist

    def test_get_fact_returns_fact_when_table_exists(self):
        """Table facts should be returned if their table exists in datastore."""
        resolver = FactResolver()

        fact = Fact(
            name="customers",
            value="12 rows",
            confidence=0.9,
            source=FactSource.DATABASE,
            table_name="customers",
            row_count=12,
        )
        resolver._cache["customers"] = fact

        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = [{"name": "customers"}]  # Table exists
        resolver._datastore = mock_datastore

        result = resolver.get_fact("customers")
        assert result is not None
        assert result.name == "customers"
        assert result.value == "12 rows"

    def test_get_fact_skips_verification_when_disabled(self):
        """When verify_tables=False, table existence is not checked."""
        resolver = FactResolver()

        fact = Fact(
            name="customers",
            value="12 rows",
            confidence=0.9,
            source=FactSource.DATABASE,
            table_name="customers",
            row_count=12,
        )
        resolver._cache["customers"] = fact

        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = []  # Table doesn't exist
        resolver._datastore = mock_datastore

        # With verify_tables=False, should return the cached fact
        result = resolver.get_fact("customers", verify_tables=False)
        assert result is not None
        assert result.name == "customers"

    def test_get_fact_returns_non_table_facts_without_verification(self):
        """Non-table facts should be returned regardless of datastore state."""
        resolver = FactResolver()

        fact = Fact(
            name="my_age",
            value=65,
            confidence=1.0,
            source=FactSource.USER_PROVIDED,
        )
        resolver._cache["my_age"] = fact

        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = []
        resolver._datastore = mock_datastore

        result = resolver.get_fact("my_age")
        assert result is not None
        assert result.value == 65


class TestImportExportCache:
    """Tests for fact cache import/export with table facts."""

    def test_export_table_fact(self):
        """Table facts should be exported with table_name and row_count."""
        resolver = FactResolver()

        fact = Fact(
            name="customers",
            value="12 rows",
            confidence=0.9,
            source=FactSource.DATABASE,
            table_name="customers",
            row_count=12,
            query="SELECT * FROM customers",
        )
        resolver._cache["customers"] = fact

        exported = resolver.export_cache()
        assert len(exported) == 1

        exported_fact = exported[0]
        assert exported_fact["name"] == "customers"
        assert exported_fact["value_type"] == "table"
        assert exported_fact["value"] == "customers"  # table_name becomes value for table type
        assert exported_fact["row_count"] == 12
        assert exported_fact["query"] == "SELECT * FROM customers"

    def test_import_table_fact(self):
        """Table facts should be imported with table_name and row_count restored."""
        resolver = FactResolver()

        fact_dict = {
            "name": "customers",
            "value": "customers",  # table name for table type
            "value_type": "table",
            "confidence": 0.9,
            "source": "database",
            "row_count": 12,
            "query": "SELECT * FROM customers",
            "resolved_at": "2024-01-01T00:00:00",
        }

        resolver.import_cache([fact_dict])

        assert "customers" in resolver._cache
        imported = resolver._cache["customers"]
        assert imported.name == "customers"
        assert imported.table_name == "customers"
        assert imported.row_count == 12
        assert imported.value == "12 rows"  # Restored display value
        assert imported.query == "SELECT * FROM customers"

    def test_import_export_roundtrip(self):
        """Exporting then importing should preserve fact data."""
        resolver1 = FactResolver()

        # Create original facts
        resolver1._cache["customers"] = Fact(
            name="customers",
            value="12 rows",
            confidence=0.9,
            source=FactSource.DATABASE,
            table_name="customers",
            row_count=12,
            query="SELECT * FROM customers",
        )
        resolver1._cache["my_age"] = Fact(
            name="my_age",
            value=65,
            confidence=1.0,
            source=FactSource.USER_PROVIDED,
            reasoning="User provided their age",
        )

        # Export
        exported = resolver1.export_cache()
        assert len(exported) == 2

        # Import into new resolver
        resolver2 = FactResolver()
        resolver2.import_cache(exported)

        # Verify customers fact
        customers = resolver2._cache.get("customers")
        assert customers is not None
        assert customers.table_name == "customers"
        assert customers.row_count == 12
        assert customers.value == "12 rows"

        # Verify my_age fact
        my_age = resolver2._cache.get("my_age")
        assert my_age is not None
        assert my_age.value == 65
        assert my_age.source == FactSource.USER_PROVIDED


class TestRedoScenario:
    """Integration test for the full redo scenario."""

    def test_redo_finds_facts_when_tables_exist(self):
        """
        Simulate the redo scenario with persistent datastore:
        1. Facts are resolved (including table facts)
        2. Facts are exported
        3. Facts are imported for redo
        4. Tables still exist in datastore, so facts should be found
        """
        # Step 1: Original resolution - create facts
        resolver = FactResolver()

        # Simulate resolving database facts
        resolver._cache["customers"] = Fact(
            name="customers",
            value="12 rows",
            confidence=0.9,
            source=FactSource.DATABASE,
            table_name="customers",
            row_count=12,
            query="SELECT * FROM customers",
        )
        resolver._cache["customer_tiers"] = Fact(
            name="customer_tiers",
            value="4 rows",
            confidence=0.9,
            source=FactSource.DATABASE,
            table_name="customer_tiers",
            row_count=4,
            query="SELECT DISTINCT tier FROM customers",
        )
        resolver._cache["my_age"] = Fact(
            name="my_age",
            value=65,
            confidence=1.0,
            source=FactSource.USER_PROVIDED,
            reasoning="User provided: 65",
        )

        # Step 2: Export facts (simulating session save)
        exported = resolver.export_cache()
        saved_facts_json = json.dumps(exported)

        # Step 3: Import facts (simulating redo)
        # Create a new resolver to simulate fresh session
        redo_resolver = FactResolver()

        # Datastore retains tables (persistent storage)
        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = [
            {"name": "customers"},
            {"name": "customer_tiers"},
        ]
        redo_resolver._datastore = mock_datastore

        # Import the saved facts
        saved_facts = json.loads(saved_facts_json)
        redo_resolver.import_cache(saved_facts)

        # Step 4: Verify facts can be retrieved (tables exist)
        customers = redo_resolver.get_fact("customers")
        assert customers is not None, "customers fact should be found"
        assert customers.value == "12 rows"
        assert customers.table_name == "customers"

        customer_tiers = redo_resolver.get_fact("customer_tiers")
        assert customer_tiers is not None, "customer_tiers fact should be found"
        assert customer_tiers.value == "4 rows"
        assert customer_tiers.table_name == "customer_tiers"

        my_age = redo_resolver.get_fact("my_age")
        assert my_age is not None, "my_age fact should be found"
        assert my_age.value == 65

    def test_table_fact_invalid_when_table_removed(self):
        """Table facts should be invalid if table no longer exists in datastore."""
        resolver = FactResolver()

        # Import facts from previous session
        facts = [
            {
                "name": "customers",
                "value": "customers",
                "value_type": "table",
                "confidence": 0.9,
                "source": "database",
                "row_count": 12,
                "resolved_at": "2024-01-01T00:00:00",
            }
        ]
        resolver.import_cache(facts)

        # Datastore no longer has the table (e.g., user deleted it)
        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = []
        resolver._datastore = mock_datastore

        # Fact should not be found because table doesn't exist
        customers = resolver.get_fact("customers")
        assert customers is None, "customers fact should NOT be found when table is missing"

    def test_redo_respects_user_fact_updates(self):
        """
        When user provides new values during redo (e.g., "change my age to 50"),
        those should override the cached values.
        """
        resolver = FactResolver()

        # Import original facts
        facts = [
            {
                "name": "my_age",
                "value": 65,
                "value_type": "integer",
                "confidence": 1.0,
                "source": "user_provided",
                "resolved_at": "2024-01-01T00:00:00",
            }
        ]
        resolver.import_cache(facts)

        # User provides new age value
        resolver.add_user_fact(
            fact_name="my_age",
            value=50,
            reasoning="User updated their age to 50",
        )

        # The new value should override the imported one
        my_age = resolver.get_fact("my_age")
        assert my_age is not None
        assert my_age.value == 50, "my_age should be updated to 50"
