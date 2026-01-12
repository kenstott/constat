"""End-to-end tests for natural language queries against MongoDB data source.

These tests require:
- Docker (for MongoDB container)
- ANTHROPIC_API_KEY (for LLM queries)

Tests the full flow: MongoDB data source -> NL query -> LLM planning -> code execution -> answer
"""

import os
import pytest
import tempfile
from pathlib import Path

# Skip all tests if API key not set
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    ),
    pytest.mark.requires_mongodb,
]


@pytest.fixture(scope="module")
def mongodb_ecommerce_data(mongodb_container):
    """Load e-commerce sample data into MongoDB for testing."""
    try:
        import pymongo
    except ImportError:
        pytest.skip("pymongo not installed")

    client = pymongo.MongoClient(mongodb_container["uri"])
    db = client["ecommerce"]

    # Clear existing data
    for coll_name in ["customers", "products", "orders"]:
        db.drop_collection(coll_name)

    # Insert customers
    db.customers.insert_many([
        {"_id": 1, "name": "Alice Johnson", "email": "alice@example.com", "country": "USA", "tier": "gold"},
        {"_id": 2, "name": "Bob Smith", "email": "bob@example.com", "country": "UK", "tier": "silver"},
        {"_id": 3, "name": "Carol White", "email": "carol@example.com", "country": "Canada", "tier": "gold"},
        {"_id": 4, "name": "David Brown", "email": "david@example.com", "country": "USA", "tier": "bronze"},
        {"_id": 5, "name": "Eve Davis", "email": "eve@example.com", "country": "Germany", "tier": "silver"},
    ])

    # Insert products
    db.products.insert_many([
        {"_id": 101, "name": "Laptop Pro", "category": "Electronics", "price": 1299.99, "stock": 50},
        {"_id": 102, "name": "Wireless Mouse", "category": "Electronics", "price": 29.99, "stock": 200},
        {"_id": 103, "name": "Desk Chair", "category": "Furniture", "price": 249.99, "stock": 30},
        {"_id": 104, "name": "USB-C Hub", "category": "Electronics", "price": 49.99, "stock": 100},
        {"_id": 105, "name": "Standing Desk", "category": "Furniture", "price": 599.99, "stock": 15},
        {"_id": 106, "name": "Mechanical Keyboard", "category": "Electronics", "price": 129.99, "stock": 75},
    ])

    # Insert orders with clear patterns for testing
    db.orders.insert_many([
        {"_id": 1001, "customer_id": 1, "product_id": 101, "quantity": 1, "total": 1299.99, "status": "delivered"},
        {"_id": 1002, "customer_id": 1, "product_id": 102, "quantity": 2, "total": 59.98, "status": "delivered"},
        {"_id": 1003, "customer_id": 2, "product_id": 103, "quantity": 1, "total": 249.99, "status": "delivered"},
        {"_id": 1004, "customer_id": 3, "product_id": 101, "quantity": 1, "total": 1299.99, "status": "shipped"},
        {"_id": 1005, "customer_id": 3, "product_id": 104, "quantity": 3, "total": 149.97, "status": "shipped"},
        {"_id": 1006, "customer_id": 4, "product_id": 106, "quantity": 1, "total": 129.99, "status": "pending"},
        {"_id": 1007, "customer_id": 5, "product_id": 105, "quantity": 1, "total": 599.99, "status": "delivered"},
        {"_id": 1008, "customer_id": 1, "product_id": 105, "quantity": 1, "total": 599.99, "status": "delivered"},
    ])

    # Create indexes
    db.customers.create_index("email", unique=True)
    db.products.create_index("category")
    db.orders.create_index("customer_id")

    yield {
        "uri": mongodb_container["uri"],
        "database": "ecommerce",
        "collections": ["customers", "products", "orders"],
    }

    # Cleanup
    client.close()


@pytest.fixture
def mongodb_session(mongodb_ecommerce_data):
    """Create a Session configured with MongoDB as the data source."""
    from constat.core.config import Config
    from constat.storage.history import SessionHistory
    from constat.session import Session, SessionConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            databases={
                "ecommerce": {
                    "type": "mongodb",
                    "uri": mongodb_ecommerce_data["uri"],
                    "database": mongodb_ecommerce_data["database"],
                    "description": "E-commerce database with customers, products, and orders",
                },
            },
            system_prompt="""You are analyzing an e-commerce MongoDB database.

## Collections:
- customers: Customer information (_id, name, email, country, tier)
- products: Product catalog (_id, name, category, price, stock)
- orders: Order records (_id, customer_id, product_id, quantity, total, status)

## IMPORTANT: How to Query MongoDB

The database is available as `db_ecommerce`. Access collections using subscript notation:

```python
# Access a collection
customers = db_ecommerce['customers']
products = db_ecommerce['products']
orders = db_ecommerce['orders']

# Find all documents
all_customers = list(customers.find())

# Find with filter
electronics = list(products.find({"category": "Electronics"}))

# Find one document
customer = customers.find_one({"_id": 1})

# Count documents
count = customers.count_documents({})

# Aggregation pipeline
pipeline = [
    {"$group": {"_id": "$category", "count": {"$sum": 1}}}
]
result = list(products.aggregate(pipeline))
```

## Converting to DataFrame
```python
# Query MongoDB and convert to pandas DataFrame
docs = list(db_ecommerce['customers'].find())
df = pd.DataFrame(docs)
```

## CRITICAL RULES:
1. DO NOT use pd.read_sql() - MongoDB is not SQL. Use pymongo methods.
2. Always use subscript notation: db_ecommerce['collection_name']
3. Always convert cursor to list: list(collection.find(...))
4. When using values from pandas/numpy in MongoDB queries, convert to Python int/float:
   - Use int(value) not numpy int64
   - Use float(value) not numpy float64
""",
            execution={
                "allowed_imports": ["pandas", "numpy", "json", "datetime", "pymongo"],
                "timeout_seconds": 60,
            },
        )

        history = SessionHistory(storage_dir=Path(tmpdir) / "sessions")
        session_config = SessionConfig(max_retries_per_step=3)

        session = Session(config, session_config=session_config, history=history)
        yield session


class TestMongoDBNaturalLanguageQueries:
    """End-to-end tests for natural language queries against MongoDB."""

    def test_simple_count_query(self, mongodb_session):
        """Test a simple count query against MongoDB."""
        result = mongodb_session.solve(
            "How many customers are in the database?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # The output should mention 5 customers
        assert "5" in result["output"], f"Expected '5' in output: {result['output']}"
        print(f"\n--- Simple Count Query ---")
        print(f"Output: {result['output']}")

    def test_aggregation_query(self, mongodb_session):
        """Test an aggregation query against MongoDB."""
        result = mongodb_session.solve(
            "What is the total revenue from all orders? Sum up the 'total' field from all orders."
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # Total should be 4389.89 (sum of all order totals)
        # 1299.99 + 59.98 + 249.99 + 1299.99 + 149.97 + 129.99 + 599.99 + 599.99 = 4389.89
        print(f"\n--- Aggregation Query ---")
        print(f"Output: {result['output']}")

    def test_filter_query(self, mongodb_session):
        """Test a filter query against MongoDB."""
        result = mongodb_session.solve(
            "List all products in the Electronics category. Show their names and prices."
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # Should include Laptop Pro, Wireless Mouse, USB-C Hub, Mechanical Keyboard
        output_lower = result["output"].lower()
        assert "laptop" in output_lower or "electronics" in output_lower, \
            f"Expected electronics products in output: {result['output']}"
        print(f"\n--- Filter Query ---")
        print(f"Output: {result['output']}")

    def test_join_like_query(self, mongodb_session):
        """Test a query that requires joining data from multiple collections."""
        result = mongodb_session.solve(
            "Which customer has placed the most orders? Show their name and order count."
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # Alice (customer_id=1) has 3 orders, Carol has 2, others have 1
        output_lower = result["output"].lower()
        assert "alice" in output_lower or "3" in result["output"], \
            f"Expected Alice with 3 orders: {result['output']}"
        print(f"\n--- Join-like Query ---")
        print(f"Output: {result['output']}")

    def test_multi_step_analysis(self, mongodb_session):
        """Test a multi-step analytical query."""
        result = mongodb_session.solve(
            "First find the average order value. Then identify which customers "
            "have placed orders above that average. List their names."
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        assert result["plan"] is not None
        assert len(result["plan"].steps) >= 2, "Expected at least 2 steps"

        print(f"\n--- Multi-Step Analysis ---")
        print(f"Plan: {len(result['plan'].steps)} steps")
        for step in result["plan"].steps:
            print(f"  Step {step.number}: {step.goal}")
        print(f"\nOutput: {result['output']}")

    def test_gold_tier_customers(self, mongodb_session):
        """Test a query about customer tiers."""
        result = mongodb_session.solve(
            "How many gold tier customers are there and what are their names?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # Alice and Carol are gold tier
        output_lower = result["output"].lower()
        assert "2" in result["output"] or "two" in output_lower, \
            f"Expected 2 gold customers: {result['output']}"
        print(f"\n--- Gold Tier Query ---")
        print(f"Output: {result['output']}")


class TestMongoDBSchemaDiscovery:
    """Tests for MongoDB schema discovery and introspection."""

    def test_schema_manager_discovers_collections(self, mongodb_ecommerce_data):
        """Test that SchemaManager correctly discovers MongoDB collections."""
        from constat.core.config import Config
        from constat.catalog.schema_manager import SchemaManager

        config = Config(
            databases={
                "ecommerce": {
                    "type": "mongodb",
                    "uri": mongodb_ecommerce_data["uri"],
                    "database": mongodb_ecommerce_data["database"],
                },
            }
        )

        schema_manager = SchemaManager(config)
        schema_manager.initialize()

        # Check that collections were discovered
        assert "ecommerce" in schema_manager.nosql_connections
        connector = schema_manager.nosql_connections["ecommerce"]
        collections = connector.get_collections()

        assert "customers" in collections
        assert "products" in collections
        assert "orders" in collections

    def test_schema_manager_infers_fields(self, mongodb_ecommerce_data):
        """Test that SchemaManager correctly infers field types from samples."""
        from constat.core.config import Config
        from constat.catalog.schema_manager import SchemaManager

        config = Config(
            databases={
                "ecommerce": {
                    "type": "mongodb",
                    "uri": mongodb_ecommerce_data["uri"],
                    "database": mongodb_ecommerce_data["database"],
                },
            }
        )

        schema_manager = SchemaManager(config)
        schema_manager.initialize()

        # Get table/collection metadata from cache
        # The metadata is cached with keys like "ecommerce.customers"
        metadata = schema_manager.metadata_cache.get("ecommerce.customers")

        assert metadata is not None
        # Check that fields were inferred
        field_names = [col.name for col in metadata.columns]
        assert "name" in field_names or "_id" in field_names

    def test_vector_search_finds_relevant_collections(self, mongodb_ecommerce_data):
        """Test that vector search finds relevant collections for queries."""
        from constat.core.config import Config
        from constat.catalog.schema_manager import SchemaManager

        config = Config(
            databases={
                "ecommerce": {
                    "type": "mongodb",
                    "uri": mongodb_ecommerce_data["uri"],
                    "database": mongodb_ecommerce_data["database"],
                },
            }
        )

        schema_manager = SchemaManager(config)
        schema_manager.initialize()

        # Search for customer-related tables
        matches = schema_manager.find_relevant_tables("customer information and email")

        assert len(matches) > 0
        # find_relevant_tables returns list of dicts with 'table' key
        table_names = [m["table"] for m in matches]
        assert any("customer" in t.lower() for t in table_names), \
            f"Expected 'customers' in matches: {table_names}"
