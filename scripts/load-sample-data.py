#!/usr/bin/env python3
"""Load sample data into MongoDB and PostgreSQL for testing.

Usage:
    python scripts/load-sample-data.py mongodb   # Load MongoDB sample data
    python scripts/load-sample-data.py postgres  # Load PostgreSQL sample data
    python scripts/load-sample-data.py all       # Load all sample data
"""

import json
import sys
from pathlib import Path

# Sample data for MongoDB (e-commerce style)
MONGODB_SAMPLE_DATA = {
    "customers": [
        {"_id": 1, "name": "Alice Johnson", "email": "alice@example.com", "country": "USA", "created_at": "2024-01-15"},
        {"_id": 2, "name": "Bob Smith", "email": "bob@example.com", "country": "UK", "created_at": "2024-02-20"},
        {"_id": 3, "name": "Carol White", "email": "carol@example.com", "country": "Canada", "created_at": "2024-03-10"},
        {"_id": 4, "name": "David Brown", "email": "david@example.com", "country": "USA", "created_at": "2024-04-05"},
        {"_id": 5, "name": "Eve Davis", "email": "eve@example.com", "country": "Germany", "created_at": "2024-05-12"},
    ],
    "products": [
        {"_id": 101, "name": "Laptop Pro", "category": "Electronics", "price": 1299.99, "stock": 50, "tags": ["computer", "work"]},
        {"_id": 102, "name": "Wireless Mouse", "category": "Electronics", "price": 29.99, "stock": 200, "tags": ["accessory"]},
        {"_id": 103, "name": "Desk Chair", "category": "Furniture", "price": 249.99, "stock": 30, "tags": ["office", "comfort"]},
        {"_id": 104, "name": "USB-C Hub", "category": "Electronics", "price": 49.99, "stock": 100, "tags": ["accessory", "connectivity"]},
        {"_id": 105, "name": "Standing Desk", "category": "Furniture", "price": 599.99, "stock": 15, "tags": ["office", "health"]},
        {"_id": 106, "name": "Mechanical Keyboard", "category": "Electronics", "price": 129.99, "stock": 75, "tags": ["accessory", "typing"]},
        {"_id": 107, "name": "Monitor 27\"", "category": "Electronics", "price": 349.99, "stock": 40, "tags": ["display", "work"]},
        {"_id": 108, "name": "Webcam HD", "category": "Electronics", "price": 79.99, "stock": 120, "tags": ["video", "meetings"]},
    ],
    "orders": [
        {"_id": 1001, "customer_id": 1, "products": [{"product_id": 101, "quantity": 1}, {"product_id": 102, "quantity": 2}], "total": 1359.97, "status": "delivered", "order_date": "2024-06-01"},
        {"_id": 1002, "customer_id": 2, "products": [{"product_id": 103, "quantity": 1}], "total": 249.99, "status": "delivered", "order_date": "2024-06-05"},
        {"_id": 1003, "customer_id": 1, "products": [{"product_id": 104, "quantity": 1}, {"product_id": 106, "quantity": 1}], "total": 179.98, "status": "shipped", "order_date": "2024-06-10"},
        {"_id": 1004, "customer_id": 3, "products": [{"product_id": 105, "quantity": 1}, {"product_id": 107, "quantity": 2}], "total": 1299.97, "status": "processing", "order_date": "2024-06-15"},
        {"_id": 1005, "customer_id": 4, "products": [{"product_id": 108, "quantity": 1}], "total": 79.99, "status": "pending", "order_date": "2024-06-20"},
        {"_id": 1006, "customer_id": 5, "products": [{"product_id": 101, "quantity": 1}, {"product_id": 103, "quantity": 1}], "total": 1549.98, "status": "delivered", "order_date": "2024-06-22"},
    ],
    "reviews": [
        {"_id": 1, "product_id": 101, "customer_id": 1, "rating": 5, "comment": "Excellent laptop, very fast!", "date": "2024-06-10"},
        {"_id": 2, "product_id": 102, "customer_id": 1, "rating": 4, "comment": "Good mouse, comfortable grip", "date": "2024-06-11"},
        {"_id": 3, "product_id": 103, "customer_id": 2, "rating": 5, "comment": "Perfect for long work sessions", "date": "2024-06-15"},
        {"_id": 4, "product_id": 106, "customer_id": 3, "rating": 4, "comment": "Great typing experience", "date": "2024-06-20"},
        {"_id": 5, "product_id": 101, "customer_id": 5, "rating": 5, "comment": "Best laptop I've ever owned", "date": "2024-06-25"},
    ],
}

# PostgreSQL sample schema and data
POSTGRESQL_SCHEMA = """
-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    country VARCHAR(50),
    created_at DATE
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock INTEGER DEFAULT 0
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    total DECIMAL(10, 2),
    status VARCHAR(20),
    order_date DATE
);

-- Order items table
CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER,
    unit_price DECIMAL(10, 2)
);

-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    customer_id INTEGER REFERENCES customers(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    review_date DATE
);
"""

POSTGRESQL_DATA = """
-- Insert customers
INSERT INTO customers (id, name, email, country, created_at) VALUES
(1, 'Alice Johnson', 'alice@example.com', 'USA', '2024-01-15'),
(2, 'Bob Smith', 'bob@example.com', 'UK', '2024-02-20'),
(3, 'Carol White', 'carol@example.com', 'Canada', '2024-03-10'),
(4, 'David Brown', 'david@example.com', 'USA', '2024-04-05'),
(5, 'Eve Davis', 'eve@example.com', 'Germany', '2024-05-12')
ON CONFLICT (id) DO NOTHING;

-- Reset sequence
SELECT setval('customers_id_seq', 5);

-- Insert products
INSERT INTO products (id, name, category, price, stock) VALUES
(101, 'Laptop Pro', 'Electronics', 1299.99, 50),
(102, 'Wireless Mouse', 'Electronics', 29.99, 200),
(103, 'Desk Chair', 'Furniture', 249.99, 30),
(104, 'USB-C Hub', 'Electronics', 49.99, 100),
(105, 'Standing Desk', 'Furniture', 599.99, 15),
(106, 'Mechanical Keyboard', 'Electronics', 129.99, 75),
(107, 'Monitor 27"', 'Electronics', 349.99, 40),
(108, 'Webcam HD', 'Electronics', 79.99, 120)
ON CONFLICT (id) DO NOTHING;

SELECT setval('products_id_seq', 108);

-- Insert orders
INSERT INTO orders (id, customer_id, total, status, order_date) VALUES
(1001, 1, 1359.97, 'delivered', '2024-06-01'),
(1002, 2, 249.99, 'delivered', '2024-06-05'),
(1003, 1, 179.98, 'shipped', '2024-06-10'),
(1004, 3, 1299.97, 'processing', '2024-06-15'),
(1005, 4, 79.99, 'pending', '2024-06-20'),
(1006, 5, 1549.98, 'delivered', '2024-06-22')
ON CONFLICT (id) DO NOTHING;

SELECT setval('orders_id_seq', 1006);

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1001, 101, 1, 1299.99),
(1001, 102, 2, 29.99),
(1002, 103, 1, 249.99),
(1003, 104, 1, 49.99),
(1003, 106, 1, 129.99),
(1004, 105, 1, 599.99),
(1004, 107, 2, 349.99),
(1005, 108, 1, 79.99),
(1006, 101, 1, 1299.99),
(1006, 103, 1, 249.99)
ON CONFLICT DO NOTHING;

-- Insert reviews
INSERT INTO reviews (id, product_id, customer_id, rating, comment, review_date) VALUES
(1, 101, 1, 5, 'Excellent laptop, very fast!', '2024-06-10'),
(2, 102, 1, 4, 'Good mouse, comfortable grip', '2024-06-11'),
(3, 103, 2, 5, 'Perfect for long work sessions', '2024-06-15'),
(4, 106, 3, 4, 'Great typing experience', '2024-06-20'),
(5, 101, 5, 5, 'Best laptop I have ever owned', '2024-06-25')
ON CONFLICT (id) DO NOTHING;

SELECT setval('reviews_id_seq', 5);
"""


def load_mongodb(uri: str = "mongodb://localhost:27017", database: str = "sample_ecommerce"):
    """Load sample data into MongoDB."""
    try:
        import pymongo
    except ImportError:
        print("Error: pymongo not installed. Run: pip install pymongo")
        return False

    print(f"Connecting to MongoDB at {uri}...")
    client = pymongo.MongoClient(uri)
    db = client[database]

    # Drop existing collections
    for collection_name in MONGODB_SAMPLE_DATA.keys():
        db.drop_collection(collection_name)
        print(f"  Dropped collection: {collection_name}")

    # Insert data
    for collection_name, documents in MONGODB_SAMPLE_DATA.items():
        result = db[collection_name].insert_many(documents)
        print(f"  Inserted {len(result.inserted_ids)} documents into {collection_name}")

    # Create indexes
    db.customers.create_index("email", unique=True)
    db.products.create_index("category")
    db.orders.create_index("customer_id")
    db.reviews.create_index([("product_id", 1), ("customer_id", 1)])
    print("  Created indexes")

    client.close()
    print(f"MongoDB sample data loaded into database: {database}")
    return True


def load_postgresql(dsn: str = "postgresql://postgres:postgres@localhost:5432/postgres"):
    """Load sample data into PostgreSQL."""
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

    print(f"Connecting to PostgreSQL...")
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()

    # Create schema
    print("  Creating schema...")
    cur.execute(POSTGRESQL_SCHEMA)

    # Load data
    print("  Loading data...")
    cur.execute(POSTGRESQL_DATA)

    # Show counts
    for table in ["customers", "products", "orders", "order_items", "reviews"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"    {table}: {count} rows")

    cur.close()
    conn.close()
    print("PostgreSQL sample data loaded")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1].lower()

    # Get connection strings from environment or use defaults
    import os

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    postgres_dsn = os.environ.get("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")

    if target in ("mongodb", "mongo", "all"):
        load_mongodb(mongodb_uri)

    if target in ("postgresql", "postgres", "pg", "all"):
        load_postgresql(postgres_dsn)

    if target not in ("mongodb", "mongo", "postgresql", "postgres", "pg", "all"):
        print(f"Unknown target: {target}")
        print("Valid targets: mongodb, postgres, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
