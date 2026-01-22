#!/usr/bin/env python3
# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Setup script for Constat demo environment.

Creates sample databases, documents, and config for testing the CLI.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR / "data"
DOCS_DIR = DEMO_DIR / "docs"


def create_sales_db():
    """Create a sales database with customers, products, and orders."""
    db_path = DATA_DIR / "sales.db"
    db_path.unlink(missing_ok=True)  # Remove existing to avoid duplicates
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Customers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            tier TEXT CHECK(tier IN ('bronze', 'silver', 'gold', 'platinum')),
            signup_date DATE,
            region TEXT
        )
    """)

    # Products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price DECIMAL(10, 2),
            cost DECIMAL(10, 2),
            stock_quantity INTEGER DEFAULT 0
        )
    """)

    # Orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(customer_id),
            order_date DATE,
            status TEXT CHECK(status IN ('pending', 'shipped', 'delivered', 'cancelled')),
            total_amount DECIMAL(10, 2)
        )
    """)

    # Order items table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER REFERENCES orders(order_id),
            product_id INTEGER REFERENCES products(product_id),
            quantity INTEGER,
            unit_price DECIMAL(10, 2)
        )
    """)

    # Sample customers
    customers = [
        ("Acme Corp", "contact@acme.com", "platinum", "2022-01-15", "North"),
        ("TechStart Inc", "info@techstart.io", "gold", "2022-03-22", "West"),
        ("Global Logistics", "sales@globallog.com", "gold", "2022-05-10", "East"),
        ("Local Shop", "owner@localshop.net", "silver", "2022-08-01", "South"),
        ("MegaMart", "procurement@megamart.com", "platinum", "2021-11-30", "North"),
        ("StartupXYZ", "hello@startupxyz.co", "bronze", "2023-02-14", "West"),
        ("Enterprise Solutions", "orders@enterprise.biz", "gold", "2022-06-20", "East"),
        ("Corner Store", "mike@cornerstore.local", "bronze", "2023-05-05", "South"),
        ("DataDriven Ltd", "info@datadriven.io", "silver", "2022-09-12", "North"),
        ("Cloud Nine Services", "support@cloudnine.dev", "silver", "2022-12-01", "West"),
        ("Prime Industries", "sales@prime-ind.com", "platinum", "2021-08-15", "East"),
        ("Quick Delivery Co", "orders@quickdeliver.net", "gold", "2022-04-18", "South"),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO customers (name, email, tier, signup_date, region) VALUES (?, ?, ?, ?, ?)",
        customers
    )

    # Sample products
    products = [
        ("Widget Pro", "Electronics", 99.99, 45.00, 150),
        ("Widget Basic", "Electronics", 49.99, 20.00, 300),
        ("Gadget X", "Electronics", 199.99, 85.00, 75),
        ("Office Chair", "Furniture", 299.99, 120.00, 45),
        ("Standing Desk", "Furniture", 599.99, 250.00, 30),
        ("Monitor Stand", "Furniture", 79.99, 30.00, 200),
        ("Notebook Pack", "Supplies", 24.99, 8.00, 500),
        ("Pen Set Premium", "Supplies", 39.99, 12.00, 350),
        ("Printer Paper", "Supplies", 29.99, 15.00, 1000),
        ("Wireless Mouse", "Electronics", 34.99, 12.00, 250),
        ("Keyboard Mechanical", "Electronics", 149.99, 60.00, 100),
        ("USB Hub", "Electronics", 44.99, 15.00, 180),
        ("Desk Lamp", "Furniture", 59.99, 22.00, 120),
        ("Filing Cabinet", "Furniture", 199.99, 80.00, 40),
        ("Whiteboard", "Supplies", 89.99, 35.00, 60),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO products (name, category, price, cost, stock_quantity) VALUES (?, ?, ?, ?, ?)",
        products
    )

    # Generate orders for the past year
    random.seed(42)
    base_date = datetime.now() - timedelta(days=365)
    statuses = ['pending', 'shipped', 'delivered', 'delivered', 'delivered', 'cancelled']

    for order_id in range(1, 201):
        customer_id = random.randint(1, 12)
        days_offset = random.randint(0, 365)
        order_date = (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        status = random.choice(statuses)

        # Generate order items
        num_items = random.randint(1, 5)
        total = 0
        items = []

        for _ in range(num_items):
            product_id = random.randint(1, 15)
            quantity = random.randint(1, 10)
            # Get product price
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
            price = cursor.fetchone()[0]
            total += price * quantity
            items.append((order_id, product_id, quantity, price))

        cursor.execute(
            "INSERT OR IGNORE INTO orders (order_id, customer_id, order_date, status, total_amount) VALUES (?, ?, ?, ?, ?)",
            (order_id, customer_id, order_date, status, round(total, 2))
        )

        cursor.executemany(
            "INSERT OR IGNORE INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
            items
        )

    conn.commit()
    conn.close()
    print(f"Created: {db_path}")


def create_inventory_db():
    """Create an inventory/warehouse database."""
    db_path = DATA_DIR / "inventory.db"
    db_path.unlink(missing_ok=True)  # Remove existing to avoid duplicates
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Warehouses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS warehouses (
            warehouse_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT,
            capacity INTEGER,
            manager TEXT
        )
    """)

    # Inventory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            inventory_id INTEGER PRIMARY KEY,
            warehouse_id INTEGER REFERENCES warehouses(warehouse_id),
            sku TEXT NOT NULL,
            product_name TEXT,
            quantity INTEGER DEFAULT 0,
            reorder_level INTEGER DEFAULT 10,
            last_restocked DATE
        )
    """)

    # Shipments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shipments (
            shipment_id INTEGER PRIMARY KEY,
            warehouse_id INTEGER REFERENCES warehouses(warehouse_id),
            direction TEXT CHECK(direction IN ('inbound', 'outbound')),
            sku TEXT,
            quantity INTEGER,
            shipment_date DATE,
            carrier TEXT
        )
    """)

    # Sample warehouses
    warehouses = [
        ("East Coast Hub", "Newark, NJ", 50000, "John Smith"),
        ("West Coast Center", "Los Angeles, CA", 75000, "Maria Garcia"),
        ("Central Distribution", "Dallas, TX", 60000, "Robert Johnson"),
        ("Southeast Facility", "Atlanta, GA", 40000, "Lisa Chen"),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO warehouses (name, location, capacity, manager) VALUES (?, ?, ?, ?)",
        warehouses
    )

    # Sample inventory
    skus = [
        ("SKU-001", "Widget Pro"),
        ("SKU-002", "Widget Basic"),
        ("SKU-003", "Gadget X"),
        ("SKU-004", "Office Chair"),
        ("SKU-005", "Standing Desk"),
        ("SKU-006", "Monitor Stand"),
        ("SKU-007", "Notebook Pack"),
        ("SKU-008", "Pen Set Premium"),
    ]

    random.seed(43)
    for warehouse_id in range(1, 5):
        for sku, name in skus:
            quantity = random.randint(0, 500)
            reorder = random.randint(20, 100)
            last_restock = (datetime.now() - timedelta(days=random.randint(1, 60))).strftime('%Y-%m-%d')
            cursor.execute(
                "INSERT INTO inventory (warehouse_id, sku, product_name, quantity, reorder_level, last_restocked) VALUES (?, ?, ?, ?, ?, ?)",
                (warehouse_id, sku, name, quantity, reorder, last_restock)
            )

    # Sample shipments
    carriers = ["FedEx", "UPS", "USPS", "DHL"]
    for shipment_id in range(1, 101):
        warehouse_id = random.randint(1, 4)
        direction = random.choice(["inbound", "outbound", "outbound"])
        sku, _ = random.choice(skus)
        quantity = random.randint(10, 200)
        ship_date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d')
        carrier = random.choice(carriers)
        cursor.execute(
            "INSERT INTO shipments (warehouse_id, direction, sku, quantity, shipment_date, carrier) VALUES (?, ?, ?, ?, ?, ?)",
            (warehouse_id, direction, sku, quantity, ship_date, carrier)
        )

    conn.commit()
    conn.close()
    print(f"Created: {db_path}")


def create_hr_db():
    """Create an HR/employees database."""
    db_path = DATA_DIR / "hr.db"
    db_path.unlink(missing_ok=True)  # Remove existing to avoid duplicates
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Departments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            department_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget DECIMAL(12, 2),
            head_count_limit INTEGER
        )
    """)

    # Employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE,
            department_id INTEGER REFERENCES departments(department_id),
            hire_date DATE,
            salary DECIMAL(10, 2),
            job_title TEXT,
            manager_id INTEGER REFERENCES employees(employee_id)
        )
    """)

    # Performance reviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_reviews (
            review_id INTEGER PRIMARY KEY,
            employee_id INTEGER REFERENCES employees(employee_id),
            review_date DATE,
            rating INTEGER CHECK(rating BETWEEN 1 AND 5),
            reviewer_id INTEGER REFERENCES employees(employee_id),
            comments TEXT
        )
    """)

    # Departments
    departments = [
        ("Engineering", 2500000, 50),
        ("Sales", 1500000, 30),
        ("Marketing", 800000, 15),
        ("Operations", 1200000, 25),
        ("Human Resources", 500000, 10),
        ("Finance", 600000, 12),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO departments (name, budget, head_count_limit) VALUES (?, ?, ?)",
        departments
    )

    # Employees
    employees = [
        ("Alice", "Johnson", "alice.johnson@company.com", 1, "2020-03-15", 150000, "VP Engineering", None),
        ("Bob", "Smith", "bob.smith@company.com", 1, "2021-06-01", 120000, "Senior Engineer", 1),
        ("Carol", "Williams", "carol.williams@company.com", 1, "2022-01-10", 95000, "Software Engineer", 1),
        ("David", "Brown", "david.brown@company.com", 1, "2022-08-20", 85000, "Junior Engineer", 2),
        ("Eva", "Martinez", "eva.martinez@company.com", 2, "2019-11-05", 140000, "VP Sales", None),
        ("Frank", "Garcia", "frank.garcia@company.com", 2, "2021-02-14", 90000, "Sales Manager", 5),
        ("Grace", "Lee", "grace.lee@company.com", 2, "2022-04-01", 65000, "Sales Rep", 6),
        ("Henry", "Wilson", "henry.wilson@company.com", 3, "2020-07-20", 110000, "Marketing Director", None),
        ("Iris", "Taylor", "iris.taylor@company.com", 3, "2021-09-15", 75000, "Marketing Specialist", 8),
        ("Jack", "Anderson", "jack.anderson@company.com", 4, "2018-05-10", 130000, "COO", None),
        ("Karen", "Thomas", "karen.thomas@company.com", 4, "2020-12-01", 80000, "Operations Manager", 10),
        ("Leo", "Jackson", "leo.jackson@company.com", 5, "2019-08-25", 95000, "HR Director", None),
        ("Maya", "White", "maya.white@company.com", 6, "2020-02-28", 115000, "CFO", None),
        ("Nathan", "Harris", "nathan.harris@company.com", 6, "2021-11-10", 85000, "Financial Analyst", 13),
        ("Olivia", "Martin", "olivia.martin@company.com", 1, "2023-01-15", 90000, "Software Engineer", 2),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO employees (first_name, last_name, email, department_id, hire_date, salary, job_title, manager_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        employees
    )

    # Performance reviews
    random.seed(44)
    ratings_comments = [
        (5, "Exceptional performance, exceeds all expectations"),
        (4, "Strong performer, consistently delivers quality work"),
        (3, "Meets expectations, solid contributor"),
        (2, "Needs improvement in some areas"),
        (4, "Good progress this quarter"),
    ]

    review_id = 1
    for emp_id in range(1, 16):
        # Each employee gets 2-3 reviews
        for _ in range(random.randint(2, 3)):
            review_date = (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d')
            rating, comment = random.choice(ratings_comments)
            reviewer_id = random.choice([1, 5, 8, 10, 12, 13])
            if reviewer_id != emp_id:
                cursor.execute(
                    "INSERT INTO performance_reviews (employee_id, review_date, rating, reviewer_id, comments) VALUES (?, ?, ?, ?, ?)",
                    (emp_id, review_date, rating, reviewer_id, comment)
                )
                review_id += 1

    conn.commit()
    conn.close()
    print(f"Created: {db_path}")


def create_business_rules_doc():
    """Create a markdown business rules document."""
    doc_path = DOCS_DIR / "business_rules.md"
    content = """# Business Rules and Policies

## Customer Tiers

Our customer tier system determines pricing and support levels:

### Tier Definitions

| Tier | Annual Spend | Discount | Support Level |
|------|--------------|----------|---------------|
| Bronze | < $10,000 | 0% | Email only |
| Silver | $10,000 - $50,000 | 5% | Email + Chat |
| Gold | $50,000 - $200,000 | 10% | Priority support |
| Platinum | > $200,000 | 15% | Dedicated account manager |

### Upgrade Criteria

Customers are automatically upgraded when:
- Their trailing 12-month spend exceeds the threshold
- They have been a customer for at least 6 months
- They have no outstanding payment issues

## Inventory Management

### Reorder Rules

- **Automatic reorder**: Triggered when quantity falls below reorder_level
- **Reorder quantity**: 2x the reorder_level or minimum order quantity
- **Lead time**: Standard 5 business days, express 2 business days

### Stock Alerts

| Alert Level | Condition | Action |
|-------------|-----------|--------|
| Warning | quantity < reorder_level * 1.5 | Email to procurement |
| Critical | quantity < reorder_level | Automatic PO generated |
| Stockout | quantity = 0 | Escalate to management |

## Revenue Recognition

### Order Status Definitions

- **Pending**: Order received, payment processing
- **Shipped**: Left warehouse, in transit
- **Delivered**: Confirmed receipt by customer
- **Cancelled**: Order cancelled before shipment

### Revenue Timing

Revenue is recognized when:
1. Order status = 'delivered'
2. Payment has been received
3. Return period (30 days) has expired for high-value items

## Performance Review Guidelines

### Rating Scale

| Rating | Description | Typical Raise |
|--------|-------------|---------------|
| 5 | Exceptional | 8-12% |
| 4 | Exceeds Expectations | 5-8% |
| 3 | Meets Expectations | 2-4% |
| 2 | Needs Improvement | 0% |
| 1 | Unsatisfactory | PIP required |

### Review Frequency

- Annual reviews: All employees in Q4
- Mid-year check-ins: Optional but recommended
- New hire reviews: 30, 60, 90 days after start
"""
    doc_path.write_text(content)
    print(f"Created: {doc_path}")


def create_metrics_csv():
    """Create a CSV file with website metrics data."""
    csv_path = DATA_DIR / "website_metrics.csv"

    import csv

    headers = ["date", "page", "visitors", "page_views", "bounce_rate", "avg_session_duration"]

    random.seed(45)
    pages = ["/home", "/products", "/pricing", "/about", "/contact", "/blog", "/docs"]

    rows = []
    base_date = datetime.now() - timedelta(days=90)

    for day_offset in range(90):
        date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        for page in pages:
            # Base traffic varies by page
            base_visitors = {"home": 500, "/products": 300, "/pricing": 150,
                           "/about": 80, "/contact": 50, "/blog": 200, "/docs": 120}.get(page, 100)
            visitors = base_visitors + random.randint(-50, 100)
            page_views = int(visitors * random.uniform(1.2, 2.5))
            bounce_rate = round(random.uniform(0.2, 0.7), 2)
            avg_duration = round(random.uniform(30, 300), 1)
            rows.append([date, page, visitors, page_views, bounce_rate, avg_duration])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Created: {csv_path}")


def create_events_json():
    """Create a JSON file with event/log data."""
    json_path = DATA_DIR / "events.json"

    import json

    random.seed(46)
    event_types = ["page_view", "click", "form_submit", "purchase", "signup", "error"]
    sources = ["web", "mobile_ios", "mobile_android", "api"]

    events = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(500):
        timestamp = base_date + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        event = {
            "event_id": f"evt_{i:06d}",
            "timestamp": timestamp.isoformat(),
            "event_type": random.choice(event_types),
            "source": random.choice(sources),
            "user_id": f"user_{random.randint(1, 100):04d}" if random.random() > 0.2 else None,
            "properties": {
                "page": random.choice(["/home", "/products", "/checkout", "/account"]),
                "duration_ms": random.randint(100, 5000),
            }
        }
        if event["event_type"] == "purchase":
            event["properties"]["amount"] = round(random.uniform(10, 500), 2)
        if event["event_type"] == "error":
            event["properties"]["error_code"] = random.choice(["E001", "E002", "E003", "E404", "E500"])
        events.append(event)

    with open(json_path, 'w') as f:
        json.dump(events, f, indent=2)

    print(f"Created: {json_path}")


def create_config():
    """Create the demo config file with relative paths."""
    config_path = DEMO_DIR / "config.yaml"
    content = """# Constat Demo Configuration
# Run from project root: constat repl -c demo/config.yaml

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}

# SQL Databases and File-based Data Sources
databases:
  # SQL Databases (SQLite)
  sales:
    uri: sqlite:///demo/data/sales.db
    description: |
      Sales data with customers (name, email, tier, region), products (name, category,
      price, cost), orders (date, status, total), and order_items (quantity, unit_price).
      Customer tiers: bronze/silver/gold/platinum. Order status: pending/shipped/delivered/cancelled.

  inventory:
    uri: sqlite:///demo/data/inventory.db
    description: |
      Warehouse inventory with warehouses (location, capacity, manager), inventory
      (sku, quantity, reorder_level), and shipments (direction, carrier, date).
      SKUs link to sales.products via naming convention.

  hr:
    uri: sqlite:///demo/data/hr.db
    description: |
      HR data with departments (budget, headcount_limit), employees (salary, title,
      manager_id for hierarchy), and performance_reviews (1-5 rating scale).

  # File-based data sources (CSV, JSON, Parquet, Arrow)
  web_metrics:
    type: csv
    path: demo/data/website_metrics.csv
    description: |
      Daily web analytics by page (90 days). Columns: date, page, visitors,
      page_views, bounce_rate, avg_session_duration

  events:
    type: json
    path: demo/data/events.json
    description: |
      Clickstream events (500 records). Fields: event_id, timestamp, event_type
      (page_view/click/form_submit/purchase/signup/error), source, user_id, properties

databases_description: |
  Three interconnected business databases and two file-based data sources:
  - sales: CRM and order management (customers.tier determines discount)
  - inventory: Multi-warehouse stock management (reorder_level triggers alerts)
  - hr: Employee and performance tracking (ratings affect compensation)
  - web_metrics: Daily web analytics CSV (use file_web_metrics path)
  - events: Clickstream events JSON (use file_events path)

# Reference Documents (unstructured files - markdown, text, PDFs)
documents:
  business_rules:
    type: file
    path: demo/docs/business_rules.md
    description: Business policies for customer tiers, inventory reorder, and performance reviews

system_prompt: |
  You are a business analyst assistant analyzing company data.

  DOMAIN KNOWLEDGE:
  - Customer tiers: platinum (15% discount), gold (10%), silver (5%), bronze (0%)
  - Revenue recognition: Only count orders with status='delivered'
  - Inventory alerts: quantity < reorder_level triggers restock
  - Performance ratings: 5=exceptional, 4=exceeds, 3=meets, 2=needs improvement, 1=unsatisfactory

  DATA RELATIONSHIPS:
  - inventory.sku links to products by name pattern (e.g., SKU-001 = Widget Pro)
  - employees.manager_id is self-referential for org hierarchy
  - orders.customer_id -> customers.customer_id
  - order_items.product_id -> products.product_id

  FILE ACCESS:
  - For CSV files: pd.read_csv(file_web_metrics)
  - For JSON files: pd.read_json(file_events)

# External APIs (examples - these are public demo APIs)
apis:
  countries:
    type: graphql
    url: https://countries.trevorblades.com/graphql
    description: |
      Public GraphQL API for country data. Query countries by code,
      get continents, languages, currencies. Useful for geo-enrichment.

execution:
  timeout_seconds: 120
  max_retries: 5
  allowed_imports:
    - pandas
    - numpy
    - json
    - datetime
    - statistics
    - polars
    # Standard library
    - re
    - math
    - collections
    - itertools
    - functools
    # Visualization libraries
    - plotly
    - altair
    - matplotlib
    - seaborn
    - folium
"""
    config_path.write_text(content)
    print(f"Created: {config_path}")


def main():
    """Set up the complete demo environment."""
    print("Setting up Constat demo environment...\n")

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    # Create databases
    print("Creating databases:")
    create_sales_db()
    create_inventory_db()
    create_hr_db()

    # Create file-based data sources
    print("\nCreating data files:")
    create_metrics_csv()
    create_events_json()

    # Create documents
    print("\nCreating documents:")
    create_business_rules_doc()

    # Create config
    print("\nCreating config:")
    create_config()

    print("\n" + "=" * 60)
    print("Demo environment ready!")
    print("=" * 60)
    print("""
DATA SOURCES CREATED:

  Databases (databases section):
    SQL:
      - sales.db: customers, products, orders, order_items (200 orders)
      - inventory.db: warehouses, inventory, shipments (4 warehouses)
      - hr.db: departments, employees, performance_reviews (15 employees)
    Files:
      - web_metrics (CSV): 90 days of web analytics by page
      - events (JSON): 500 clickstream events with properties

  Documents (unstructured reference):
    - business_rules.md: Tier discounts, inventory policies, review guidelines

  APIs:
    - countries (GraphQL): Public API for country/continent data

GENERATED CODE ACCESS:
  - SQL databases: pd.read_sql("...", db_sales)
  - CSV files: pd.read_csv(file_web_metrics)
  - JSON files: pd.read_json(file_events)

TO USE THE DEMO:

  1. Set your API key:
     export ANTHROPIC_API_KEY=your_key_here

  2. Run from project root:
     constat schema -c demo/config.yaml    # View schema
     constat repl -c demo/config.yaml      # Interactive mode

  3. Example queries:
     - "Top 5 customers by total order value"
     - "Products with low inventory across warehouses"
     - "Average performance rating by department"
     - "Which pages have the highest bounce rate?"
     - "Count events by type from the events data"
""")


if __name__ == "__main__":
    main()
