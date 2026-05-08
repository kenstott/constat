# Fix Demo-Specific Advantages in LLM Prompts

## Problem Statement

The product code contains LLM prompt examples that exactly match the demo schema (customers, orders, tier, revenue, sales_db). This creates artificial performance inflation during demos that won't generalize to production use cases.

**Key insight**: The bias primarily affects *general/world knowledge* rather than *technical knowledge*. SQL syntax and Python patterns transfer well across model sizes and domains. But soft knowledge (recognizing sample databases, understanding business domain patterns, inferring schema semantics) is where larger models have training-data priors that get unfairly reinforced by demo-specific examples.

---

## Locations Requiring Changes

### 1. `constat/session.py:134` - Database Name Examples

**Current:**
```python
- Database connections: `db_<name>` for each database (e.g., `db_chinook`, `db_northwind`)
```

**Problem:** `chinook` and `northwind` are famous sample databases extensively covered in LLM training data. Large models have strong priors about their schemas.

**Fix:**a 
```python
- Database connections: `db_<name>` for each database (e.g., `db_main`, `db_analytics`)
```

---

### 2. `constat/session.py:201-207` - DataFrame/Query Examples

**Current:**
```python
store.save_dataframe('customers', df, step_number=1, description='Customer data')
customers = store.load_dataframe('customers')
result = store.query('SELECT * FROM customers WHERE revenue > 1000')
```

**Problem:** Uses `customers` table with `revenue` column - exact demo schema.

**Fix:**
```python
store.save_dataframe('results', df, step_number=1, description='Query results')
data = store.load_dataframe('results')
result = store.query('SELECT * FROM results WHERE value > 1000')
```

---

### 3. `constat/session.py:235-243` - Column Existence Check Example

**Current:**
```python
df = store.load_dataframe('customers')
# BAD: df['tier'].nunique()  # May fail if 'tier' doesn't exist
# GOOD:
if 'tier' in df.columns:
    result = df['tier'].nunique()
else:
    tier_cols = [c for c in df.columns if 'tier' in c.lower()]
    if tier_cols:
        result = df[tier_cols[0]].nunique()
```

**Problem:** Teaches LLM to specifically look for `tier` columns - matches demo's customer tier concept.

**Fix:**
```python
df = store.load_dataframe('data')
# BAD: df['category'].nunique()  # May fail if column doesn't exist
# GOOD:
target_col = 'category'
if target_col in df.columns:
    result = df[target_col].nunique()
else:
    # Try alternative column names or handle gracefully
    similar = [c for c in df.columns if target_col in c.lower()]
    if similar:
        result = df[similar[0]].nunique()
```

---

### 4. `constat/session.py:251-254` - Date Filtering Example

**Current:**
```python
result = store.query('''
    SELECT * FROM orders
    WHERE CAST(order_date AS DATE) >= '2024-01-01'
    AND CAST(order_date AS DATE) < '2025-01-01'
''')
```

**Problem:** Uses `orders` table with `order_date` - demo schema.

**Fix:**
```python
result = store.query('''
    SELECT * FROM events
    WHERE CAST(event_date AS DATE) >= '2024-01-01'
    AND CAST(event_date AS DATE) < '2025-01-01'
''')
```

---

### 5. `constat/session.py:4011-4035` - Audit Mode Derivation Examples

**Current:**
```python
EXAMPLE 1 - "What is revenue multiplied by Pi?":

PREMISES:
P1: orders = ? (All orders with amounts) [source: database:sales_db]
P2: pi_value = 3.14159 (Mathematical constant) [source: knowledge]

INFERENCE:
I1: total_revenue = sum(P1.amount) -- Sum all order amounts
I2: adjusted_revenue = multiply(I1, P2) -- Multiply by Pi

EXAMPLE 2 - "Monthly revenue trend for last 12 months":

PREMISES:
P1: orders = ? (All orders with date and amount) [source: database:sales_db]

INFERENCE:
I1: recent_orders = filter(P1, last_12_months) -- Filter to last 12 months
I2: monthly_revenue = group_sum(I1, month, amount) -- Group by month, sum amounts
```

**Problem:** Both examples use `orders`, `sales_db`, `amount`, `revenue` - exact demo terminology.

**Fix:**
```python
EXAMPLE 1 - "What is the total multiplied by Pi?":

PREMISES:
P1: records = ? (All records with numeric values) [source: database:main_db]
P2: pi_value = 3.14159 (Mathematical constant) [source: knowledge]

INFERENCE:
I1: total = sum(P1.value) -- Sum all values
I2: adjusted_total = multiply(I1, P2) -- Multiply by Pi

EXAMPLE 2 - "Monthly trend for last 12 months":

PREMISES:
P1: events = ? (All events with date and metric) [source: database:main_db]

INFERENCE:
I1: recent_events = filter(P1, last_12_months) -- Filter to last 12 months
I2: monthly_totals = group_sum(I1, month, metric) -- Group by month, sum metric
```

---

### 6. `constat/execution/fact_resolver.py:2486-2499` - Revenue by Tier Example

**Current:**
```python
Example for "What was revenue by customer tier in Q3?":

GOAL: revenue_by_tier(q3, Tier, Revenue)

revenue_by_tier(Quarter, Tier, Revenue) :-
    tier_criteria(Tier, Criteria),
```

**Problem:** Directly matches demo's business logic (customer tiers, revenue analysis).

**Fix:**
```python
Example for "What was the metric by category in Q3?":

GOAL: metric_by_category(q3, Category, Value)

metric_by_category(Quarter, Category, Value) :-
    category_criteria(Category, Criteria),
```

---

### 7. `constat/execution/problog_resolver.py:19, 179` - Fact Resolution Examples

**Current:**
```python
result = resolver.resolve_fact("monthly_revenue_by_tier", params={})
fact_name: The fact to resolve (e.g., "monthly_revenue_by_tier")
```

**Problem:** `monthly_revenue_by_tier` is demo-specific terminology.

**Fix:**
```python
result = resolver.resolve_fact("monthly_metric_by_group", params={})
fact_name: The fact to resolve (e.g., "monthly_metric_by_group")
```

---

## Validation Strategy

After making changes, compare performance:

| Test Case | Schema | Expected Outcome |
|-----------|--------|------------------|
| Demo schema | customers/orders/tier | Should still work (schema provided at runtime) |
| HR schema | employees/departments/level | Should work equally well |
| IoT schema | sensors/readings/device_type | Should work equally well |
| Finance schema | transactions/accounts/category | Should work equally well |

### Cross-Model Validation

Test with both large (Claude) and small (Mistral 12B) models:
- If changes are effective, performance gap between demo and non-demo schemas should narrow
- Small models should show more consistent performance across schema types

---

## Principles for Future Prompt Examples

1. **Use generic terminology**: `data`, `records`, `events`, `results` instead of domain-specific terms
2. **Avoid famous sample databases**: No `chinook`, `northwind`, `sakila`, `adventureworks`
3. **Use neutral column names**: `value`, `metric`, `category`, `group` instead of `revenue`, `tier`, `customer`
4. **Vary examples**: If multiple examples needed, use different domains (HR, IoT, finance) rather than repeating e-commerce
5. **Focus on pattern, not domain**: Examples should teach the *structure* of a solution, not domain knowledge

---

## Files to Modify

- [ ] `constat/session.py` - Lines 134, 201-207, 235-243, 251-254, 4011-4035
- [ ] `constat/execution/fact_resolver.py` - Lines 2486-2499
- [ ] `constat/execution/problog_resolver.py` - Lines 19, 179
