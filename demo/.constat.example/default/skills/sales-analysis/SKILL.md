---
name: sales-analysis
description: SQL patterns and metrics for analyzing sales data
allowed-tools:
  - list_tables
  - get_table_schema
  - run_sql
---

# Sales Analysis Patterns

## Key Metrics

| Metric | Calculation |
|--------|-------------|
| Revenue | `SUM(amount)` |
| AOV (Avg Order Value) | `SUM(amount) / COUNT(DISTINCT order_id)` |
| Units Sold | `SUM(quantity)` |
| Order Count | `COUNT(DISTINCT order_id)` |

## Common Queries

### Revenue by Period
```sql
SELECT
    DATE_TRUNC('month', order_date) as period,
    SUM(amount) as revenue,
    COUNT(DISTINCT order_id) as orders,
    SUM(amount) / COUNT(DISTINCT order_id) as aov
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY 1
ORDER BY 1
```

### Product Performance
```sql
SELECT
    p.category,
    p.name,
    SUM(oi.quantity) as units,
    SUM(oi.quantity * oi.unit_price) as revenue,
    ROUND(100.0 * SUM(oi.quantity * oi.unit_price) / SUM(SUM(oi.quantity * oi.unit_price)) OVER (), 2) as pct_total
FROM order_items oi
JOIN products p ON oi.product_id = p.id
GROUP BY 1, 2
ORDER BY revenue DESC
```

### Period-over-Period Growth
```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', order_date) as month, SUM(amount) as revenue
    FROM orders GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) as prev_month,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / LAG(revenue) OVER (ORDER BY month), 1) as growth_pct
FROM monthly
ORDER BY month DESC
```

## Related: customer-insights

Use with `customer-insights` skill to segment sales by customer behavior.
