---
name: customer-insights
description: RFM segmentation and customer lifetime value patterns
allowed-tools:
  - list_tables
  - get_table_schema
  - run_sql
---

# Customer Analysis Patterns

## Key Metrics

| Metric | Calculation |
|--------|-------------|
| CLV | `AVG(total_spend) * AVG(lifespan_months)` |
| Churn Rate | `churned_customers / total_customers` |
| Repeat Rate | `customers_with_2plus_orders / total_customers` |
| Avg Order Frequency | `total_orders / unique_customers` |

## RFM Segmentation

Score customers 1-5 on each dimension:
- **Recency**: Days since last purchase (1=longest ago, 5=most recent)
- **Frequency**: Order count (1=fewest, 5=most)
- **Monetary**: Total spend (1=lowest, 5=highest)

### RFM Score Query
```sql
WITH customer_metrics AS (
    SELECT
        customer_id,
        DATEDIFF('day', MAX(order_date), CURRENT_DATE) as recency_days,
        COUNT(DISTINCT order_id) as frequency,
        SUM(amount) as monetary
    FROM orders
    GROUP BY customer_id
)
SELECT
    customer_id,
    NTILE(5) OVER (ORDER BY recency_days DESC) as r_score,
    NTILE(5) OVER (ORDER BY frequency) as f_score,
    NTILE(5) OVER (ORDER BY monetary) as m_score,
    CONCAT(
        NTILE(5) OVER (ORDER BY recency_days DESC),
        NTILE(5) OVER (ORDER BY frequency),
        NTILE(5) OVER (ORDER BY monetary)
    ) as rfm_segment
FROM customer_metrics
```

### Segment Definitions

| RFM Pattern | Segment | Action |
|-------------|---------|--------|
| 555, 554, 545 | Champions | Reward, early access |
| 543, 444, 435 | Loyal | Upsell, referral program |
| 512, 411, 311 | New | Onboard, nurture |
| 155, 154, 145 | At Risk | Win-back campaign |
| 111, 112, 121 | Lost | Survey, last attempt |

## Cohort Retention

```sql
WITH first_order AS (
    SELECT customer_id, DATE_TRUNC('month', MIN(order_date)) as cohort_month
    FROM orders GROUP BY customer_id
),
monthly_orders AS (
    SELECT customer_id, DATE_TRUNC('month', order_date) as order_month
    FROM orders GROUP BY 1, 2
)
SELECT
    f.cohort_month,
    DATEDIFF('month', f.cohort_month, m.order_month) as month_number,
    COUNT(DISTINCT m.customer_id) as customers
FROM first_order f
JOIN monthly_orders m ON f.customer_id = m.customer_id
GROUP BY 1, 2
ORDER BY 1, 2
```

## Related: sales-analysis

Use with `sales-analysis` skill to analyze revenue by customer segment.
