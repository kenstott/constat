# Business Rules and Policies

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
in fac
