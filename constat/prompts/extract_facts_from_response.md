Extract key facts/metrics from this analysis response that would be useful to remember.

Question asked: {problem}

Response:
{answer}

Extract facts like:
- Numeric results (e.g., "total revenue was $2.4M" -> total_revenue: 2400000)
- Counts (e.g., "found 150 customers" -> customer_count: 150)
- Percentages (e.g., "growth rate of 15%" -> growth_rate: 0.15)
- Key findings (e.g., "top product is Widget Pro" -> top_product: Widget Pro)
- Time periods analyzed (e.g., "for Q4 2024" -> analysis_period: Q4 2024)

Only extract concrete, specific values. Skip vague or uncertain statements.

Format each fact as:
FACT_NAME: value | brief description
---

Example:
total_revenue: 2400000 | Sum of all order amounts in the period
customer_count: 150 | Number of unique customers who made purchases
growth_rate: 0.15 | Year-over-year revenue growth percentage
---

If no concrete facts to extract, respond with: NO_FACTS