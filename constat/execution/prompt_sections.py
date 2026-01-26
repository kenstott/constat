# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Prompt sections for conditional injection based on query concepts.

This module defines specialized prompt sections that are injected only when
the user's query semantically matches the concept. This reduces prompt bloat
by ~30% for queries that don't need specialized guidance.

Sections are detected using embedding similarity against exemplar sentences.
"""

from dataclasses import dataclass, field
from typing import Literal

PromptTarget = Literal["engine", "planner", "step"]


@dataclass
class PromptSection:
    """A prompt section that can be conditionally injected."""

    concept_id: str
    """Unique identifier for this concept (e.g., 'dashboard_layout')."""

    content: str
    """The actual prompt content to inject."""

    targets: list[PromptTarget]
    """Which prompts this section applies to: 'engine', 'planner', 'step'."""

    exemplars: list[str] = field(default_factory=list)
    """Example sentences that indicate this concept should be injected."""


# ============================================================================
# Prompt Sections Registry
# ============================================================================

PROMPT_SECTIONS: dict[str, PromptSection] = {
    # -------------------------------------------------------------------------
    # Dashboard Layout Guidelines
    # -------------------------------------------------------------------------
    "dashboard_layout": PromptSection(
        concept_id="dashboard_layout",
        targets=["step"],
        exemplars=[
            "Create a dashboard showing sales metrics",
            "Build a dashboard for executive KPIs",
            "I need a dashboard with multiple charts",
            "Generate an analytics dashboard",
            "Make a dashboard visualization",
            "Build a 2x2 grid of charts",
            "Create a multi-panel view",
            "Show me a dashboard overview",
        ],
        content="""## Dashboard Generation Rules

When the user requests a "dashboard":

### Default Layout (2x2)
Generate 4 complementary visualizations arranged in a 2x2 grid using `make_subplots(rows=2, cols=2)`:
- Top-left: Primary metric over time (line/bar)
- Top-right: Breakdown/composition (pie/bar)
- Bottom-left: Comparison or ranking (bar/table)
- Bottom-right: Trend or KPI summary

### Layout Variations
Adjust based on data characteristics:

| Data Available | Layout | Panels |
|----------------|--------|--------|
| Single metric, time series | 1x2 | Trend + Summary stats |
| Multiple categories | 2x2 | Overview, breakdown, comparison, detail |
| Hierarchical data | 1x3 | High-level -> Mid -> Detail |
| KPI-focused | 3x2 | Top row: KPI cards, Bottom: supporting charts |

### Panel Selection Priority
1. **Critical/requested metrics** - Always include
2. **Time-based trends** - If temporal data exists
3. **Comparisons** - If categorical groupings exist
4. **Distributions** - If numerical spread is relevant

### Code Pattern
```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=2, subplot_titles=('Metric 1', 'Metric 2', 'Metric 3', 'Metric 4'))
fig.add_trace(go.Bar(...), row=1, col=1)
fig.add_trace(go.Pie(...), row=1, col=2)
fig.update_layout(height=600, showlegend=True)
viz.save_chart('dashboard', fig, title='Dashboard Title')
```
""",
    ),
    # -------------------------------------------------------------------------
    # Email Policy
    # -------------------------------------------------------------------------
    "email_policy": PromptSection(
        concept_id="email_policy",
        targets=["planner"],
        exemplars=[
            "Email the report to the team",
            "Send this analysis to marketing",
            "Email the results to john@example.com",
            "Send a summary to the CFO",
            "Notify stakeholders via email",
            "Can you mail this to finance",
            "Send the data to my manager",
            "Email me when this is done",
        ],
        content="""## Email Policy (CRITICAL)

ONLY include email steps when the user EXPLICITLY requests emailing results (e.g., "email this to...", "send to...").
NEVER proactively add email steps to plans. Do NOT interpret phrases like "for CFO review" or "analysis for the team" as email requests.

NEVER include email steps for data that contains:
- Salary, compensation, or pay information
- Personal identifiable information (SSN, addresses, phone numbers)
- Performance reviews or disciplinary records
- Medical or health information
- Financial account numbers
- Passwords or authentication credentials

If the user explicitly requests emailing sensitive data, add a WARNING note to the step that manual review is required before sending.
""",
    ),
    # -------------------------------------------------------------------------
    # API Filtering Patterns
    # -------------------------------------------------------------------------
    "api_filtering": PromptSection(
        concept_id="api_filtering",
        targets=["engine", "step"],
        exemplars=[
            "Query the GraphQL API for active users",
            "Fetch orders from the REST endpoint",
            "Get data from the API with filtering",
            "Use the countries API",
            "Call the orders endpoint",
            "Filter the API results by status",
            "Query only pending items from the API",
            "Get filtered data from the external service",
        ],
        content="""## API Filtering Patterns

**CRITICAL: Always filter at the source!**
Use API filters/arguments instead of fetching all data and filtering in Python. This is faster and uses less memory.

### GraphQL APIs
```python
# Query - pass the GraphQL query string
result = api_<name>('query { ... }')
# result is the 'data' payload directly (outer wrapper stripped)
df = pd.DataFrame(result['<field>'])  # NOT result['data']['<field>']

# GOOD - filter in the query (check schema for exact filter syntax):
result = api_orders('{ orders(status: "pending") { id total } }')

# BAD - fetching all then filtering in Python:
result = api_orders('{ orders { id total status } }')
df = df[df['status'] == 'pending']  # Wasteful!
```

### REST APIs
```python
# Use query parameters for filtering
result = api_<name>('GET /endpoint', {'param': 'value', 'filter': 'active'})
# result is the parsed JSON response
```

Check API schema for available filter arguments before writing queries.
""",
    ),
    # -------------------------------------------------------------------------
    # Visualization (Charts and Maps)
    # -------------------------------------------------------------------------
    "visualization": PromptSection(
        concept_id="visualization",
        targets=["step"],
        exemplars=[
            "Create a chart showing revenue trends",
            "Plot the distribution of customers",
            "Visualize sales by region",
            "Generate a map of store locations",
            "Show a pie chart of market share",
            "Create a bar chart comparison",
            "Make a line graph of monthly data",
            "Display a scatter plot",
            "Create an interactive map",
        ],
        content="""## File Output & Visualizations (via viz)

Save files and interactive visualizations to ~/.constat/outputs/ with clickable file:// URIs.

### Documents and Data Files (text formats)
```python
viz.save_file('quarterly_report', markdown_content, ext='md', title='Q4 Report')
viz.save_file('export', df.to_csv(index=False), ext='csv', title='Data Export')
viz.save_file('report', json.dumps(data, indent=2), ext='json', title='JSON Report')
```

### Excel, PDF, and Binary Files
```python
# Excel: Create in-memory, then save binary content
from io import BytesIO
buffer = BytesIO()
df.to_excel(buffer, index=False, engine='openpyxl')
viz.save_binary('sales_report', buffer.getvalue(), ext='xlsx', title='Sales Report')

# Multiple sheets in Excel
with BytesIO() as buffer:
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Summary', index=False)
        df2.to_excel(writer, sheet_name='Details', index=False)
    viz.save_binary('report', buffer.getvalue(), ext='xlsx', title='Full Report')
```

### Interactive Maps (using folium)
```python
import folium

m = folium.Map(location=[50, 10], zoom_start=4)
for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['name'],
        tooltip=row['name']
    ).add_to(m)

viz.save_map('locations', m, title='Store Locations')
```

### Interactive Charts (using Plotly)
```python
import plotly.express as px

# Bar chart
fig = px.bar(df, x='country', y='population', title='Population by Country')

# Pie chart
fig = px.pie(df, values='count', names='category')

# Line chart
fig = px.line(df, x='date', y='value', color='series')

# Scatter plot
fig = px.scatter(df, x='x', y='y', color='category', size='value')

# Choropleth map
fig = px.choropleth(df, locations='iso_code', color='value',
                    locationmode='ISO-3', title='World Map')

viz.save_chart('chart_name', fig, title='Chart Title')
```

**IMPORTANT**: Never save files directly to disk (e.g., `df.to_excel('file.xlsx')`). Always use viz methods so files are properly tracked and stored in the artifacts directory.
""",
    ),
    # -------------------------------------------------------------------------
    # File Output (Reports, Exports, Results)
    # -------------------------------------------------------------------------
    "file_output": PromptSection(
        concept_id="file_output",
        targets=["step"],
        exemplars=[
            "Calculate employee raises and save the results",
            "Generate a report of sales data",
            "Export the analysis to Excel",
            "Create a summary report",
            "Save the results to a file",
            "Produce a salary report",
            "Output the data to CSV",
            "Make a report showing the totals",
            "Generate an Excel spreadsheet",
            "Save this as a markdown report",
            "Create an export of the data",
            "Prepare a report for management",
        ],
        content="""## Final Results & Reports (via viz)

For any final output that should be viewable by the user, use `viz` to create clickable file:// URIs.

### Summary Reports (Markdown)
```python
# Use llm_ask for narrative prose (adapts to data at runtime)
summary = llm_ask(f"Write 2-3 sentence summary of analysis with {len(df)} records, avg value ${df['amount'].mean():,.2f}")
insights = llm_ask(f"List 3 key insights from: {df.groupby('category')['amount'].sum().to_dict()}")

# Build report with generated prose + data
report = f\"\"\"# Analysis Report

## Summary
{summary}

- Total records: {len(df)}
- Average value: ${df['amount'].mean():,.2f}

## Key Insights
{insights}

## Details
{df.head(10).to_markdown(index=False)}
\"\"\"
viz.save_file('report', report, ext='md', title='Analysis Report')
```

**IMPORTANT:** Never write literal prose that describes data. Use `llm_ask()` so prose adapts when data changes.

### Data Exports (CSV/Excel)
```python
# CSV export
viz.save_file('export', df.to_csv(index=False), ext='csv', title='Data Export')

# Excel export (single sheet)
from io import BytesIO
buffer = BytesIO()
df.to_excel(buffer, index=False, engine='openpyxl')
viz.save_binary('report', buffer.getvalue(), ext='xlsx', title='Excel Report')
```

### Key Principles
1. **Use viz for all final outputs** - creates clickable file:// URIs in terminal
2. **Don't print full tables** - use viz.save_file() instead
3. **Print only brief summaries** - e.g., "Saved 150 records to report"
4. Tables saved to `store` appear automatically in the artifacts panel
""",
    ),
    # -------------------------------------------------------------------------
    # State Management (store)
    # -------------------------------------------------------------------------
    "state_management": PromptSection(
        concept_id="state_management",
        targets=["step"],
        exemplars=[
            "Use the data from the previous step",
            "Load the customer dataframe we saved",
            "Save this result for the next step",
            "Query the stored orders table",
            "Pass this value to step 3",
            "Get the results from step 1",
            "Store this for later use",
            "Access the intermediate table",
        ],
        content="""## State Management (via store)

Each step runs in complete isolation. The ONLY way to share data between steps is through `store`.

### DataFrames
```python
# Save a DataFrame for later steps
store.save_dataframe('customers', df, step_number=1, description='Customer data')

# Load a DataFrame from a previous step
customers = store.load_dataframe('customers')

# Query saved data with SQL (DuckDB syntax)
result = store.query('SELECT * FROM customers WHERE revenue > 1000')

# List available tables
tables = store.list_tables()
```

### Simple Values (numbers, strings, lists, dicts)
```python
# Save state (NOTE: set_state does NOT have a 'description' parameter)
store.set_state('total_revenue', total, step_number=1)
store.set_state('top_genres', ['Rock', 'Latin', 'Metal'], step_number=1)

# Load state - ALWAYS check for None!
total = store.get_state('total_revenue')
if total is None:
    total = 0  # Handle missing value
```

**IMPORTANT:** Nothing in local variables persists between steps!
""",
    ),
    # -------------------------------------------------------------------------
    # LLM Batching
    # -------------------------------------------------------------------------
    "llm_batching": PromptSection(
        concept_id="llm_batching",
        targets=["step"],
        exemplars=[
            "Enrich each row with additional information",
            "Add descriptions to each item",
            "Get details for every country",
            "Look up facts about each product",
            "Add context to the data using LLM",
            "Augment the dataframe with LLM knowledge",
            "Get AI-generated descriptions for each",
            "Use llm_ask to add information",
        ],
        content="""## LLM Knowledge (via llm_ask)

Use `llm_ask(question)` to get general knowledge not available in databases.

**CRITICAL: Batch LLM calls for multiple items!**
NEVER call llm_ask() in a loop - it's extremely slow. Instead, batch all questions into ONE call:

```python
# BAD - 10 separate LLM calls (very slow!)
for country in countries:
    attractions[country] = llm_ask(f"Tourist attractions in {country}")

# GOOD - 1 batched LLM call (fast!)
countries_list = ", ".join(df['name'].tolist())
result = llm_ask(f"For each country, list 2-3 tourist attractions. Countries: {countries_list}. Format: Country: attraction1, attraction2")
# Then parse the result
```

Note: llm_ask returns a string. Parse numeric values or structured data if needed.
""",
    ),
    # -------------------------------------------------------------------------
    # Prose Generation (Reports, Summaries, Narratives)
    # -------------------------------------------------------------------------
    "prose_generation": PromptSection(
        concept_id="prose_generation",
        targets=["step"],
        exemplars=[
            "Create a markdown report",
            "Generate a summary document",
            "Write an executive summary",
            "Produce a narrative analysis",
            "Create a report with insights",
            "Generate documentation",
            "Write a comprehensive report",
            "Build a formatted report",
            "Create an analysis report",
            "Generate a summary of findings",
        ],
        content="""## Prose Generation (CRITICAL)

**NEVER write literal prose/narrative text in code.** The code may be re-run with different data, making literals incorrect.

**ALWAYS use `llm_ask()` to generate prose that describes or analyzes data:**

```python
# BAD - Literal prose becomes wrong if data changes!
summary = "This analysis covers 10 cat breeds including the Abyssinian and Siamese..."

# GOOD - Prose adapts to actual data at runtime
summary = llm_ask(
    f"Write a 2-3 sentence executive summary for a cat breeds analysis. "
    f"Number of breeds: {len(df)}. "
    f"Countries represented: {df['country'].nunique()}. "
    f"Top coat types: {df['coat'].value_counts().head(3).to_dict()}"
)
```

### What to Generate with llm_ask()
- Executive summaries
- Section narratives ("The data reveals...")
- Analytical insights
- Conclusions and recommendations
- Any prose that references data values or patterns

### What Can Be Literal
- Section headers/titles
- Table column headers
- Static methodology descriptions
- Fixed labels and formatting

### Pattern for Reports
```python
# Generate prose sections with llm_ask
intro = llm_ask(f"Write intro paragraph for {topic} analysis with {len(df)} records")
insights = llm_ask(f"List 3-4 key insights from this data: {df.describe().to_dict()}")
conclusion = llm_ask(f"Write conclusion for analysis showing {key_finding}")

# Combine with structure
report = f\"\"\"# {title}

## Introduction
{intro}

## Data Summary
{df.describe().to_markdown()}

## Key Insights
{insights}

## Conclusion
{conclusion}
\"\"\"
viz.save_file('report', report, ext='md', title=title)
```
""",
    ),
    # -------------------------------------------------------------------------
    # Sensitive Data Handling
    # -------------------------------------------------------------------------
    "sensitive_data": PromptSection(
        concept_id="sensitive_data",
        targets=["planner", "step"],
        exemplars=[
            "Show employee salaries",
            "List compensation packages",
            "Get customer social security numbers",
            "Display patient health records",
            "Export payroll data",
            "Show performance reviews",
            "Get credit card numbers",
            "List employee addresses",
            "Show personal phone numbers",
            "Display medical information",
            "Export PII data",
            "Get financial account details",
        ],
        content="""## Sensitive Data Handling (CRITICAL)

This request may involve sensitive data. Apply these safeguards:

### Data Classification
Mark plan as `contains_sensitive_data: true` for:
- **PII**: Names + SSN, addresses, phone numbers, email
- **Financial**: Salary, compensation, account numbers, credit cards
- **Health**: Medical records, diagnoses, prescriptions (HIPAA)
- **HR**: Performance reviews, disciplinary records, termination reasons

### Protection Rules
1. **No email with sensitive data** - Never include email steps for sensitive data unless explicitly requested AND add WARNING
2. **Minimize exposure** - Only query/display the specific fields needed
3. **Aggregate when possible** - Use averages, counts, percentiles instead of individual records
4. **Audit trail** - Print what sensitive data was accessed for logging

### Code Patterns
```python
# GOOD: Aggregate sensitive data
avg_salary = df['salary'].mean()
print(f"Average salary: ${avg_salary:,.0f}")

# BAD: Expose individual sensitive records
print(df[['name', 'salary', 'ssn']])  # Don't do this!

# If individual display is required, mask partial data:
df['ssn_masked'] = df['ssn'].apply(lambda x: f"***-**-{x[-4:]}")
```
""",
    ),
}


def get_section(concept_id: str) -> PromptSection | None:
    """Get a prompt section by ID."""
    return PROMPT_SECTIONS.get(concept_id)


def get_sections_for_target(target: PromptTarget) -> list[PromptSection]:
    """Get all sections that apply to a specific prompt target."""
    return [s for s in PROMPT_SECTIONS.values() if target in s.targets]
