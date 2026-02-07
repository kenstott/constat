# Constat Quick Reference

*For data practitioners and AI-assisted analysis*

**Note:** Constat prioritizes accuracy and provability over response time. Analysis may take longer than typical AI tools, but results are auditable and reproducible.

## Getting Started

Type natural language questions in the input box. Constat will analyze your data sources and execute analysis steps to answer your question.

**Examples:**
- "What are the top 10 customers by revenue?"
- "Show employees hired in the last 6 months"
- "Calculate the average order value by region"

## Workflow

1. **Ask a question** - Type your analysis request
2. **Review the plan** - Constat shows the steps it will take
3. **Approve or modify** - Click Approve to proceed, or edit steps
4. **View results** - Tables, charts, and insights appear in the right panel

## Commands

Use `/` commands for quick actions:

| Command | Description |
|---------|-------------|
| `/tables` | List available data tables |
| `/show <table>` | Preview table contents |
| `/artifacts` | View generated outputs |
| `/facts` | Show discovered facts |
| `/prove` | Generate auditable proof of claims |
| `/reset` | Start a new analysis |
| `/help` | Show this help |

## Plan Editing

When reviewing a plan:
- **Edit step goals** - Click on a step to modify its objective
- **Delete steps** - Click the X to remove unwanted steps
- **Provide feedback** - Add comments for replanning
- **Role assignment** - Steps can be assigned to specialist roles

## Side Panel

The right panel shows:
- **Results** - Tables and artifacts from your analysis
- **Code** - Generated Python code for each step
- **Entities** - Business terms and concepts discovered
- **Facts** - Key values and findings

Click any result to expand it. Use the download button to export.

### Code Export

The Code section lets you download all step code as a single Python script. This is useful for:
- Reproducing analysis outside Constat
- Integrating generated code into existing pipelines
- Auditing exactly what was executed

## Roles & Skills

Constat uses specialist roles and skills to improve analysis quality:

### Roles

Domain-specific personas that guide how analysis is performed:
- Each role has expertise in a particular domain (e.g., HR, Finance, Sales)
- Roles influence code generation style, terminology, and approach
- Steps in a plan can be assigned to specific roles
- Examples: `hr_analyst`, `financial_controller`, `data_engineer`

### Skills

Reusable capabilities that can be applied to analysis steps:
- Encapsulate common operations or domain knowledge
- Can be shared across roles
- Automatically matched to queries based on relevance
- Examples: `sql_optimization`, `date_handling`, `currency_conversion`

Roles and skills are configured in your project's YAML files and can be customized for your organization's needs.

## Tips

- **Follow-up questions** - After an analysis, ask refinement questions
- **Corrections** - Use `/correct` to teach Constat your preferences
- **Proof mode** - Click "Proof" to verify claims with step-by-step reasoning
- **Save plans** - Use `/save <name>` to reuse analysis workflows

## Knowledge Types

Constat uses five types of stored knowledge:

### Sources

Metadata about where to find information:
- Databases, APIs, and documents configured for your session
- Can be **system-level** (available to all sessions) or **session-level** (temporary)
- Provide schema and connection info for fact resolution
- Examples: `hr_database`, `sales_api`, `company_handbook.pdf`

### Entities

Abstraction over sources for progressive discovery:
- Tables, columns, concepts, business terms discovered from sources
- Enable semantic search and relationship mapping
- Automatically extracted and indexed for fast lookup
- Help phrase questions more precisely (autocomplete, suggestions)
- Examples: `customer` (table), `revenue` (column), `churn rate` (concept)

### Facts

Concrete values with full provenance:
- Provide *what* values to use in calculations
- Referenced by name in proof/auditable mode
- Include source attribution and confidence scores
- Examples: `my_age = 35`, `raise_budget = 50000`
- Use for: specific values that should appear in proof derivations

### Learnings

Patterns learned from interactions:
- Influence *how* the system approaches problems
- Injected as guidance during planning and code generation
- Created automatically from corrections or manually via `/learn`
- Examples: "Always use LEFT JOIN on customer_orders", "fiscal_year starts in April"
- Use for: domain knowledge, preferences, avoiding past mistakes

### Rules

Consolidated learnings with higher confidence:
- Automatically compacted from multiple related learnings
- Have tags and source counts for discoverability
- Higher confidence than individual learnings
- Applied automatically when relevant to a query

| Aspect | Source | Entity | Fact | Learning | Rule |
|--------|--------|--------|------|----------|------|
| Provides data | Yes | Discovery | Yes | No | No |
| Appears in proof tree | No | No | Yes | No | No |
| Has confidence score | No | No | Yes | No | Yes |
| Influences behavior | No | No | No | Yes | Yes |
| Created automatically | Config | Indexing | Query | Corrections | Compaction |

**When to use which:**
- **Source**: Configure to enable data access (system or session level)
- **Entity**: Automatic - discovered from sources during indexing
- **Fact**: Store values needed in calculations with audit trails
- **Learning**: Teach preferences and domain knowledge
- **Rule**: Automatic - system consolidates learnings over time

## Data Sources

Your configured data sources appear in the toolbar. Add new sources using:
- `/database` - Connect a database
- `/api` - Add an API endpoint
- `/doc` - Add a document
