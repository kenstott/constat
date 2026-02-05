# Constat Quick Reference

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
- **Entities** - Business terms and concepts discovered
- **Facts** - Key values and findings

Click any result to expand it. Use the download button to export.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Submit query |
| `Ctrl+C` | Cancel execution |
| `Ctrl+Left` | Shrink side panel |
| `Ctrl+Right` | Expand side panel |

## Tips

- **Follow-up questions** - After an analysis, ask refinement questions
- **Corrections** - Use `/correct` to teach Constat your preferences
- **Proof mode** - Click "Proof" to verify claims with step-by-step reasoning
- **Save plans** - Use `/save <name>` to reuse analysis workflows

## Data Sources

Your configured data sources appear in the toolbar. Add new sources using:
- `/database` - Connect a database
- `/api` - Add an API endpoint
- `/doc` - Add a document
