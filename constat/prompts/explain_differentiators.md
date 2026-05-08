**What Makes Constat Different**

Constat is designed for serious data analysis where accuracy, transparency, and collaboration matter.

**1. Universal Data Connectivity**

Connect almost any data source as a fact source:
- **Databases**: PostgreSQL, MySQL, SQLite, DuckDB, and more
- **Structured files**: CSV, Excel, Parquet, JSON
- **Documents**: PDF, Word, PowerPoint for context
- **APIs**: REST endpoints for live data

All sources become queryable facts that ground your analysis.

**2. Parallel Execution**

Plans are directed acyclic graphs (DAGs), not linear scripts:
- Independent steps execute in parallel automatically
- Complex analyses complete faster
- Dependencies are tracked and respected

**3. Reproducibility**

Plans are deterministic code, not chat transcripts:
- Save any analysis as a replayable plan
- Re-run against current data to see how results change
- Version control your analytical workflows

**4. Intelligent Caching**

Extensive caching minimizes costs and latency:
- Facts are cached and reused across steps
- Database query results are stored
- LLM calls are minimized through smart planning

**5. Collaboration**

Share your work with others:
- Share plans with specific users or make them public
- Resume sessions from where you left off
- Teams can build on each other's analyses

**6. Auditable Reasoning**

Formal verification for high-stakes decisions:
- Claims are recursively decomposed until grounded in verifiable facts
- Full audit trail for compliance and review
- Ask "How do you reason about problems?" for details

**7. Breadth of Data**

Handle large data estates without upfront metadata loading:
- Progressive discovery drills into metadata only when needed for a specific question
- Connect hundreds of tables without overwhelming context
- Metadata is fetched on-demand, not preloaded

**8. Extensible Skills**

Extend reasoning capabilities using the familiar Skills pattern:
- Add custom skills for domain-specific analyses
- Reuse existing AI agent skills you've already built
- Skills plug into the fact-gathering and reasoning pipeline