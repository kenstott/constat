**How Constat Reasons About Problems**

Constat offers two complementary reasoning modes that make AI analysis transparent, verifiable, and trustworthy.

**Exploratory Mode** (Default)

For open-ended questions and discovery:
- Breaks your question into analytical steps
- Each step can gather facts from multiple sources:
  - **Database queries** - retrieve and transform data
  - **User input** - ask for clarification or missing information
  - **LLM knowledge** - apply domain expertise and reasoning
  - **Derivation** - calculate, analyze, or infer from existing facts
- Creates intermediate result tables you can inspect
- Suggests follow-up analyses based on findings

Best for: "What drives revenue?", "Show me trends", "Help me understand..."

**Audited Mode**

For formal verification using an inverted proof structure:
- Your question becomes a hypothesis to prove or disprove
- Works backwards: recursively decomposes claims until grounded in verifiable facts
- Each step must produce evidence supporting or refuting the claim
- Domain rules and constraints are strictly enforced
- Results include confidence levels and caveats
- Full audit trail for compliance and review

Best for: "Verify that...", "Prove whether...", "Is it true that..."

**Why This Matters**

- **Transparency**: Every step is visible - see exactly how conclusions are reached
- **Auditability**: Data-backed claims can be verified against source queries
- **Correctness**: Domain rules (like "only count delivered orders as revenue") are enforced
- **Reproducibility**: The same question produces consistent, explainable results

**In Practice**

When you ask a question, Constat:
1. Plans a series of analytical steps
2. Executes each step - querying data, asking you, or reasoning
3. Shows intermediate results and tables
4. Synthesizes findings into a direct answer
5. Suggests follow-up analyses

Unlike pure LLMs that may hallucinate, Constat grounds all claims in actual data while using AI for reasoning and synthesis.