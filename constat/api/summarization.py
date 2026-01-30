# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Summarization functions for plans, sessions, facts, and tables.

Provides LLM-powered summarization of various session components.
"""

from typing import TYPE_CHECKING, Optional

from constat.api.types import SummarizeResult

if TYPE_CHECKING:
    from constat.session import Session
    from constat.storage.facts import FactStore


def summarize_plan(session: "Session", llm) -> SummarizeResult:
    """Generate a summary of the current execution plan.

    Args:
        session: Session instance with plan to summarize
        llm: LLM provider for generating summary

    Returns:
        SummarizeResult with plan summary or error
    """
    if not session.plan:
        return SummarizeResult(success=False, error="No plan to summarize")

    plan = session.plan
    plan_text = f"Goal: {plan.goal}\n\nSteps:\n"

    step_results = getattr(session, 'step_results', [])
    for i, step in enumerate(plan.steps, 1):
        status = "completed" if i <= len(step_results) else "pending"
        plan_text += f"{i}. [{status}] {step.description}\n"
        if step.code:
            code_preview = step.code[:100] + "..." if len(step.code) > 100 else step.code
            plan_text += f"   Code: {code_preview}\n"

    prompt = f"""Summarize this execution plan concisely:

{plan_text}

Provide a 2-3 sentence summary covering:
1. The overall goal
2. Key steps and current progress
3. Any notable approach or methodology"""

    try:
        result = llm.generate(
            system="You are a concise technical summarizer.",
            user_message=prompt,
            max_tokens=300,
        )
        return SummarizeResult(success=True, summary=result)
    except Exception as e:
        return SummarizeResult(success=False, error=str(e))


def summarize_session(session: "Session", llm) -> SummarizeResult:
    """Generate a summary of the current session state.

    Args:
        session: Session instance to summarize
        llm: LLM provider for generating summary

    Returns:
        SummarizeResult with session summary or error
    """
    session_info = []

    # Session ID and mode
    session_info.append(f"Session ID: {session.session_id or 'Not started'}")
    if hasattr(session, 'current_mode'):
        session_info.append(f"Mode: {session.current_mode}")

    # Plan status
    if session.plan:
        session_info.append(f"Plan: {session.plan.goal}")
        session_info.append(f"Steps: {len(session.plan.steps)}")

    # Tables in datastore
    if session.datastore:
        tables = session.datastore.list_tables()
        if tables:
            session_info.append(f"Tables: {', '.join(tables)}")

    # Facts count
    if session.fact_resolver:
        facts = session.fact_resolver.get_all_facts()
        if facts:
            session_info.append(f"Facts cached: {len(facts)}")

    # Execution history
    if hasattr(session, 'execution_history') and session.execution_history:
        session_info.append(f"Queries executed: {len(session.execution_history)}")

    if not session_info:
        return SummarizeResult(success=False, error="No session state to summarize")

    prompt = f"""Summarize this session state concisely:

{chr(10).join(session_info)}

Provide a 2-3 sentence summary of what this session has accomplished and its current state."""

    try:
        result = llm.generate(
            system="You are a concise technical summarizer.",
            user_message=prompt,
            max_tokens=300,
        )
        return SummarizeResult(success=True, summary=result)
    except Exception as e:
        return SummarizeResult(success=False, error=str(e))


def summarize_facts(session: "Session", llm) -> SummarizeResult:
    """Generate a summary of all cached facts.

    Args:
        session: Session instance with fact resolver
        llm: LLM provider for generating summary

    Returns:
        SummarizeResult with facts summary or error
    """
    if not session.fact_resolver:
        return SummarizeResult(success=False, error="No fact resolver available")

    facts = session.fact_resolver.get_all_facts()
    if not facts:
        return SummarizeResult(success=False, error="No facts to summarize")

    facts_text = []
    for name, fact in facts.items():
        source = fact.source.value if hasattr(fact.source, 'value') else str(fact.source)
        value_str = str(fact.value)[:100] if fact.value else "None"
        facts_text.append(f"- {name}: {value_str} (source: {source})")

    prompt = f"""Summarize these cached facts concisely:

{chr(10).join(facts_text)}

Provide a summary covering:
1. How many facts are cached
2. Types of facts (data sources, user-provided, computed)
3. Key facts that drive the analysis"""

    try:
        result = llm.generate(
            system="You are a concise technical summarizer.",
            user_message=prompt,
            max_tokens=300,
        )
        return SummarizeResult(success=True, summary=result)
    except Exception as e:
        return SummarizeResult(success=False, error=str(e))


def summarize_table(
    session: "Session",
    table_name: str,
    llm,
) -> SummarizeResult:
    """Generate a summary of a specific table's contents.

    Args:
        session: Session instance with datastore
        table_name: Name of the table to summarize
        llm: LLM provider for generating summary

    Returns:
        SummarizeResult with table summary or error
    """
    if not session.datastore:
        return SummarizeResult(success=False, error="No datastore available")

    tables = session.datastore.list_tables()
    if table_name not in tables:
        available = ', '.join(tables) if tables else 'None'
        return SummarizeResult(
            success=False,
            error=f"Table '{table_name}' not found. Available tables: {available}",
        )

    try:
        # Get table info
        df = session.datastore.load_dataframe(table_name)
        row_count = len(df)
        columns = list(df.columns)

        # Get sample data
        sample = df.head(5).to_string() if row_count > 0 else "Empty table"

        # Get basic stats for numeric columns
        stats = []
        numeric_cols = df.select_dtypes(include=['number']).columns[:3]
        for col in numeric_cols:
            stats.append(
                f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                f"mean={df[col].mean():.2f}"
            )

        stats_section = f"Numeric stats: {'; '.join(stats)}" if stats else ""

        prompt = f"""Summarize this table concisely:

Table: {table_name}
Rows: {row_count}
Columns: {', '.join(columns)}

Sample data:
{sample}

{stats_section}

Provide a 2-3 sentence summary covering:
1. What kind of data this table contains
2. Key columns and their purpose
3. Notable patterns or ranges in the data"""

        result = llm.generate(
            system="You are a concise data analyst.",
            user_message=prompt,
            max_tokens=300,
        )
        return SummarizeResult(success=True, summary=result)

    except Exception as e:
        return SummarizeResult(success=False, error=f"Error reading table: {e}")
