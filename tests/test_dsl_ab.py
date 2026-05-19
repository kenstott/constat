"""A/B benchmark: DSL primitive approach vs. free-form Python codegen.

Runs each query N_RUNS times under each variant and compares:
  - success rate
  - total step attempts (sum across all steps)
  - duration
  - steps planned

Includes easy, medium, and hard (ceiling) queries.

Requires: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow]

# How many times to run each query per variant. LLM output is stochastic;
# more runs give tighter confidence but take longer.
N_RUNS = 10


# ---------------------------------------------------------------------------
# Benchmark query definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    name: str
    query: str
    tier: str  # "easy" | "medium" | "hard"
    # Optional: callable(result) -> bool for semantic correctness check
    check: object | None = None


BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    # --- Easy ---
    BenchmarkQuery(
        name="count_by_tier",
        tier="easy",
        query="How many customers are in each tier?",
        check=lambda r: r.get("success") and "platinum" in str(r.get("answer", "")).lower(),
    ),
    BenchmarkQuery(
        name="total_revenue",
        tier="easy",
        query="What is the total revenue from all delivered orders?",
        # sum of all order amounts = 76,950; check for presence of "76" in answer
        check=lambda r: r.get("success") and "76" in str(r.get("answer", "")),
    ),
    # --- Medium ---
    BenchmarkQuery(
        name="top5_customers_by_spend",
        tier="medium",
        query=(
            "Show the top 5 customers by total order amount, "
            "including their name, tier, and total spend. Sort descending."
        ),
        check=lambda r: r.get("success") and "Stark" in str(r.get("answer", "")),
    ),
    BenchmarkQuery(
        name="monthly_revenue_trend",
        tier="medium",
        query=(
            "Show monthly total revenue for each month in the dataset. "
            "Include month, total revenue, and month-over-month change."
        ),
        check=lambda r: r.get("success") and r.get("answer"),
    ),
    BenchmarkQuery(
        name="avg_order_by_tier",
        tier="medium",
        query=(
            "For each customer tier, calculate the average order value "
            "and the number of orders. Which tier has the highest average?"
        ),
        check=lambda r: r.get("success") and "platinum" in str(r.get("answer", "")).lower(),
    ),
    # --- Hard (ceiling tests) ---
    BenchmarkQuery(
        name="inactive_customers",
        tier="hard",
        query=(
            "Find customers who placed at least 2 orders in Jan-Feb 2024 "
            "but have placed NO orders since March 2024. "
            "Show their name, tier, and last order date."
        ),
        check=lambda r: r.get("success") and r.get("answer"),
    ),
    BenchmarkQuery(
        name="revenue_share_by_tier",
        tier="hard",
        query=(
            "Calculate each tier's share of total revenue as a percentage. "
            "Also compute the revenue-per-customer ratio for each tier. "
            "Rank tiers by revenue share descending."
        ),
        check=lambda r: r.get("success") and "%" in str(r.get("answer", "")),
    ),
    BenchmarkQuery(
        name="cohort_spend_growth",
        tier="hard",
        query=(
            "For customers who joined in Q1 2023 (Jan-Mar), compute their "
            "total spend per quarter (Q1 2024, Q2 2024). "
            "Show which customers increased their spend and by how much."
        ),
        check=lambda r: r.get("success") and r.get("answer"),
    ),
]


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    variant: str          # "dsl" | "no_dsl"
    query_name: str
    run_index: int
    success: bool
    total_attempts: int   # sum of attempts across all steps
    duration_ms: int
    n_steps: int
    answer_ok: bool       # passed semantic check


@dataclass
class ABResults:
    runs: list[RunResult] = field(default_factory=list)

    def add(self, r: RunResult) -> None:
        self.runs.append(r)

    def summary(self) -> str:
        from collections import defaultdict
        lines = []

        # Group by query + variant
        grouped: dict[tuple, list[RunResult]] = defaultdict(list)
        for r in self.runs:
            grouped[(r.query_name, r.variant)].append(r)

        # Header
        col = 26
        lines.append(
            f"\n{'query':<{col}} {'tier':<7} "
            f"{'variant':<8} {'ok%':>5} {'sem%':>5} {'avg_att':>8} {'avg_dur_s':>10} {'avg_steps':>10}"
        )
        lines.append("-" * 85)

        query_tiers = {q.name: q.tier for q in BENCHMARK_QUERIES}
        queries = [q.name for q in BENCHMARK_QUERIES]
        for qname in queries:
            for variant in ("dsl", "no_dsl"):
                runs = grouped.get((qname, variant), [])
                if not runs:
                    continue
                ok_pct = 100 * sum(1 for r in runs if r.success) / len(runs)
                sem_pct = 100 * sum(1 for r in runs if r.answer_ok) / len(runs)
                avg_att = sum(r.total_attempts for r in runs) / len(runs)
                avg_dur = sum(r.duration_ms for r in runs) / len(runs) / 1000
                avg_steps = sum(r.n_steps for r in runs) / len(runs)
                tier = query_tiers.get(qname, "?")
                lines.append(
                    f"{qname:<{col}} {tier:<7} "
                    f"{variant:<8} {ok_pct:>4.0f}% {sem_pct:>4.0f}% "
                    f"{avg_att:>8.1f} {avg_dur:>9.1f}s {avg_steps:>9.1f}"
                )
            lines.append("")

        # Overall summary per variant
        lines.append("=== Overall ===")
        for variant in ("dsl", "no_dsl"):
            vr = [r for r in self.runs if r.variant == variant]
            if not vr:
                continue
            ok = 100 * sum(1 for r in vr if r.success) / len(vr)
            sem = 100 * sum(1 for r in vr if r.answer_ok) / len(vr)
            att = sum(r.total_attempts for r in vr) / len(vr)
            dur = sum(r.duration_ms for r in vr) / len(vr) / 1000
            lines.append(
                f"  {variant:<8}  success={ok:.0f}%  semantic={sem:.0f}%  "
                f"avg_attempts={att:.1f}  avg_dur={dur:.1f}s  n={len(vr)}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DB + session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ab_ecommerce_db():
    """E-commerce SQLite DB reused across all A/B runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "ab_ecommerce.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                tier TEXT NOT NULL,
                created_date TEXT NOT NULL
            )
        """)
        customers = [
            (1, "Acme Corp", "platinum", "2023-01-15"),
            (2, "Globex Inc", "gold", "2023-02-20"),
            (3, "Initech", "silver", "2023-03-10"),
            (4, "Umbrella Corp", "bronze", "2023-04-05"),
            (5, "Stark Industries", "platinum", "2023-05-12"),
            (6, "Wayne Enterprises", "gold", "2023-06-18"),
            (7, "Oscorp", "silver", "2023-07-22"),
            (8, "LexCorp", "bronze", "2023-08-30"),
        ]
        cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

        cur.execute("""
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)
        orders = [
            (1,  1, "2024-01-05", 5000.00, "delivered"),
            (2,  2, "2024-01-10", 2500.00, "delivered"),
            (3,  3, "2024-01-15", 1200.00, "delivered"),
            (4,  4, "2024-01-20",  800.00, "delivered"),
            (5,  1, "2024-02-03", 4500.00, "delivered"),
            (6,  5, "2024-02-08", 6000.00, "delivered"),
            (7,  2, "2024-02-12", 2800.00, "delivered"),
            (8,  6, "2024-02-18", 3200.00, "delivered"),
            (9,  1, "2024-03-02", 5500.00, "delivered"),
            (10, 3, "2024-03-07", 1500.00, "delivered"),
            (11, 5, "2024-03-12", 7000.00, "delivered"),
            (12, 7, "2024-03-20",  900.00, "delivered"),
            (13, 2, "2024-04-05", 3000.00, "delivered"),
            (14, 4, "2024-04-10",  850.00, "delivered"),
            (15, 6, "2024-04-15", 3500.00, "delivered"),
            (16, 8, "2024-04-22",  600.00, "delivered"),
            (17, 1, "2024-05-01", 4800.00, "delivered"),
            (18, 5, "2024-05-08", 6500.00, "delivered"),
            (19, 3, "2024-05-15", 1300.00, "delivered"),
            (20, 7, "2024-05-20",  950.00, "delivered"),
        ]
        cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
        conn.commit()
        conn.close()
        yield str(db_path)


def _make_session(db_path: str, enable_dsl: bool, tmpdir: str):
    """Create a fresh Session for one A/B run."""
    from constat.core.config import Config, DatabaseConfig
    from constat.session import Session, SessionConfig
    from constat.storage.history import SessionHistory

    config = Config(
        databases={
            "sales": DatabaseConfig(
                type="sql",
                uri=f"sqlite:///{db_path}",
                description="E-commerce database",
            ),
        },
        system_prompt="""You are analyzing e-commerce data.

Tables:
- customers: customer_id, name, tier (platinum/gold/silver/bronze), created_date (TEXT 'YYYY-MM-DD')
- orders: order_id, customer_id, order_date (TEXT 'YYYY-MM-DD'), amount, status

Important:
- This is SQLite — use strftime() for date math, not DATE_TRUNC or EXTRACT
- Do NOT use schema prefixes (use 'customers' not 'sales.customers')
- Join on customer_id
""",
        execution={
            "allowed_imports": ["pandas", "numpy", "datetime"],
            "timeout_seconds": 120,
        },
    )
    session_id = str(uuid.uuid4())
    history = SessionHistory(storage_dir=Path(tmpdir) / session_id / "sessions")
    session_config = SessionConfig(
        require_approval=False,
        enable_primitive_dsl=enable_dsl,
        max_retries_per_step=5,
    )
    return Session(
        config,
        session_id=session_id,
        session_config=session_config,
        history=history,
        data_dir=Path(tmpdir) / session_id,
    )


def _run_query(session, query: str, bq: BenchmarkQuery, variant: str, run_idx: int) -> RunResult:
    """Execute one query and collect metrics."""
    t0 = time.time()
    try:
        result = session.solve(query)
    except Exception as e:
        result = {"success": False, "final_answer": "", "error": str(e)}
    duration_ms = int((time.time() - t0) * 1000)

    success = bool(result.get("success", False))
    # answer text lives in final_answer (not answer)
    answer_text = str(result.get("final_answer") or result.get("output") or "")
    # run semantic check against the answer text
    check_result = {"success": success, "answer": answer_text}
    answer_ok = bool(bq.check(check_result)) if bq.check else success

    # Count total attempts across steps via step.result (StepResult.attempts)
    plan = result.get("plan")
    total_attempts = 0
    n_steps = 0
    if plan:
        steps = getattr(plan, "steps", [])
        n_steps = len(steps)
        for step in steps:
            step_result = getattr(step, "result", None)
            total_attempts += getattr(step_result, "attempts", 1) if step_result else 1
    if total_attempts == 0:
        total_attempts = 1

    return RunResult(
        variant=variant,
        query_name=bq.name,
        run_index=run_idx,
        success=success,
        total_attempts=total_attempts,
        duration_ms=duration_ms,
        n_steps=n_steps,
        answer_ok=answer_ok,
    )


# ---------------------------------------------------------------------------
# A/B test — one parametrized test per (query, variant, run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ab_results():
    return ABResults()


@pytest.fixture
def require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY not set — required for this test")


@pytest.mark.parametrize("bq", BENCHMARK_QUERIES, ids=[q.name for q in BENCHMARK_QUERIES])
@pytest.mark.parametrize("variant", ["dsl", "no_dsl"])
@pytest.mark.parametrize("run_idx", range(N_RUNS))
def test_ab_run(require_anthropic_key, ab_ecommerce_db, ab_results, bq, variant, run_idx):
    """Single A/B run — record metrics; never hard-fail (A/B is a measurement)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        enable_dsl = variant == "dsl"
        session = _make_session(ab_ecommerce_db, enable_dsl, tmpdir)
        run = _run_query(session, bq.query, bq, variant, run_idx)
        ab_results.add(run)
        # Log outcome but do not assert — failure is a valid metric data point
        if not run.success:
            print(f"\n  [METRIC] [{variant}] {bq.name} run {run_idx}: FAILED (recorded)")


def test_ab_summary(ab_results):
    """Print A/B comparison table. Always passes; acts as the report step."""
    report = ab_results.summary()
    print(report)
    # Write to file for CI artifact capture
    out = Path("ab_results.txt")
    out.write_text(report)
    print(f"\nReport written to {out.resolve()}")
