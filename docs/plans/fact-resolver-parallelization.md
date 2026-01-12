# Fact Resolver Parallelization Plan

## Overview

This document outlines the plan to parallelize fact resolution in the auditable proof system. The key insight is that **top-level assumed facts are independent** and can be fetched in parallel, followed by a single computation step.

## Current Architecture

```
User Question
     │
     ▼
Sequential Fact Resolution (SLOW)
     │
     ├── resolve(fact_a)  ← 2 sec (LLM)
     ├── resolve(fact_b)  ← 2 sec (LLM)
     ├── resolve(fact_c)  ← 0.5 sec (DB)
     └── resolve(fact_d)  ← 0 sec (cache)
     │
     Total: ~4.5 seconds
     │
     ▼
Execute Computation
```

## Proposed Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│  LLM: Identify Required Facts           │
│                                         │
│  Returns: [fact_a, fact_b, fact_c, ...] │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Parallel Fact Resolution (FAST)        │
│                                         │
│  asyncio.gather(                        │
│      resolve(fact_a),  ─┐               │
│      resolve(fact_b),  ─┼─► ~2 sec      │
│      resolve(fact_c),  ─┤   (parallel)  │
│      resolve(fact_d),  ─┘               │
│  )                                      │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Execute Computation (single step)      │
│                                         │
│  answer = f(fact_a, fact_b, fact_c, d)  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Return Answer + Auditable Proof        │
│                                         │
│  {                                      │
│    answer: "...",                       │
│    facts: [...],                        │
│    computation: "...",                  │
│    provenance: [...]                    │
│  }                                      │
└─────────────────────────────────────────┘
```

## Key Insight: Facts Are Independent

In an auditable proof system:

| Concept | Definition | Dependencies |
|---------|------------|--------------|
| **Assumed Fact** | Ground truth from external source (DB, API, config, user) | None - independent |
| **Computation** | Code that combines facts to produce answer | Requires all facts |

Example proof structure:
```
CLAIM: Acme Corp is a good acquisition target

ASSUMED FACTS (all independent, fetch in parallel):
  - acme_revenue = $50M         (source: database)
  - acme_growth_rate = 15%      (source: database)
  - industry_avg_growth = 8%    (source: market_data_api)
  - acquisition_threshold = 10% (source: user_config)

COMPUTATION:
  growth_premium = acme_growth_rate - industry_avg_growth  # 15% - 8% = 7%
  is_good_target = growth_premium > 0  # 7% > 0 = True

ANSWER: Yes (confidence: 0.85)
```

All four facts can be resolved simultaneously. No topological sort needed.

## When Would Facts Have Dependencies?

After rigorous analysis, genuine fact-to-fact dependencies are **rare**:

| Scenario | Example | Resolution |
|----------|---------|------------|
| Multi-hop entity lookup | parent_id → parent_revenue | Collapse into single SQL JOIN |
| Parameterized queries | quarter → quarterly_revenue | Embed parameter in query |
| Pagination | page_1 → cursor → page_2 | Encapsulate in single fetch operation |

**Conclusion**: Design for the common case (all facts parallel). Handle rare dependencies as exceptions.

## Implementation Plan

### Phase 1: Async LLM Provider

**Files to modify:**
- `constat/providers/base.py`
- `constat/providers/anthropic.py`

**Changes:**
```python
# base.py
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, ...) -> str:
        """Synchronous generation (existing)."""
        ...

    @abstractmethod
    async def generate_async(self, prompt: str, ...) -> str:
        """Asynchronous generation (new)."""
        ...

# anthropic.py
from anthropic import AsyncAnthropic

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, ...):
        self._client = Anthropic(...)
        self._async_client = AsyncAnthropic(...)

    async def generate_async(self, prompt: str, ...) -> str:
        response = await self._async_client.messages.create(...)
        return response.content[0].text
```

### Phase 2: Parallel Fact Resolution

**Files to modify:**
- `constat/execution/fact_resolver.py`

**New method:**
```python
class FactResolver:
    async def resolve_async(self, fact_name: str, **params) -> Fact:
        """Resolve a single fact asynchronously."""
        cache_key = self._cache_key(fact_name, params)

        # Check cache first (sync, fast)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Async resolution through sources
        for source in self.strategy.source_priority:
            if source == FactSource.CACHE:
                continue  # Already checked
            elif source == FactSource.DATABASE:
                fact = await self._resolve_from_database_async(fact_name, params)
            elif source == FactSource.LLM_KNOWLEDGE:
                fact = await self._resolve_from_llm_async(fact_name, params)
            # ... other sources

            if fact and fact.is_resolved:
                self._cache[cache_key] = fact
                return fact

        return Fact(name=fact_name, value=None, source=FactSource.UNRESOLVED)

    async def resolve_all(
        self,
        facts: list[tuple[str, dict]],
        max_concurrent: int = 5,
    ) -> list[Fact]:
        """Resolve multiple facts in parallel.

        Args:
            facts: List of (fact_name, params) tuples
            max_concurrent: Max concurrent LLM calls (rate limiting)

        Returns:
            List of resolved Facts in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def resolve_with_limit(fact_name: str, params: dict) -> Fact:
            async with semaphore:
                return await self.resolve_async(fact_name, **params)

        tasks = [
            resolve_with_limit(name, params)
            for name, params in facts
        ]

        return await asyncio.gather(*tasks)
```

### Phase 3: Rate Limiting

**New file:** `constat/execution/rate_limiter.py`

```python
import asyncio
import time

class RateLimiter:
    """Rate limiter for LLM API calls."""

    def __init__(
        self,
        max_concurrent: int = 5,
        requests_per_minute: int = 60,
    ):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rpm_limit = requests_per_minute
        self._request_times: list[float] = []

    async def acquire(self):
        """Acquire permission to make a request."""
        await self._semaphore.acquire()

        # RPM limiting
        now = time.time()
        minute_ago = now - 60
        self._request_times = [t for t in self._request_times if t > minute_ago]

        if len(self._request_times) >= self._rpm_limit:
            sleep_time = self._request_times[0] - minute_ago
            await asyncio.sleep(sleep_time)

        self._request_times.append(time.time())

    def release(self):
        """Release after request completes."""
        self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        self.release()
```

### Phase 4: Integration with Session

**Files to modify:**
- `constat/session.py`

**Changes:**
```python
class Session:
    async def solve_async(self, problem: str) -> dict:
        """Async version of solve with parallel fact resolution."""
        # 1. Generate plan (identifies needed facts)
        plan = await self._generate_plan_async(problem)

        # 2. Extract all assumed facts from plan
        assumed_facts = self._extract_assumed_facts(plan)

        # 3. Resolve all facts in parallel
        resolved_facts = await self.fact_resolver.resolve_all(assumed_facts)

        # 4. Execute computation with resolved facts
        result = await self._execute_with_facts_async(plan, resolved_facts)

        # 5. Build auditable proof
        return {
            "answer": result.answer,
            "facts": [f.to_dict() for f in resolved_facts],
            "computation": result.code,
            "provenance": self._build_provenance(resolved_facts),
        }
```

## Auditable Proof Structure

The system produces an auditable proof with each response:

```python
@dataclass
class AuditableProof:
    """Complete audit trail for a response."""

    # The question asked
    question: str

    # All assumed facts with provenance
    facts: list[Fact]

    # The computation/code that combined facts
    computation: str

    # The final answer
    answer: Any

    # Confidence score
    confidence: float

    # Timestamp
    resolved_at: datetime

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "facts": [
                {
                    "name": f.name,
                    "value": f.value,
                    "source": f.source.value,
                    "query": f.query,  # SQL if from database
                    "reasoning": f.reasoning,  # If from LLM
                }
                for f in self.facts
            ],
            "computation": self.computation,
            "resolved_at": self.resolved_at.isoformat(),
        }
```

## Performance Expectations

| Scenario | Sequential | Parallel (5 concurrent) | Speedup |
|----------|-----------|------------------------|---------|
| 5 LLM facts | ~10 sec | ~2 sec | 5x |
| 3 LLM + 2 DB facts | ~7 sec | ~2.5 sec | 2.8x |
| 10 LLM facts | ~20 sec | ~4 sec | 5x |
| 5 cached facts | ~0 sec | ~0 sec | 1x |

## Error Handling

When a fact fails to resolve:

```python
async def resolve_all(self, facts: list[tuple[str, dict]]) -> list[Fact]:
    results = await asyncio.gather(
        *[self.resolve_async(name, **params) for name, params in facts],
        return_exceptions=True,
    )

    resolved = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append((facts[i], result))
            resolved.append(Fact(
                name=facts[i][0],
                value=None,
                source=FactSource.UNRESOLVED,
                reasoning=str(result),
            ))
        else:
            resolved.append(result)

    if errors:
        # Log but continue - partial results are valid
        logger.warning(f"Failed to resolve {len(errors)} facts: {errors}")

    return resolved
```

## Testing Strategy

### Unit Tests

```python
# tests/test_parallel_resolution.py

@pytest.mark.asyncio
async def test_resolve_all_parallel():
    """Test that facts are resolved in parallel."""
    resolver = FactResolver(...)

    # Mock LLM with 1 second delay
    async def slow_llm(*args):
        await asyncio.sleep(1)
        return "resolved"

    resolver._resolve_from_llm_async = slow_llm

    facts = [("fact_1", {}), ("fact_2", {}), ("fact_3", {})]

    start = time.time()
    results = await resolver.resolve_all(facts)
    elapsed = time.time() - start

    assert len(results) == 3
    assert elapsed < 2  # Should be ~1 sec, not 3 sec

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test that concurrent requests are limited."""
    resolver = FactResolver(...)

    call_times = []
    async def track_calls(*args):
        call_times.append(time.time())
        await asyncio.sleep(0.1)
        return "resolved"

    resolver._resolve_from_llm_async = track_calls

    facts = [("fact", {"i": i}) for i in range(10)]
    await resolver.resolve_all(facts, max_concurrent=2)

    # With max_concurrent=2, calls should be staggered
    # Not all 10 starting at the same time
    assert max(call_times) - min(call_times) > 0.3
```

### Integration Tests

```python
@pytest.mark.asyncio
@pytest.mark.requires_anthropic_key
async def test_parallel_llm_resolution():
    """Test parallel resolution with real LLM."""
    config = Config(...)
    resolver = FactResolver(llm=AnthropicProvider(config.llm), ...)

    facts = [
        ("us_population", {}),
        ("uk_population", {}),
        ("france_population", {}),
    ]

    start = time.time()
    results = await resolver.resolve_all(facts)
    elapsed = time.time() - start

    assert all(f.is_resolved for f in results)
    assert elapsed < 5  # Should be ~2-3 sec parallel, not 6-9 sec serial
```

## Migration Path

1. **Phase 1**: Add async methods alongside sync methods (no breaking changes)
2. **Phase 2**: Add `resolve_all()` for batch resolution (opt-in)
3. **Phase 3**: Add `solve_async()` to Session (opt-in)
4. **Phase 4**: Deprecate sync `resolve()` for fact batches (optional)

Existing code continues to work. New code can opt into parallelization.

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `constat/providers/base.py` | Modify | Add `generate_async()` abstract method |
| `constat/providers/anthropic.py` | Modify | Implement async using `AsyncAnthropic` |
| `constat/execution/fact_resolver.py` | Modify | Add `resolve_async()`, `resolve_all()` |
| `constat/execution/rate_limiter.py` | Create | Rate limiting for API calls |
| `constat/session.py` | Modify | Add `solve_async()` with parallel facts |
| `tests/test_parallel_resolution.py` | Create | Unit and integration tests |

## Initial Proof in Layman's English

In auditable mode, the LLM first generates a proof structure in plain English that explains:
1. What question is being answered
2. What facts are needed and why
3. How those facts will be combined to reach a conclusion

This is presented to the user BEFORE resolution, so they understand the reasoning approach.

### Example: Acquisition Target Analysis

**User asks:** "Is Acme Corp a good acquisition target for us?"

**LLM generates initial proof (layman's English):**

```
To determine if Acme Corp is a good acquisition target, I'll evaluate
the following:

WHAT I NEED TO FIND OUT:

1. Financial Health
   - Acme's annual revenue (from your financial database)
   - Acme's revenue growth rate over the past 3 years (from your financial database)
   - Acme's profit margin (from your financial database)

2. Customer Value
   - Acme's average customer lifetime value (I'll calculate this from their
     order history - specifically their average order size, how often
     customers buy, and typical customer retention period)

3. Market Context
   - The typical growth rate for companies in Acme's industry (from market data)
   - Your company's acquisition budget (from your configuration)
   - Your minimum acceptable growth premium for acquisitions (from your configuration)

HOW I'LL DECIDE:

I'll recommend Acme as a GOOD acquisition target if:
  • Their growth rate exceeds the industry average by at least your
    required premium (showing they're outperforming the market)
  • Their customer lifetime value is healthy (indicating sustainable revenue)
  • Their valuation fits within your acquisition budget

I'll recommend AGAINST acquisition if:
  • Growth is below industry average (why pay a premium for underperformance?)
  • Customer value is declining (suggests future revenue problems)
  • The price exceeds your budget with no strategic justification

Shall I proceed with this analysis?
```

### Example: Simpler Query

**User asks:** "Which sales region performed best last quarter?"

**LLM generates initial proof:**

```
To find your best-performing sales region last quarter, I need:

FACTS TO GATHER:
  • Total sales for each region in Q4 2024 (from your sales database)
  • Number of sales reps per region (from your HR database)
  • Regional sales targets for Q4 (from your planning database)

HOW I'LL EVALUATE "BEST":
  I'll rank regions by:
  1. Total revenue (absolute performance)
  2. Revenue per sales rep (efficiency)
  3. Percentage of target achieved (relative to expectations)

  The "best" region is the one that ranks highest across these metrics.
  If there's no clear winner, I'll explain the trade-offs.

Shall I proceed?
```

### Example: Query Requiring Sub-Proof

**User asks:** "What's our customer churn risk?"

**LLM generates initial proof:**

```
To assess your customer churn risk, I need to understand several things:

FACTS TO GATHER:

1. Current Customer Behavior
   - Number of active customers (from your CRM)
   - Average days since last purchase per customer (from order history)
   - Customer support ticket trends (from your support system)

2. Historical Churn Patterns
   - Your churn rate over the past 12 months (from your metrics database)
   - Industry average churn rate (from market research)

3. Leading Indicators (these require some calculation)
   - Customer engagement score
     → I'll derive this from: login frequency + feature usage + support interactions
   - Customer health score
     → I'll derive this from: payment history + usage trends + satisfaction surveys

HOW I'LL ASSESS RISK:

  LOW RISK: Churn below industry average, engagement scores stable/improving
  MEDIUM RISK: Churn near industry average, some engagement decline
  HIGH RISK: Churn above average, significant engagement drops,
             increasing support tickets

For any customers flagged as high-risk, I'll identify the specific
warning signs so you can take action.

Shall I proceed with this analysis?
```

### Structure of Initial Proof

The LLM output follows this pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│  QUESTION RESTATED                                              │
│  (Confirm understanding of what user wants to know)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FACTS NEEDED                                                   │
│  - Fact 1: description (source)                                 │
│  - Fact 2: description (source)                                 │
│  - Derived Fact: description                                    │
│    → How it will be calculated from other facts                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  DECISION LOGIC                                                 │
│  - How facts will be combined                                   │
│  - What thresholds or criteria apply                            │
│  - What different outcomes mean                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONFIRMATION REQUEST                                           │
│  "Shall I proceed?" or "Does this approach make sense?"         │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Matters

1. **Transparency**: User sees the reasoning BEFORE execution
2. **Validation**: User can correct misunderstandings early
3. **Trust**: No "black box" - the logic is explicit
4. **Auditability**: The proof structure is part of the audit trail
5. **Efficiency**: Catch wrong approaches before wasting compute

## CLI and UI Presentation

Since facts resolve in parallel, we need to show progress in real-time. The user should see:
1. Which facts are being resolved
2. Progress as each completes
3. Final proof with all resolved values

### Fact Sources

Facts can come from various sources, and the display should clearly indicate which:

| Source Type | Display Format | Example |
|-------------|----------------|---------|
| SQL Database | `db_name (type)` | `financial_db (PostgreSQL)` |
| Document Store | `db_name (type)` | `orders_db (MongoDB)` |
| Search Index | `index_name (type)` | `products (Elasticsearch)` |
| REST API | `api_name (REST)` | `market_api (REST)` |
| GraphQL API | `api_name (GraphQL)` | `analytics_api (GraphQL)` |
| Config | `config` | `config` |
| User Input | `user_provided` | `user_provided` |
| LLM Knowledge | `LLM knowledge` | `LLM knowledge` |
| Derived | `derived` | `derived` (shows sub-facts) |

For database sources, also show:
- **SQL**: The actual query executed
- **MongoDB**: The aggregation pipeline or find query
- **Elasticsearch**: The search query
- **API**: The endpoint called

This transparency is crucial for auditability - users need to verify WHERE data came from.

### CLI Representation (Rich Library)

Using the existing Rich-based feedback system, we can show a live-updating display:

**Initial State (after user confirms proof):**
```
Resolving facts...

  ◐ acme_revenue          resolving...
  ◐ acme_growth_rate      resolving...
  ◐ industry_avg_growth   resolving...
  ◐ acquisition_threshold resolving...
  ◐ acme_customer_ltv     resolving...
```

**As facts resolve (live updates):**
```
Resolving facts...

  ✓ acme_revenue          $50,000,000         financial_db (postgres)     0.3s
  ✓ acme_growth_rate      15%                 financial_db (postgres)     0.4s
  ◐ industry_avg_growth   resolving...        market_api (REST)
  ✓ acquisition_threshold 10%                 config                      0.1s
  ◐ acme_customer_ltv     deriving...         (sub-proof)
      ├─ ✓ avg_order_value     $150           orders_db (mongodb)
      ├─ ✓ purchase_frequency  4/year         orders_db (mongodb)
      └─ ◐ retention_years     resolving...   LLM knowledge
```

**Completed State:**
```
Resolving facts... done (2.3s)

  ✓ acme_revenue          $50,000,000         financial_db (postgres)     0.3s
  ✓ acme_growth_rate      15%                 metrics_db (postgres)       0.4s
  ✓ industry_avg_growth   8%                  market_api (REST)           1.2s
  ✓ acquisition_threshold 10%                 config                      0.1s
  ✓ acme_customer_ltv     $1,800              derived                     1.8s
      ├─ avg_order_value     $150             orders_db (mongodb)
      ├─ purchase_frequency  4/year           orders_db (mongodb)
      └─ retention_years     3 years          LLM knowledge

Running analysis...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESULT: Acme Corp is a GOOD acquisition target

REASONING:
  • Growth rate (15%) exceeds industry average (8%) by 7 percentage points
  • This 7% premium exceeds your minimum threshold of 10%... wait, no.
  • Actually, 15% - 8% = 7%, which is BELOW your 10% threshold.

REVISED RESULT: Acme Corp is NOT recommended for acquisition

  The growth premium (7%) does not meet your minimum requirement (10%).
  Consider if strategic factors justify an exception.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Error State (partial resolution):**
```
Resolving facts... 4 of 5 completed

  ✓ acme_revenue          $50,000,000         from database     0.3s
  ✓ acme_growth_rate      15%                 from database     0.4s
  ✗ industry_avg_growth   FAILED              API timeout
  ✓ acquisition_threshold 10%                 from config       0.1s
  ✓ acme_customer_ltv     $1,800              derived           1.8s

⚠ Could not resolve: industry_avg_growth
  Error: Market data API timed out after 30s

Options:
  [1] Retry failed facts
  [2] Provide value manually
  [3] Continue without (reduced confidence)
  [4] Abort

Choice:
```

### CLI Implementation (Rich)

```python
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.spinner import Spinner
from rich.text import Text

class FactResolutionDisplay:
    """Live CLI display for parallel fact resolution."""

    def __init__(self, console: Console):
        self.console = console
        self.facts: dict[str, FactDisplayState] = {}

    def add_fact(self, name: str, source_hint: str = ""):
        """Register a fact to be resolved."""
        self.facts[name] = FactDisplayState(
            name=name,
            status="pending",
            source_hint=source_hint,
        )

    def update_fact(
        self,
        name: str,
        status: str,  # "resolving", "resolved", "failed", "deriving"
        value: Any = None,
        source: str = None,
        elapsed: float = None,
        error: str = None,
        sub_facts: list = None,
    ):
        """Update a fact's display state."""
        self.facts[name].status = status
        self.facts[name].value = value
        self.facts[name].source = source
        self.facts[name].elapsed = elapsed
        self.facts[name].error = error
        self.facts[name].sub_facts = sub_facts or []

    def render(self) -> Table:
        """Render current state as a Rich Table."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("status", width=3)
        table.add_column("name", width=25)
        table.add_column("value", width=20)
        table.add_column("source", width=15)
        table.add_column("time", width=8)

        for fact in self.facts.values():
            icon = self._status_icon(fact.status)
            value_str = self._format_value(fact.value) if fact.value else ""
            source_str = fact.source or ""
            time_str = f"{fact.elapsed:.1f}s" if fact.elapsed else ""

            table.add_row(icon, fact.name, value_str, source_str, time_str)

            # Show sub-facts for derived values
            for i, sub in enumerate(fact.sub_facts):
                prefix = "└─" if i == len(fact.sub_facts) - 1 else "├─"
                sub_icon = self._status_icon(sub.status)
                table.add_row(
                    "",
                    f"  {prefix} {sub.name}",
                    self._format_value(sub.value),
                    sub.source or "",
                    "",
                )

        return table

    def _status_icon(self, status: str) -> Text:
        icons = {
            "pending": Text("○", style="dim"),
            "resolving": Text("◐", style="yellow"),
            "deriving": Text("◐", style="cyan"),
            "resolved": Text("✓", style="green"),
            "failed": Text("✗", style="red"),
        }
        return icons.get(status, Text("?"))

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if value > 1_000_000:
                return f"${value/1_000_000:.1f}M"
            elif value < 1:
                return f"{value*100:.0f}%"
        return str(value)


# Usage during resolution
async def resolve_with_display(resolver, facts, console):
    display = FactResolutionDisplay(console)

    # Register all facts
    for name, params in facts:
        display.add_fact(name)

    with Live(display.render(), console=console, refresh_per_second=4) as live:
        async def resolve_and_update(name, params):
            display.update_fact(name, status="resolving")
            live.update(display.render())

            try:
                fact = await resolver.resolve_async(name, **params)
                display.update_fact(
                    name,
                    status="resolved",
                    value=fact.value,
                    source=fact.source.value,
                    elapsed=fact.resolved_at - start_time,
                    sub_facts=[...] if fact.because else [],
                )
            except Exception as e:
                display.update_fact(name, status="failed", error=str(e))

            live.update(display.render())
            return fact

        results = await asyncio.gather(
            *[resolve_and_update(name, params) for name, params in facts]
        )

    return results
```

### React UI Representation (Future Phase)

For the React UI, a similar live-updating view:

```
┌─────────────────────────────────────────────────────────────────┐
│  Resolving Facts                                         2.3s   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✓ acme_revenue           $50,000,000                          │
│    └─ Source: financial_db.companies                           │
│                                                                 │
│  ✓ acme_growth_rate       15%                                  │
│    └─ Source: financial_db.metrics                             │
│                                                                 │
│  ● industry_avg_growth    ████████░░░░░░░░  resolving...       │
│    └─ Source: market_data_api                                  │
│                                                                 │
│  ✓ acquisition_threshold  10%                                  │
│    └─ Source: config.acquisition_rules                         │
│                                                                 │
│  ◐ acme_customer_ltv      deriving...                          │
│    ├─ ✓ avg_order_value      $150                              │
│    ├─ ✓ purchase_frequency   4/year                            │
│    └─ ● retention_years      resolving...                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

React component structure:

```typescript
interface FactDisplayProps {
  facts: FactState[];
  onRetry?: (factName: string) => void;
  onProvideValue?: (factName: string) => void;
}

interface FactState {
  name: string;
  status: 'pending' | 'resolving' | 'resolved' | 'failed' | 'deriving';
  value?: any;
  source?: string;
  elapsed?: number;
  error?: string;
  subFacts?: FactState[];
}

function FactResolutionPanel({ facts, onRetry, onProvideValue }: FactDisplayProps) {
  return (
    <div className="fact-resolution-panel">
      {facts.map(fact => (
        <FactRow
          key={fact.name}
          fact={fact}
          onRetry={() => onRetry?.(fact.name)}
          onProvideValue={() => onProvideValue?.(fact.name)}
        />
      ))}
    </div>
  );
}

function FactRow({ fact, onRetry, onProvideValue }) {
  return (
    <div className={`fact-row fact-${fact.status}`}>
      <StatusIcon status={fact.status} />
      <span className="fact-name">{fact.name}</span>

      {fact.status === 'resolving' && <ProgressBar />}
      {fact.status === 'resolved' && <span className="fact-value">{formatValue(fact.value)}</span>}
      {fact.status === 'failed' && (
        <div className="fact-error">
          <span>{fact.error}</span>
          <button onClick={onRetry}>Retry</button>
          <button onClick={onProvideValue}>Provide Value</button>
        </div>
      )}

      {fact.subFacts && (
        <div className="sub-facts">
          {fact.subFacts.map(sub => (
            <FactRow key={sub.name} fact={sub} indent />
          ))}
        </div>
      )}
    </div>
  );
}
```

### Completed Proof Display (CLI)

After all facts resolve, show the complete proof:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                           AUDITABLE PROOF

Question: Is Acme Corp a good acquisition target?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FACTS GATHERED:

  acme_revenue ................ $50,000,000
                                └─ financial_db (PostgreSQL)
                                   SELECT revenue FROM companies WHERE name='Acme'
                                   [0.3s]

  acme_growth_rate ............ 15%
                                └─ metrics_db (PostgreSQL)
                                   SELECT growth_rate FROM metrics WHERE company_id=123
                                   [0.4s]

  industry_avg_growth ......... 8%
                                └─ market_api (REST API)
                                   GET https://api.marketdata.com/v1/industry/tech/growth
                                   [1.2s]

  acquisition_threshold ....... 10%
                                └─ config
                                   acquisition_rules.min_growth_premium
                                   [0.1s]

  acme_customer_ltv ........... $1,800 (derived)
                                ├─ avg_order_value = $150
                                │  └─ orders_db (MongoDB)
                                │     db.orders.aggregate([{$group: {_id: null, avg: {$avg: "$total"}}}])
                                ├─ purchase_frequency = 4/year
                                │  └─ orders_db (MongoDB)
                                │     db.orders.aggregate([{$group: {_id: "$customer_id", count: {$sum: 1}}}])
                                └─ retention_years = 3
                                   └─ LLM Knowledge
                                      "Based on industry benchmarks, typical B2B SaaS customer
                                       retention is 3-5 years. Using conservative estimate of 3."
                                └─ Calculation: $150 × 4 × 3 = $1,800
                                   [1.8s total]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALYSIS:

  growth_premium = acme_growth_rate - industry_avg_growth
                 = 15% - 8%
                 = 7%

  is_good_target = growth_premium >= acquisition_threshold
                 = 7% >= 10%
                 = FALSE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCLUSION: Acme Corp is NOT recommended for acquisition.

  The growth premium (7%) does not meet your minimum threshold (10%).

  However, consider:
    • Customer LTV ($1,800) is healthy
    • Revenue ($50M) shows established market presence

  Recommendation: Re-evaluate if strategic factors (market access,
  technology, talent) justify accepting below-threshold growth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[P] Show Python code  [J] Export as JSON  [S] Save report  [Q] New question
```

### Event-Driven Architecture

The fact resolver emits events that the UI subscribes to:

```python
class FactResolutionEvent:
    """Events emitted during fact resolution."""
    STARTED = "started"           # Resolution batch started
    FACT_STARTED = "fact_started" # Single fact resolution started
    FACT_RESOLVED = "fact_resolved"
    FACT_FAILED = "fact_failed"
    SUB_PROOF_STARTED = "sub_proof_started"
    SUB_PROOF_COMPLETED = "sub_proof_completed"
    COMPLETED = "completed"       # All facts resolved

@dataclass
class FactEvent:
    event_type: str
    fact_name: str
    timestamp: float
    data: dict  # value, source, error, etc.

class FactResolver:
    def __init__(self, ...):
        self._event_handlers: list[Callable[[FactEvent], None]] = []

    def on_event(self, handler: Callable[[FactEvent], None]):
        self._event_handlers.append(handler)

    def _emit(self, event: FactEvent):
        for handler in self._event_handlers:
            handler(event)

    async def resolve_async(self, fact_name: str, **params) -> Fact:
        self._emit(FactEvent(
            event_type=FactResolutionEvent.FACT_STARTED,
            fact_name=fact_name,
            timestamp=time.time(),
            data={"params": params},
        ))

        try:
            fact = await self._do_resolve(fact_name, params)
            self._emit(FactEvent(
                event_type=FactResolutionEvent.FACT_RESOLVED,
                fact_name=fact_name,
                timestamp=time.time(),
                data={"value": fact.value, "source": fact.source.value},
            ))
            return fact
        except Exception as e:
            self._emit(FactEvent(
                event_type=FactResolutionEvent.FACT_FAILED,
                fact_name=fact_name,
                timestamp=time.time(),
                data={"error": str(e)},
            ))
            raise
```

### Implementation Note

The initial proof is generated by a dedicated prompt:

```python
PROOF_GENERATION_PROMPT = """
You are generating an auditable proof for a data analysis question.

Given the user's question, explain in plain English:
1. What facts you need to gather (and where they come from)
2. Any facts that need to be derived from other facts
3. How you'll combine the facts to reach a conclusion
4. What different conclusions would mean

Use simple language a non-technical business user would understand.
Do NOT use jargon, SQL, or code.

Format your response as a clear explanation, not a technical spec.
End by asking if the user wants you to proceed.
"""
```

## Recursive Fact Resolution (Sub-Proofs)

A top-level fact may not be directly resolvable from a database or API. In this case, the system generates a **sub-proof** - a new proof structure to derive the fact. This makes fact resolution **recursive**.

### Example: Derived Fact Requiring Sub-Proof

```
User: "Is Acme a good acquisition target?"

Top-Level Proof:
├── acme_revenue = $50M          (DATABASE - direct)
├── acme_growth_rate = 15%       (DATABASE - direct)
├── industry_threshold = 10%     (CONFIG - direct)
└── acme_customer_ltv = ???      (NOT DIRECT - needs sub-proof)
         │
         ▼
    ┌─────────────────────────────────────────┐
    │  Sub-Proof for customer_ltv             │
    │                                         │
    │  Assumed Facts (parallel):              │
    │  ├── avg_order_value = $150  (DATABASE) │
    │  ├── purchase_frequency = 4  (DATABASE) │
    │  └── retention_years = 3     (LLM)      │
    │                                         │
    │  Computation:                           │
    │  ltv = 150 * 4 * 3 = $1,800             │
    └─────────────────────────────────────────┘
         │
         ▼
    acme_customer_ltv = $1,800 (DERIVED via sub-proof)
```

### Recursive Structure

```
resolve(fact)
    │
    ├── Try CACHE → return if hit
    ├── Try DATABASE → return if found
    ├── Try API → return if found
    ├── Try LLM_KNOWLEDGE → return if answered
    │
    └── Try SUB_PROOF (recursive):
            │
            ├── LLM generates sub-proof structure
            │   └── Returns: [sub_facts], computation
            │
            ├── resolve_all(sub_facts)  ← RECURSIVE
            │   └── Each sub_fact may itself need sub-proofs
            │
            ├── Execute computation with resolved sub_facts
            │
            └── Return derived fact with provenance chain
```

### Parallelization with Recursion

The key insight remains: **at each level of recursion, the assumed facts are independent**.

```
Level 0 (Top-Level Proof):
┌─────────────────────────────────────────────────────────┐
│  Parallel: revenue, growth_rate, threshold              │
│  Sequential: customer_ltv (needs sub-proof)             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
Level 1 (Sub-Proof for customer_ltv):
┌─────────────────────────────────────────────────────────┐
│  Parallel: avg_order_value, purchase_frequency,         │
│            retention_years                              │
│  Then: compute ltv                                      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
Back to Level 0:
┌─────────────────────────────────────────────────────────┐
│  Now customer_ltv is resolved                           │
│  Execute top-level computation                          │
└─────────────────────────────────────────────────────────┘
```

### Implementation for Recursive Resolution

```python
class FactResolver:
    async def resolve_async(self, fact_name: str, **params) -> Fact:
        """Resolve a fact, potentially recursively via sub-proofs."""

        # Check cache
        cache_key = self._cache_key(fact_name, params)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try direct sources (parallel-safe)
        for source in [FactSource.DATABASE, FactSource.CONFIG, FactSource.LLM_KNOWLEDGE]:
            fact = await self._try_source_async(source, fact_name, params)
            if fact and fact.is_resolved:
                self._cache[cache_key] = fact
                return fact

        # Need sub-proof (recursive)
        return await self._resolve_via_sub_proof_async(fact_name, params)

    async def _resolve_via_sub_proof_async(self, fact_name: str, params: dict) -> Fact:
        """Generate and execute a sub-proof to derive a fact."""

        # 1. LLM generates sub-proof structure
        sub_proof = await self._generate_sub_proof_async(fact_name, params)
        # sub_proof = {
        #     "assumed_facts": [("avg_order_value", {}), ("frequency", {}), ...],
        #     "computation": "ltv = avg * freq * retention"
        # }

        # 2. Resolve all sub-facts in parallel (RECURSIVE)
        sub_facts = await self.resolve_all(sub_proof["assumed_facts"])

        # 3. Check for unresolved sub-facts
        unresolved = [f for f in sub_facts if not f.is_resolved]
        if unresolved:
            return Fact(
                name=fact_name,
                value=None,
                source=FactSource.UNRESOLVED,
                reasoning=f"Could not resolve dependencies: {[f.name for f in unresolved]}",
            )

        # 4. Execute computation
        result = self._execute_computation(sub_proof["computation"], sub_facts)

        # 5. Return derived fact with full provenance
        return Fact(
            name=fact_name,
            value=result,
            source=FactSource.SUB_PLAN,
            because=sub_facts,  # Provenance chain
            reasoning=f"Derived via: {sub_proof['computation']}",
        )
```

### Depth Limiting

To prevent infinite recursion:

```python
async def resolve_async(self, fact_name: str, _depth: int = 0, **params) -> Fact:
    if _depth > self.max_sub_proof_depth:  # Default: 3
        return Fact(
            name=fact_name,
            value=None,
            source=FactSource.UNRESOLVED,
            reasoning=f"Max sub-proof depth ({self.max_sub_proof_depth}) exceeded",
        )

    # ... resolution logic ...

    # Pass incremented depth to recursive calls
    sub_facts = await self.resolve_all(
        sub_proof["assumed_facts"],
        _depth=_depth + 1,
    )
```

### Provenance Chain for Recursive Proofs

The `because` field creates a tree of provenance:

```python
Fact(
    name="is_good_target",
    value=True,
    source=FactSource.SUB_PLAN,
    because=[
        Fact(name="acme_revenue", value=50_000_000, source=FactSource.DATABASE),
        Fact(name="acme_growth_rate", value=0.15, source=FactSource.DATABASE),
        Fact(
            name="acme_customer_ltv",
            value=1800,
            source=FactSource.SUB_PLAN,
            because=[
                Fact(name="avg_order_value", value=150, source=FactSource.DATABASE),
                Fact(name="purchase_frequency", value=4, source=FactSource.DATABASE),
                Fact(name="retention_years", value=3, source=FactSource.LLM_KNOWLEDGE),
            ],
            reasoning="ltv = avg_order_value * purchase_frequency * retention_years",
        ),
    ],
    reasoning="growth_rate > industry_threshold",
)
```

### Audit Trail Visualization

```
is_good_target = True
├── BECAUSE acme_revenue = $50M (DATABASE: SELECT revenue FROM companies WHERE name='Acme')
├── BECAUSE acme_growth_rate = 15% (DATABASE: SELECT growth FROM metrics WHERE company_id=123)
├── BECAUSE acme_customer_ltv = $1,800 (SUB_PLAN)
│   ├── BECAUSE avg_order_value = $150 (DATABASE: SELECT AVG(total) FROM orders WHERE...)
│   ├── BECAUSE purchase_frequency = 4 (DATABASE: SELECT COUNT(*)/years FROM orders WHERE...)
│   └── BECAUSE retention_years = 3 (LLM_KNOWLEDGE: "Typical SaaS retention is 3 years")
│   └── COMPUTATION: ltv = 150 * 4 * 3 = $1,800
└── COMPUTATION: 15% > 10% threshold = True
```

## Open Questions

1. **Should we also parallelize the other providers?** (OpenAI, Gemini, etc.)
   - Recommendation: Yes, but lower priority

2. **What are the actual Anthropic rate limits for the user's tier?**
   - Needed to set sensible `max_concurrent` default

3. **Should partial results be returned on fact resolution failures?**
   - Current plan: Yes, with unresolved facts marked

4. **How deep should sub-proofs be allowed to recurse?**
   - Current plan: Default max depth of 3

## Success Criteria

1. **Correctness**: Same results as sequential resolution
2. **Performance**: 3-5x speedup for 5+ LLM-resolved facts
3. **Rate Limiting**: No 429 errors under normal load
4. **Backward Compatible**: Sync API unchanged
5. **Auditable**: Full provenance chain preserved
