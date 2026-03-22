---
name: async-patterns
description: Asyncio conventions and patterns used in this project. Auto-triggers when working with async code.
---

# Async Patterns

## Async Fact Resolution
- Located in `constat/execution/fact_resolver/_async.py`
- Pattern: `AsyncFactResolver(FactResolver)` extends base with async methods
- `ThreadPoolExecutor` for parallel I/O (facts, DB queries)
- `RateLimiter` class for API rate limiting

## FastAPI Routes
- Async handlers in `constat/server/routes/`
- WebSocket support in `constat/server/app.py`
- Use `async def` for route handlers that do I/O

## Sync/Async Boundary
- `ThreadPoolExecutor` bridges sync code into async context
- Never block the event loop with sync I/O
- Use `asyncio.to_thread()` or executor for CPU-bound work
- `run_in_executor()` for sync DB calls from async handlers

## Patterns
```python
# Parallel async execution
async def resolve_facts(facts: list[Fact]) -> list[Result]:
    tasks = [resolve_one(f) for f in facts]
    return await asyncio.gather(*tasks)

# Rate-limited API calls
async with rate_limiter:
    result = await api_call()

# Sync → async bridge
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(executor, sync_function, arg)
```

## Anti-Patterns
- `time.sleep()` in async code (use `asyncio.sleep()`)
- Blocking DB calls without executor
- Creating new event loops inside async functions
- Fire-and-forget tasks without error handling