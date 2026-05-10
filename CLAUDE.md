CRITICAL: Never add fallback values or silent error handling. This has caused repeated production issues.
CRITICAL: We are currently in version 1 development. Never add migrations.
CRITICAL: Maximum brevity. No pleasantries. No explanations unless asked. Code and facts only.
CRITICAL: Files must stay under 1000 lines. If a file approaches or exceeds this, split it by separation of concerns. This applies to all languages (Python, TypeScript, etc.).
CRITICAL: "Audit" for UI features must include browser rendering and functionality testing (vitest + Playwright), not just code review.
CRITICAL: When the user asks to "audit" code, always spawn the `code-reviewer` agent (subagent_type: code-reviewer). Do not perform audits inline.
CRITICAL: All test errors must be resolved, whether they are new failures or preexisting conditions. Never skip or ignore a failing test.
CRITICAL: Tests that pass individually but fail in the full suite (or vice versa) indicate improper test isolation and must be fixed. Every test must pass both in isolation and as part of the full suite.
CRITICAL: Tests involving LLM responses must include a retry loop (up to 3 attempts) to account for probabilistic output. If a test still fails after retries, revise the LLM instructions/prompts to improve result reliability. Assertions may be loosened only if they retain probative value — e.g., accepting multiple valid phrasings is fine, but asserting "returned a string" is not meaningful.
CRITICAL: `pytest.skip()` is forbidden in all forms — infrastructure unavailable, missing environment variable, missing API key, missing Docker. A skipped test is a silent lie about coverage. If infrastructure is needed, start it via Docker. If a service is absent, call `pytest.fail()` so the gap is visible.
CRITICAL: Mocks in unit tests are valid and useful for fast CI/CD runs, but a mocked unit test does NOT replace an integration test. Any component that talks to a real service (DB, queue, API) must have both: a unit test with mocks AND an integration test against the real service.
CRITICAL: Test placement is mandatory. See `.claude/skills/pytest-patterns/SKILL.md` for tier definitions. Wrong-tier tests create false confidence: a unit test that hits the network is not a unit test.

# Test Tiers
- `tests/unit/` — pure logic, zero I/O, runs in milliseconds, no fixtures beyond in-memory objects
- `tests/integration/` — real DB + services (started via Docker in fixtures), no HTTP server required
- `tests/e2e/` — full HTTP round-trips through the running app (Playwright + live backend + Vite)

# Requirements Tracking
When the user states a new requirement, constraint, feature request, or design decision, spawn a general-purpose agent (model: haiku) in the background to append it to `docs/arch/requirements.md`. The agent should first read `.claude/agents/requirements-tracker.md` for format rules, then read the current requirements file, then append the new requirement(s). Do this silently — no confirmation needed. Do NOT spawn for implementation details, bug reports, or questions.

# Architecture
@.claude/refs/architecture.ref

# Verification Commands
- Unit tests: `python -m pytest tests/unit/ -x -q` (fast, no services needed)
- Integration tests: `python -m pytest tests/integration/ -x -q` (requires Docker)
- E2E tests: `python -m pytest tests/e2e/ -x -q` (requires running backend + Vite)
- All backend tests: `python -m pytest tests/ -x -q`
- Frontend build: `cd constat-ui && npm run build`
- Frontend lint: `cd constat-ui && npm run lint`
- Type check: `cd constat-ui && npx tsc --noEmit`
- Frontend unit tests: `cd constat-ui && npm test`
- Server: `python -m constat.server -c demo/config.yaml`
- Demo config: `demo/config.yaml` (domains: sales-analytics, hr-reporting)
- Server logs: `.logs/server.log`, `.logs/ui.log`
- Session data: `.constat/{user-id}/sessions/{timestamped-dir}/`

# Module Boundaries
@.claude/refs/boundaries.ref

# Swarm Mode
@.claude/refs/swarm-mode.ref

# Agent Isolation
@.claude/refs/agent-isolation.ref

# Token Management
- CRITICAL: Run /compact after every 5 tool calls or 10 messages.
- CRITICAL: Use /context every 15 minutes to verify 'Messages' < 40k tokens.
- If 'Messages' > 50k, stop work and run /compact immediately.
