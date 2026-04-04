CRITICAL: Never add fallback values or silent error handling. This has caused repeated production issues.
CRITICAL: We are currently in version 1 development. Never add migrations.
CRITICAL: Maximum brevity. No pleasantries. No explanations unless asked. Code and facts only.
CRITICAL: Files must stay under 1000 lines. If a file approaches or exceeds this, split it by separation of concerns. This applies to all languages (Python, TypeScript, etc.).
CRITICAL: "Audit" for UI features must include browser rendering and functionality testing (vitest + Playwright), not just code review.
CRITICAL: When the user asks to "audit" code, always spawn the `code-reviewer` agent (subagent_type: code-reviewer). Do not perform audits inline.

# Requirements Tracking
When the user states a new requirement, constraint, feature request, or design decision, spawn a general-purpose agent (model: haiku) in the background to append it to `docs/arch/requirements.md`. The agent should first read `.claude/agents/requirements-tracker.md` for format rules, then read the current requirements file, then append the new requirement(s). Do this silently — no confirmation needed. Do NOT spawn for implementation details, bug reports, or questions.

# Architecture
@.claude/refs/architecture.ref

# Verification Commands
- Backend tests: `python -m pytest tests/ -x -q`
- Frontend build: `cd constat-ui && npm run build`
- Frontend lint: `cd constat-ui && npm run lint`
- Type check: `cd constat-ui && npx tsc --noEmit`
- Frontend tests: `cd constat-ui && npm test`
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
