---
name: clarification-harvest
description: Use when Kimi needs to gather requirements before implementing a task that has ambiguity, multiple valid approaches, or unclear scope. This skill enforces a strict three-phase workflow: (1) Harvest ALL clarifying questions in a single comprehensive batch with defaults, (2) Lock in answers and confirm assumptions, (3) Execute autonomously to completion with zero further questions. Triggers for: new features, architectural decisions, integration work, deployment setup, API design, database schemas, or any task where upfront clarity prevents rework.
---

# Clarification Harvest Protocol

Harvest all requirements upfront, then execute autonomously without interruptions.

## Phase 0 — Understand

Parse the request and identify:
- Deliverables (what will exist after completion)
- Constraints (technical, temporal, organizational)
- Success criteria (how to verify correctness)
- Ambiguity gaps (what could change the approach)

## Phase 1 — Question Harvest (SINGLE MESSAGE)

Produce a comprehensive list of clarifying questions. Do not implement anything yet.

### Question Categories (check all that apply)

| Category | Check | Description |
|----------|-------|-------------|
| Correctness | ☐ | Data validation, error handling, edge cases |
| Scope | ☐ | What's in/out, MVP vs full feature |
| UX/API Design | ☐ | User flows, interface contracts, patterns |
| Integration | ☐ | External systems, dependencies, APIs |
| Performance | ☐ | Latency, throughput, resource limits |
| Security | ☐ | Auth, encryption, secrets management |
| Deployment | ☐ | Environment, CI/CD, infrastructure |
| Testing | ☐ | Coverage requirements, test types, acceptance criteria |

### Question Format

Each question must follow this exact format:

```
- Q{N} ({P0/P1/P2}) [Category: {Name}]
  Question: {Clear, specific question}
  Options: {A / B / C} (if applicable)
  Default if unanswered: {Specific default value}
```

**Priority Levels:**
- **P0**: Blocks correctness or requires external access (API keys, credentials)
- **P1**: Major design choice affecting architecture
- **P2**: Polish, optimization, or nice-to-have

### Output Template

```
Reply with answers to any subset; I'll use defaults for the rest.

## Clarifying Questions

### Critical (P0)
- Q1 (P0) [Category: Integration]
  Question: Do you have API credentials for {service}?
  Options: Yes (provide below) / No / Not needed
  Default if unanswered: No — will implement with env var placeholders

### Major Design (P1)
- Q2 (P1) [Category: Scope]
  Question: Should this be MVP or production-ready?
  Options: MVP (core only) / Production (full error handling, tests, docs)
  Default if unanswered: Production

### Polish (P2)
- Q3 (P2) [Category: UX]
  Question: Preferred output format?
  Options: JSON / YAML / Table / Custom
  Default if unanswered: JSON
```

## Phase 2 — Lock-In

After user responds:

1. **Apply defaults** to all unanswered questions
2. **Confirm assumptions** in a short bullet list:
   ```
   ## Locked-In Assumptions
   
   - Using: {framework/approach from Q2}
   - Environment: {local/cloud from Q5}
   - Auth: {OAuth/API key from Q7} → using env var: {NAME}
   - Scope: {MVP/full from Q8}
   ```
3. **Hard rule**: After this message, zero additional questions

## Phase 3 — Autonomous Execution

Execute end-to-end without interruptions:

### A) Repo Inventory
- Read README, manifests, configs
- Identify conventions and patterns
- Find entry points and interfaces

### B) Plan
- Write a short plan (2-3 bullets max)
- Execute immediately

### C) Implement
- Complete changes across all files
- No TODOs unless user explicitly requested scaffolding
- If blocked by missing access: implement with clear placeholders + docs

### D) Verify
- Run available checks (lint, typecheck, tests, build)
- Provide exact verification commands if not runnable

### E) Report

```
## ✅ Complete

### What Changed
{Summary}

### Files Modified
- `path/to/file.py` — {purpose}

### How to Verify
```bash
{commands}
```

### Locked Assumptions + Defaults Used
| Question | Answer Used |
|----------|-------------|
| Q2 (framework) | FastAPI (default) |
| Q5 (environment) | Local (default) |

### ⚠️ Tradeoffs/Risks
- {Risk and mitigation}
```

## Reference Templates

For domain-specific question templates, see:
- [references/web-api-questions.md](references/web-api-questions.md)
- [references/database-questions.md](references/database-questions.md)
- [references/deployment-questions.md](references/deployment-questions.md)

Load these when the user's task matches the domain.

## Critical Rules

1. **Never self-censor** — include all plausible questions, even if long
2. **Always provide defaults** — every question must have a fallback
3. **No mid-flight questions** — after lock-in, make decisions autonomously
4. **Missing access = placeholders** — don't ask for secrets, implement env var patterns
5. **Repo conventions win** — if evidence contradicts a default, follow the repo
