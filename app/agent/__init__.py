"""
Agent layer (see master spec sections 2 and 9).

Phase 0 isolates the *existing* keyword-matching command router behind a
proper interface (`LegacyRuleBasedOrchestrator`) so the rest of the
system (API layer, tests) depends on an `AgentOrchestrator` abstraction
rather than on ad-hoc string matching scattered across the codebase.

Phase 9 (REAL AGENT ORCHESTRATOR) adds `planner_orchestrator.PlannerOrchestrator`:
genuine intent understanding, tool discovery/selection, and bounded
multi-step planning against the tool registry built out in Phases 3-7,
driven by the provider-independent `LLMProvider` from Phase 8. It is now
the primary orchestrator (wired in `app.create_app`).
`LegacyRuleBasedOrchestrator` is no longer the primary path, but it is
kept and wired in as `PlannerOrchestrator`'s own fallback for sessions
where no LLM provider is configured/reachable (master spec section 15 -
free-first: the agent must still handle basic commands with zero LLM
configured, not fail outright).

Per-session working memory (`state.AgentState`/`AgentStateStore`) and
per-request context (`context.AgentRequestContext`) support the planner's
reference resolution ("play the third result", "reply to that",
"send it") and confirmation gate for sensitive tools.
"""
