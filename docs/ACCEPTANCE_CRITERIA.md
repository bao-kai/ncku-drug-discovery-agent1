# Agent 1 Acceptance Criteria

## Level 1 — Structural

- Objects pass versioned schemas.
- Entity IDs are tool-verified.
- Operations are approved or blocked.
- The plan is acyclic and dependencies reference prior step IDs.
- Filters, metrics, aggregation, safeguards, and outputs match contracts.
- Facts, derivations, and interpretations remain separate.

## Level 2 — Execution

- Every approved operation maps to an adapter.
- Every source records parameters, time, pagination, limits, truncation, validation, and errors.
- Retries and timeouts are bounded.
- Partial failure preserves successful results.
- Reruns add snapshots without altering prior runs.
- Stage 1 references the records it summarizes.
- Plan-only acceptance performs zero retrieval.

## Level 3 — Scientific review

- Every claim is traceable to source evidence.
- Missing evidence remains unknown unless a bounded negative claim is justified.
- Association scores are not presented as causality, efficacy, or success rates.
- Conflicts and low coverage are surfaced.
- Scientific claims use team-approved sources.
- Results answer the original question within declared scope.

## Release evidence

A release records the commit, dependency versions, model digest, schema versions, test count, representative runtime, known limitations, and unresolved team decisions.