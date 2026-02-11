# Instructions for an AI coding agent

You are working in a Rust workspace. Keep changes small, compile frequently, and preserve existing behavior unless explicitly changing it.

## Ground rules

1. **Run `cargo test` and `cargo fmt`** before proposing a change.
2. Prefer **adding tests** or a reproducible demo over “just code”.
3. Keep crates decoupled: do not import `cli` from library crates.
4. Avoid long prompts or giant config files. Trace format should remain stable.

## Repository intent

We are building a DyTopo-inspired multi-agent system:
- Each round:
  - Manager sets a `RoundGoal`
  - Each agent returns `(query, key, draft)`
  - Router builds a sparse directed graph from key->query similarity
  - Orchestrator delivers messages along edges and logs a trace
- The **graph is the product**: it must be inspectable and logged.

## What is “done”?

A task is done when:
- `cargo test` passes,
- The CLI demo works,
- Trace JSONL includes the expected fields.

## Safe iteration workflow

1. Pick a milestone from `docs/PLAN.md`.
2. Implement it behind a CLI flag if it’s risky.
3. Add/adjust tests in the relevant crate.
4. Update docs if behavior changed.

## Coding style

- Use `anyhow::Result` in binaries; use typed errors in libraries only if needed.
- Favor explicit structs over tuples for anything crossing crate boundaries.
- No unsafe code.

## Common pitfalls

- Context explosion: keep message payloads small; implement caps/top-K.
- Non-determinism: prefer deterministic routing for reproducible demos unless otherwise requested.
