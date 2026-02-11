# dytopo-rs (Rust-first DyTopo-style multi-agent routing)

This repository is a **Rust workspace** that implements a **DyTopo-inspired** dynamic communication topology for multi-agent systems.

**Goal:** each round, agents emit:
- **Query** (what I need)
- **Key** (what I offer)

A router embeds Query/Key text, computes semantic similarity, and builds a **sparse directed graph** deciding who talks to whom that round. The orchestrator runs message passing along those edges and logs an interpretable trace.

This repo starts with a **fully Rust, zero-Python** baseline:
- A lightweight embedder (deterministic, hash-based) to get a working system fast
- A similarity router (cosine + top-K sparsification)
- A CLI demo that prints routing graphs and writes traces

Then you can upgrade:
- Embedder: `fastembed`/ONNX or `candle` (Rust ML) (planned)
- Agents: plug in local LLM via Ollama or llama.cpp HTTP (scaffold included)

## Quick start

```bash
cargo run -p dytopo-cli -- demo --rounds 3 --agents 5 --topk 2
```

Outputs:
- stdout: per-round topology + messages
- `./traces/trace_*.jsonl`: machine-readable trace

## Docs for your coding agent

- **AGENTS.md** — how to iterate safely, in small PR-sized steps
- **docs/PLAN.md** — milestone plan + acceptance checks
- **docs/PROTOCOL.md** — message schemas and invariants
- **docs/ARCHITECTURE.md** — crate boundaries + design rationale
- **docs/RUN.md** — how to run demos, tests, and trace export

## License

Dual-licensed MIT OR Apache-2.0 (pick one).
