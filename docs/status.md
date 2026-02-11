# Project Status

**Last Updated:** 2026-02-11

## Quick Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Core Types | Done | AgentId, AgentIO, Edge, Topology, TraceEvent |
| Hash Embedder | Done | Deterministic baseline |
| Semantic Embedder | Done | OllamaEmbedder with nomic-embed-text |
| Router | Done | Top-K sparsification, force-connect, baselines |
| Stub Agents | Done | Domain-tagged templates |
| LLM Agents | Done | OllamaPool + LlmWorker with retry logic |
| Orchestrator | Done | Full round loop with traces |
| DOT Export | Done | GraphViz visualization |
| Trace Analysis | Done | Metrics computation, SVG plots |
| CLI | Done | `demo`, `analyze`, `benchmark` commands |
| JSONL Traces | Done | Valid format, all events |

## Milestone Progress

### Milestone 0 - Working Skeleton

**Status: COMPLETE**

- [x] `cargo build` compiles all crates
- [x] `cargo test` passes
- [x] `cargo clippy` clean
- [x] `cargo run -p dytopo-cli -- demo` runs
- [x] JSONL trace written to `./traces/`
- [x] DOT files generated per round

### Milestone 1 - Semantic Embeddings

**Status: COMPLETE**

- [x] OllamaEmbedder using /api/embeddings endpoint
- [x] Embedding cache to avoid redundant API calls
- [x] OllamaEmbedderPool for multi-host embedding
- [x] `--embedder hash|ollama` CLI flag
- [x] `--embed-model` CLI option (default: nomic-embed-text)
- [x] `--embed-url` CLI option
- [x] Semantic routing demo verified (0.6-0.97 similarity scores)

### Milestone 2 - LLM Agents

**Status: COMPLETE**

- [x] OllamaClient implementation with blocking HTTP
- [x] OllamaPool with multi-host load balancing
- [x] LlmWorker agent with prompt construction
- [x] `--llm ollama` CLI flag
- [x] JSON extraction from markdown/embedded responses
- [x] Retry with repair prompt on parse failure
- [x] Fallback to stub behavior on error
- [x] Stagger delay to avoid GPU overload
- [x] Host capacity tracking

### Milestone 3 - Message Caps

**Status: PARTIAL**

- [x] `max_inbox` config exists
- [x] Oldest messages dropped when cap exceeded
- [ ] `--summarize-inbox` flag
- [ ] Recency vs relevance option

### Milestone 4 - Evaluation

**Status: COMPLETE**

- [x] Baseline topologies (fully-connected, chain, star, ring)
- [x] `analyze` command with metrics and plots
- [x] `benchmark` command for comparative runs
- [x] Comparison report generation
- [x] SVG plot generation (scores, timing, density)

---

## Crate Status

### dytopo-core (v0.1.0)
- **Lines:** ~70
- **Tests:** 0 (relies on integration tests)
- **Status:** Stable, minimal API

### dytopo-embed (v0.1.0)
- **Lines:** ~350
- **Tests:** 3 (hash embedder tests)
- **Status:** HashEmbedder + OllamaEmbedder both working

### dytopo-router (v0.1.0)
- **Lines:** ~85
- **Tests:** 1 (builds_edges_and_respects_topk)
- **Status:** Stable

### dytopo-llm (v0.1.0)
- **Lines:** ~380
- **Tests:** 3 (JSON extraction)
- **Status:** Fully functional with OllamaPool

### dytopo-agents (v0.1.0)
- **Lines:** ~335
- **Tests:** 0 (tested via orchestrator)
- **Status:** StubWorker + LlmWorker both working

### dytopo-orchestrator (v0.1.0)
- **Lines:** ~155
- **Tests:** 0 (CLI is the test)
- **Status:** Core loop working

### dytopo-viz (v0.1.0)
- **Lines:** ~22
- **Tests:** 0
- **Status:** DOT export working

### dytopo-analyze (v0.1.0)
- **Lines:** ~350
- **Tests:** 1
- **Status:** Trace analysis, metrics, SVG plots

### dytopo-cli (v0.1.0)
- **Lines:** ~490
- **Tests:** 0
- **Status:** `demo`, `analyze`, `benchmark` commands

---

## Demo Commands

### Stub Demo (Hash Embeddings, No LLM)

```bash
cargo run -p dytopo-cli -- demo --rounds 3 --agents 5
```

### Semantic Embeddings Only (No LLM)

```bash
cargo run -p dytopo-cli -- demo \
  --embedder ollama \
  --embed-url http://localhost:11434 \
  --agents 5 --rounds 2
```

### Full Demo (LLM + Semantic Embeddings)

```bash
cargo run -p dytopo-cli -- demo \
  --llm ollama \
  --model "qwen2.5:7b-instruct" \
  --ollama-hosts "manager=http://manager.local:11434:2,curiosity=http://curiosity:11434:3" \
  --embedder ollama \
  --embed-model "nomic-embed-text" \
  --embed-url "http://manager.local:11434" \
  --agents 5 \
  --rounds 3 \
  --task "Write a Python function to calculate Fibonacci numbers"
```

### Analyze a Trace

```bash
cargo run -p dytopo-cli -- analyze --trace traces/trace_demo_*.jsonl
```

Options:
- `--format json|csv|text` - Output format
- `--no-plots` - Skip SVG plot generation

### Run Benchmark (Dynamic vs Baselines)

```bash
cargo run -p dytopo-cli -- benchmark \
  --task "Write a Python function to sort a list" \
  --agents 5 \
  --rounds 3 \
  --embed-url http://localhost:11434
```

---

## Embedding Comparison

| Embedder | Typical Score Range | Semantic? | Speed |
|----------|---------------------|-----------|-------|
| Hash | 0.1 - 0.5 | No (bag-of-words) | Very fast |
| Ollama/nomic | 0.5 - 0.97 | Yes | ~50ms/embed |

With semantic embeddings:
- Agent offering "matrix exponentiation" → Agent needing "efficient algorithm" = **0.97**
- True semantic matching, not just keyword overlap

---

## Next Actions

1. **Verify process:** `cargo clippy && cargo test`
2. **Run full demo:** See commands above
3. **Run benchmark:** `cargo run -p dytopo-cli -- benchmark --embed-url http://localhost:11434`
4. **View plots:** Open `benchmark/scores.svg`, `benchmark/timing.svg`, `benchmark/density.svg`
5. **Next milestone:** Message summarization (M3)

---

## Change Log

### 2026-02-11 (Evening)
- **Milestone 4 Complete:**
  - Created dytopo-analyze crate for trace analysis
  - Added `analyze` CLI command with metrics and SVG plots
  - Added `benchmark` CLI command for comparison runs
  - Added baseline topologies (fully-connected, star, chain, ring)
  - Generates markdown benchmark reports
  - SVG plots: scores, timing, density over rounds

### 2026-02-11 (PM - Late)
- **Milestone 1 Complete:**
  - Added OllamaEmbedder using /api/embeddings
  - Implemented embedding cache
  - Added `--embedder hash|ollama` CLI flag
  - Verified semantic routing (0.97 similarity for matching concepts)
  - Fixed clippy warnings

### 2026-02-11 (PM)
- **Milestone 2 Complete:**
  - Implemented OllamaClient with blocking HTTP
  - Added OllamaPool for multi-host load balancing
  - Created LlmWorker with prompt construction and JSON parsing
  - Added retry logic with repair prompts
  - Integrated into CLI with `--llm ollama` flag
  - Added stagger delay for GPU protection

### 2026-02-11 (AM)
- Initial project creation
- Milestone 0 complete
- Documentation added (architecture, prd, design, plan, status)
