# Project Status

**Last Updated:** 2026-02-11

## Quick Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Core Types | Done | AgentId, AgentIO, Edge, Topology, TraceEvent |
| Hash Embedder | Done | Deterministic baseline |
| Router | Done | Top-K sparsification, force-connect |
| Stub Agents | Done | Domain-tagged templates |
| LLM Agents | Done | OllamaPool + LlmWorker with retry logic |
| Orchestrator | Done | Full round loop with traces |
| DOT Export | Done | GraphViz visualization |
| CLI | Done | `demo` command with LLM support |
| JSONL Traces | Done | Valid format, all events |

## Milestone Progress

### Milestone 0 - Working Skeleton

**Status: COMPLETE**

- [x] `cargo build` compiles all crates
- [x] `cargo test` passes
- [x] `cargo run -p dytopo-cli -- demo` runs
- [x] JSONL trace written to `./traces/`
- [x] DOT files generated per round

### Milestone 1 - Real Embeddings

**Status: NOT STARTED**

- [ ] fastembed feature flag
- [ ] candle feature flag
- [ ] `--embedder` CLI flag
- [ ] Semantic routing demo

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

**Status: NOT STARTED**

- [ ] Baseline topologies
- [ ] `analyze-trace` command
- [ ] Benchmark task suite
- [ ] Comparison report

---

## Crate Status

### dytopo-core (v0.1.0)
- **Lines:** ~70
- **Tests:** 0 (relies on integration tests)
- **Status:** Stable, minimal API

### dytopo-embed (v0.1.0)
- **Lines:** ~60
- **Tests:** 0 (tested via router)
- **Status:** Stable, extensible via trait

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

### dytopo-cli (v0.1.0)
- **Lines:** ~220
- **Tests:** 0
- **Status:** Demo command with LLM support working

---

## Demo Commands

### Stub Demo (No LLM)

```bash
cargo run -p dytopo-cli -- demo --rounds 3 --agents 5
```

### LLM Demo with Ollama

```bash
# Using two hosts: manager (2 concurrent) and curiosity (3 concurrent)
cargo run -p dytopo-cli -- demo \
  --llm ollama \
  --model llama2 \
  --ollama-hosts "manager=http://manager.local:11434:2,curiosity=http://curiosity.local:11434:3" \
  --agents 5 \
  --rounds 3 \
  --stagger-ms 500 \
  --task "Write a function to sort a list of integers"
```

### Single Host

```bash
cargo run -p dytopo-cli -- demo \
  --llm ollama \
  --model mistral \
  --ollama-hosts "local=http://localhost:11434:2" \
  --agents 3 \
  --rounds 2
```

---

## Host Configuration

The `--ollama-hosts` format is: `name=url:concurrent`

- **name**: Human-readable identifier for logs
- **url**: Full Ollama API URL (including port)
- **concurrent**: Max simultaneous requests to this host

Example for your setup:
- manager: 2 concurrent (GPU can handle 2 models loaded)
- curiosity: 3 concurrent (larger GPU)

The pool automatically:
1. Selects the host with most available capacity
2. Staggers requests by `--stagger-ms` to avoid GPU spikes
3. Falls back to stub output on connection errors

---

## Next Actions

1. **Test with Ollama:** Ensure `ollama serve` is running on both hosts
2. **Run LLM demo:** Use command above with your model
3. **Check traces:** `cat traces/trace_demo_*.jsonl | jq .`
4. **Visualize:** `dot -Tsvg traces/*.dot -o topology.svg`

---

## Change Log

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
