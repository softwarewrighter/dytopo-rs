# Implementation Plan

## Current Status: Milestone 0 Complete

The baseline skeleton is implemented and functional. The system runs end-to-end with stub agents and hash-based embeddings.

## Milestone Roadmap

### Milestone 0 - Working Skeleton [DONE]

**Deliverables:**
- [x] Core types (AgentId, AgentIO, Edge, Topology, TraceEvent)
- [x] HashEmbedder for deterministic routing
- [x] Router with top-K sparsification
- [x] StubWorker agents with domain templates
- [x] Orchestrator loop with message delivery
- [x] DOT export for visualization
- [x] CLI with `demo` command
- [x] JSONL trace output

**Verification:**
```bash
cargo test                                    # All tests pass
cargo run -p dytopo-cli -- demo --rounds 3    # Produces trace + DOT
```

---

### Milestone 1 - Real Embedding Backends [NOT STARTED]

**Goal:** Replace hash embeddings with semantic embeddings for meaningful routing.

**Tasks:**

1. **Add fastembed feature** (2-3 hours)
   - Add `fastembed` dependency with feature flag
   - Implement `FastEmbedder` struct
   - Wire CLI `--embedder fastembed`

2. **Add candle feature** (3-4 hours)
   - Add `candle-core` + `candle-transformers` dependencies
   - Implement `CandleEmbedder` struct
   - Wire CLI `--embedder candle`

3. **Comparison demo** (1 hour)
   - Run same task with hash vs semantic
   - Document topology differences

**Acceptance Criteria:**
- [ ] `--embedder hash|fastembed|candle` works
- [ ] With semantic embeddings, similar domains cluster
- [ ] Feature flags compile cleanly when disabled

**Dependencies:** None (can start immediately)

---

### Milestone 2 - LLM Agent IO [NOT STARTED]

**Goal:** Replace stub agents with LLM-backed agents.

**Tasks:**

1. **Implement OllamaClient** (2 hours)
   - POST to /api/generate
   - Parse response.response field
   - Add timeout handling

2. **Create LlmWorker agent** (3 hours)
   - Prompt construction with goal + inbox
   - JSON output parsing
   - Retry logic with repair prompt

3. **CLI integration** (1 hour)
   - `--llm ollama --model <name>` flags
   - Fallback to stub if Ollama unavailable

4. **Add structured output tests** (2 hours)
   - Valid JSON parsing
   - Character limit enforcement
   - Repair prompt effectiveness

**Acceptance Criteria:**
- [ ] Agents emit valid JSON: `{query, key, draft}`
- [ ] Parse failures trigger retry with repair prompt
- [ ] Graceful fallback when Ollama unavailable

**Dependencies:** Milestone 0 (complete)

---

### Milestone 3 - Message Caps + Summarization [NOT STARTED]

**Goal:** Prevent context explosion with inbox management.

**Tasks:**

1. **Configurable inbox cap** (1 hour)
   - Already partially implemented (max_inbox)
   - Add tests for cap behavior
   - Document in CLI help

2. **Summarize inbox step** (3-4 hours)
   - Optional LLM call to condense inbox
   - Trigger when inbox exceeds threshold
   - `--summarize-inbox` CLI flag

3. **Recency vs relevance tradeoff** (2 hours)
   - Option to keep newest vs highest-score messages
   - `--inbox-strategy recent|relevant` flag

**Acceptance Criteria:**
- [ ] `--max-inbox N` caps messages
- [ ] `--summarize-inbox` produces condensed context
- [ ] No message loss without explicit configuration

**Dependencies:** Milestone 2 (for LLM summarization)

---

### Milestone 4 - Evaluation Harness [NOT STARTED]

**Goal:** Compare DyTopo against baseline topologies.

**Tasks:**

1. **Baseline topology implementations** (2 hours)
   - `FullyConnectedRouter` - all-to-all
   - `ChainRouter` - sequential
   - `StarRouter` - hub-and-spoke

2. **Trace analysis tool** (3 hours)
   - `analyze-trace` CLI command
   - Metrics: message count, unique edges, convergence
   - Comparison across topologies

3. **Benchmark task suite** (2-3 hours)
   - Define toy tasks with known solutions
   - Math problem, code generation, planning
   - Ground truth for evaluation

4. **Summary report** (2 hours)
   - Markdown or HTML output
   - Topology visualization comparisons
   - Performance metrics table

**Acceptance Criteria:**
- [ ] `--baseline fully-connected|chain|star` works
- [ ] `analyze-trace` produces meaningful metrics
- [ ] Documentation of findings

**Dependencies:** Milestone 2 (for meaningful agent output)

---

## Demo Development Plan

### Demo 1: Hash Routing (Available Now)

```bash
cargo run -p dytopo-cli -- demo --rounds 3 --agents 5 --topk 2
```

Shows: Basic topology construction and message routing.

### Demo 2: Semantic Routing (Milestone 1)

```bash
cargo run -p dytopo-cli -- demo --embedder fastembed --agents 5
```

Shows: Semantically meaningful routing decisions.

### Demo 3: LLM Agents (Milestone 2)

```bash
cargo run -p dytopo-cli -- demo --llm ollama --model llama2 --task "Write a sorting algorithm"
```

Shows: Real reasoning with dynamic coordination.

### Demo 4: Topology Comparison (Milestone 4)

```bash
# Run same task with different topologies
cargo run -p dytopo-cli -- demo --llm ollama --task "..." --out traces/dytopo
cargo run -p dytopo-cli -- demo --llm ollama --task "..." --baseline fully-connected --out traces/fc
cargo run -p dytopo-cli -- demo --llm ollama --task "..." --baseline chain --out traces/chain

# Compare results
cargo run -p dytopo-cli -- analyze-trace traces/
```

Shows: DyTopo advantages over static topologies.

---

## Risk Mitigation

### Risk: Embedding model download issues
**Mitigation:** Hash embedder always available as fallback

### Risk: Ollama not running
**Mitigation:** Clear error message, fallback to stub agents

### Risk: LLM JSON output unreliable
**Mitigation:** Repair prompt + strict validation + fallback defaults

### Risk: Context explosion with many agents
**Mitigation:** Top-K sparsification + inbox caps

---

## Next Steps (Recommended Order)

1. **Immediate:** Verify M0 works with `cargo test && cargo run -p dytopo-cli -- demo`

2. **Short-term (M2):** Implement LLM agent support
   - Most impactful for demonstrable results
   - Shows real reasoning, not just routing

3. **Medium-term (M1):** Add semantic embeddings
   - Makes routing decisions meaningful
   - Can be done in parallel with M2

4. **Later (M3, M4):** Polish and evaluation
   - Message caps for robustness
   - Evaluation harness for claims
