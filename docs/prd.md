# Product Requirements Document (PRD)

## Product Vision

dytopo-rs implements DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning, a framework that dynamically reconstructs communication patterns between AI agents based on semantic matching of needs and offerings.

## Problem Statement

Existing multi-agent LLM systems suffer from:

1. **Fixed Communication Patterns** - Static topologies (all-to-all, chain, star) don't adapt to evolving problem states
2. **Context Explosion** - Fully-connected agents overwhelm context windows with irrelevant messages
3. **Opaque Coordination** - Difficult to interpret why agents communicated

## Solution

DyTopo creates sparse, dynamic communication graphs:
- Each round, agents emit **Query** (what I need) and **Key** (what I offer)
- Semantic matching determines who should talk to whom
- The evolving topology becomes an interpretable trace of reasoning

## Target Users

1. **Researchers** - Experimenting with multi-agent coordination strategies
2. **Developers** - Building agentic systems with controllable communication
3. **AI Engineers** - Comparing DyTopo against baseline topologies

## Core Requirements

### P0 - Must Have (Milestone 0-1)

#### R1: Stub Demo (Milestone 0 - Complete)
- CLI runs end-to-end with deterministic agents
- Hash-based embeddings produce consistent results
- JSONL trace captures all events
- DOT files visualize topology

#### R2: Configurable Routing
- Top-K incoming edges per receiver
- Minimum similarity threshold
- Force-connect option for isolated agents

#### R3: Trace Format
- Round start/end with timestamps
- Agent IO (query, key, draft)
- Topology edges with scores
- Message delivery events

### P1 - Should Have (Milestone 2-3)

#### R4: Real Embeddings
- Feature flag for fastembed (ONNX)
- Feature flag for candle (pure Rust)
- CLI flag to select embedder

#### R5: LLM Agents
- Ollama integration
- Structured JSON output
- Retry with repair prompt on parse failure

#### R6: Message Caps
- Configurable inbox size per agent
- Optional summarization step

### P2 - Nice to Have (Milestone 4+)

#### R7: Evaluation Harness
- Baseline topologies (fully-connected, chain, star)
- Comparison metrics from traces
- Summary report generation

#### R8: Web Visualization
- Interactive topology viewer
- Timeline scrubbing
- Agent state inspection

## Success Metrics

### Demo Quality
- `cargo run -p dytopo-cli -- demo` completes without error
- Topology graphs show sensible routing (matching domains)
- Traces are valid JSONL

### Performance
- 5-agent, 3-round demo runs in < 1 second (hash embeddings)
- No memory leaks over 100 rounds

### Interpretability
- DOT files render correctly in GraphViz
- Edge labels show meaningful similarity scores

## Non-Goals

- Production-grade LLM integration (research prototype)
- Web UI for non-technical users
- Real-time streaming visualization
- Distributed multi-machine execution

## Technical Constraints

- Rust 2021 edition
- Zero Python dependencies for baseline
- JSONL trace format is stable API
- No unsafe code

## Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | Working skeleton with stub agents | Done |
| M1 | Real embedding backends (fastembed/candle) | Not started |
| M2 | LLM agent IO via Ollama | Not started |
| M3 | Message caps + summarization | Not started |
| M4 | Evaluation harness | Not started |

## Acceptance Criteria

### Milestone 0 (Complete)
- [x] `cargo run -p dytopo-cli -- demo` prints topology per round
- [x] JSONL trace written to `./traces/`
- [x] DOT files generated for each round
- [x] Tests pass

### Milestone 1
- [ ] `--embedder hash|fastembed|candle` flag works
- [ ] Routing differs meaningfully with semantic embeddings
- [ ] Feature flags compile cleanly when disabled

### Milestone 2
- [ ] `--llm ollama --model <name>` produces agent output
- [ ] Agents emit valid JSON: `{query, key, draft}`
- [ ] Parse failures trigger one retry with repair prompt

### Milestone 3
- [ ] `--max-inbox N` caps messages per agent
- [ ] Optional `--summarize-inbox` condenses messages

### Milestone 4
- [ ] `--baseline fully-connected|chain|star` for comparison
- [ ] `analyze-trace` command produces summary report
