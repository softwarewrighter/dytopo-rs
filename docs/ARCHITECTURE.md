# Architecture

## Overview

dytopo-rs is a Rust implementation of the DyTopo (Dynamic Topology) framework for multi-agent reasoning. The system dynamically reconstructs communication topology at each reasoning round based on semantic matching between agent needs (queries) and offerings (keys).

## System Architecture

```
+------------------+     +------------------+     +------------------+
|     Manager      |     |     Agents       |     |    Embedder      |
|   (RoundGoal)    |     | (Query/Key/Draft)|     |  (Hash/Semantic) |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+------------------------------------------------------------------------+
|                         Orchestrator                                    |
|  - Collects AgentIO per round                                          |
|  - Builds topology via Router                                          |
|  - Delivers messages along edges                                       |
|  - Writes trace events                                                 |
+------------------------------------------------------------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|     Router       |     |      Viz         |     |   Trace (JSONL)  |
| (Sparse Graph)   |     |  (DOT Export)    |     |                  |
+------------------+     +------------------+     +------------------+
```

## Crate Boundaries

### dytopo-core (crates/core)

**Purpose:** Shared types across all crates.

**Key Types:**
- `AgentId` - Unique agent identifier
- `RoundGoal` - Manager's objective for a round
- `AgentIO` - Agent output: query, key, draft
- `Edge` - Directed connection with similarity score
- `Topology` - Graph of edges for a round
- `Message` - Content delivered along an edge
- `TraceEvent` - JSONL log events

**Dependencies:** serde only (no crate dependencies)

### dytopo-embed (crates/embed)

**Purpose:** Text embedding for semantic similarity.

**Trait:**
```rust
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Embedding>;
}
```

**Implementations:**
- `HashEmbedder` - Deterministic, dependency-free baseline
- (Planned) `FastEmbedder` - ONNX-based semantic embeddings
- (Planned) `CandleEmbedder` - Pure Rust ML embeddings

### dytopo-router (crates/router)

**Purpose:** Build sparse directed graphs from agent keys/queries.

**Algorithm:**
1. Embed all keys and queries
2. Compute cosine similarity: `score(i -> j) = cos(key_i, query_j)`
3. For each receiver j, select top-K senders by score
4. Apply minimum threshold (with optional force_connect)

**Configuration:**
- `topk_per_receiver` - Max incoming edges per agent
- `min_score` - Similarity threshold
- `force_connect` - Ensure at least one edge per receiver

### dytopo-llm (crates/llm)

**Purpose:** LLM provider abstractions.

**Trait:**
```rust
pub trait LlmClient: Send + Sync {
    fn complete_json(&self, prompt: &str) -> Result<String>;
}
```

**Implementations:**
- `OllamaClient` - Local LLM via Ollama HTTP API (scaffold)

### dytopo-agents (crates/agents)

**Purpose:** Agent implementations.

**Trait:**
```rust
pub trait Agent: Send + Sync {
    fn id(&self) -> AgentId;
    fn step(&mut self, round: usize, goal: &RoundGoal, inbox: &[String]) -> Result<AgentIO>;
}
```

**Implementations:**
- `Manager` - Generates round goals
- `StubWorker` - Deterministic stub for testing
- (Planned) `LlmWorker` - LLM-backed agent

### dytopo-orchestrator (crates/orchestrator)

**Purpose:** Main execution loop.

**Flow:**
1. Manager generates RoundGoal
2. Each agent produces AgentIO (query, key, draft)
3. Router builds sparse topology
4. Orchestrator delivers messages along edges
5. Trace events written to JSONL

### dytopo-viz (crates/viz)

**Purpose:** Visualization exports.

**Outputs:**
- GraphViz DOT files
- (Planned) JSON for web viewers

### dytopo-cli (crates/cli)

**Purpose:** Command-line interface.

**Commands:**
- `demo` - Run stub demo with configurable parameters

## Data Flow

```
Round N:
  1. Manager.round_goal(N) -> RoundGoal
  2. For each agent:
     agent.step(N, goal, inbox) -> AgentIO{query, key, draft}
  3. Router:
     embed(keys), embed(queries) -> similarity matrix
     sparsify -> Topology{edges}
  4. For each edge:
     build Message from sender's draft+key
     deliver to receiver's inbox
  5. Write TraceEvents
```

## Dependency Graph

```
cli
 |
 +-- orchestrator
 |    |-- core
 |    |-- embed
 |    |-- router
 |    +-- agents
 |
 +-- viz
      +-- core
```

## Design Decisions

### Why Rust?
- Zero-cost abstractions for performance-critical embedding/routing
- Strong type system catches protocol mismatches at compile time
- No Python dependency for baseline demo

### Why Hash Embeddings First?
- Deterministic for reproducible demos
- No external ML dependencies
- Validates end-to-end pipeline before adding complexity

### Why Sparse Graphs?
- Reduces context explosion (each agent receives limited messages)
- Makes communication patterns interpretable
- Matches DyTopo paper's approach

### Why JSONL Traces?
- Append-only for streaming
- Line-based for easy grep/filtering
- JSON for machine parsing
- Standard format for analysis tools
