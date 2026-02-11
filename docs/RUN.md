# Running and developing

## Build

```bash
cargo build
```

## Tests

```bash
cargo test
```

## Format

```bash
cargo fmt
```

## Demo: dynamic topology

```bash
cargo run -p dytopo-cli -- demo --rounds 3 --agents 5 --topk 2 --min-score 0.10
```

## Output artifacts

- `./traces/trace_<timestamp>.jsonl` (JSONL)
- `./traces/topology_<timestamp>_round<N>.dot` (GraphViz)

To render DOT:
```bash
dot -Tsvg traces/topology_*.dot -o topology.svg
```
