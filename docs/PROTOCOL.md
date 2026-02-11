# Protocol (schemas + invariants)

## Agent output schema

Agents return:

```json
{
  "query": "What I need next",
  "key": "What I can contribute",
  "draft": "Optional partial solution or notes"
}
```

Invariants:
- `query` and `key` must be <= 280 chars for the baseline demo (configurable later)
- `draft` may be longer but should remain < 2k chars in traces

## Router semantics

We compute a directed score:

`score(i -> j) = cos(embed(key_i), embed(query_j))`

Edges:
- For each receiver j, select top-K senders i by score
- Apply minimum threshold `min_score` (but allow at least 1 incoming edge if `force_connect=true`)

## Trace format (JSONL)

Each line is a JSON object with a `type` field:

- `round_start`: { round, goal, agent_count, ts }
- `agent_io`: { round, agent_id, query, key, draft }
- `topology`: { round, edges: [{from,to,score}] }
- `message`: { round, from, to, score, content }
- `round_end`: { round, ts }

Traces must be append-only and valid JSONL.
