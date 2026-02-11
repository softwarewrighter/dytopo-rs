use anyhow::Result;
use dytopo_core::{AgentId, Edge, Topology};
use dytopo_embed::{Embedder, Embedding};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct RouterConfig {
    pub topk_per_receiver: usize,
    pub min_score: f32,
    pub force_connect: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            topk_per_receiver: 2,
            min_score: 0.10,
            force_connect: true,
        }
    }
}

/// Cosine similarity between two L2-normalized vectors.
/// If not normalized, this is still cosine-ish but may exceed [-1,1].
fn cosine(a: &Embedding, b: &Embedding) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Build a sparse directed topology from agent keys -> agent queries.
pub fn build_topology(
    round: usize,
    embedder: &dyn Embedder,
    keys: &[(AgentId, String)],
    queries: &[(AgentId, String)],
    cfg: &RouterConfig,
) -> Result<Topology> {
    let key_embs: Vec<(AgentId, Embedding)> = keys
        .par_iter()
        .map(|(id, k)| Ok((*id, embedder.embed(k)?)))
        .collect::<Result<Vec<_>>>()?;

    let query_embs: Vec<(AgentId, Embedding)> = queries
        .par_iter()
        .map(|(id, q)| Ok((*id, embedder.embed(q)?)))
        .collect::<Result<Vec<_>>>()?;

    // For each receiver (query owner), select top-K senders by cosine(key, query)
    let mut edges: Vec<Edge> = Vec::new();

    for (to_id, qv) in &query_embs {
        let mut scored: Vec<(AgentId, f32)> = key_embs
            .iter()
            .filter(|(from_id, _)| from_id != to_id) // avoid self-edge by default
            .map(|(from_id, kv)| (*from_id, cosine(kv, qv)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut picked = 0usize;
        for (from_id, score) in scored.iter().copied() {
            if picked >= cfg.topk_per_receiver {
                break;
            }
            if score >= cfg.min_score {
                edges.push(Edge {
                    from: from_id,
                    to: *to_id,
                    score,
                });
                picked += 1;
            }
        }

        if cfg.force_connect && picked == 0 {
            if let Some((best_from, best_score)) = scored.first().copied() {
                edges.push(Edge {
                    from: best_from,
                    to: *to_id,
                    score: best_score,
                });
            }
        }
    }

    Ok(Topology { round, edges })
}
