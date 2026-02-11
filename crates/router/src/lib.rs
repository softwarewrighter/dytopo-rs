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

/// Baseline topology types for comparison experiments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BaselineTopology {
    /// Every agent connected to every other agent
    FullyConnected,
    /// One central agent (first agent) connected to all others
    Star,
    /// Sequential chain: 1->2->3->...->N
    Chain,
    /// Circular: 1->2->3->...->N->1
    Ring,
}

/// Build a baseline topology for comparison with dynamic routing.
pub fn build_baseline_topology(
    round: usize,
    baseline: BaselineTopology,
    agent_ids: &[AgentId],
) -> Topology {
    let edges = match baseline {
        BaselineTopology::FullyConnected => {
            let mut e = Vec::new();
            for &from in agent_ids {
                for &to in agent_ids {
                    if from != to {
                        e.push(Edge {
                            from,
                            to,
                            score: 1.0, // uniform score for baselines
                        });
                    }
                }
            }
            e
        }
        BaselineTopology::Star => {
            // First agent is the hub
            let mut e = Vec::new();
            if let Some(&hub) = agent_ids.first() {
                for &other in agent_ids.iter().skip(1) {
                    // Hub -> other
                    e.push(Edge {
                        from: hub,
                        to: other,
                        score: 1.0,
                    });
                    // Other -> hub
                    e.push(Edge {
                        from: other,
                        to: hub,
                        score: 1.0,
                    });
                }
            }
            e
        }
        BaselineTopology::Chain => {
            let mut e = Vec::new();
            for i in 0..agent_ids.len().saturating_sub(1) {
                e.push(Edge {
                    from: agent_ids[i],
                    to: agent_ids[i + 1],
                    score: 1.0,
                });
            }
            e
        }
        BaselineTopology::Ring => {
            let mut e = Vec::new();
            let n = agent_ids.len();
            if n > 0 {
                for i in 0..n {
                    e.push(Edge {
                        from: agent_ids[i],
                        to: agent_ids[(i + 1) % n],
                        score: 1.0,
                    });
                }
            }
            e
        }
    };

    Topology { round, edges }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_fully_connected() {
        let ids = vec![AgentId(1), AgentId(2), AgentId(3)];
        let topo = build_baseline_topology(0, BaselineTopology::FullyConnected, &ids);
        // 3 agents = 3*(3-1) = 6 edges
        assert_eq!(topo.edges.len(), 6);
    }

    #[test]
    fn test_baseline_star() {
        let ids = vec![AgentId(1), AgentId(2), AgentId(3)];
        let topo = build_baseline_topology(0, BaselineTopology::Star, &ids);
        // Hub connects to 2 others bidirectionally = 4 edges
        assert_eq!(topo.edges.len(), 4);
    }

    #[test]
    fn test_baseline_chain() {
        let ids = vec![AgentId(1), AgentId(2), AgentId(3)];
        let topo = build_baseline_topology(0, BaselineTopology::Chain, &ids);
        // 1->2, 2->3 = 2 edges
        assert_eq!(topo.edges.len(), 2);
    }

    #[test]
    fn test_baseline_ring() {
        let ids = vec![AgentId(1), AgentId(2), AgentId(3)];
        let topo = build_baseline_topology(0, BaselineTopology::Ring, &ids);
        // 1->2, 2->3, 3->1 = 3 edges
        assert_eq!(topo.edges.len(), 3);
    }
}
