use dytopo_core::AgentId;
use dytopo_embed::HashEmbedder;
use dytopo_router::{build_topology, RouterConfig};

#[test]
fn builds_edges_and_respects_topk() {
    let embedder = HashEmbedder::new(64, 1);
    let cfg = RouterConfig { topk_per_receiver: 1, min_score: -1.0, force_connect: true };

    let keys = vec![
        (AgentId(1), "offer math".to_string()),
        (AgentId(2), "offer code".to_string()),
        (AgentId(3), "offer plan".to_string()),
    ];
    let queries = vec![
        (AgentId(1), "need code".to_string()),
        (AgentId(2), "need math".to_string()),
        (AgentId(3), "need plan".to_string()),
    ];

    let topo = build_topology(0, &embedder, &keys, &queries, &cfg).unwrap();
    // One incoming edge per receiver (topk=1), excluding self edges.
    assert_eq!(topo.edges.len(), 3);
    for e in topo.edges {
        assert_ne!(e.from, e.to);
    }
}
