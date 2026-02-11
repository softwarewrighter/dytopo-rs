use anyhow::Result;
use dytopo_agents::{Agent, Manager};
use dytopo_core::{AgentId, Message, Topology, TraceEvent};
use dytopo_embed::Embedder;
use dytopo_router::{build_topology, RouterConfig};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct OrchestratorConfig {
    pub rounds: usize,
    pub router: RouterConfig,
    pub max_inbox: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            rounds: 3,
            router: RouterConfig::default(),
            max_inbox: 3,
        }
    }
}

pub struct RunArtifacts {
    pub trace_path: String,
    pub topologies: Vec<Topology>,
    pub messages: Vec<Message>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn run_stub(
    task: &str,
    mut agents: Vec<Box<dyn Agent>>,
    embedder: &dyn Embedder,
    cfg: &OrchestratorConfig,
    trace_dir: &str,
    run_tag: &str,
) -> Result<RunArtifacts> {
    create_dir_all(trace_dir)?;
    let ts = now_ms();
    let trace_path = format!("{}/trace_{}_{}.jsonl", trace_dir, run_tag, ts);
    let f = File::create(&trace_path)?;
    let mut w = BufWriter::new(f);

    let manager = Manager::new(task);

    // inbox per agent id
    let mut inboxes: Vec<Vec<String>> = vec![Vec::new(); agents.len() + 1]; // + manager placeholder

    let mut topologies = Vec::new();
    let mut messages = Vec::new();

    for round in 0..cfg.rounds {
        let goal = manager.round_goal(round);
        let start = TraceEvent::RoundStart {
            round,
            goal: goal.0.clone(),
            agent_count: agents.len(),
            ts_unix_ms: now_ms(),
        };
        serde_json::to_writer(&mut w, &start)?;
        w.write_all(b"\n")?;

        // step agents
        let mut ios = Vec::with_capacity(agents.len());
        for a in agents.iter_mut() {
            let id = a.id();
            let inbox = inboxes.get(id.0).map(|v| v.as_slice()).unwrap_or(&[]);
            let io = a.step(round, &goal, inbox)?;
            let ev = TraceEvent::AgentIO {
                round,
                agent_id: io.agent_id.0,
                query: io.query.clone(),
                key: io.key.clone(),
                draft: io.draft.clone(),
                tokens_in: 0,  // TODO: get from agent response
                tokens_out: 0,
            };
            serde_json::to_writer(&mut w, &ev)?;
            w.write_all(b"\n")?;
            ios.push(io);
        }

        let keys: Vec<(AgentId, String)> = ios.iter().map(|io| (io.agent_id, io.key.clone())).collect();
        let queries: Vec<(AgentId, String)> = ios.iter().map(|io| (io.agent_id, io.query.clone())).collect();

        let topo = build_topology(round, embedder, &keys, &queries, &cfg.router)?;
        let topo_ev = TraceEvent::Topology {
            round,
            edges: topo.edges.clone(),
        };
        serde_json::to_writer(&mut w, &topo_ev)?;
        w.write_all(b"\n")?;

        // deliver messages along edges
        for e in topo.edges.iter() {
            let content = format!(
                "From agent {}: {} // {}",
                e.from.0,
                ios.iter().find(|x| x.agent_id == e.from).map(|x| x.draft.as_str()).unwrap_or(""),
                ios.iter().find(|x| x.agent_id == e.from).map(|x| x.key.as_str()).unwrap_or("")
            );

            let msg = Message {
                round,
                from: e.from,
                to: e.to,
                score: e.score,
                content: content.clone(),
            };
            messages.push(msg.clone());

            // push into inbox with cap
            let inbox = inboxes.get_mut(e.to.0).unwrap();
            inbox.push(content.clone());
            if inbox.len() > cfg.max_inbox {
                // drop oldest
                let excess = inbox.len() - cfg.max_inbox;
                inbox.drain(0..excess);
            }

            let me = TraceEvent::Message {
                round,
                from: e.from.0,
                to: e.to.0,
                score: e.score,
                content,
            };
            serde_json::to_writer(&mut w, &me)?;
            w.write_all(b"\n")?;
        }

        let end = TraceEvent::RoundEnd {
            round,
            ts_unix_ms: now_ms(),
        };
        serde_json::to_writer(&mut w, &end)?;
        w.write_all(b"\n")?;

        topologies.push(topo);
    }

    w.flush()?;

    Ok(RunArtifacts {
        trace_path,
        topologies,
        messages,
    })
}
