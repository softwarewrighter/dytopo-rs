use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub usize);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundGoal(pub String);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentIO {
    pub agent_id: AgentId,
    pub query: String,
    pub key: String,
    pub draft: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub from: AgentId,
    pub to: AgentId,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Topology {
    pub round: usize,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub round: usize,
    pub from: AgentId,
    pub to: AgentId,
    pub score: f32,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TraceEvent {
    RoundStart {
        round: usize,
        goal: String,
        agent_count: usize,
        ts_unix_ms: u64,
    },
    AgentIO {
        round: usize,
        agent_id: usize,
        query: String,
        key: String,
        draft: String,
    },
    Topology {
        round: usize,
        edges: Vec<Edge>,
    },
    Message {
        round: usize,
        from: usize,
        to: usize,
        score: f32,
        content: String,
    },
    RoundEnd {
        round: usize,
        ts_unix_ms: u64,
    },
}
