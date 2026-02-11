//! Agent implementations for dytopo multi-agent coordination.

use anyhow::Result;
use dytopo_core::{AgentId, AgentIO, RoundGoal};
use dytopo_llm::{extract_json, LlmClient};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

/// Trait for agents that participate in multi-round coordination.
pub trait Agent: Send + Sync {
    fn id(&self) -> AgentId;
    fn step(&mut self, round: usize, goal: &RoundGoal, inbox: &[String]) -> Result<AgentIO>;
}

/// A simple manager that emits round goals.
pub struct Manager {
    task: String,
}

impl Manager {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
        }
    }

    pub fn round_goal(&self, round: usize) -> RoundGoal {
        RoundGoal(format!(
            "Round {} goal: advance solution for task: {}",
            round, self.task
        ))
    }
}

/// A stub worker agent that produces query/key from templates.
/// This makes routing behavior visible without needing an LLM.
pub struct StubWorker {
    id: AgentId,
    rng: StdRng,
}

impl StubWorker {
    pub fn new(id: AgentId, seed: u64) -> Self {
        Self {
            id,
            rng: StdRng::seed_from_u64(seed ^ (id.0 as u64).wrapping_mul(0x9E3779B97F4A7C15)),
        }
    }
}

impl Agent for StubWorker {
    fn id(&self) -> AgentId {
        self.id
    }

    fn step(&mut self, round: usize, goal: &RoundGoal, inbox: &[String]) -> Result<AgentIO> {
        let domains = [
            ("math", "algebra, arithmetic, unit checking"),
            ("code", "patch ideas, debugging, compile errors"),
            ("planning", "step-by-step plan, milestones, acceptance tests"),
            ("writing", "clear explanation, narrative, concise summary"),
            ("review", "edge cases, counterexamples, sanity checks"),
        ];

        let (tag, offer) = domains[self.rng.gen_range(0..domains.len())];

        let query = format!(
            "[{}] Need: one missing piece for round {} given goal: {}",
            tag, round, goal.0
        );
        let key = format!("[{}] Offer: {}", tag, offer);

        let draft = if inbox.is_empty() {
            format!("No inbox yet. I will focus on {tag} aspects.")
        } else {
            format!(
                "Inbox has {} msgs. I will integrate and refine {tag}.",
                inbox.len()
            )
        };

        Ok(AgentIO {
            agent_id: self.id,
            query,
            key,
            draft,
        })
    }
}

/// Configuration for LLM-backed agents.
#[derive(Clone)]
pub struct LlmWorkerConfig {
    /// Agent specialty/role description
    pub specialty: String,
    /// Max characters for query field
    pub max_query_len: usize,
    /// Max characters for key field
    pub max_key_len: usize,
    /// Number of retry attempts on parse failure
    pub max_retries: usize,
}

impl Default for LlmWorkerConfig {
    fn default() -> Self {
        Self {
            specialty: "general problem solving".to_string(),
            max_query_len: 280,
            max_key_len: 280,
            max_retries: 1,
        }
    }
}

impl LlmWorkerConfig {
    pub fn with_specialty(mut self, specialty: impl Into<String>) -> Self {
        self.specialty = specialty.into();
        self
    }
}

/// LLM-backed worker agent that generates query/key/draft via Ollama.
pub struct LlmWorker {
    id: AgentId,
    client: Arc<dyn LlmClient>,
    config: LlmWorkerConfig,
    fallback_rng: StdRng,
}

impl LlmWorker {
    pub fn new(id: AgentId, client: Arc<dyn LlmClient>, config: LlmWorkerConfig) -> Self {
        Self {
            id,
            client,
            config,
            fallback_rng: StdRng::seed_from_u64(id.0 as u64 * 12345),
        }
    }

    /// Build the prompt for the LLM.
    fn build_prompt(&self, round: usize, goal: &RoundGoal, inbox: &[String]) -> String {
        let inbox_section = if inbox.is_empty() {
            "No messages received yet.".to_string()
        } else {
            let msgs: Vec<String> = inbox
                .iter()
                .enumerate()
                .map(|(i, m)| format!("{}. {}", i + 1, m))
                .collect();
            format!("Messages from other agents:\n{}", msgs.join("\n"))
        };

        format!(
            r#"You are Agent {id} with specialty: {specialty}

This is round {round} of the collaboration.

Current goal: {goal}

{inbox}

Based on the goal and any messages you received, respond with a JSON object:
{{
  "query": "What specific information or help do you need from other agents?",
  "key": "What specific knowledge or capability can you offer to help others?",
  "draft": "Your current thoughts, progress, or partial solution"
}}

IMPORTANT RULES:
1. query must be <= {max_q} characters - be specific about what you need
2. key must be <= {max_k} characters - clearly state what you can contribute
3. draft can be longer but focus on substance
4. Use domain tags like [math], [code], [planning] to help routing
5. Reference inbox messages when relevant
6. Respond with ONLY the JSON object, no other text

JSON response:"#,
            id = self.id.0,
            round = round,
            specialty = self.config.specialty,
            goal = goal.0,
            inbox = inbox_section,
            max_q = self.config.max_query_len,
            max_k = self.config.max_key_len,
        )
    }

    /// Build a repair prompt after a parse failure.
    fn build_repair_prompt(&self, original_response: &str) -> String {
        format!(
            r#"Your previous response could not be parsed as JSON:
---
{original_response}
---

Please respond with ONLY a valid JSON object in this exact format:
{{"query": "...", "key": "...", "draft": "..."}}

No explanation, no markdown, just the JSON object:"#
        )
    }

    /// Parse the LLM response into AgentIO.
    fn parse_response(&self, response: &str) -> Result<AgentIO> {
        let json = extract_json(response)?;

        let query = json["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'query' field"))?
            .to_string();

        let key = json["key"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'key' field"))?
            .to_string();

        let draft = json["draft"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'draft' field"))?
            .to_string();

        // Truncate if needed
        let query = if query.len() > self.config.max_query_len {
            query[..self.config.max_query_len].to_string()
        } else {
            query
        };

        let key = if key.len() > self.config.max_key_len {
            key[..self.config.max_key_len].to_string()
        } else {
            key
        };

        Ok(AgentIO {
            agent_id: self.id,
            query,
            key,
            draft,
        })
    }

    /// Generate fallback output using stub logic.
    fn fallback(&mut self, round: usize, goal: &RoundGoal, inbox: &[String]) -> AgentIO {
        let domains = [
            ("general", "problem-solving assistance"),
            ("analysis", "breaking down complex problems"),
            ("synthesis", "combining information from multiple sources"),
        ];

        let (tag, offer) = domains[self.fallback_rng.gen_range(0..domains.len())];

        let query = format!(
            "[{tag}] Need: guidance for round {round} on: {}",
            &goal.0[..goal.0.len().min(100)]
        );
        let key = format!("[{tag}] Offer: {offer}");
        let draft = if inbox.is_empty() {
            format!("Starting fresh with {tag} approach.")
        } else {
            format!("Received {} messages, applying {tag} skills.", inbox.len())
        };

        AgentIO {
            agent_id: self.id,
            query,
            key,
            draft,
        }
    }
}

impl Agent for LlmWorker {
    fn id(&self) -> AgentId {
        self.id
    }

    fn step(&mut self, round: usize, goal: &RoundGoal, inbox: &[String]) -> Result<AgentIO> {
        let prompt = self.build_prompt(round, goal, inbox);

        // Try initial request
        let response = match self.client.complete(&prompt) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Agent {} LLM error: {e}, using fallback", self.id.0);
                return Ok(self.fallback(round, goal, inbox));
            }
        };

        // Try to parse response
        match self.parse_response(&response) {
            Ok(io) => return Ok(io),
            Err(e) => {
                eprintln!("Agent {} parse error: {e}, attempting repair", self.id.0);
            }
        }

        // Retry with repair prompt if configured
        for attempt in 0..self.config.max_retries {
            let repair_prompt = self.build_repair_prompt(&response);
            let retry_response = match self.client.complete(&repair_prompt) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!(
                        "Agent {} retry {} failed: {e}",
                        self.id.0,
                        attempt + 1
                    );
                    continue;
                }
            };

            match self.parse_response(&retry_response) {
                Ok(io) => return Ok(io),
                Err(e) => {
                    eprintln!(
                        "Agent {} retry {} parse failed: {e}",
                        self.id.0,
                        attempt + 1
                    );
                }
            }
        }

        // All retries failed, use fallback
        eprintln!("Agent {} all retries failed, using fallback", self.id.0);
        Ok(self.fallback(round, goal, inbox))
    }
}

/// Predefined agent specialties for common roles.
pub mod specialties {
    pub const MATHEMATICIAN: &str = "mathematical reasoning, algebra, arithmetic, proofs, and formal logic";
    pub const CODER: &str = "software development, debugging, code review, and implementation";
    pub const PLANNER: &str = "strategic planning, task decomposition, milestone definition, and project management";
    pub const WRITER: &str = "clear communication, documentation, explanations, and summarization";
    pub const REVIEWER: &str = "critical analysis, edge case identification, testing, and quality assurance";
    pub const RESEARCHER: &str = "information gathering, fact-checking, and knowledge synthesis";
}
