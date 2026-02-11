//! LLM provider abstractions for dytopo agents.
//!
//! Provides `OllamaClient` for single-host and `OllamaPool` for multi-host
//! load-balanced LLM inference.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Response from an LLM completion with token counts.
#[derive(Clone, Debug, Default)]
pub struct LlmResponse {
    pub text: String,
    pub tokens_in: usize,
    pub tokens_out: usize,
}

/// Trait for LLM completion clients.
pub trait LlmClient: Send + Sync {
    /// Send a prompt and get a completion response.
    fn complete(&self, prompt: &str) -> Result<String>;

    /// Send a prompt and get a completion with token counts.
    fn complete_with_tokens(&self, prompt: &str) -> Result<LlmResponse> {
        // Default implementation for backward compatibility
        let text = self.complete(prompt)?;
        Ok(LlmResponse {
            text,
            tokens_in: 0,
            tokens_out: 0,
        })
    }
}

/// Configuration for an Ollama host.
#[derive(Clone, Debug)]
pub struct OllamaHost {
    pub name: String,
    pub base_url: String,
    pub max_concurrent: usize,
}

impl OllamaHost {
    pub fn new(name: impl Into<String>, base_url: impl Into<String>, max_concurrent: usize) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            max_concurrent,
        }
    }
}

/// Single-host Ollama client.
#[derive(Clone)]
pub struct OllamaClient {
    pub host: OllamaHost,
    pub model: String,
    pub timeout: Duration,
    client: reqwest::blocking::Client,
}

#[derive(Debug, Serialize)]
struct OllamaReq<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: i32,
}

#[derive(Debug, Deserialize)]
struct OllamaResp {
    response: String,
    #[serde(default)]
    prompt_eval_count: usize,
    #[serde(default)]
    eval_count: usize,
}

impl OllamaClient {
    pub fn new(host: OllamaHost, model: impl Into<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            host,
            model: model.into(),
            timeout: Duration::from_secs(120),
            client,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl LlmClient for OllamaClient {
    fn complete(&self, prompt: &str) -> Result<String> {
        Ok(self.complete_with_tokens(prompt)?.text)
    }

    fn complete_with_tokens(&self, prompt: &str) -> Result<LlmResponse> {
        let url = format!("{}/api/generate", self.host.base_url);

        let req_body = OllamaReq {
            model: &self.model,
            prompt,
            stream: false,
            options: OllamaOptions {
                temperature: 0.7,
                num_predict: 1024,
            },
        };

        let resp = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .with_context(|| format!("Failed to connect to Ollama at {}", self.host.base_url))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("Ollama returned {status}: {body}");
        }

        let ollama_resp: OllamaResp = resp
            .json()
            .context("Failed to parse Ollama response")?;

        Ok(LlmResponse {
            text: ollama_resp.response,
            tokens_in: ollama_resp.prompt_eval_count,
            tokens_out: ollama_resp.eval_count,
        })
    }
}

/// Tracks in-flight requests per host for load balancing.
#[derive(Debug, Default)]
struct HostState {
    in_flight: usize,
    last_request: Option<Instant>,
}

/// Multi-host Ollama pool with load balancing and staggering.
///
/// Distributes requests across hosts based on capacity and staggers
/// concurrent requests to avoid GPU overload.
pub struct OllamaPool {
    hosts: Vec<OllamaHost>,
    model: String,
    state: Arc<Mutex<Vec<HostState>>>,
    stagger_delay: Duration,
    client: reqwest::blocking::Client,
}

impl OllamaPool {
    /// Create a new pool with the given hosts and model.
    pub fn new(hosts: Vec<OllamaHost>, model: impl Into<String>) -> Self {
        let state = Arc::new(Mutex::new(
            hosts.iter().map(|_| HostState::default()).collect(),
        ));

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            hosts,
            model: model.into(),
            state,
            stagger_delay: Duration::from_millis(500),
            client,
        }
    }

    /// Set the delay between staggered requests to the same host.
    pub fn with_stagger_delay(mut self, delay: Duration) -> Self {
        self.stagger_delay = delay;
        self
    }

    /// Select the best host for the next request.
    /// Returns host index and whether we should wait before sending.
    fn select_host(&self) -> (usize, Duration) {
        let mut state = self.state.lock();
        let now = Instant::now();

        // Find host with most available capacity
        let mut best_idx = 0;
        let mut best_available = 0i64;

        for (idx, host) in self.hosts.iter().enumerate() {
            let available = host.max_concurrent as i64 - state[idx].in_flight as i64;
            if available > best_available {
                best_available = available;
                best_idx = idx;
            }
        }

        // Calculate stagger delay if needed
        let wait = if let Some(last) = state[best_idx].last_request {
            let elapsed = now.duration_since(last);
            if elapsed < self.stagger_delay {
                self.stagger_delay - elapsed
            } else {
                Duration::ZERO
            }
        } else {
            Duration::ZERO
        };

        // Update state
        state[best_idx].in_flight += 1;
        state[best_idx].last_request = Some(now + wait);

        (best_idx, wait)
    }

    /// Mark a request as complete for the given host.
    fn release_host(&self, host_idx: usize) {
        let mut state = self.state.lock();
        if state[host_idx].in_flight > 0 {
            state[host_idx].in_flight -= 1;
        }
    }

    /// Send a completion request with load balancing.
    pub fn complete_with_tokens(&self, prompt: &str) -> Result<LlmResponse> {
        let (host_idx, wait) = self.select_host();

        // Stagger if needed
        if !wait.is_zero() {
            thread::sleep(wait);
        }

        let host = &self.hosts[host_idx];
        let url = format!("{}/api/generate", host.base_url);

        let req_body = OllamaReq {
            model: &self.model,
            prompt,
            stream: false,
            options: OllamaOptions {
                temperature: 0.7,
                num_predict: 1024,
            },
        };

        let result = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .with_context(|| format!("Failed to connect to {} at {}", host.name, host.base_url));

        // Release host on completion (success or failure)
        let resp = match result {
            Ok(r) => r,
            Err(e) => {
                self.release_host(host_idx);
                return Err(e);
            }
        };

        if !resp.status().is_success() {
            self.release_host(host_idx);
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("{} returned {status}: {body}", host.name);
        }

        let ollama_resp: OllamaResp = match resp.json() {
            Ok(r) => r,
            Err(e) => {
                self.release_host(host_idx);
                return Err(e).context("Failed to parse Ollama response");
            }
        };

        self.release_host(host_idx);
        Ok(LlmResponse {
            text: ollama_resp.response,
            tokens_in: ollama_resp.prompt_eval_count,
            tokens_out: ollama_resp.eval_count,
        })
    }

    /// Send a completion request (convenience wrapper).
    pub fn complete(&self, prompt: &str) -> Result<String> {
        Ok(self.complete_with_tokens(prompt)?.text)
    }

    /// Get current load statistics for monitoring.
    pub fn stats(&self) -> Vec<(String, usize, usize)> {
        let state = self.state.lock();
        self.hosts
            .iter()
            .enumerate()
            .map(|(idx, host)| {
                (host.name.clone(), state[idx].in_flight, host.max_concurrent)
            })
            .collect()
    }
}

impl LlmClient for OllamaPool {
    fn complete(&self, prompt: &str) -> Result<String> {
        OllamaPool::complete(self, prompt)
    }

    fn complete_with_tokens(&self, prompt: &str) -> Result<LlmResponse> {
        OllamaPool::complete_with_tokens(self, prompt)
    }
}

/// Parse JSON from LLM response, handling markdown code blocks.
pub fn extract_json(response: &str) -> Result<serde_json::Value> {
    // Try direct parse first
    if let Ok(v) = serde_json::from_str(response) {
        return Ok(v);
    }

    // Try to extract from markdown code block
    let trimmed = response.trim();

    // Handle ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let content_start = start + 7;
        if let Some(end) = trimmed[content_start..].find("```") {
            let json_str = &trimmed[content_start..content_start + end].trim();
            if let Ok(v) = serde_json::from_str(json_str) {
                return Ok(v);
            }
        }
    }

    // Handle ``` ... ``` blocks (no language specifier)
    if let Some(start) = trimmed.find("```") {
        let content_start = start + 3;
        // Skip any language identifier on the same line
        let content = &trimmed[content_start..];
        let actual_start = content.find('\n').map(|i| i + 1).unwrap_or(0);
        if let Some(end) = content[actual_start..].find("```") {
            let json_str = content[actual_start..actual_start + end].trim();
            if let Ok(v) = serde_json::from_str(json_str) {
                return Ok(v);
            }
        }
    }

    // Try to find JSON object in the text
    if let Some(start) = trimmed.find('{') {
        // Find matching closing brace
        let mut depth = 0;
        let mut end_idx = start;
        for (i, c) in trimmed[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end_idx = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            let json_str = &trimmed[start..end_idx];
            if let Ok(v) = serde_json::from_str(json_str) {
                return Ok(v);
            }
        }
    }

    anyhow::bail!("Could not extract valid JSON from response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_direct() {
        let input = r#"{"query": "test", "key": "value", "draft": "content"}"#;
        let v = extract_json(input).unwrap();
        assert_eq!(v["query"], "test");
    }

    #[test]
    fn test_extract_json_markdown() {
        let input = r#"Here is the response:
```json
{"query": "test", "key": "value", "draft": "content"}
```
"#;
        let v = extract_json(input).unwrap();
        assert_eq!(v["query"], "test");
    }

    #[test]
    fn test_extract_json_embedded() {
        let input = r#"I think the answer is {"query": "test", "key": "value", "draft": "content"} based on the context."#;
        let v = extract_json(input).unwrap();
        assert_eq!(v["query"], "test");
    }
}
