//! Embedding backends for semantic similarity routing.
//!
//! Provides `HashEmbedder` for deterministic testing and `OllamaEmbedder`
//! for semantic embeddings via Ollama's embedding API.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Fixed-length vector for similarity routing.
pub type Embedding = Vec<f32>;

/// Trait for text embedding backends.
pub trait Embedder: Send + Sync {
    /// Return the dimensionality of embeddings.
    fn dim(&self) -> usize;
    /// Embed a text string into a vector.
    fn embed(&self, text: &str) -> Result<Embedding>;
}

/// A deterministic, dependency-free baseline embedder.
/// It is *not* semantically meaningful, but it lets the system run end-to-end.
#[derive(Clone)]
pub struct HashEmbedder {
    dim: usize,
    seed: u64,
}

impl HashEmbedder {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    fn hash64(&self, s: &str) -> u64 {
        // Simple FNV-1a-ish hash, seeded. Deterministic across platforms.
        let mut h = 1469598103934665603u64 ^ self.seed;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211u64);
        }
        h
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        let mut v = vec![0f32; self.dim];
        // Tokenize on whitespace and punctuation-ish boundaries.
        for tok in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
        {
            let h = self.hash64(tok);
            let idx = (h as usize) % self.dim;
            // +/- sign bit to avoid all-positive vectors
            let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        // L2 normalize for cosine compatibility
        let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
        Ok(v)
    }
}

/// Ollama embedding request.
#[derive(Debug, Serialize)]
struct OllamaEmbedReq<'a> {
    model: &'a str,
    prompt: &'a str,
}

/// Ollama embedding response.
#[derive(Debug, Deserialize)]
struct OllamaEmbedResp {
    embedding: Vec<f64>,
}

/// Semantic embedder using Ollama's embedding API.
///
/// Uses models like `nomic-embed-text` to produce meaningful embeddings
/// for semantic similarity routing.
pub struct OllamaEmbedder {
    base_url: String,
    model: String,
    client: reqwest::blocking::Client,
    cache: Arc<Mutex<HashMap<String, Embedding>>>,
    dim: usize,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder.
    ///
    /// # Arguments
    /// * `base_url` - Ollama API base URL (e.g., "http://localhost:11434")
    /// * `model` - Embedding model name (e.g., "nomic-embed-text")
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let model = model.into();

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        // Probe the model to get embedding dimension
        let dim = Self::probe_dimension(&client, &base_url, &model)?;

        Ok(Self {
            base_url,
            model,
            client,
            cache: Arc::new(Mutex::new(HashMap::new())),
            dim,
        })
    }

    /// Probe the embedding model to determine its dimension.
    fn probe_dimension(
        client: &reqwest::blocking::Client,
        base_url: &str,
        model: &str,
    ) -> Result<usize> {
        let url = format!("{}/api/embeddings", base_url);
        let req = OllamaEmbedReq {
            model,
            prompt: "test",
        };

        let resp = client
            .post(&url)
            .json(&req)
            .send()
            .with_context(|| format!("Failed to connect to Ollama at {base_url}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("Ollama embedding probe failed ({status}): {body}");
        }

        let embed_resp: OllamaEmbedResp = resp
            .json()
            .context("Failed to parse Ollama embedding response")?;

        Ok(embed_resp.embedding.len())
    }

    /// Create with a known dimension (skips probe).
    pub fn with_known_dim(
        base_url: impl Into<String>,
        model: impl Into<String>,
        dim: usize,
    ) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url: base_url.into(),
            model: model.into(),
            client,
            cache: Arc::new(Mutex::new(HashMap::new())),
            dim,
        }
    }

    /// Clear the embedding cache.
    pub fn clear_cache(&self) {
        self.cache.lock().clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock();
        (cache.len(), cache.capacity())
    }
}

impl Embedder for OllamaEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        // Check cache first
        {
            let cache = self.cache.lock();
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }

        // Call Ollama API
        let url = format!("{}/api/embeddings", self.base_url);
        let req = OllamaEmbedReq {
            model: &self.model,
            prompt: text,
        };

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .with_context(|| format!("Failed to get embedding from {}", self.base_url))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("Ollama embedding failed ({status}): {body}");
        }

        let embed_resp: OllamaEmbedResp = resp
            .json()
            .context("Failed to parse Ollama embedding response")?;

        // Convert f64 to f32 and normalize
        let mut embedding: Embedding = embed_resp.embedding.iter().map(|&x| x as f32).collect();

        // L2 normalize for cosine similarity
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut embedding {
            *x /= norm;
        }

        // Cache the result
        {
            let mut cache = self.cache.lock();
            cache.insert(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }
}

/// Multi-host Ollama embedder with load balancing.
///
/// Distributes embedding requests across multiple Ollama instances.
pub struct OllamaEmbedderPool {
    embedders: Vec<OllamaEmbedder>,
    next_idx: Arc<Mutex<usize>>,
}

impl OllamaEmbedderPool {
    /// Create a pool from multiple Ollama hosts.
    ///
    /// All hosts must have the same embedding model with the same dimension.
    pub fn new(hosts: &[&str], model: &str) -> Result<Self> {
        if hosts.is_empty() {
            anyhow::bail!("At least one host required for embedding pool");
        }

        let mut embedders = Vec::with_capacity(hosts.len());
        let mut expected_dim = None;

        for host in hosts {
            let embedder = OllamaEmbedder::new(*host, model)?;
            let dim = embedder.dim();

            if let Some(expected) = expected_dim {
                if dim != expected {
                    anyhow::bail!(
                        "Embedding dimension mismatch: {} has dim {}, expected {}",
                        host,
                        dim,
                        expected
                    );
                }
            } else {
                expected_dim = Some(dim);
            }

            embedders.push(embedder);
        }

        Ok(Self {
            embedders,
            next_idx: Arc::new(Mutex::new(0)),
        })
    }

    /// Get the next embedder in round-robin order.
    fn next_embedder(&self) -> &OllamaEmbedder {
        let mut idx = self.next_idx.lock();
        let embedder = &self.embedders[*idx];
        *idx = (*idx + 1) % self.embedders.len();
        embedder
    }
}

impl Embedder for OllamaEmbedderPool {
    fn dim(&self) -> usize {
        self.embedders[0].dim()
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        // Check all caches first
        for embedder in &self.embedders {
            let cache = embedder.cache.lock();
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }

        // Use round-robin selection
        self.next_embedder().embed(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embedder_deterministic() {
        let embedder = HashEmbedder::new(64, 42);
        let e1 = embedder.embed("hello world").unwrap();
        let e2 = embedder.embed("hello world").unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_hash_embedder_normalized() {
        let embedder = HashEmbedder::new(64, 42);
        let e = embedder.embed("test embedding").unwrap();
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hash_embedder_different_texts() {
        let embedder = HashEmbedder::new(64, 42);
        let e1 = embedder.embed("hello").unwrap();
        let e2 = embedder.embed("goodbye").unwrap();
        assert_ne!(e1, e2);
    }
}
