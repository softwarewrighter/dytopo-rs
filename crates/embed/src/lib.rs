use anyhow::Result;

/// Fixed-length vector for similarity routing.
pub type Embedding = Vec<f32>;

pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Embedding>;
}

/// A deterministic, dependency-free baseline embedder.
/// It is *not* semantically meaningful, but it lets the system run end-to-end.
/// Replace with a real embedder backend later (fastembed/candle).
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
