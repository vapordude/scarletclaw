use crate::tensor::Tensor;

/// Converts discrete token IDs into dense continuous vector representations.
pub struct Embedding {
    weight: Tensor, // (vocab_size, hidden_size)
    hidden_size: usize,
    vocab_size: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            // Dummy initialization for scaffolding
            weight: Tensor::zeros(vec![vocab_size, hidden_size]),
            hidden_size,
            vocab_size,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Forward pass: looks up the embedding vector for a given token ID.
    pub fn forward(&self, mut token_id: usize) -> Tensor {
        if token_id >= self.vocab_size {
            // Map out-of-vocabulary to token 0 (typically UNK)
            token_id = 0;
        }

        let mut out = Tensor::zeros(vec![1, self.hidden_size]);
        // Simple row extraction
        let offset = token_id * self.hidden_size;
        for i in 0..self.hidden_size {
            out.data[i] = self.weight.data[offset + i];
        }
        out
    }
}

/// The final Language Model Head.
/// Projects the final hidden state back into the vocabulary space to produce logits.
pub struct LmHead {
    weight: Tensor, // (hidden_size, vocab_size)
}

impl LmHead {
    pub fn new(hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            weight: Tensor::zeros(vec![hidden_size, vocab_size]),
        }
    }

    /// Forward pass: `x` is (1, hidden_size), returns logits (1, vocab_size).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight)
    }
}

/// A mathematical implementation of Root Mean Square Layer Normalization (RMSNorm).
/// Popularized by LLaMA models, it stabilizes the activations by scaling them
/// by the root mean square of the vector elements, rather than mean and variance.
pub struct RMSNorm {
    weight: Tensor, // (hidden_size,)
    eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            weight: Tensor::new(vec![1.0; hidden_size], vec![hidden_size]),
            eps: 1e-5,
        }
    }

    /// Forward pass of RMSNorm.
    /// `x` is a 1D tensor (hidden_size).
    pub fn forward(&self, x: &mut Tensor) {
        let size = x.data.len();

        // Calculate sum of squares
        let mut ss = 0.0;
        for i in 0..size {
            ss += x.data[i] * x.data[i];
        }

        // Calculate Root Mean Square
        ss /= size as f32;
        ss += self.eps;
        let inv_rms = 1.0 / ss.sqrt();

        // Scale and apply learned weights
        for i in 0..size {
            x.data[i] = (inv_rms * x.data[i]) * self.weight.data[i];
        }
    }
}

/// A standard Feed Forward Network (SwiGLU / SiLU based, a la Llama).
pub struct FeedForward {
    w1: Tensor, // (hidden_size, intermediate_size)
    w2: Tensor, // (intermediate_size, hidden_size)
    w3: Tensor, // (hidden_size, intermediate_size)
}

impl FeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        // Initialize weights (normally these would be loaded from crimson-core or a GGUF file)
        Self {
            w1: Tensor::zeros(vec![hidden_size, intermediate_size]),
            w2: Tensor::zeros(vec![intermediate_size, hidden_size]),
            w3: Tensor::zeros(vec![hidden_size, intermediate_size]),
        }
    }

    /// `x` is (1, hidden_size). Returns (1, hidden_size).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Project to intermediate space: h1 = x * w1
        let mut h1 = x.matmul(&self.w1);

        // 2. Apply Swish/SiLU: h1 = silu(x * w1)
        h1.silu();

        // 3. Project with gating: h2 = x * w3
        let h2 = x.matmul(&self.w3);

        // 4. Element-wise multiplication of the two intermediate states
        for i in 0..h1.data.len() {
            h1.data[i] *= h2.data[i];
        }

        // 5. Final projection back to hidden size: out = h1 * w2
        h1.matmul(&self.w2)
    }
}

/// Rotary Positional Embeddings (RoPE) mathematical structure.
/// Injects positional information by rotating the queries and keys in the complex plane.
pub fn apply_rope(q: &mut Tensor, k: &mut Tensor, pos: usize, head_dim: usize) {
    debug_assert!(head_dim > 0 && head_dim % 2 == 0, "head_dim must be positive and even");
    debug_assert!(q.shape[0] % head_dim == 0, "q dimension must be divisible by head_dim");
    debug_assert!(k.shape[0] % head_dim == 0, "k dimension must be divisible by head_dim");

    // In a real implementation, theta frequencies are precomputed.
    // We apply rotations to pairs of dimensions.
    let theta_base = 10000.0f32;
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / theta_base.powf((i as f32) / (head_dim as f32));
        let val = (pos as f32) * freq;
        let cos_val = val.cos();
        let sin_val = val.sin();

        // Rotate queries
        for h in 0..(q.shape[0] / head_dim) {
            let offset = h * head_dim + i;
            let q0 = q.data[offset];
            let q1 = q.data[offset + 1];
            q.data[offset] = q0 * cos_val - q1 * sin_val;
            q.data[offset + 1] = q0 * sin_val + q1 * cos_val;
        }

        // Rotate keys
        for h in 0..(k.shape[0] / head_dim) {
            let offset = h * head_dim + i;
            let k0 = k.data[offset];
            let k1 = k.data[offset + 1];
            k.data[offset] = k0 * cos_val - k1 * sin_val;
            k.data[offset + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}
