use anyhow::Result;
use async_trait::async_trait;

use crate::{
    engine::InferenceEngine,
    mamba::SsmBlock,
    models::Message,
    tensor::Tensor,
    transformer::{Embedding, FeedForward, LmHead, RMSNorm},
};
use tokio::sync::Mutex;

/// Represents a hybrid BitMamba2 architecture native inference engine.
/// This utilizes zero external dependencies, running solely on our `tensor.rs` math primitives.
pub struct CrimsonEngineAdapter {
    // Architecture Blocks
    embedding: Embedding,
    norm: RMSNorm,
    ssm: Mutex<SsmBlock>,
    ffn: FeedForward,
    lm_head: LmHead,
}

impl CrimsonEngineAdapter {
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        expand_size: usize,
        state_size: usize,
    ) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, hidden_size),
            norm: RMSNorm::new(hidden_size),
            // We use Mutex here because SSM is stateful (carries `h_state` across tokens)
            ssm: Mutex::new(SsmBlock::new(hidden_size, expand_size, state_size)),
            ffn: FeedForward::new(hidden_size, intermediate_size),
            lm_head: LmHead::new(hidden_size, vocab_size),
        }
    }
}

#[async_trait]
impl InferenceEngine for CrimsonEngineAdapter {
    async fn generate(&self, _messages: &[Message]) -> Result<String> {
        // 1. Tokenize messages (Placeholder)
        let tokens = vec![1, 2, 3]; // Example token IDs

        let mut ssm = self.ssm.lock().await;
        // Ensure state is clean before processing a new sequence
        ssm.reset_state();

        // 2. We keep track of the final logits to sample from
        let mut _logits = Tensor::zeros(vec![1, 1]); // Dummy init

        // 3. Forward Pass Loop (Iterating over sequence)
        for token_id in tokens {
            // a. Embed Token -> (1, hidden_size)
            let mut activation = self.embedding.forward(token_id);

            // b. Apply Pre-Norm (We must copy to preserve the residual identity)
            let mut norm_activation = activation.clone();
            self.norm.forward(&mut norm_activation);

            // c. BitMamba SSM Step (Handles temporal dynamics + mixing)
            let ssm_out = ssm.forward_step(&norm_activation);

            // d. Residual Connection 1 (Add to original un-normalized activation)
            activation.add_assign(&ssm_out);

            // e. Post-Norm
            let mut norm_activation_2 = activation.clone();
            self.norm.forward(&mut norm_activation_2);

            // f. FeedForward (SwiGLU)
            let ffn_out = self.ffn.forward(&norm_activation_2);

            // g. Residual Connection 2
            activation.add_assign(&ffn_out);

            // h. Final pre-head norm
            self.norm.forward(&mut activation);

            // i. Final Language Model Head projection -> (1, vocab_size)
            _logits = self.lm_head.forward(&activation);

            // In a full runtime, we would sample the next token from `logits` here
            // using argmax or temperature sampling, and feed it back into the loop.
        }

        Ok("Crimson-Core BitMamba2 Engine fully connected. Awaiting real 1.58-bit Safetensor weights!".to_string())
    }
}
