use anyhow::Result;
use async_trait::async_trait;

use crate::{
    engine::InferenceEngine,
    models::Message,
    tensor::Tensor,
    transformer::{FeedForward, RMSNorm},
    mamba::SsmBlock,
};
use tokio::sync::Mutex;

/// Represents a hybrid BitMamba2 architecture native inference engine.
/// This utilizes zero external dependencies, running solely on our `tensor.rs` math primitives.
pub struct CrimsonEngineAdapter {
    hidden_size: usize,

    // Architecture Blocks
    norm: RMSNorm,
    ssm: Mutex<SsmBlock>,
    ffn: FeedForward,
}

impl CrimsonEngineAdapter {
    pub fn new(hidden_size: usize, intermediate_size: usize, expand_size: usize, state_size: usize) -> Self {
        Self {
            hidden_size,
            norm: RMSNorm::new(hidden_size),
            // We use Mutex here because SSM is stateful (carries `h_state` across tokens)
            ssm: Mutex::new(SsmBlock::new(hidden_size, expand_size, state_size)),
            ffn: FeedForward::new(hidden_size, intermediate_size),
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

        // 2. Allocate an activation tensor (batch_size=1, hidden_size)
        let mut activation = Tensor::zeros(vec![1, self.hidden_size]);

        // 3. Forward Pass Loop (Iterating over sequence)
        for _token in tokens {
            // a. Apply Pre-Norm
            self.norm.forward(&mut activation);

            // b. BitMamba SSM Step (Handles temporal dynamics + mixing)
            let ssm_out = ssm.forward_step(&activation);

            // c. Residual Connection 1
            activation.add_assign(&ssm_out);

            // d. Post-Norm
            self.norm.forward(&mut activation);

            // e. FeedForward (SwiGLU)
            let ffn_out = self.ffn.forward(&activation);

            // f. Residual Connection 2
            activation.add_assign(&ffn_out);

            // In a full implementation, we'd map `activation` to vocab logits here.
        }

        Ok("Crimson-Core BitMamba2 Engine initialized. Awaiting real 1.58-bit Safetensor weights!".to_string())
    }
}
