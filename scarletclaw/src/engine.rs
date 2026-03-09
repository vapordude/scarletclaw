use anyhow::Result;
use async_trait::async_trait;

use crate::models::Message;

/// A generic inference engine trait.
/// This allows us to swap out cloud providers (like OpenAI/Anthropic)
/// with a local native Rust engine like `crimson-core` when ready.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn generate(&self, messages: &[Message]) -> Result<String>;
}

pub struct DummyEngine;

#[async_trait]
impl InferenceEngine for DummyEngine {
    async fn generate(&self, messages: &[Message]) -> Result<String> {
        let last_msg = messages.last().map(|m| m.content.as_str()).unwrap_or("");
        Ok(format!("(Dummy Engine Reply to: '{}')", last_msg))
    }
}
