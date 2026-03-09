use anyhow::Result;
use std::sync::Arc;

use crate::{
    engine::InferenceEngine,
    models::Message,
    sandbox::Sandbox,
};

pub struct Agent {
    engine: Arc<dyn InferenceEngine>,
    sandbox: Sandbox,
    history: Vec<Message>,
}

impl Agent {
    pub fn new(engine: Arc<dyn InferenceEngine>, sandbox: Sandbox) -> Self {
        Self {
            engine,
            sandbox,
            history: Vec::new(),
        }
    }

    /// Appends a system prompt to guide the agent behavior safely.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.history.push(Message::system(prompt));
    }

    /// Process a new user message and generate an assistant reply.
    pub async fn chat(&mut self, message: &str) -> Result<String> {
        self.history.push(Message::user(message));

        let response = self.engine.generate(&self.history).await?;
        self.history.push(Message::assistant(&response));

        Ok(response)
    }

    // Agent could use the sandbox safely:
    // This demonstrates an architectural pattern where we route *all*
    // actions through the secure Sandbox component.
    pub fn try_run_command(&self, cmd: &str) -> Result<String> {
        self.sandbox.execute_command(cmd)
    }
}
