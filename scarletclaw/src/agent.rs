use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;

use crate::{
    engine::InferenceEngine,
    models::Message,
    sandbox::Sandbox,
    memory::Memory,
    tools::Tool,
};

pub struct Agent {
    engine: Arc<dyn InferenceEngine>,
    sandbox: Sandbox,
    memory: Memory,
    tools: HashMap<String, Box<dyn Tool>>,
}

impl Agent {
    pub fn new(engine: Arc<dyn InferenceEngine>, sandbox: Sandbox) -> Self {
        Self {
            engine,
            sandbox,
            memory: Memory::with_limit(4000), // Default limit for now
            tools: HashMap::new(),
        }
    }

    /// Registers a tool for the agent to use securely.
    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Appends a system prompt to guide the agent behavior safely.
    pub fn add_system_prompt(&mut self, prompt: &str) {
        self.memory.push(Message::system(prompt));
    }

    /// Process a new user message and generate an assistant reply.
    pub async fn chat(&mut self, message: &str) -> Result<String> {
        self.memory.push(Message::user(message));

        // Let the engine process the context.
        let mut response = self.engine.generate(self.memory.get_context()).await?;

        // Mock Tool Invocation Detection (very simplified parsing for now)
        // e.g. "TOOLCALL: read_file /etc/passwd"
        if response.starts_with("TOOLCALL:") {
            let parts: Vec<&str> = response.splitn(3, ' ').collect();
            if parts.len() >= 2 {
                let tool_name = parts[1];
                let tool_args = if parts.len() == 3 { parts[2] } else { "" };

                if let Some(tool) = self.tools.get(tool_name) {
                    match tool.execute(&self.sandbox, tool_args).await {
                        Ok(tool_result) => {
                            let tool_msg = format!("Tool {} result: {}", tool_name, tool_result);
                            self.memory.push(Message::assistant(&response));
                            self.memory.push(Message::system(&tool_msg));

                            // Re-feed back into engine so it interprets the result
                            response = self.engine.generate(self.memory.get_context()).await?;
                        }
                        Err(e) => {
                            let err_msg = format!("Tool execution failed: {}", e);
                            self.memory.push(Message::assistant(&response));
                            self.memory.push(Message::system(&err_msg));

                            response = self.engine.generate(self.memory.get_context()).await?;
                        }
                    }
                } else {
                    response = format!("Error: Tool '{}' not found", tool_name);
                }
            }
        }

        self.memory.push(Message::assistant(&response));

        Ok(response)
    }
}
