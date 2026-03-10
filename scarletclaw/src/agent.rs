use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::{
    engine::InferenceEngine,
    memory::{DummyEpisodicMemory, EpisodicMemory, Memory},
    models::Message,
    react::AgentThought,
    sandbox::Sandbox,
    tools::Tool,
};

/// Events that can be sent to the Agent's inbox.
pub enum AgentEvent {
    /// A new chat message from a user or channel.
    UserMessage {
        content: String,
        /// Optional channel to send the reply back to the caller.
        reply_tx: Option<tokio::sync::oneshot::Sender<String>>,
    },
    /// A system-level event or cron trigger.
    SystemTrigger { description: String },
    /// Command to cleanly shutdown the agent loop.
    Shutdown,
}

pub struct Agent {
    engine: Arc<dyn InferenceEngine>,
    sandbox: Sandbox,
    memory: Memory,
    episodic_memory: Arc<dyn EpisodicMemory>,
    tools: HashMap<String, Box<dyn Tool>>,
}

impl Agent {
    pub fn new(engine: Arc<dyn InferenceEngine>, sandbox: Sandbox) -> Self {
        Self {
            engine,
            sandbox,
            memory: Memory::with_limit(4000), // Default limit for now
            episodic_memory: Arc::new(DummyEpisodicMemory::new()),
            tools: HashMap::new(),
        }
    }

    pub fn with_episodic_memory(mut self, mem: Arc<dyn EpisodicMemory>) -> Self {
        self.episodic_memory = mem;
        self
    }

    /// Registers a tool for the agent to use securely.
    pub fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Appends a system prompt to guide the agent behavior safely.
    pub fn add_system_prompt(&mut self, prompt: &str) {
        self.memory.push(Message::system(prompt));
    }

    /// Spawns the agent as a continuous, autonomous background task.
    /// Returns a Sender to drop events into its inbox, and a JoinHandle.
    pub fn spawn(mut self) -> (mpsc::Sender<AgentEvent>, JoinHandle<()>) {
        let (tx, mut rx) = mpsc::channel::<AgentEvent>(100);

        let handle = tokio::spawn(async move {
            println!("🤖 Agent autonomous loop started.");

            while let Some(event) = rx.recv().await {
                match event {
                    AgentEvent::UserMessage { content, reply_tx } => {
                        println!("🤖 Agent received user message: {}", content);
                        match self.process_turn(&content).await {
                            Ok(reply) => {
                                if let Some(tx) = reply_tx {
                                    let _ = tx.send(reply);
                                }
                            }
                            Err(e) => {
                                eprintln!("Agent error processing turn: {}", e);
                                if let Some(tx) = reply_tx {
                                    let _ = tx.send(format!("Error: {}", e));
                                }
                            }
                        }
                    }
                    AgentEvent::SystemTrigger { description } => {
                        println!("🤖 Agent received system trigger: {}", description);
                        let prompt = format!(
                            "System Event Triggered: {}. What should you do?",
                            description
                        );
                        let _ = self.process_turn(&prompt).await;
                    }
                    AgentEvent::Shutdown => {
                        println!("🤖 Agent shutting down.");
                        break;
                    }
                }
            }
        });

        (tx, handle)
    }

    /// Internal method to process a single conversational turn.
    /// Uses a basic ReAct loop (Reason -> Act -> Observe -> Repeat) up to a max step limit.
    async fn process_turn(&mut self, message: &str) -> Result<String> {
        self.memory.push(Message::user(message));

        let max_steps = 5;
        let mut current_step = 0;

        while current_step < max_steps {
            current_step += 1;

            // 1. Generate the next thought/action from the engine.
            let raw_response = self.engine.generate(self.memory.get_context()).await?;
            self.memory.push(Message::assistant(&raw_response));

            // Attempt to parse the response as JSON ReAct structure
            // Fallback to raw text generation if parsing fails (for dummy engine/testing)
            match AgentThought::parse(&raw_response) {
                Ok(thought) => {
                    // Check if the agent wants to act
                    if let Some(action) = thought.action {
                        println!("Agent Thought: {}", thought.thought);
                        println!(
                            "Agent executing Tool: {} with args '{}'",
                            action.tool_name, action.args
                        );

                        let tool_observation = if let Some(tool) = self.tools.get(&action.tool_name)
                        {
                            match tool.execute(&self.sandbox, &action.args).await {
                                Ok(res) => {
                                    format!("Observation: Execution successful. Output: {}", res)
                                }
                                Err(e) => format!("Observation: Execution failed. Error: {}", e),
                            }
                        } else {
                            format!("Observation: Tool '{}' does not exist.", action.tool_name)
                        };

                        // Push the observation back into memory and loop again
                        self.memory.push(Message::system(&tool_observation));
                    } else if let Some(final_response) = thought.response {
                        // Task complete! Return the final response to the user.
                        return Ok(final_response);
                    } else {
                        // The agent just thought but didn't act or reply. Ask it to continue.
                        self.memory.push(Message::system("You thought without acting or replying. Please produce a final response or take an action."));
                    }
                }
                Err(_) => {
                    // If we couldn't parse it as JSON, we just treat it as a conversational
                    // response. E.g., Dummy engine output will hit this.
                    return Ok(raw_response);
                }
            }
        }

        Ok("Agent exceeded maximum thinking steps before reaching a conclusion.".to_string())
    }
}
