use async_trait::async_trait;
use crate::models::Message;

/// Represents a persistent episodic memory system (e.g., Vector DB or SQLite).
/// This allows the agent to recall facts over long time horizons.
#[async_trait]
pub trait EpisodicMemory: Send + Sync {
    async fn store_memory(&self, content: &str) -> anyhow::Result<()>;
    async fn recall_memories(&self, query: &str, limit: usize) -> anyhow::Result<Vec<String>>;
}

/// A dummy in-memory implementation of long-term memory.
pub struct DummyEpisodicMemory {
    memories: std::sync::Mutex<Vec<String>>,
}

impl DummyEpisodicMemory {
    pub fn new() -> Self {
        Self {
            memories: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl EpisodicMemory for DummyEpisodicMemory {
    async fn store_memory(&self, content: &str) -> anyhow::Result<()> {
        let mut mems = self.memories.lock().unwrap();
        mems.push(content.to_string());
        Ok(())
    }

    async fn recall_memories(&self, query: &str, limit: usize) -> anyhow::Result<Vec<String>> {
        let mems = self.memories.lock().unwrap();
        // Extremely naive "search" for demonstration purposes.
        let results = mems.iter()
            .filter(|m| m.contains(query))
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }
}

/// Represents the short-term working context window of the agent.
pub struct Memory {
    messages: Vec<Message>,
    max_tokens: Option<usize>, // A simple token limit simulation
}

impl Memory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: None,
        }
    }

    pub fn with_limit(max_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: Some(max_tokens),
        }
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
        self.prune();
    }

    pub fn get_context(&self) -> &[Message] {
        &self.messages
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Very simple pruning logic based on a rough character count (simulating tokens).
    /// In a real system, you would use a tokenizer like `tiktoken`.
    fn prune(&mut self) {
        if let Some(limit) = self.max_tokens {
            // Rough estimate: 4 chars = 1 token
            let char_limit = limit * 4;

            let mut current_chars: usize = self.messages.iter().map(|m| m.content.len()).sum();

            while current_chars > char_limit && self.messages.len() > 1 {
                // Never remove the first system message if it exists
                let index_to_remove = if self.messages[0].role == crate::models::Role::System {
                    1
                } else {
                    0
                };

                if index_to_remove < self.messages.len() {
                    let removed = self.messages.remove(index_to_remove);
                    current_chars -= removed.content.len();
                } else {
                    break;
                }
            }
        }
    }
}
