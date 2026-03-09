use crate::models::Message;

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
