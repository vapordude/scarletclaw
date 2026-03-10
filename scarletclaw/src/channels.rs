use anyhow::Result;
use async_trait::async_trait;

/// Represents a communication channel where ScarletClaw listens for and sends messages.
/// Examples: Discord, Slack, Telegram, Webhook
#[async_trait]
pub trait Channel: Send + Sync {
    /// Start listening on this channel for incoming events.
    async fn listen(&self) -> Result<()>;

    /// Send a message back to a specific target within this channel.
    async fn send(&self, target_id: &str, message: &str) -> Result<()>;
}

/// A dummy webhook channel used for local testing.
pub struct WebhookChannel {
    pub name: String,
}

impl WebhookChannel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl Channel for WebhookChannel {
    async fn listen(&self) -> Result<()> {
        println!(
            "[{}] Channel is listening for incoming webhooks...",
            self.name
        );
        // In reality, this would bind a port or hook into the Gateway to receive HTTP requests.
        Ok(())
    }

    async fn send(&self, target_id: &str, message: &str) -> Result<()> {
        println!(
            "[{}] Sending message to {}: {}",
            self.name, target_id, message
        );
        Ok(())
    }
}
