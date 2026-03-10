use std::time::Duration;
use tokio::sync::mpsc::Sender;
use tokio::time::interval;

use crate::agent::AgentEvent;

/// A simple background scheduler that wakes the agent up periodically
/// so it can act autonomously even when the user isn't interacting with it.
pub struct Scheduler {
    inbox: Sender<AgentEvent>,
    interval_seconds: u64,
}

impl Scheduler {
    pub fn new(inbox: Sender<AgentEvent>, interval_seconds: u64) -> Self {
        Self {
            inbox,
            interval_seconds,
        }
    }

    /// Spawns the scheduler as a background task.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(self.interval_seconds));

            // Skip the immediate first tick so we don't spam on startup
            ticker.tick().await;

            println!("⏰ Scheduler started. Agent will be pinged every {} seconds.", self.interval_seconds);

            loop {
                ticker.tick().await;

                let event = AgentEvent::SystemTrigger {
                    description: "Periodic internal check-in. Review your memory, plan your next actions, or report anything notable.".to_string(),
                };

                // If the inbox is closed, the agent has shut down, so we should too.
                if self.inbox.send(event).await.is_err() {
                    println!("⏰ Scheduler shutting down (Agent inbox closed).");
                    break;
                }
            }
        })
    }
}
