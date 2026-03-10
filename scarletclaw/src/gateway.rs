use anyhow::Result;
use axum::{
    routing::{get, post},
    Router, Json,
    extract::State,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agent::Agent;

/// A simple gateway state managing a shared agent context.
/// In a real scenario, you'd likely map connection IDs to different Agent sessions.
struct GatewayState {
    agent: Arc<Mutex<Agent>>,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    message: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    reply: String,
}

pub struct Gateway {
    port: u16,
    agent: Arc<Mutex<Agent>>,
}

impl Gateway {
    pub fn new(port: u16, agent: Agent) -> Self {
        Self {
            port,
            agent: Arc::new(Mutex::new(agent)),
        }
    }

    pub async fn run(&self) -> Result<()> {
        let state = Arc::new(GatewayState {
            agent: self.agent.clone(),
        });

        let app = Router::new()
            .route("/health", get(health_check))
            .route("/chat", post(chat_endpoint))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", self.port)).await?;
        println!("🚀 Gateway listening on {}", listener.local_addr()?);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn health_check() -> &'static str {
    "OK"
}

async fn chat_endpoint(
    State(state): State<Arc<GatewayState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let mut agent = state.agent.lock().await;

    let reply = match agent.chat(&payload.message).await {
        Ok(r) => r,
        Err(e) => format!("Error communicating with agent: {}", e),
    };

    Json(ChatResponse { reply })
}
