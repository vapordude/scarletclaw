use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

use crate::agent::AgentEvent;

/// A simple gateway state managing a shared agent context.
/// In a real scenario, you'd likely map connection IDs to different Agent sessions.
struct GatewayState {
    inbox: Sender<AgentEvent>,
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
    inbox: Sender<AgentEvent>,
}

impl Gateway {
    pub fn new(port: u16, inbox: Sender<AgentEvent>) -> Self {
        Self { port, inbox }
    }

    pub async fn run(&self) -> Result<()> {
        let state = Arc::new(GatewayState {
            inbox: self.inbox.clone(),
        });

        let app = Router::new()
            .route("/health", get(health_check))
            .route("/chat", post(chat_endpoint))
            .with_state(state);

        // By default, only bind to loopback for security.
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", self.port)).await?;
        println!("🚀 Gateway listening on {}", listener.local_addr()?);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn health_check() -> &'static str {
    "OK"
}

use axum::http::StatusCode;

async fn chat_endpoint(
    State(state): State<Arc<GatewayState>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    // Send event asynchronously to the agent's background loop
    let event = AgentEvent::UserMessage {
        content: payload.message,
        reply_tx: Some(tx),
    };

    // Use a short timeout for enqueueing the event (the inbox might be full/blocked)
    if let Err(e) = tokio::time::timeout(std::time::Duration::from_secs(5), state.inbox.send(event)).await {
        return Err((StatusCode::SERVICE_UNAVAILABLE, format!("Timeout queueing message to agent: {}", e)));
    }

    // Await the response from the agent loop with a strict timeout (e.g. 60s)
    let reply = match tokio::time::timeout(std::time::Duration::from_secs(60), rx).await {
        Ok(Ok(res)) => res,
        Ok(Err(_)) => return Err((StatusCode::INTERNAL_SERVER_ERROR, "Agent channel dropped the request.".to_string())),
        Err(_) => return Err((StatusCode::GATEWAY_TIMEOUT, "Agent timed out computing a response.".to_string())),
    };

    Ok(Json(ChatResponse { reply }))
}
