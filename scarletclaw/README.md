# 🦀 ScarletClaw

**ScarletClaw** is a fully native, zero-dependency, bare-metal Rust implementation of the OpenClaw AI assistant framework. It is specifically designed to operate as a **Level 6 Autonomous Agent**, capable of proactive thought, dynamic tool creation, and highly secure host isolation.

Unlike standard ML projects, ScarletClaw intentionally avoids heavy frameworks like PyTorch or Candle. Its internal engine (`crimson-core`) is built entirely from mathematical scratch using the standard library, focusing specifically on **hybrid 1.58-bit BitNet and Mamba-2 architectures**.

## 🚀 Core Architectural Features

*   **Autonomous Asynchronous Loop:** The agent does not wait for HTTP requests. It runs continuously via `tokio::spawn` with an MPSC inbox. A cron-like `Scheduler` pings the agent, allowing it to "wake up", think, and perform background tasks proactively.
*   **Zero-Dependency Mathematics (`tensor.rs`):** Implements `Tensor` and `TernaryTensor` entirely using standard `Vec<f32>` and `Vec<i8>`. Matrix multiplication for 1.58-bit weights completely avoids floating-point multiplication in favor of pure integer addition/subtraction.
*   **Mamba-2 State Space Model (`mamba.rs`):** Avoids $O(n^2)$ attention matrices by implementing the discretized linear state-space scanning equations of Mamba-2 architectures.
*   **WASM Sandbox (`wasm.rs`):** Bypasses arbitrary bash-execution vulnerabilities found in older agents. ScarletClaw's `WriteAndCompileWasmTool` allows the agent to dynamically *write its own Rust code*, compile it to WASM on the host, and execute it strictly within a mathematically bounded `wasmtime` environment.
*   **SQLite Episodic Memory:** Maintains long-term persistence across reboots.
*   **ReAct Cognitive Framework:** Forces the agent into a structured JSON `Thought -> Action -> Observation` loop before responding.

## 🛠️ Getting Started

To run the agent locally with a dummy inference engine (for testing the loop):
```bash
cargo run -- chat --system "You are a helpful Level 6 autonomous assistant."
```

To run the agent proactively with the background scheduler firing every 10 seconds:
```bash
cargo run -- chat --cron-interval 10
```

To start the Gateway for external channel communication (e.g., Slack/Discord Webhooks):
```bash
cargo run -- gateway --port 18789
```

To use the experimental zero-dependency `crimson-core` BitMamba-2 math engine:
```bash
cargo run -- chat --use-crimson-engine
```

## 🔒 Security Posture

By default, the `Sandbox` disables all host-level Shell execution and File I/O. If you want to allow the agent to execute real shell commands on your host (not recommended), you must explicitly pass:
```bash
cargo run -- chat --unsafe-shell
```