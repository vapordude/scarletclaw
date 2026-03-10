use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;

use scarletclaw::{
    Agent, AgentEvent, CrimsonEngineAdapter, DummyEngine, Gateway, Sandbox, SandboxConfig,
    Scheduler, SqliteEpisodicMemory, InferenceEngine,
};

/// ScarletClaw - The native Rust local AI assistant
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start a chat session with the assistant
    Chat {
        /// Optional system prompt to initialize the agent
        #[arg(short, long)]
        system: Option<String>,

        /// Whether to enable unsafe shell command execution (use with extreme caution!)
        #[arg(long, default_value_t = false)]
        unsafe_shell: bool,

        /// Developer explicit opt-in confirmation required for unsafe operations
        #[arg(long, default_value_t = false)]
        dev_confirm_unsafe: bool,

        /// Automatically trigger background reasoning loops every N seconds
        #[arg(long)]
        cron_interval: Option<u64>,

        /// Use the experimental BitMamba-2 mathematical engine instead of Dummy Engine
        #[arg(long, default_value_t = false)]
        use_crimson_engine: bool,
    },

    /// Start the Gateway server to accept remote connections
    Gateway {
        /// Port to run the server on
        #[arg(short, long, default_value_t = 18789)]
        port: u16,

        /// Optional system prompt to initialize the agent
        #[arg(short, long)]
        system: Option<String>,
    },

    /// Test the sandbox security constraints
    TestSandbox {
        #[arg(short, long)]
        command: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Chat {
            system,
            unsafe_shell,
            dev_confirm_unsafe,
            cron_interval,
            use_crimson_engine,
        } => {
            println!("🦀 Welcome to ScarletClaw!");
            let mut config = SandboxConfig::default();

            if *unsafe_shell {
                if *dev_confirm_unsafe {
                    println!("⚠️ WARNING: Unsafe shell execution is ENABLED.");
                    config.allow_shell_execution = true;
                } else {
                    println!("🛡️ '--unsafe-shell' passed without '--dev-confirm-unsafe'. Shell execution remains DISABLED.");
                }
            }

            let sandbox = Sandbox::new(config);

            let engine: Arc<dyn InferenceEngine> = if *use_crimson_engine {
                println!("🧠 Initializing hybrid BitMamba-2 mathematical engine...");
                // Dummy dimension values matching typical 0.25b models roughly
                // vocab_size: 32000, hidden: 512, intermediate: 1024, expand: 1024, state: 64
                Arc::new(CrimsonEngineAdapter::new(32000, 512, 1024, 1024, 64))
            } else {
                Arc::new(DummyEngine)
            };

            let memory_db =
                SqliteEpisodicMemory::new("scarletclaw_memory.db").expect("Failed to init DB");

            let mut agent = Agent::new(engine, sandbox).with_episodic_memory(Arc::new(memory_db));

            if let Some(sys) = system {
                agent.add_system_prompt(sys);
            }

            let (tx, _handle) = agent.spawn();

            if let Some(secs) = cron_interval {
                let scheduler = Scheduler::new(tx.clone(), *secs);
                scheduler.spawn();
            }

            println!("Type your message below (type 'exit' to quit):");
            loop {
                use std::io::{self, Write};
                print!("> ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();

                if input.eq_ignore_ascii_case("exit") {
                    let _ = tx.send(AgentEvent::Shutdown).await;
                    println!("Goodbye!");
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                let event = AgentEvent::UserMessage {
                    content: input.to_string(),
                    reply_tx: Some(reply_tx),
                };

                if tx.send(event).await.is_err() {
                    println!("Agent inbox closed unexpectedly.");
                    break;
                }

                if let Ok(reply) = reply_rx.await {
                    println!("🤖 {}", reply);
                } else {
                    println!("Failed to get reply from agent.");
                }
            }
        }
        Commands::Gateway { port, system } => {
            let config = SandboxConfig::default();
            let sandbox = Sandbox::new(config);
            let engine = Arc::new(DummyEngine);

            let memory_db =
                SqliteEpisodicMemory::new("scarletclaw_memory.db").expect("Failed to init DB");

            let mut agent = Agent::new(engine, sandbox).with_episodic_memory(Arc::new(memory_db));

            if let Some(sys) = system {
                agent.add_system_prompt(sys);
            }

            let (tx, agent_handle) = agent.spawn();
            let gateway = Gateway::new(*port, tx.clone());

            // Run the gateway server. If it errors out or exits, we shut down the agent.
            if let Err(e) = gateway.run().await {
                println!("Gateway exited with error: {}", e);
            }

            println!("Shutting down agent loop...");
            let _ = tx.send(AgentEvent::Shutdown).await;

            // Wait for agent to exit properly.
            if let Err(e) = agent_handle.await {
                println!("Agent task panicked: {}", e);
            }
        }
        Commands::TestSandbox { command } => {
            let config = SandboxConfig::default();
            let sandbox = Sandbox::new(config);

            println!(
                "Attempting to execute '{}' in a default sandbox...",
                command
            );
            match sandbox.execute_command(command) {
                Ok(res) => println!("Success: {}", res),
                Err(e) => println!("Sandbox blocked the action successfully: {}", e),
            }
        }
    }

    Ok(())
}
