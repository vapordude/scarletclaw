use clap::{Parser, Subcommand};
use std::sync::Arc;
use anyhow::Result;

use scarletclaw::{
    agent::Agent,
    engine::DummyEngine,
    sandbox::{Sandbox, SandboxConfig},
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
        Commands::Chat { system, unsafe_shell } => {
            println!("🦀 Welcome to ScarletClaw!");
            let mut config = SandboxConfig::default();

            if *unsafe_shell {
                println!("⚠️ WARNING: Unsafe shell execution is ENABLED.");
                config.allow_shell_execution = true;
            }

            let sandbox = Sandbox::new(config);
            let engine = Arc::new(DummyEngine);
            let mut agent = Agent::new(engine, sandbox);

            if let Some(sys) = system {
                agent.set_system_prompt(sys);
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
                    println!("Goodbye!");
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                let reply = agent.chat(input).await?;
                println!("🤖 {}", reply);
            }
        }
        Commands::TestSandbox { command } => {
            let config = SandboxConfig::default();
            let sandbox = Sandbox::new(config);

            println!("Attempting to execute '{}' in a default sandbox...", command);
            match sandbox.execute_command(command) {
                Ok(res) => println!("Success: {}", res),
                Err(e) => println!("Sandbox blocked the action successfully: {}", e),
            }
        }
    }

    Ok(())
}
