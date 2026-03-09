use anyhow::Result;
use async_trait::async_trait;

use crate::sandbox::Sandbox;

/// A trait defining a safe tool that the agent can execute within the sandbox.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The unique identifier or name of the tool (e.g., "read_file")
    fn name(&self) -> &str;

    /// A description explaining to the agent when and how to use this tool
    fn description(&self) -> &str;

    /// Execute the tool. It is provided a reference to the secure sandbox
    /// and any arguments passed by the agent.
    async fn execute(&self, sandbox: &Sandbox, args: &str) -> Result<String>;
}

/// A safe tool for reading files, governed by Sandbox constraints.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the contents of a file at the given path."
    }

    async fn execute(&self, sandbox: &Sandbox, args: &str) -> Result<String> {
        let path = args.trim();
        sandbox.read_file(path)
    }
}
