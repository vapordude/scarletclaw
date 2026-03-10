use anyhow::{Result, bail};
use async_trait::async_trait;
use std::fs;
use std::process::Command;

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
#[allow(dead_code)]
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
        if path.contains("..") || path.starts_with('/') {
            bail!("Invalid path: absolute paths and traversal sequences are not allowed");
        }
        sandbox.read_file(path)
    }
}

/// A tool allowing the agent to write raw Rust code, compile it to WASM,
/// and execute it. In a Level 6 agent, this permits unbounded dynamic capability
/// creation while keeping the execution itself safely bounded within the WasmSandbox.
#[allow(dead_code)]
pub struct WriteAndCompileWasmTool;

#[async_trait]
impl Tool for WriteAndCompileWasmTool {
    fn name(&self) -> &str {
        "compile_and_run_wasm"
    }

    fn description(&self) -> &str {
        "Compiles raw Rust source code into a WASM module and executes it safely. Args: <raw rust source code>"
    }

    async fn execute(&self, sandbox: &Sandbox, args: &str) -> Result<String> {
        let agent_code = args;

        // Block simple attempts to exfiltrate host data via rustc macros
        if agent_code.contains("include_str!") || agent_code.contains("include_bytes!") || agent_code.contains("env!") {
            bail!("Security violation: compile-time environment/file macros are disabled for dynamic tools.");
        }

        // Ensure the host has rustc installed for this dynamic capability
        if Command::new("rustc").arg("--version").output().is_err() {
            bail!("rustc is not available on the host to compile dynamic tools.");
        }

        // We wrap the agent's code in standard WASM cdylib boilerplate
        // so the agent only has to write the internal logic of `pub fn run() -> ...`
        let full_source = format!(
            "
            #[no_mangle]
            pub extern \"C\" fn run() {{
                {}
            }}
            ",
            agent_code
        );

        // 1. Write the agent's code to a unique temporary file
        let temp_dir = std::env::temp_dir().join("scarletclaw_wasm");
        fs::create_dir_all(&temp_dir)?;
        let run_id = uuid::Uuid::new_v4();
        let src_path = temp_dir.join(format!("dynamic_tool_{}.rs", run_id));
        let out_path = temp_dir.join(format!("dynamic_tool_{}.wasm", run_id));

        fs::write(&src_path, full_source)?;

        // 2. Compile the source to WASM targeting wasm32-unknown-unknown
        // We use standard Command here, NOT the agent's sandbox, because compilation
        // is an authorized host-level orchestration action triggered by this specific tool.
        let compile_output = Command::new("rustc")
            .arg("--target")
            .arg("wasm32-unknown-unknown")
            .arg("-O")
            .arg("--crate-type")
            .arg("cdylib")
            .arg("-o")
            .arg(&out_path)
            .arg(&src_path)
            .output()?;

        if !compile_output.status.success() {
            let stderr = String::from_utf8_lossy(&compile_output.stderr);
            bail!("WASM Compilation failed: {}", stderr);
        }

        // 3. Read the compiled WASM binary
        let wasm_bytes = fs::read(&out_path)?;

        // 4. Pass the binary into the secure WasmSandbox for execution!
        // This is where the actual un-trusted agent logic runs.
        let execution_result = sandbox.wasm().execute_wasm_module(&wasm_bytes, "");

        // Cleanup
        let _ = fs::remove_file(src_path);
        let _ = fs::remove_file(out_path);

        execution_result
    }
}
