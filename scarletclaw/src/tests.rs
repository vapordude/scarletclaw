#[cfg(test)]
mod tests {
    use crate::{
        agent::Agent,
        engine::DummyEngine,
        sandbox::{Sandbox, SandboxConfig},
    };
    use std::sync::Arc;

    #[tokio::test]
    async fn test_agent_chat() {
        let sandbox_config = SandboxConfig::default();
        let sandbox = Sandbox::new(sandbox_config);
        let engine = Arc::new(DummyEngine);

        let mut agent = Agent::new(engine, sandbox);

        // Setup system prompt
        agent.set_system_prompt("You are a helpful assistant.");

        // Have a chat
        let response = agent.chat("Hello!").await.unwrap();

        // Assert response matches our Dummy Engine behavior
        assert_eq!(response, "(Dummy Engine Reply to: 'Hello!')");
    }

    #[test]
    fn test_sandbox_default_blocks_execution() {
        let sandbox = Sandbox::new(SandboxConfig::default());
        let result = sandbox.execute_command("ls -la");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Security violation: shell execution is disabled"));
    }

    #[test]
    fn test_sandbox_allows_execution_if_configured() {
        let mut config = SandboxConfig::default();
        config.allow_shell_execution = true;
        let sandbox = Sandbox::new(config);

        let result = sandbox.execute_command("ls -la");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Executed: ls -la");
    }
}
