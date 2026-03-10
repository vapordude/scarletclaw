use serde::{Deserialize, Serialize};

/// Represents the structured output of an agent during the ReAct loop.
/// The agent should output JSON matching this structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentThought {
    /// The agent's internal monologue, reasoning, or planning.
    pub thought: String,

    /// A chosen action/tool to execute, if any.
    pub action: Option<AgentAction>,

    /// The final response to send back to the user, if the task is complete.
    pub response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    pub tool_name: String,
    pub args: String,
}

impl AgentThought {
    /// Helper to parse the JSON string back into the struct.
    pub fn parse(json_str: &str) -> anyhow::Result<Self> {
        let thought: Self = serde_json::from_str(json_str)?;

        // Enforce the ReAct invariant: it must either act OR respond, not both or neither.
        if thought.action.is_some() == thought.response.is_some() {
            anyhow::bail!("Malformed ReAct JSON: Agent must provide exactly one of 'action' or 'response'.");
        }

        Ok(thought)
    }
}
