use crate::memory::EpisodicMemory;
use anyhow::Result;
use async_trait::async_trait;
use rusqlite::{Connection, params};
use std::sync::Mutex;

pub struct SqliteEpisodicMemory {
    conn: Mutex<Connection>,
}

impl SqliteEpisodicMemory {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        // Initialize the episodic memory table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

#[async_trait]
impl EpisodicMemory for SqliteEpisodicMemory {
    async fn store_memory(&self, content: &str) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire memory DB lock"))?;
        conn.execute(
            "INSERT INTO memories (content) VALUES (?1)",
            params![content],
        )?;
        Ok(())
    }

    async fn recall_memories(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire memory DB lock"))?;
        let mut stmt = conn.prepare(
            "SELECT content FROM memories WHERE content LIKE ?1 ORDER BY timestamp DESC LIMIT ?2",
        )?;

        // In a real system, you'd use FTS5 (Full Text Search) or Vector embeddings.
        // For now, we use a simple LIKE query.
        let search_pattern = format!("%{}%", query);

        let limit_i64 = limit as i64;
        let memory_iter = stmt.query_map(params![search_pattern, limit_i64], |row| row.get(0))?;

        let mut results = Vec::new();
        for mem in memory_iter {
            results.push(mem?);
        }

        Ok(results)
    }
}
