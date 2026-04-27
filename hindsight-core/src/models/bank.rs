use serde::{Deserialize, Serialize};

/// Disposition traits that affect reflect behavior (1-5 scale)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disposition {
    pub skepticism: u8,
    pub literalism: u8,
    pub empathy: u8,
}

impl Default for Disposition {
    fn default() -> Self {
        Self {
            skepticism: 3,
            literalism: 3,
            empathy: 3,
        }
    }
}

/// A memory bank - an isolated memory store for one user/agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bank {
    pub id: String,
    pub name: String,
    pub disposition: Disposition,
    pub background_context: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}
