use serde::{Deserialize, Serialize};

/// An entity extracted from memory content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub bank_id: String,
    pub name: String,
    pub entity_type: String,
    pub labels: Vec<String>,
    pub created_at: String,
}
