use serde::{Deserialize, Serialize};

/// Type of memory fact
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FactType {
    World,
    Experience,
    MentalModel,
}

/// A single memory unit (fact)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUnit {
    pub id: String,
    pub bank_id: String,
    pub fact_type: FactType,
    pub content: String,
    pub summary: Option<String>,
    pub source_document_id: Option<String>,
    pub confidence: f32,
    pub tags: Vec<String>,
    pub occurred_start: Option<String>,
    pub occurred_end: Option<String>,
    pub mentioned_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Type of link between memories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LinkType {
    Semantic,
    Temporal,
    Entity,
    Causal,
}

/// A link between two memory units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub link_type: LinkType,
    pub weight: f32,
    pub created_at: String,
}

impl AsRef<str> for FactType {
    fn as_ref(&self) -> &str {
        match self {
            FactType::World => "world",
            FactType::Experience => "experience",
            FactType::MentalModel => "mental_model",
        }
    }
}
