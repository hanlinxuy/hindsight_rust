use crate::error::{HindsightError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ── Public types for tool calling ──────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// Messages in a conversation (supports tool calling flow)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: String,
        #[serde(rename = "tool_call_id")]
        tool_call_id: String,
    },
}

/// Result from a chat completion that may contain text or tool calls
#[derive(Debug)]
pub enum ChatCompletion {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

// ── Internal API types ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// ── Client implementation ──────────────────────────────────────────────

/// Unified LLM client for remote API calls (OpenAI-compatible)
pub struct LlmClient {
    http: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl LlmClient {
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .connect_timeout(Duration::from_secs(15))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            http,
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Simple chat completion (system + user → text response)
    pub async fn chat(&self, system: &str, user: &str) -> Result<String> {
        let messages = vec![
            ChatMessage::System { content: system.into() },
            ChatMessage::User { content: user.into() },
        ];
        match self.chat_raw(messages, None).await? {
            ChatCompletion::Text(t) => Ok(t),
            ChatCompletion::ToolCalls(_) => Err(HindsightError::Llm("Unexpected tool call in simple chat".into())),
        }
    }

    /// Chat completion with tools support
    pub async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: &[ToolDefinition],
    ) -> Result<ChatCompletion> {
        self.chat_raw(messages, Some(tools.to_vec())).await
    }

    /// Raw chat completion API call
    async fn chat_raw(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<ChatCompletion> {
        let req = ChatRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.3,
            tools,
        };

        let resp = self
            .http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(HindsightError::Llm(format!("LLM API error {}: {}", status, &body[..body.len().min(500)])));
        }

        let chat_resp: ChatResponse = resp.json().await?;
        let choice = chat_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| HindsightError::Llm("No response from LLM".into()))?;

        // If tool calls present, return them
        if let Some(tool_calls) = choice.message.tool_calls {
            if !tool_calls.is_empty() {
                return Ok(ChatCompletion::ToolCalls(tool_calls));
            }
        }

        // Otherwise return text
        choice
            .message
            .content
            .map(ChatCompletion::Text)
            .ok_or_else(|| HindsightError::Llm("Empty response from LLM".into()))
    }

    /// Get embeddings for a batch of texts
    pub async fn embed(
        &self,
        texts: &[String],
        embedding_model: &str,
        base_url: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let req = EmbeddingRequest {
            model: embedding_model.to_string(),
            input: texts.to_vec(),
        };

        let resp = self
            .http
            .post(format!("{}/v1/embeddings", base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(HindsightError::Llm(format!("Embedding API error {}: {}", status, body)));
        }

        let emb_resp: EmbeddingResponse = resp.json().await?;
        if emb_resp.data.len() != texts.len() {
            return Err(HindsightError::Llm(format!(
                "Embedding count mismatch: expected {}, got {}",
                texts.len(),
                emb_resp.data.len()
            )));
        }
        Ok(emb_resp.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Rerank documents using Cohere/Jina-compatible API
    pub async fn rerank(&self, query: &str, documents: &[String], base_url: &str) -> Result<Vec<usize>> {
        #[derive(Serialize)]
        struct RerankRequest {
            model: String,
            query: String,
            documents: Vec<String>,
            top_n: usize,
        }

        #[derive(Deserialize)]
        struct RerankResponse {
            results: Vec<RerankResult>,
        }

        #[derive(Deserialize)]
        struct RerankResult {
            index: usize,
        }

        let req = RerankRequest {
            model: "rerank-v3.5".to_string(),
            query: query.to_string(),
            documents: documents.to_vec(),
            top_n: documents.len().min(20),
        };

        let resp = self
            .http
            .post(format!("{}/v1/rerank", base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(HindsightError::Llm(format!("Rerank API error {}: {}", status, body)));
        }

        let rerank_resp: RerankResponse = resp.json().await?;
        Ok(rerank_resp.results.into_iter().map(|r| r.index).collect())
    }
}
