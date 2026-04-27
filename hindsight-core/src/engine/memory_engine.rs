use crate::error::{HindsightError, Result};
use crate::llm::{ChatCompletion, ChatMessage, LlmClient, ToolDefinition, ToolFunction};
use crate::models::*;
use crate::storage::SqliteStore;
use std::collections::HashMap;
use tracing::info;

/// Core memory engine - orchestrates retain/recall/reflect
pub struct MemoryEngine {
    store: SqliteStore,
    llm: LlmClient,
    embedding_model: String,
    embedding_base_url: String,
}

impl MemoryEngine {
    pub fn new(store: SqliteStore, llm: LlmClient) -> Self {
        Self {
            store,
            llm,
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_base_url: "https://api.openai.com".to_string(),
        }
    }

    pub fn init(db_path: &str, api_key: &str, base_url: &str, model: &str) -> Result<Self> {
        let store = SqliteStore::open(db_path)?;
        let llm = LlmClient::new(api_key, base_url, model);
        Ok(Self {
            store,
            llm,
            embedding_base_url: base_url.to_string(),
            embedding_model: "text-embedding-3-small".to_string(),
        })
    }

    pub fn with_embedding(mut self, model: &str, base_url: &str) -> Self {
        self.embedding_model = model.to_string();
        self.embedding_base_url = base_url.to_string();
        self
    }

    // ─── RETAIN: Store content into memory bank ──────────────────────────

    pub async fn retain(&self, bank_id: &str, content: &str) -> Result<RetainResult> {
        info!("Retain: bank={}, content_len={}", bank_id, content.len());

        if self.store.get_bank(bank_id)?.is_none() {
            return Err(HindsightError::NotFound(format!("Bank {} not found", bank_id)));
        }

        // 0. Store raw document
        let doc_id = uuid::Uuid::new_v4().to_string();
        let _ = self.store.store_document(&doc_id, bank_id, content, "text/plain");

        // 1. Extract facts via LLM
        let facts = self.extract_facts(content).await?;
        info!("Retain: extracted {} facts", facts.len());

        let now = chrono::Utc::now().to_rfc3339();
        let mut memory_ids = Vec::new();

        // 2. Store each fact
        for fact in &facts {
            let id = uuid::Uuid::new_v4().to_string();
            let unit = MemoryUnit {
                id: id.clone(),
                bank_id: bank_id.to_string(),
                fact_type: fact.fact_type.clone(),
                content: fact.content.clone(),
                summary: fact.summary.clone(),
                source_document_id: Some(doc_id.clone()),
                confidence: fact.confidence.unwrap_or(1.0),
                tags: fact.tags.clone(),
                occurred_start: fact.occurred_start.clone(),
                occurred_end: fact.occurred_end.clone(),
                mentioned_at: Some(now.clone()),
                created_at: now.clone(),
                updated_at: now.clone(),
            };
            self.store.store_memory(&unit)?;
            memory_ids.push(id);
        }

        // 3. Generate embeddings for all facts
        let texts: Vec<String> = facts.iter().map(|f| f.content.clone()).collect();
        let embeddings = self.llm.embed(&texts, &self.embedding_model, &self.embedding_base_url).await;
        if let Ok(ref emb_vec) = embeddings {
            for (i, emb) in emb_vec.iter().enumerate() {
                if i < memory_ids.len() {
                    let emb_id = uuid::Uuid::new_v4().to_string();
                    let _ = self.store.store_embedding(&emb_id, &memory_ids[i], emb, &self.embedding_model);
                }
            }
            info!("Retain: stored {} embeddings", emb_vec.len());
        }

        // 4. Extract entities and associate with facts
        let entities = self.extract_entities(content).await.unwrap_or_default();
        let mut entity_ids: Vec<(String, String)> = Vec::new();
        for entity in &entities {
            let e_id = uuid::Uuid::new_v4().to_string();
            let _ = self.store.store_entity(&e_id, bank_id, &entity.name, &entity.entity_type, &entity.labels);
            entity_ids.push((e_id, entity.name.clone()));
        }

        // 4b. Link entities to facts that mention them
        for (mem_idx, fact) in facts.iter().enumerate() {
            for (e_id, e_name) in &entity_ids {
                if fact.content.to_lowercase().contains(&e_name.to_lowercase()) {
                    let _ = self.store.store_unit_entity(&memory_ids[mem_idx], e_id);
                }
            }
        }

        // 5. Build temporal links between consecutive facts
        for i in 1..memory_ids.len() {
            let link = MemoryLink {
                id: uuid::Uuid::new_v4().to_string(),
                source_id: memory_ids[i - 1].clone(),
                target_id: memory_ids[i].clone(),
                link_type: LinkType::Temporal,
                weight: 1.0,
                created_at: now.clone(),
            };
            self.store.store_link(&link)?;
        }

        // 6. Build entity links between facts that share entities
        if !entity_ids.is_empty() {
            let mut fact_entity_map: HashMap<usize, Vec<String>> = HashMap::new();
            for (i, fact) in facts.iter().enumerate() {
                for (_, e_name) in &entity_ids {
                    if fact.content.to_lowercase().contains(&e_name.to_lowercase()) {
                        fact_entity_map.entry(i).or_default().push(e_name.clone());
                    }
                }
            }
            let mut linked = std::collections::HashSet::new();
            let keys: Vec<usize> = fact_entity_map.keys().cloned().collect();
            for i in &keys {
                for j in &keys {
                    if i >= j { continue; }
                    let shared = fact_entity_map[i].iter().any(|e| fact_entity_map[j].contains(e));
                    if shared {
                        let key = (*i, *j);
                        if linked.insert(key) {
                            let link = MemoryLink {
                                id: uuid::Uuid::new_v4().to_string(),
                                source_id: memory_ids[*i].clone(),
                                target_id: memory_ids[*j].clone(),
                                link_type: LinkType::Entity,
                                weight: 0.7,
                                created_at: now.clone(),
                            };
                            self.store.store_link(&link)?;
                        }
                    }
                }
            }
        }

        // 7. Build semantic links based on embedding similarity
        if let Ok(ref emb_vec) = embeddings {
            let sem_threshold = 0.75;
            let mut sem_linked = std::collections::HashSet::new();
            for i in 0..emb_vec.len() {
                for j in (i + 1)..emb_vec.len() {
                    let sim = cosine_similarity(&emb_vec[i], &emb_vec[j]);
                    if sim >= sem_threshold {
                        let key = (i, j);
                        if sem_linked.insert(key) {
                            let link = MemoryLink {
                                id: uuid::Uuid::new_v4().to_string(),
                                source_id: memory_ids[i].clone(),
                                target_id: memory_ids[j].clone(),
                                link_type: LinkType::Semantic,
                                weight: sim,
                                created_at: now.clone(),
                            };
                            self.store.store_link(&link)?;
                        }
                    }
                }
            }
            info!("Retain: built {} semantic links", sem_linked.len());
        }

        // 8. Build causal links via LLM analysis
        if facts.len() >= 2 {
            if let Ok(causal_pairs) = self.extract_causal_links(&facts).await {
                for (src_idx, tgt_idx) in causal_pairs {
                    if src_idx < memory_ids.len() && tgt_idx < memory_ids.len() && src_idx != tgt_idx {
                        let link = MemoryLink {
                            id: uuid::Uuid::new_v4().to_string(),
                            source_id: memory_ids[src_idx].clone(),
                            target_id: memory_ids[tgt_idx].clone(),
                            link_type: LinkType::Causal,
                            weight: 0.8,
                            created_at: now.clone(),
                        };
                        self.store.store_link(&link)?;
                    }
                }
            }
        }

        Ok(RetainResult {
            document_id: doc_id,
            memory_ids,
            facts_extracted: facts.len(),
            entities_extracted: entities.len(),
        })
    }

    // ─── RECALL: Retrieve relevant memories ──────────────────────────────

    pub async fn recall(&self, bank_id: &str, query: &str, limit: usize) -> Result<Vec<ScoredMemory>> {
        self.recall_filtered(bank_id, query, limit, &[], &[], None, None).await
    }

    pub async fn recall_filtered(
        &self, bank_id: &str, query: &str, limit: usize,
        fact_types: &[FactType], tags: &[String],
        date_start: Option<&str>, date_end: Option<&str>,
    ) -> Result<Vec<ScoredMemory>> {
        info!("Recall: bank={}, query='{}', limit={}, fact_types={:?}, tags={:?}, date={:?}", bank_id, query, limit, fact_types, tags, (date_start, date_end));

        let k = 60.0;
        let mut score_map: HashMap<String, (MemoryUnit, f32)> = HashMap::new();

        // Extract temporal constraint from query if not explicitly provided
        let (ds, de) = if date_start.is_none() && date_end.is_none() {
            extract_temporal_range(query)
        } else {
            (date_start.map(|s| s.to_string()), date_end.map(|s| s.to_string()))
        };

        // Strategy 1: Full-text search (FTS5)
        if let Ok(fts_results) = self.store.search_fulltext(
            bank_id, query, limit * 2, fact_types, tags,
            ds.as_deref(), de.as_deref()
        ) {
            for (rank, unit) in fts_results.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                score_map
                    .entry(unit.id.clone())
                    .and_modify(|(_, s)| *s += rrf_score)
                    .or_insert((unit.clone(), rrf_score));
            }
            info!("Recall: FTS found {} results", fts_results.len());
        }

        // Strategy 2: Semantic (vector) search
        if let Ok(query_emb) = self.get_query_embedding(query).await {
            if let Ok(similar) = self.store.search_similar(bank_id, &query_emb, limit * 2, fact_types, tags) {
                for (rank, (unit, _sim)) in similar.iter().enumerate() {
                    let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                    score_map
                        .entry(unit.id.clone())
                        .and_modify(|(_, s)| *s += rrf_score)
                        .or_insert((unit.clone(), rrf_score));
                }
                info!("Recall: semantic search found {} results", similar.len());
            }
        }

        // Strategy 3: Graph expansion (entity + temporal + causal links)
        let mut scored_list: Vec<(String, f32)> = score_map.iter()
            .map(|(id, (_, score))| (id.clone(), *score))
            .collect();
        scored_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_count = 5.min(scored_list.len());
        let top_ids: Vec<String> = scored_list[..top_k_count].iter().map(|(id, _)| id.clone()).collect();

        for source_id in &top_ids {
            if let Ok(links) = self.store.get_links_for(source_id) {
                for link in &links {
                    let target = if link.source_id == *source_id { &link.target_id } else { &link.source_id };
                    if let Some(unit) = self.store.get_memory(target)? {
                        let rrf_score = (1.0 / 61.0) * link.weight;
                        score_map
                            .entry(unit.id.clone())
                            .and_modify(|(_, s)| *s += rrf_score)
                            .or_insert((unit.clone(), rrf_score));
                    }
                }
            }
        }

        // Strategy 4: Temporal proximity boost for memories with occurred_start
        if ds.is_some() || de.is_some() {
            for (_, (unit, score)) in score_map.iter_mut() {
                // Boost memories that have temporal info when querying with time
                if unit.occurred_start.is_some() {
                    *score += 0.01;
                }
            }
        }

        let mut results: Vec<ScoredMemory> = score_map
            .into_iter()
            .map(|(_, (unit, score))| ScoredMemory {
                unit,
                score,
                source: "rrf_fusion".to_string(),
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        // Strategy 5: Rerank top results via remote API
        if results.len() > 1 {
            results = self.rerank(query, results).await;
        }

        info!("Recall: returning {} results", results.len());
        Ok(results)
    }

    // ─── REFLECT: Agent-style multi-turn reasoning with tools ───────────

    pub async fn reflect(&self, bank_id: &str, query: &str) -> Result<ReflectResult> {
        info!("Reflect: bank={}, query='{}'", bank_id, query);

        let bank = self.store.get_bank(bank_id)?
            .ok_or_else(|| HindsightError::NotFound(format!("Bank {} not found", bank_id)))?;

        let bg_context = bank.background_context.as_deref().unwrap_or("");

        let system_prompt = format!(
            "You are a reflective memory analyst with the following disposition:
- Skepticism: {}/5 (how critically to evaluate claims)
- Literalism: {}/5 (how literally to interpret)
- Empathy: {}/5 (how empathetically to respond)
{bg_section}
You have access to tools for searching memories. Use them iteratively to gather comprehensive information before answering.
Analyze contradictions, patterns, and insights across memories.
After gathering enough information, provide your final answer WITHOUT using any tool calls.
Your final answer must be in JSON: {{\"analysis\": \"...\", \"conclusion\": \"...\", \"confidence\": 0.0-1.0}}",
            bank.disposition.skepticism,
            bank.disposition.literalism,
            bank.disposition.empathy,
            bg_section = if bg_context.is_empty() { String::new() } else { format!("\nBackground context: {}\n", bg_context) },
        );

        // Define tools the agent can use
        let tools = vec![
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "recall".into(),
                    description: "Search for relevant memories using semantic + keyword search. Returns scored memory units.".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results", "default": 10}
                        },
                        "required": ["query"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "search_mental_models".into(),
                    description: "Find consolidated mental models and insights previously synthesized from memories.".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What kind of mental models to search for"}
                        },
                        "required": ["query"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "search_observations".into(),
                    description: "Search consolidated observations (auto-generated knowledge from consolidation). Returns mental models with confidence info. Use AFTER search_mental_models.".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for in observations"}
                        },
                        "required": ["query"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "get_entity_relations".into(),
                    description: "Get all memories related to a specific named entity (person, place, concept).".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string", "description": "Name of the entity to look up"}
                        },
                        "required": ["entity_name"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "expand_context".into(),
                    description: "Get the source document and linked memories for a specific memory unit. Use this to get more context around a memory.".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "ID of the memory unit to expand"}
                        },
                        "required": ["memory_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "done".into(),
                    description: "Signal completion with your final answer. Use this when you have gathered enough information.".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Your response as well-formatted text. Include key insights and evidence."},
                            "memory_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of memory IDs that support your answer"
                            }
                        },
                        "required": ["answer"]
                    }),
                },
            },
        ];

        // Build conversation
        let mut messages: Vec<ChatMessage> = vec![
            ChatMessage::System { content: system_prompt },
            ChatMessage::User { content: query.to_string() },
        ];

        let max_iterations = 6;
        let mut total_memories_used = 0;
        let mut final_answer = String::new();

        for iteration in 0..max_iterations {
            info!("Reflect agent: iteration {}", iteration + 1);

            let result = match self.llm.chat_with_tools(messages.clone(), &tools).await {
                Ok(r) => r,
                Err(e) => {
                    info!("Reflect agent LLM error at iteration {}: {}", iteration + 1, e);
                    // Fallback to simple reflect on last iteration
                    break;
                }
            };

            match result {
                ChatCompletion::Text(text) => {
                    // LLM gave a direct answer - we're done
                    info!("Reflect agent: got text response at iteration {}", iteration + 1);
                    final_answer = text;
                    break;
                }
                ChatCompletion::ToolCalls(tool_calls) => {
                    info!("Reflect agent: {} tool calls at iteration {}", tool_calls.len(), iteration + 1);

                    // Add assistant message with tool calls to conversation
                    messages.push(ChatMessage::Assistant {
                        content: None,
                        tool_calls: Some(tool_calls.clone()),
                    });

                    // Execute each tool call
                    for tc in &tool_calls {
                        let tool_result = self.execute_reflect_tool(bank_id, &tc.function.name, &tc.function.arguments, &mut total_memories_used).await;

                        // Check if the agent called "done"
                        if tool_result.starts_with("DONE:") {
                            final_answer = tool_result[5..].to_string();
                            break;
                        }

                        messages.push(ChatMessage::Tool {
                            content: tool_result,
                            tool_call_id: tc.id.clone(),
                        });
                    }

                    // If done was called, exit the loop
                    if !final_answer.is_empty() {
                        break;
                    }
                }
            }
        }

        // If no answer yet (max iterations), do one final summary call without tools
        if final_answer.is_empty() {
            info!("Reflect agent: max iterations reached, forcing final answer");
            let memories = self.recall(bank_id, query, 10).await.unwrap_or_default();
            total_memories_used = total_memories_used.max(memories.len());
            let memory_text: String = memories.iter()
                .map(|m| format!("- [{:?}] {}", m.unit.fact_type, m.unit.content))
                .collect::<Vec<_>>()
                .join("\n");

            let fallback_prompt = format!(
                "Based on these memories, answer the query. Respond in JSON: {{\"analysis\": \"...\", \"conclusion\": \"...\", \"confidence\": 0.0-1.0}}\n\nMemories:\n{}\n\nQuery: {}",
                memory_text, query
            );
            final_answer = self.llm.chat("You are a memory analyst.", &fallback_prompt).await.unwrap_or_default();
        }

        Ok(ReflectResult {
            answer: final_answer,
            memories_used: total_memories_used,
            disposition: bank.disposition,
        })
    }

    /// Execute a single tool call from the reflect agent
    async fn execute_reflect_tool(&self, bank_id: &str, tool_name: &str, arguments: &str, memories_used: &mut usize) -> String {
        let args: serde_json::Value = serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));

        match tool_name {
            "done" => {
                // The done tool's answer is returned as the final answer
                let answer = args.get("answer").and_then(|v| v.as_str()).unwrap_or("");
                let _memory_ids: Vec<String> = args.get("memory_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default();
                format!("DONE:{}", answer)
            }
            "recall" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                // Recall only raw facts (world + experience), not mental models
                let fact_types = vec![FactType::World, FactType::Experience];
                match self.recall_filtered(bank_id, query, limit, &fact_types, &[], None, None).await {
                    Ok(results) => {
                        *memories_used += results.len();
                        let text: String = results.iter()
                            .map(|r| format!("[{:.3}] [{}] {} (id: {})", r.score, r.unit.fact_type.as_ref(), r.unit.content, r.unit.id))
                            .collect::<Vec<_>>()
                            .join("\n");
                        if text.is_empty() { "No memories found.".into() } else { text }
                    }
                    Err(e) => format!("Recall error: {}", e),
                }
            }
            "search_observations" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                // Observations are mental models — consolidated knowledge
                match self.recall_filtered(bank_id, query, 10, &[FactType::MentalModel], &[], None, None).await {
                    Ok(results) => {
                        *memories_used += results.len();
                        if results.is_empty() {
                            "No observations found.".into()
                        } else {
                            results.iter()
                                .map(|r| format!("[observation] {} (confidence: {:.2}, id: {})", r.unit.content, r.unit.confidence, r.unit.id))
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    }
                    Err(e) => format!("Search observations error: {}", e),
                }
            }
            "search_mental_models" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                match self.recall_filtered(bank_id, query, 10, &[FactType::MentalModel], &[], None, None).await {
                    Ok(results) => {
                        let models: Vec<&ScoredMemory> = results.iter()
                            .filter(|r| matches!(r.unit.fact_type, FactType::MentalModel))
                            .collect();
                        *memories_used += models.len();
                        if models.is_empty() {
                            // No mental models yet, list all facts
                            let text: String = results.iter()
                                .take(5)
                                .map(|r| format!("[{}] {}", r.unit.fact_type.as_ref(), r.unit.content))
                                .collect::<Vec<_>>()
                                .join("\n");
                            if text.is_empty() { "No mental models or memories found.".into() } else { format!("No mental models yet. Available facts:\n{}", text) }
                        } else {
                            models.iter()
                                .map(|m| format!("[mental_model] {} (confidence: {:.2})", m.unit.content, m.unit.confidence))
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    }
                    Err(e) => format!("Search error: {}", e),
                }
            }
            "get_entity_relations" => {
                let entity_name = args.get("entity_name").and_then(|v| v.as_str()).unwrap_or("");
                match self.get_related(bank_id, entity_name) {
                    Ok(memories) => {
                        *memories_used += memories.len();
                        let text: String = memories.iter()
                            .map(|m| format!("[{}] {}", m.fact_type.as_ref(), m.content))
                            .collect::<Vec<_>>()
                            .join("\n");
                        if text.is_empty() { format!("No memories found for entity '{}'.", entity_name) } else { text }
                    }
                    Err(e) => format!("Entity lookup error: {}", e),
                }
            }
            "expand_context" => {
                let memory_id = args.get("memory_id").and_then(|v| v.as_str()).unwrap_or("");
                let mut context_parts = Vec::new();

                if let Ok(Some(unit)) = self.store.get_memory(memory_id) {
                    context_parts.push(format!("Memory: [{}] {}", unit.fact_type.as_ref(), unit.content));

                    // Get source document
                    if let Some(doc_id) = &unit.source_document_id {
                        // We don't have a get_document method, just note it
                        context_parts.push(format!("Source document: {}", doc_id));
                    }

                    // Get linked memories via graph
                    if let Ok(links) = self.store.get_links_for(memory_id) {
                        for link in &links {
                            let target = if link.source_id == memory_id { &link.target_id } else { &link.source_id };
                            if let Ok(Some(linked_unit)) = self.store.get_memory(target) {
                                context_parts.push(format!("  --[{:?}]--> [{}] {}", link.link_type, linked_unit.fact_type.as_ref(), linked_unit.content));
                                *memories_used += 1;
                            }
                        }
                    }
                } else {
                    return format!("Memory '{}' not found.", memory_id);
                }

                context_parts.join("\n")
            }
            _ => format!("Unknown tool: {}", tool_name),
        }
    }

    // ─── Bank management ────────────────────────────────────────────────

    pub fn create_bank(&self, name: &str, disposition: Option<Disposition>) -> Result<Bank> {
        let id = uuid::Uuid::new_v4().to_string();
        let disp = disposition.unwrap_or_default();
        self.store.create_bank(&id, name, &disp)
    }

    pub fn get_bank(&self, id: &str) -> Result<Option<Bank>> {
        self.store.get_bank(id)
    }

    pub fn list_banks(&self) -> Result<Vec<Bank>> {
        self.store.list_banks()
    }

    pub fn delete_bank(&self, id: &str) -> Result<()> {
        self.store.delete_bank(id)
    }

    pub fn update_bank_context(&self, id: &str, context: &str) -> Result<()> {
        self.store.update_bank_context(id, context)
    }

    // ─── Memory management ──────────────────────────────────────────────

    pub fn delete_memory(&self, id: &str) -> Result<()> {
        self.store.delete_memory(id)
    }

    pub fn list_memories(&self, bank_id: &str, limit: usize, offset: usize) -> Result<Vec<MemoryUnit>> {
        self.store.list_memories(bank_id, limit, offset, None, None)
    }

    pub fn list_memories_filtered(&self, bank_id: &str, limit: usize, offset: usize,
                                   fact_type: Option<&FactType>, search_query: Option<&str>) -> Result<Vec<MemoryUnit>> {
        self.store.list_memories(bank_id, limit, offset, fact_type, search_query)
    }

    // ─── Observation injection (bypass LLM) ─────────────────────────────

    pub async fn inject_observation(
        &self,
        bank_id: &str,
        content: &str,
        fact_type: Option<FactType>,
        tags: Vec<String>,
        confidence: Option<f32>,
    ) -> Result<String> {
        info!("Inject: bank={}, content_len={}", bank_id, content.len());

        if self.store.get_bank(bank_id)?.is_none() {
            return Err(HindsightError::NotFound(format!("Bank {} not found", bank_id)));
        }

        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let unit = MemoryUnit {
            id: id.clone(),
            bank_id: bank_id.to_string(),
            fact_type: fact_type.unwrap_or(FactType::Experience),
            content: content.to_string(),
            summary: None,
            source_document_id: None,
            confidence: confidence.unwrap_or(1.0),
            tags,
            occurred_start: None,
            occurred_end: None,
            mentioned_at: Some(now.clone()),
            created_at: now.clone(),
            updated_at: now,
        };
        self.store.store_memory(&unit)?;

        if let Ok(embeddings) = self.llm.embed(&[content.to_string()], &self.embedding_model, &self.embedding_base_url).await {
            if let Some(emb) = embeddings.into_iter().next() {
                let emb_id = uuid::Uuid::new_v4().to_string();
                let _ = self.store.store_embedding(&emb_id, &id, &emb, &self.embedding_model);
            }
        }

        info!("Inject: created memory {}", id);
        Ok(id)
    }

    // ─── Graph query ────────────────────────────────────────────────────

    pub fn query_graph(&self, memory_id: &str, max_hops: usize, link_types: Vec<LinkType>) -> Result<Vec<(MemoryUnit, usize)>> {
        info!("QueryGraph: start={}, max_hops={}, types={:?}", memory_id, max_hops, link_types);
        self.store.graph_traverse(memory_id, max_hops, &link_types)
    }

    // ─── Entity-related memories ─────────────────────────────────────────

    pub fn get_related(&self, bank_id: &str, entity_name: &str) -> Result<Vec<MemoryUnit>> {
        info!("GetRelated: bank={}, entity={}", bank_id, entity_name);
        self.store.get_entity_memories(bank_id, entity_name)
    }

    pub fn list_entities(&self, bank_id: &str, limit: usize, offset: usize) -> Result<Vec<Entity>> {
        self.store.list_entities(bank_id, limit, offset)
    }

    pub fn get_entity(&self, bank_id: &str, entity_id: &str) -> Result<Option<Entity>> {
        self.store.get_entity(bank_id, entity_id)
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    async fn extract_facts(&self, content: &str) -> Result<Vec<ExtractedFact>> {
        let system = "You are a memory extraction engine. Extract discrete facts from the given content.
Each fact should be categorized as:
- world: general knowledge facts (\"The sky is blue\")
- experience: personal experiences or events (\"I visited Paris in 2023\")
- mental_model: consolidated knowledge patterns (\"User prefers functional programming\")

Return a JSON array of objects with:
- fact_type: one of world, experience, mental_model
- content: the fact text
- tags: array of strings
- summary: optional short version
- confidence: 0.0-1.0
- occurred_start: ISO 8601 datetime if the fact has a specific time, else null
- occurred_end: ISO 8601 datetime if the fact spans a time range, else null

Examples of temporal extraction:
- \"昨天去了北京\" → occurred_start: yesterday's date
- \"2024年3月到5月在实习\" → occurred_start: 2024-03, occurred_end: 2024-05
- \"天空是蓝色的\" → no temporal info (null)

If the content is too vague to extract facts, return the whole content as a single experience fact.
Use ensure_ascii=False in JSON output.";

        let response = self.llm.chat(system, content).await?;

        let json_str = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        serde_json::from_str(json_str).map_err(|e| {
            info!("Fact extraction parse error: {}, raw response: {}", e, &response[..response.len().min(200)]);
            HindsightError::Llm(format!("Failed to parse extracted facts: {}", e))
        })
    }

    async fn extract_entities(&self, content: &str) -> Result<Vec<ExtractedEntity>> {
        let system = "Extract named entities from the given text. Return a JSON array of objects with:
- name: the entity name
- entity_type: person, place, organization, concept, or thing
- labels: array of descriptive labels

If no clear entities exist, return an empty array [].";

        let response = self.llm.chat(system, content).await?;
        let json_str = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        Ok(serde_json::from_str(json_str).unwrap_or_else(|_| Vec::new()))
    }

    /// Extract causal relationships between facts (returns (source_idx, target_idx) pairs)
    async fn extract_causal_links(&self, facts: &[ExtractedFact]) -> Result<Vec<(usize, usize)>> {
        if facts.len() < 2 { return Ok(Vec::new()); }

        let fact_list: String = facts.iter().enumerate()
            .map(|(i, f)| format!("{}. {}", i, f.content))
            .collect::<Vec<_>>()
            .join("\n");

        let system = "Given the following list of facts, identify any causal relationships (one fact causes or enables another).
Return a JSON array of [source_index, target_index] pairs where source causes/enables target.
If no causal relationships exist, return an empty array [].
Only return clear causal links, not mere correlations.";

        let response = self.llm.chat(system, &fact_list).await?;
        let json_str = response.trim()
            .trim_start_matches("```json").trim_start_matches("```")
            .trim_end_matches("```").trim();

        let pairs: Vec<Vec<usize>> = serde_json::from_str(json_str).unwrap_or_default();
        Ok(pairs.into_iter()
            .filter_map(|p| if p.len() == 2 { Some((p[0], p[1])) } else { None })
            .collect())
    }

    async fn get_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        let embeddings = self.llm
            .embed(&[query.to_string()], &self.embedding_model, &self.embedding_base_url)
            .await?;
        embeddings.into_iter().next()
            .ok_or_else(|| HindsightError::Llm("No embedding returned".into()))
    }

    async fn rerank(&self, query: &str, results: Vec<ScoredMemory>) -> Vec<ScoredMemory> {
        if results.len() <= 1 { return results; }

        let documents: Vec<String> = results.iter().map(|r| r.unit.content.clone()).collect();
        match self.llm.rerank(query, &documents, &self.embedding_base_url).await {
            Ok(reranked_indices) => {
                let mut reranked = Vec::new();
                for (new_rank, idx) in reranked_indices.iter().enumerate() {
                    if let Some(item) = results.get(*idx) {
                        let mut scored = item.clone();
                        scored.score = 1.0 / (60.0 + new_rank as f32 + 1.0);
                        scored.source = "rrf_reranked".to_string();
                        reranked.push(scored);
                    }
                }
                let reranked_set: std::collections::HashSet<usize> = reranked_indices.iter().cloned().collect();
                for (i, item) in results.iter().enumerate() {
                    if !reranked_set.contains(&i) { reranked.push(item.clone()); }
                }
                reranked
            }
            Err(e) => {
                info!("Rerank failed (using original order): {}", e);
                results
            }
        }
    }

    // ─── CONSOLIDATE: Merge similar memories into mental models ─────────

    pub async fn consolidate(&self, bank_id: &str, similarity_threshold: f32, max_groups: usize) -> Result<ConsolidateResult> {
        info!("Consolidate: bank={}, threshold={}, max_groups={}", bank_id, similarity_threshold, max_groups);

        let bank = self.store.get_bank(bank_id)?
            .ok_or_else(|| HindsightError::NotFound(format!("Bank {} not found", bank_id)))?;

        let memories = self.store.list_memories(bank_id, 200, 0, None, None)?;
        let world_and_experience: Vec<&MemoryUnit> = memories.iter()
            .filter(|m| m.fact_type != FactType::MentalModel)
            .collect();

        if world_and_experience.len() < 2 {
            return Ok(ConsolidateResult { groups_found: 0, mental_models_created: 0 });
        }

        let texts: Vec<String> = world_and_experience.iter().map(|m| m.content.clone()).collect();
        let embeddings = match self.llm.embed(&texts, &self.embedding_model, &self.embedding_base_url).await {
            Ok(e) => e,
            Err(e) => return Err(e),
        };

        let mut assigned: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();

        for i in 0..embeddings.len() {
            if assigned.contains(&i) { continue; }
            let mut group = vec![i];
            assigned.insert(i);
            for j in (i + 1)..embeddings.len() {
                if assigned.contains(&j) { continue; }
                if cosine_similarity(&embeddings[i], &embeddings[j]) >= similarity_threshold {
                    group.push(j);
                    assigned.insert(j);
                }
            }
            if group.len() >= 2 { groups.push(group); }
            if groups.len() >= max_groups { break; }
        }

        info!("Consolidate: found {} groups", groups.len());

        let mut mental_model_ids = Vec::new();
        let now = chrono::Utc::now().to_rfc3339();

        for group in &groups {
            let group_memories: Vec<&MemoryUnit> = group.iter()
                .filter_map(|&i| world_and_experience.get(i))
                .map(|m| *m)
                .collect();

            let memory_text: String = group_memories.iter()
                .map(|m| format!("- [{}] {}", m.fact_type.as_ref(), m.content))
                .collect::<Vec<_>>()
                .join("\n");

            let system = format!(
                "You are synthesizing related memories into a consolidated mental model.
The bank has this disposition: skepticism={}/5, literalism={}/5, empathy={}/5.
Combine the following related memories into a single insight or pattern.
Respond in JSON: {{\"content\": \"...\", \"tags\": [...], \"confidence\": 0.0-1.0}}",
                bank.disposition.skepticism, bank.disposition.literalism, bank.disposition.empathy,
            );

            match self.llm.chat(&system, &memory_text).await {
                Ok(response) => {
                    let json_str = response.trim()
                        .trim_start_matches("```json").trim_start_matches("```")
                        .trim_end_matches("```").trim();

                    if let Ok(synthesized) = serde_json::from_str::<ConsolidatedMentalModel>(json_str) {
                        let mm_id = uuid::Uuid::new_v4().to_string();
                        let unit = MemoryUnit {
                            id: mm_id.clone(),
                            bank_id: bank_id.to_string(),
                            fact_type: FactType::MentalModel,
                            content: synthesized.content,
                            summary: None,
                            source_document_id: None,
                            confidence: synthesized.confidence,
                            tags: synthesized.tags,
                            occurred_start: None,
                            occurred_end: None,
                            mentioned_at: Some(now.clone()),
                            created_at: now.clone(),
                            updated_at: now.clone(),
                        };
                        self.store.store_memory(&unit)?;
                        mental_model_ids.push(mm_id.clone());

                        for src_mem in &group_memories {
                            let link = MemoryLink {
                                id: uuid::Uuid::new_v4().to_string(),
                                source_id: src_mem.id.clone(),
                                target_id: mm_id.clone(),
                                link_type: LinkType::Semantic,
                                weight: 0.8,
                                created_at: now.clone(),
                            };
                            self.store.store_link(&link)?;
                        }
                    }
                }
                Err(e) => info!("Consolidate LLM error for group: {}", e),
            }
        }

        Ok(ConsolidateResult {
            groups_found: groups.len(),
            mental_models_created: mental_model_ids.len(),
        })
    }
}

// ─── Response types ─────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ExtractedFact {
    pub fact_type: FactType,
    pub content: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub occurred_start: Option<String>,
    #[serde(default)]
    pub occurred_end: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub labels: Vec<String>,
}

#[derive(Debug, serde::Serialize, Clone)]
pub struct ScoredMemory {
    pub unit: MemoryUnit,
    pub score: f32,
    pub source: String,
}

#[derive(Debug, serde::Serialize)]
pub struct RetainResult {
    pub document_id: String,
    pub memory_ids: Vec<String>,
    pub facts_extracted: usize,
    pub entities_extracted: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct ReflectResult {
    pub answer: String,
    pub memories_used: usize,
    pub disposition: Disposition,
}

#[derive(Debug, serde::Serialize)]
pub struct ConsolidateResult {
    pub groups_found: usize,
    pub mental_models_created: usize,
}

#[derive(Debug, serde::Deserialize)]
struct ConsolidatedMentalModel {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    confidence: f32,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

/// Extract temporal date range from query text.
/// Returns (start_date, end_date) as ISO strings, or None if no temporal hint found.
fn extract_temporal_range(query: &str) -> (Option<String>, Option<String>) {
    use chrono::{Datelike, Duration, Utc};

    let lower = query.to_lowercase();
    let now = Utc::now();

    // Chinese temporal expressions
    if lower.contains("今天") || lower.contains("今日") || lower.contains("today") {
        let today = now.format("%Y-%m-%d").to_string();
        let tomorrow = (now + Duration::days(1)).format("%Y-%m-%d").to_string();
        return (Some(today), Some(tomorrow));
    }
    if lower.contains("昨天") || lower.contains("昨日") || lower.contains("yesterday") {
        let yesterday = (now - Duration::days(1)).format("%Y-%m-%d").to_string();
        let today = now.format("%Y-%m-%d").to_string();
        return (Some(yesterday), Some(today));
    }
    if lower.contains("前天") {
        let day = (now - Duration::days(2)).format("%Y-%m-%d").to_string();
        let yesterday = (now - Duration::days(1)).format("%Y-%m-%d").to_string();
        return (Some(day), Some(yesterday));
    }
    if lower.contains("上周") || lower.contains("上星期") || lower.contains("last week") {
        let start = (now - Duration::weeks(1)).format("%Y-%m-%d").to_string();
        let end = now.format("%Y-%m-%d").to_string();
        return (Some(start), Some(end));
    }
    if lower.contains("上个月") || lower.contains("last month") {
        let start = (now - Duration::days(30)).format("%Y-%m-%d").to_string();
        let end = now.format("%Y-%m-%d").to_string();
        return (Some(start), Some(end));
    }
    if lower.contains("最近") || lower.contains("latest") || lower.contains("recent") {
        let start = (now - Duration::days(7)).format("%Y-%m-%d").to_string();
        let end = now.format("%Y-%m-%d").to_string();
        return (Some(start), Some(end));
    }
    if lower.contains("去年") || lower.contains("last year") {
        let year = now.year() - 1;
        return (Some(format!("{}-01-01", year)), Some(format!("{}-12-31", year)));
    }
    if lower.contains("今年") || lower.contains("this year") {
        let year = now.year();
        return (Some(format!("{}-01-01", year)), Some(format!("{}-12-31", year)));
    }
    if lower.contains("这周") || lower.contains("本周") || lower.contains("this week") {
        let start = (now - Duration::days(7)).format("%Y-%m-%d").to_string();
        let end = now.format("%Y-%m-%d").to_string();
        return (Some(start), Some(end));
    }

    (None, None)
}
