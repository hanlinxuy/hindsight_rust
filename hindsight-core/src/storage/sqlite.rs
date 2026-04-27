use crate::error::{HindsightError, Result};
use crate::models::*;
use crate::storage::schema;
use jieba_rs::Jieba;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::OnceLock;
use tracing::info;

/// Global jieba instance (lazy-initialized, thread-safe via OnceLock)
static JIEBA: OnceLock<Jieba> = OnceLock::new();

fn get_jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

/// Segment Chinese text into space-separated tokens for FTS5 indexing.
/// Mixed Chinese/English text is handled correctly — Chinese chars are segmented,
/// English words are preserved as-is.
fn segment_for_fts(text: &str) -> String {
    let jieba = get_jieba();
    let words = jieba.cut(text, false);
    words.into_iter().filter(|w| !w.trim().is_empty()).collect::<Vec<_>>().join(" ")
}

/// SQLite-based storage backend
pub struct SqliteStore {
    conn: Connection,
}

/// Common SELECT columns for memory_units (keeps all row-reading consistent)
const MU_COLS: &str = "id, bank_id, fact_type, content, summary, source_document_id, \
    confidence, tags, occurred_start, occurred_end, mentioned_at, created_at, updated_at";

/// Map a row to MemoryUnit. Row must match MU_COLS column order.
fn row_to_memory_unit(row: &rusqlite::Row) -> rusqlite::Result<MemoryUnit> {
    let fact_str: String = row.get(2)?;
    let tags_str: String = row.get(7)?;
    Ok(MemoryUnit {
        id: row.get(0)?,
        bank_id: row.get(1)?,
        fact_type: str_to_fact_type(&fact_str),
        content: row.get(3)?,
        summary: row.get(4)?,
        source_document_id: row.get(5)?,
        confidence: row.get(6)?,
        tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        occurred_start: row.get(8)?,
        occurred_end: row.get(9)?,
        mentioned_at: row.get(10)?,
        created_at: row.get(11)?,
        updated_at: row.get(12)?,
    })
}

impl SqliteStore {
    /// Open or create database at the given path
    pub fn open(db_path: &str) -> Result<Self> {
        let path = Path::new(db_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;

        // Enable WAL mode for better concurrent performance
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        // Create schema
        conn.execute_batch(schema::create_tables())?;

        info!("Database opened at {}", db_path);
        Ok(Self { conn })
    }

    // --- Bank operations ---

    pub fn create_bank(&self, id: &str, name: &str, disposition: &Disposition) -> Result<Bank> {
        let disp_json = serde_json::to_string(disposition)?;
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO banks (id, name, disposition, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, name, disp_json, now, now],
        )?;
        Ok(Bank {
            id: id.to_string(),
            name: name.to_string(),
            disposition: disposition.clone(),
            background_context: None,
            created_at: now.clone(),
            updated_at: now,
        })
    }

    pub fn get_bank(&self, id: &str) -> Result<Option<Bank>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, disposition, background_context, created_at, updated_at FROM banks WHERE id = ?1",
        )?;
        let bank = stmt
            .query_row(params![id], |row| {
                let disp_str: String = row.get(2)?;
                let disposition: Disposition = serde_json::from_str(&disp_str).unwrap_or_default();
                Ok(Bank {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    disposition,
                    background_context: row.get(3)?,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                })
            })
            .optional()?;
        Ok(bank)
    }

    // --- Memory operations ---

    pub fn store_memory(&self, unit: &MemoryUnit) -> Result<()> {
        let tags_json = serde_json::to_string(&unit.tags)?;
        let content_search = segment_for_fts(&unit.content);
        let summary_search = unit.summary.as_deref().map(|s| segment_for_fts(s)).unwrap_or_default();
        self.conn.execute(
            "INSERT INTO memory_units (id, bank_id, fact_type, content, content_search, summary, summary_search,
                source_document_id, confidence, tags, occurred_start, occurred_end, mentioned_at, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                unit.id, unit.bank_id,
                fact_type_to_str(&unit.fact_type),
                unit.content, content_search,
                unit.summary, summary_search,
                unit.source_document_id,
                unit.confidence, tags_json,
                unit.occurred_start, unit.occurred_end, unit.mentioned_at,
                unit.created_at, unit.updated_at
            ],
        )?;
        Ok(())
    }

    pub fn search_similar(&self, bank_id: &str, query_embedding: &[f32], limit: usize,
                          fact_types: &[FactType], tags: &[String]) -> Result<Vec<(MemoryUnit, f32)>> {
        let dims = query_embedding.len();

        // Build fact_type filter
        let ft_clause = if fact_types.is_empty() { String::new() } else {
            let list: Vec<&str> = fact_types.iter().map(|ft| fact_type_to_str(ft)).collect();
            format!(" AND mu.fact_type IN ({})", list.iter().map(|f| format!("'{}'", f)).collect::<Vec<_>>().join(","))
        };

        // Build tags filter (any match)
        let tags_clause = if tags.is_empty() { String::new() } else {
            let conditions: Vec<String> = tags.iter().map(|t| format!("mu.tags LIKE '%\"{}\"%'", t.replace('\'', "''"))).collect();
            format!(" AND ({})", conditions.join(" OR "))
        };

        let sql = format!(
            "SELECT mu.id, mu.bank_id, mu.fact_type, mu.content, mu.summary, mu.source_document_id, \
    mu.confidence, mu.tags, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.created_at, mu.updated_at, e.embedding
             FROM memory_units mu
             JOIN embeddings e ON e.memory_unit_id = mu.id
             WHERE mu.bank_id = ?1 AND e.dimensions = ?2{}{}
             ", ft_clause, tags_clause
        );

        let mut stmt = self.conn.prepare(&sql)?;

        let rows = stmt.query_map(params![bank_id, dims as i32], |row| {
            let unit = row_to_memory_unit(row)?;
            // embedding is column 13 (after the 13 MU_COLS)
            let embedding_blob: Vec<u8> = row.get(13)?;
            Ok((unit, embedding_blob))
        })?;

        let mut scored: Vec<(MemoryUnit, f32)> = rows
            .filter_map(|r| r.ok())
            .map(|(unit, emb_blob)| {
                let emb: Vec<f32> = emb_blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let sim = cosine_similarity(query_embedding, &emb);
                (unit, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    pub fn store_embedding(&self, id: &str, memory_unit_id: &str, embedding: &[f32], model: &str) -> Result<()> {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let dims = embedding.len() as i32;
        self.conn.execute(
            "INSERT INTO embeddings (id, memory_unit_id, embedding, model, dimensions) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, memory_unit_id, bytes, model, dims],
        )?;
        Ok(())
    }

    // --- Entity operations ---

    pub fn store_entity(&self, id: &str, bank_id: &str, name: &str, entity_type: &str, labels: &[String]) -> Result<String> {
        let labels_json = serde_json::to_string(labels)?;
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO entities (id, bank_id, name, entity_type, labels, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, bank_id, name, entity_type, labels_json, now],
        )?;
        Ok(id.to_string())
    }

    pub fn store_unit_entity(&self, memory_unit_id: &str, entity_id: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO unit_entities (memory_unit_id, entity_id) VALUES (?1, ?2)",
            params![memory_unit_id, entity_id],
        )?;
        Ok(())
    }

    // --- Single memory lookup ---

    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryUnit>> {
        let sql = format!("SELECT {} FROM memory_units WHERE id = ?1", MU_COLS);
        let mut stmt = self.conn.prepare(&sql)?;
        let unit = stmt.query_row(params![id], |row| row_to_memory_unit(row)).optional()?;
        Ok(unit)
    }

    // --- Link operations ---

    pub fn store_link(&self, link: &MemoryLink) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_links (id, source_id, target_id, link_type, weight, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![link.id, link.source_id, link.target_id,
                    link_type_to_str(&link.link_type),
                    link.weight, link.created_at],
        )?;
        Ok(())
    }

    // --- Graph traversal ---

    pub fn get_links_for(&self, memory_id: &str) -> Result<Vec<MemoryLink>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_id, target_id, link_type, weight, created_at
             FROM memory_links WHERE source_id = ?1 OR target_id = ?1",
        )?;
        let links = stmt.query_map(params![memory_id], |row| {
            Ok(MemoryLink {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                link_type: match row.get::<_, String>(3)?.as_str() {
                    "semantic" => LinkType::Semantic,
                    "temporal" => LinkType::Temporal,
                    "entity" => LinkType::Entity,
                    "causal" => LinkType::Causal,
                    _ => LinkType::Semantic,
                },
                weight: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(links)
    }

    // --- FTS search ---

    pub fn search_fulltext(&self, bank_id: &str, query: &str, limit: usize,
                           fact_types: &[FactType], tags: &[String],
                           date_start: Option<&str>, date_end: Option<&str>) -> Result<Vec<MemoryUnit>> {
        // Segment the query with jieba, then build FTS5 MATCH expression
        let segmented = segment_for_fts(query);
        let fts_query: String = segmented
            .split_whitespace()
            .filter_map(|w| {
                let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                if clean.len() >= 2 { Some(clean) } else { None }
            })
            .take(10)
            .collect::<Vec<_>>()
            .join(" OR ");

        if fts_query.is_empty() {
            return Ok(Vec::new());
        }

        let ft_clause = if fact_types.is_empty() { String::new() } else {
            let list: Vec<&str> = fact_types.iter().map(|ft| fact_type_to_str(ft)).collect();
            format!(" AND mu.fact_type IN ({})", list.iter().map(|f| format!("'{}'", f)).collect::<Vec<_>>().join(","))
        };
        let tags_clause = if tags.is_empty() { String::new() } else {
            let conditions: Vec<String> = tags.iter().map(|t| format!("mu.tags LIKE '%\"{}\"%'", t.replace('\'', "''"))).collect();
            format!(" AND ({})", conditions.join(" OR "))
        };
        let date_clause = match (date_start, date_end) {
            (Some(s), Some(e)) => format!(" AND mu.occurred_start >= '{}' AND mu.occurred_end <= '{}'", s, e),
            (Some(s), None) => format!(" AND mu.occurred_start >= '{}'", s),
            (None, Some(e)) => format!(" AND mu.occurred_end <= '{}'", e),
            _ => String::new(),
        };

        let sql = format!(
            "SELECT mu.{}
             FROM memory_units_fts fts
             JOIN memory_units mu ON mu.rowid = fts.rowid
             WHERE fts.memory_units_fts MATCH ?1 AND mu.bank_id = ?2{}{}{}
             ORDER BY rank
             LIMIT ?3", MU_COLS, ft_clause, tags_clause, date_clause
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map(params![fts_query, bank_id, limit as i32], |row| {
            row_to_memory_unit(row)
        })?;

        results.collect::<std::result::Result<_, _>>().map_err(HindsightError::from)
    }

    // --- Document operations ---

    pub fn store_document(&self, id: &str, bank_id: &str, content: &str, content_type: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO documents (id, bank_id, content, content_type, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, bank_id, content, content_type, now],
        )?;
        Ok(())
    }

    // --- Bank management ---

    pub fn update_bank_context(&self, id: &str, context: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE banks SET background_context = ?1, updated_at = ?2 WHERE id = ?3",
            params![context, now, id],
        )?;
        Ok(())
    }

    pub fn list_banks(&self) -> Result<Vec<Bank>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, disposition, background_context, created_at, updated_at FROM banks ORDER BY created_at DESC",
        )?;
        let banks = stmt.query_map([], |row| {
            let disp_str: String = row.get(2)?;
            let disposition: Disposition = serde_json::from_str(&disp_str).unwrap_or_default();
            Ok(Bank {
                id: row.get(0)?,
                name: row.get(1)?,
                disposition,
                background_context: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(banks)
    }

    pub fn delete_bank(&self, id: &str) -> Result<()> {
        // Cascade deletes via foreign keys
        self.conn.execute("DELETE FROM memory_units WHERE bank_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM entities WHERE bank_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM documents WHERE bank_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM banks WHERE id = ?1", params![id])?;
        Ok(())
    }

    // --- Memory deletion ---

    pub fn delete_memory(&self, id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM embeddings WHERE memory_unit_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM memory_links WHERE source_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM memory_links WHERE target_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM unit_entities WHERE memory_unit_id = ?1", params![id])?;
        self.conn.execute("DELETE FROM memory_units WHERE id = ?1", params![id])?;
        Ok(())
    }

    // --- List memories for a bank ---

    pub fn list_memories(&self, bank_id: &str, limit: usize, offset: usize,
                        fact_type: Option<&FactType>, search_query: Option<&str>) -> Result<Vec<MemoryUnit>> {
        let mut where_clauses = vec!["mu.bank_id = ?1".to_string()];
        if let Some(ft) = fact_type {
            where_clauses.push(format!("mu.fact_type = '{}'", fact_type_to_str(ft)));
        }
        if let Some(q) = search_query {
            where_clauses.push(format!("(mu.content LIKE '%{}%' OR mu.summary LIKE '%{}%')",
                q.replace('\'', "''"), q.replace('\'', "''")));
        }
        let where_sql = where_clauses.join(" AND ");

        let sql = format!("SELECT mu.id, mu.bank_id, mu.fact_type, mu.content, mu.summary, mu.source_document_id, \
    mu.confidence, mu.tags, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.created_at, mu.updated_at
             FROM memory_units mu WHERE {} ORDER BY mu.created_at DESC LIMIT ?2 OFFSET ?3",
                          where_sql);
        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map(params![bank_id, limit as i32, offset as i32], |row| {
            row_to_memory_unit(row)
        })?;
        results.collect::<std::result::Result<_, _>>().map_err(HindsightError::from)
    }

    // --- Link type filtering ---

    pub fn get_links_by_type(&self, memory_id: &str, link_type: &LinkType) -> Result<Vec<MemoryLink>> {
        let lt = link_type_to_str(link_type);
        let mut stmt = self.conn.prepare(
            "SELECT id, source_id, target_id, link_type, weight, created_at
             FROM memory_links WHERE (source_id = ?1 OR target_id = ?1) AND link_type = ?2",
        )?;
        let links = stmt.query_map(params![memory_id, lt], |row| {
            Ok(MemoryLink {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                link_type: match row.get::<_, String>(3)?.as_str() {
                    "semantic" => LinkType::Semantic,
                    "temporal" => LinkType::Temporal,
                    "entity" => LinkType::Entity,
                    "causal" => LinkType::Causal,
                    _ => LinkType::Semantic,
                },
                weight: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(links)
    }

    // --- Entity → Memory lookup ---

    pub fn get_entity_memories(&self, bank_id: &str, entity_name: &str) -> Result<Vec<MemoryUnit>> {
        let sql = format!(
            "SELECT mu.id, mu.bank_id, mu.fact_type, mu.content, mu.summary, mu.source_document_id, \
    mu.confidence, mu.tags, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.created_at, mu.updated_at
             FROM memory_units mu
             JOIN unit_entities ue ON ue.memory_unit_id = mu.id
             JOIN entities e ON e.id = ue.entity_id
             WHERE e.bank_id = ?1 AND LOWER(e.name) = LOWER(?2)"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map(params![bank_id, entity_name], |row| {
            row_to_memory_unit(row)
        })?;
        results.collect::<std::result::Result<_, _>>().map_err(HindsightError::from)
    }

    // --- Entity listing and lookup ---

    pub fn list_entities(&self, bank_id: &str, limit: usize, offset: usize) -> Result<Vec<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, bank_id, name, entity_type, labels, created_at
             FROM entities WHERE bank_id = ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
        )?;
        let results = stmt.query_map(params![bank_id, limit as i32, offset as i32], |row| {
            let labels_str: String = row.get(4)?;
            Ok(Entity {
                id: row.get(0)?,
                bank_id: row.get(1)?,
                name: row.get(2)?,
                entity_type: row.get(3)?,
                labels: serde_json::from_str(&labels_str).unwrap_or_default(),
                created_at: row.get(5)?,
            })
        })?;
        results.collect::<std::result::Result<_, _>>().map_err(HindsightError::from)
    }

    pub fn get_entity(&self, bank_id: &str, entity_id: &str) -> Result<Option<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, bank_id, name, entity_type, labels, created_at
             FROM entities WHERE id = ?1 AND bank_id = ?2",
        )?;
        let entity = stmt.query_row(params![entity_id, bank_id], |row| {
            let labels_str: String = row.get(4)?;
            Ok(Entity {
                id: row.get(0)?,
                bank_id: row.get(1)?,
                name: row.get(2)?,
                entity_type: row.get(3)?,
                labels: serde_json::from_str(&labels_str).unwrap_or_default(),
                created_at: row.get(5)?,
            })
        }).optional()?;
        Ok(entity)
    }

    // --- BFS graph traversal ---

    pub fn graph_traverse(&self, start_id: &str, max_hops: usize, link_types: &[LinkType]) -> Result<Vec<(MemoryUnit, usize)>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut results: Vec<(MemoryUnit, usize)> = Vec::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        // Include starting node
        if let Some(start_unit) = self.get_memory(start_id)? {
            visited.insert(start_id.to_string());
            results.push((start_unit, 0));
            queue.push_back((start_id.to_string(), 0));
        }

        while let Some((node_id, hops)) = queue.pop_front() {
            if hops >= max_hops { continue; }

            let links = self.get_links_for(&node_id)?;
            for link in &links {
                // Filter by link type if specified
                if !link_types.is_empty() && !link_types.iter().any(|lt| std::mem::discriminant(lt) == std::mem::discriminant(&link.link_type)) {
                    continue;
                }
                let neighbor = if link.source_id == node_id { &link.target_id } else { &link.source_id };
                if visited.contains(neighbor) { continue; }
                visited.insert(neighbor.clone());
                if let Some(unit) = self.get_memory(neighbor)? {
                    results.push((unit.clone(), hops + 1));
                    queue.push_back((neighbor.clone(), hops + 1));
                }
            }
        }

        Ok(results)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

fn fact_type_to_str(ft: &FactType) -> &'static str {
    match ft {
        FactType::World => "world",
        FactType::Experience => "experience",
        FactType::MentalModel => "mental_model",
    }
}

fn str_to_fact_type(s: &str) -> FactType {
    match s {
        "world" => FactType::World,
        "mental_model" => FactType::MentalModel,
        _ => FactType::Experience,
    }
}

fn link_type_to_str(lt: &LinkType) -> &'static str {
    match lt {
        LinkType::Semantic => "semantic",
        LinkType::Temporal => "temporal",
        LinkType::Entity => "entity",
        LinkType::Causal => "causal",
    }
}
