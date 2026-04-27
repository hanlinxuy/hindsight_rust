/// Integration test that calls a real LLM API.
/// Set HINDSIGHT_TEST_API_KEY, HINDSIGHT_TEST_BASE_URL, HINDSIGHT_TEST_MODEL to run.
/// If not set, tests are skipped.
use hindsight_core::engine::MemoryEngine;
use hindsight_core::models::Disposition;
use std::env;

fn create_engine() -> Option<MemoryEngine> {
    let api_key = env::var("HINDSIGHT_TEST_API_KEY").ok()?;
    if api_key.is_empty() { return None; }
    let base_url = env::var("HINDSIGHT_TEST_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com".into());
    let model = env::var("HINDSIGHT_TEST_MODEL")
        .unwrap_or_else(|_| "gpt-4o-mini".into());
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("integration_test.db");
    std::mem::forget(dir);

    let mut engine = MemoryEngine::init(db_path.to_str().unwrap(), &api_key, &base_url, &model).ok()?;

    // Override embedding config if provided (e.g. local embedding server)
    if let Ok(emb_url) = env::var("HINDSIGHT_TEST_EMBEDDING_BASE_URL") {
        let emb_model = env::var("HINDSIGHT_TEST_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "bge-large-zh-v1.5".into());
        let emb_key = env::var("HINDSIGHT_TEST_EMBEDDING_API_KEY")
            .unwrap_or_else(|_| "not-needed".into());
        engine = engine.with_embedding(&emb_model, &emb_url);
        // Note: if embedding service requires auth, the LLM client uses the same api_key
        // For local services that don't need auth, this is fine
        let _ = emb_key; // suppress unused warning
    }

    Some(engine)
}

macro_rules! maybe_skip {
    () => {
        if create_engine().is_none() {
            eprintln!("Skipping - set HINDSIGHT_TEST_API_KEY to run");
            return;
        }
    };
}

#[tokio::test]
async fn test_retain_and_recall() {
    maybe_skip!();
    let engine = create_engine().unwrap();

    let bank = engine.create_bank("integration_test", None).unwrap();
    let bank_id = &bank.id;

    let result = engine.retain(bank_id,
        "I visited Tokyo in March 2024. The cherry blossoms were beautiful. \
         I had ramen at a small shop in Shinjuku. Tokyo is the capital of Japan."
    ).await.unwrap();

    assert!(result.facts_extracted >= 1, "Should extract at least 1 fact, got {}", result.facts_extracted);
    assert!(!result.memory_ids.is_empty());
    println!("Retain: {} facts, {} entities, doc_id={}",
        result.facts_extracted, result.entities_extracted, result.document_id);

    let results = engine.recall(bank_id, "What did I eat in Japan?", 5).await.unwrap();
    // With embedding API available, recall should return results
    if env::var("HINDSIGHT_TEST_EMBEDDING_BASE_URL").is_ok() {
        assert!(!results.is_empty(), "Recall should return results with embedding API, got 0");
    }
    println!("Recall: {} results", results.len());
    for r in &results {
        println!("  [{:.3}] {}", r.score, r.unit.content.chars().take(80).collect::<String>());
    }

    let all = engine.list_memories(bank_id, 100, 0).unwrap();
    println!("Total memories in bank: {}", all.len());
    for m in &all {
        println!("  [{}] {}", m.fact_type.as_ref(), m.content.chars().take(80).collect::<String>());
    }
}

#[tokio::test]
async fn test_reflect() {
    maybe_skip!();
    let engine = create_engine().unwrap();

    let bank = engine.create_bank("reflect_test", None).unwrap();
    let bank_id = &bank.id;

    engine.retain(bank_id,
        "User prefers dark mode in all applications. \
         User has been using VS Code for 3 years. \
         User writes code in Rust and Python."
    ).await.unwrap();

    let result = engine.reflect(bank_id, "What are the user's development preferences?").await.unwrap();
    assert!(!result.answer.is_empty());
    println!("Reflect (memories_used={}): {}", result.memories_used,
        result.answer.chars().take(200).collect::<String>());
}

#[tokio::test]
async fn test_background_context() {
    maybe_skip!();
    let engine = create_engine().unwrap();

    let bank = engine.create_bank("ctx_test", Some(Disposition {
        skepticism: 5, literalism: 1, empathy: 3,
    })).unwrap();
    let bank_id = &bank.id;

    engine.update_bank_context(bank_id, "This is a skeptical science-focused assistant").unwrap();

    engine.retain(bank_id, "Studies show that coffee may reduce risk of heart disease.").await.unwrap();
    let result = engine.reflect(bank_id, "Should I drink more coffee?").await.unwrap();
    println!("Reflect with high skepticism: {}",
        result.answer.chars().take(200).collect::<String>());
}

#[tokio::test]
async fn test_delete_operations() {
    maybe_skip!();
    let engine = create_engine().unwrap();

    let bank = engine.create_bank("delete_test", None).unwrap();
    let bank_id = &bank.id;

    let retain = engine.retain(bank_id, "This is a temporary memory about the weather.").await.unwrap();
    assert!(!retain.memory_ids.is_empty());

    engine.delete_memory(&retain.memory_ids[0]).unwrap();
    engine.delete_bank(bank_id).unwrap();
    assert!(engine.get_bank(bank_id).unwrap().is_none());
    println!("Delete operations passed");
}

#[tokio::test]
async fn test_inject_and_recall() {
    maybe_skip!();
    let engine = create_engine().unwrap();

    let bank = engine.create_bank("inject_test", None).unwrap();
    let bank_id = &bank.id;

    // Inject observations directly (no LLM extraction)
    let id1 = engine.inject_observation(
        bank_id,
        "User prefers dark mode in all IDEs and terminals.",
        None,
        vec!["preference".into()],
        None,
    ).await.unwrap();

    let id2 = engine.inject_observation(
        bank_id,
        "User's favorite programming language is Rust.",
        None,
        vec!["preference".into(), "programming".into()],
        Some(0.95),
    ).await.unwrap();

    let id3 = engine.inject_observation(
        bank_id,
        "User commutes by bicycle every day.",
        None,
        vec!["lifestyle".into()],
        Some(0.8),
    ).await.unwrap();

    println!("Injected: {}, {}, {}", id1, id2, id3);

    // Verify stored
    let all = engine.list_memories(bank_id, 100, 0).unwrap();
    assert_eq!(all.len(), 3);

    // Recall should find them via FTS + embedding
    let results = engine.recall(bank_id, "programming language preference", 5).await.unwrap();
    println!("Recall after inject: {} results", results.len());
    for r in &results {
        println!("  [{:.3}] {}", r.score, r.unit.content);
    }
    // At least the Rust preference should be found
    assert!(!results.is_empty(), "Recall should find injected observations");
}
