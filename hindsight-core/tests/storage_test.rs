use hindsight_core::storage::SqliteStore;
use hindsight_core::models::*;

fn test_store() -> SqliteStore {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db").to_str().unwrap().to_string();
    // Prevent dir from being dropped
    std::mem::forget(dir);
    SqliteStore::open(&db_path).unwrap()
}

#[test]
fn test_create_and_get_bank() {
    let store = test_store();
    let disp = Disposition { skepticism: 4, literalism: 2, empathy: 5 };
    let bank = store.create_bank("id1", "test_bank", &disp).unwrap();

    assert_eq!(bank.name, "test_bank");
    assert_eq!(bank.disposition.skepticism, 4);
    assert_eq!(bank.disposition.empathy, 5);

    let fetched = store.get_bank("id1").unwrap().unwrap();
    assert_eq!(fetched.id, "id1");
    assert_eq!(fetched.disposition.literalism, 2);
}

#[test]
fn test_get_nonexistent_bank() {
    let store = test_store();
    assert!(store.get_bank("nope").unwrap().is_none());
}

#[test]
fn test_store_and_get_memory() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    let unit = MemoryUnit {
        id: "m1".into(),
        bank_id: "b1".into(),
        fact_type: FactType::Experience,
        content: "I visited Tokyo in 2024".into(),
        summary: Some("Tokyo trip".into()),
        source_document_id: None,
        confidence: 0.9,
        tags: vec!["travel".into(), "japan".into()],
        occurred_start: None,
        occurred_end: None,
        mentioned_at: None,
        created_at: "2024-01-01T00:00:00Z".into(),
        updated_at: "2024-01-01T00:00:00Z".into(),
    };
    store.store_memory(&unit).unwrap();

    let fetched = store.get_memory("m1").unwrap().unwrap();
    assert_eq!(fetched.content, "I visited Tokyo in 2024");
    assert_eq!(fetched.fact_type, FactType::Experience);
    assert_eq!(fetched.tags.len(), 2);
}

#[test]
fn test_embedding_and_search() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    let unit = MemoryUnit {
        id: "m1".into(),
        bank_id: "b1".into(),
        fact_type: FactType::World,
        content: "Rust is a systems programming language".into(),
        summary: None,
        source_document_id: None,
        confidence: 1.0,
        tags: vec![],
        occurred_start: None,
        occurred_end: None,
        mentioned_at: None,
        created_at: "2024-01-01T00:00:00Z".into(),
        updated_at: "2024-01-01T00:00:00Z".into(),
    };
    store.store_memory(&unit).unwrap();

    // Store a fake embedding (128 dims)
    let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    store.store_embedding("e1", "m1", &embedding, "test-model").unwrap();

    // Search with a similar vector
    let results = store.search_similar("b1", &embedding, 10, &[], &[]).unwrap();
    assert_eq!(results.len(), 1);
    assert!((results[0].1 - 1.0).abs() < 0.001); // should be ~1.0 similarity
}

#[test]
fn test_links() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    // Create two memories
    for id in &["m1", "m2"] {
        let unit = MemoryUnit {
            id: id.to_string(),
            bank_id: "b1".into(),
            fact_type: FactType::Experience,
            content: format!("Fact {}", id),
            summary: None,
            source_document_id: None,
            confidence: 1.0,
            tags: vec![],
            occurred_start: None,
            occurred_end: None,
            mentioned_at: None,
            created_at: "2024-01-01T00:00:00Z".into(),
            updated_at: "2024-01-01T00:00:00Z".into(),
        };
        store.store_memory(&unit).unwrap();
    }

    let link = MemoryLink {
        id: "l1".into(),
        source_id: "m1".into(),
        target_id: "m2".into(),
        link_type: LinkType::Temporal,
        weight: 1.0,
        created_at: "2024-01-01T00:00:00Z".into(),
    };
    store.store_link(&link).unwrap();

    let links = store.get_links_for("m1").unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].target_id, "m2");
}

#[test]
fn test_entity_operations() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    let e_id = store.store_entity("e1", "b1", "Tokyo", "place", &["city".into(), "japan".into()]).unwrap();
    assert_eq!(e_id, "e1");

    // Create memory and link entity
    let unit = MemoryUnit {
        id: "m1".into(),
        bank_id: "b1".into(),
        fact_type: FactType::Experience,
        content: "I visited Tokyo".into(),
        summary: None,
        source_document_id: None,
        confidence: 1.0,
        tags: vec![],
        occurred_start: None,
        occurred_end: None,
        mentioned_at: None,
        created_at: "2024-01-01T00:00:00Z".into(),
        updated_at: "2024-01-01T00:00:00Z".into(),
    };
    store.store_memory(&unit).unwrap();
    store.store_unit_entity("m1", "e1").unwrap();
}

#[test]
fn test_document_storage() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    store.store_document("d1", "b1", "Hello world content", "text/plain").unwrap();
}

#[test]
fn test_delete_memory() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    let unit = MemoryUnit {
        id: "m1".into(),
        bank_id: "b1".into(),
        fact_type: FactType::World,
        content: "temporary fact".into(),
        summary: None,
        source_document_id: None,
        confidence: 1.0,
        tags: vec![],
        occurred_start: None,
        occurred_end: None,
        mentioned_at: None,
        created_at: "2024-01-01T00:00:00Z".into(),
        updated_at: "2024-01-01T00:00:00Z".into(),
    };
    store.store_memory(&unit).unwrap();
    store.store_embedding("e1", "m1", &vec![0.1; 64], "test").unwrap();

    store.delete_memory("m1").unwrap();
    assert!(store.get_memory("m1").unwrap().is_none());
}

#[test]
fn test_list_banks() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "first", &disp).unwrap();
    store.create_bank("b2", "second", &disp).unwrap();

    let banks = store.list_banks().unwrap();
    assert_eq!(banks.len(), 2);
}

#[test]
fn test_list_memories() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    for i in 0..5 {
        let unit = MemoryUnit {
            id: format!("m{}", i),
            bank_id: "b1".into(),
            fact_type: FactType::World,
            content: format!("Fact number {}", i),
            summary: None,
            source_document_id: None,
            confidence: 1.0,
            tags: vec![],
            occurred_start: None,
            occurred_end: None,
            mentioned_at: None,
            created_at: "2024-01-01T00:00:00Z".into(),
            updated_at: "2024-01-01T00:00:00Z".into(),
        };
        store.store_memory(&unit).unwrap();
    }

    let page1 = store.list_memories("b1", 3, 0, None, None).unwrap();
    assert_eq!(page1.len(), 3);

    let page2 = store.list_memories("b1", 3, 3, None, None).unwrap();
    assert_eq!(page2.len(), 2);
}

#[test]
fn test_update_bank_context() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    store.update_bank_context("b1", "This is a personal assistant").unwrap();
    let bank = store.get_bank("b1").unwrap().unwrap();
    assert_eq!(bank.background_context, Some("This is a personal assistant".into()));
}

#[test]
fn test_delete_bank() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    let unit = MemoryUnit {
        id: "m1".into(),
        bank_id: "b1".into(),
        fact_type: FactType::World,
        content: "test".into(),
        summary: None,
        source_document_id: None,
        confidence: 1.0,
        tags: vec![],
        occurred_start: None,
        occurred_end: None,
        mentioned_at: None,
        created_at: "2024-01-01T00:00:00Z".into(),
        updated_at: "2024-01-01T00:00:00Z".into(),
    };
    store.store_memory(&unit).unwrap();

    store.delete_bank("b1").unwrap();
    assert!(store.get_bank("b1").unwrap().is_none());
    assert!(store.get_memory("m1").unwrap().is_none());
}

#[test]
fn test_get_links_by_type() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    for id in &["m1", "m2", "m3"] {
        let unit = MemoryUnit {
            id: id.to_string(),
            bank_id: "b1".into(),
            fact_type: FactType::Experience,
            content: format!("Fact {}", id),
            summary: None,
            source_document_id: None,
            confidence: 1.0,
            tags: vec![],
            occurred_start: None,
            occurred_end: None,
            mentioned_at: None,
            created_at: "2024-01-01T00:00:00Z".into(),
            updated_at: "2024-01-01T00:00:00Z".into(),
        };
        store.store_memory(&unit).unwrap();
    }

    // m1 --temporal--> m2, m1 --entity--> m3
    store.store_link(&MemoryLink {
        id: "l1".into(), source_id: "m1".into(), target_id: "m2".into(),
        link_type: LinkType::Temporal, weight: 1.0, created_at: "2024-01-01T00:00:00Z".into(),
    }).unwrap();
    store.store_link(&MemoryLink {
        id: "l2".into(), source_id: "m1".into(), target_id: "m3".into(),
        link_type: LinkType::Entity, weight: 0.7, created_at: "2024-01-01T00:00:00Z".into(),
    }).unwrap();

    let temporal = store.get_links_by_type("m1", &LinkType::Temporal).unwrap();
    assert_eq!(temporal.len(), 1);
    assert_eq!(temporal[0].target_id, "m2");

    let entity = store.get_links_by_type("m1", &LinkType::Entity).unwrap();
    assert_eq!(entity.len(), 1);
    assert_eq!(entity[0].target_id, "m3");

    let semantic = store.get_links_by_type("m1", &LinkType::Semantic).unwrap();
    assert_eq!(semantic.len(), 0);
}

#[test]
fn test_get_entity_memories() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    // Create entity "Tokyo"
    store.store_entity("e1", "b1", "Tokyo", "place", &["city".into()]).unwrap();

    // Create two memories mentioning Tokyo
    for (mid, content) in &[("m1", "I visited Tokyo in 2024"), ("m2", "Tokyo is the capital of Japan")] {
        let unit = MemoryUnit {
            id: mid.to_string(),
            bank_id: "b1".into(),
            fact_type: FactType::Experience,
            content: content.to_string(),
            summary: None,
            source_document_id: None,
            confidence: 1.0,
            tags: vec![],
            occurred_start: None,
            occurred_end: None,
            mentioned_at: None,
            created_at: "2024-01-01T00:00:00Z".into(),
            updated_at: "2024-01-01T00:00:00Z".into(),
        };
        store.store_memory(&unit).unwrap();
        store.store_unit_entity(mid, "e1").unwrap();
    }

    // Unrelated memory
    let unit = MemoryUnit {
        id: "m3".into(), bank_id: "b1".into(), fact_type: FactType::World,
        content: "The sky is blue".into(), summary: None, source_document_id: None,
        confidence: 1.0, tags: vec![], occurred_start: None, occurred_end: None, mentioned_at: None, created_at: "2024-01-01T00:00:00Z".into(),
        updated_at: "2024-01-01T00:00:00Z".into(),
    };
    store.store_memory(&unit).unwrap();

    let tokyo_memories = store.get_entity_memories("b1", "Tokyo").unwrap();
    assert_eq!(tokyo_memories.len(), 2);
}

#[test]
fn test_graph_traverse() {
    let store = test_store();
    let disp = Disposition::default();
    store.create_bank("b1", "bank", &disp).unwrap();

    // Build chain: m1 --temporal--> m2 --temporal--> m3 --entity--> m4
    for id in &["m1", "m2", "m3", "m4"] {
        let unit = MemoryUnit {
            id: id.to_string(),
            bank_id: "b1".into(),
            fact_type: FactType::Experience,
            content: format!("Fact {}", id),
            summary: None,
            source_document_id: None,
            confidence: 1.0,
            tags: vec![],
            occurred_start: None,
            occurred_end: None,
            mentioned_at: None,
            created_at: "2024-01-01T00:00:00Z".into(),
            updated_at: "2024-01-01T00:00:00Z".into(),
        };
        store.store_memory(&unit).unwrap();
    }

    store.store_link(&MemoryLink {
        id: "l1".into(), source_id: "m1".into(), target_id: "m2".into(),
        link_type: LinkType::Temporal, weight: 1.0, created_at: "2024-01-01T00:00:00Z".into(),
    }).unwrap();
    store.store_link(&MemoryLink {
        id: "l2".into(), source_id: "m2".into(), target_id: "m3".into(),
        link_type: LinkType::Temporal, weight: 1.0, created_at: "2024-01-01T00:00:00Z".into(),
    }).unwrap();
    store.store_link(&MemoryLink {
        id: "l3".into(), source_id: "m3".into(), target_id: "m4".into(),
        link_type: LinkType::Entity, weight: 0.7, created_at: "2024-01-01T00:00:00Z".into(),
    }).unwrap();

    // Traverse all types, 3 hops from m1
    let results = store.graph_traverse("m1", 3, &[]).unwrap();
    assert_eq!(results.len(), 4); // m1(0) + m2(1) + m3(2) + m4(3)

    // Traverse only temporal links, 2 hops
    let temporal_only = store.graph_traverse("m1", 2, &[LinkType::Temporal]).unwrap();
    assert_eq!(temporal_only.len(), 3); // m1(0) + m2(1) + m3(2), m4 is entity link

    // 1 hop only
    let one_hop = store.graph_traverse("m1", 1, &[]).unwrap();
    assert_eq!(one_hop.len(), 2); // m1(0) + m2(1)
}
