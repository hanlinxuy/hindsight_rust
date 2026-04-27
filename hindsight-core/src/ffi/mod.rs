use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::MemoryEngine;

/// Thread-safe multi-tenant engine registry
static ENGINES: std::sync::OnceLock<std::sync::Mutex<HashMap<u64, MemoryEngine>>> =
    std::sync::OnceLock::new();

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn engines() -> &'static std::sync::Mutex<HashMap<u64, MemoryEngine>> {
    ENGINES.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Opaque handle type returned to callers
pub type HindsightHandle = u64;

macro_rules! cstr_or_else {
    ($ptr:expr, $on_null:expr) => {
        if $ptr.is_null() {
            $on_null
        } else {
            unsafe { CStr::from_ptr($ptr) }.to_str().unwrap_or("")
        }
    };
}

macro_rules! with_engine {
    ($handle:expr, $var:ident, $on_err:expr) => {
        let map = match engines().lock() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("engine registry lock error: {}", e);
                return $on_err;
            }
        };
        let $var = match map.get(&$handle) {
            Some(e) => e,
            None => {
                eprintln!("invalid handle: {}", $handle);
                return $on_err;
            }
        };
    };
}

fn json_to_cchar<T: serde::Serialize>(val: T) -> *mut c_char {
    match serde_json::to_string(&val) {
        Ok(json) => CString::new(json).unwrap().into_raw(),
        Err(e) => {
            eprintln!("json serialize error: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Lifecycle: init / destroy / list tenants
// ──────────────────────────────────────────────────────────────────────

/// Initialize a new memory engine instance for a tenant.
/// Returns a positive handle on success, -1 on failure.
#[no_mangle]
pub extern "C" fn hindsight_init(
    db_path: *const c_char,
    api_key: *const c_char,
    base_url: *const c_char,
    model: *const c_char,
) -> i64 {
    let db_path = cstr_or_else!(db_path, "./hindsight.db");
    let api_key = cstr_or_else!(api_key, "");
    let base_url = cstr_or_else!(base_url, "https://api.openai.com");
    let model = cstr_or_else!(model, "gpt-4o-mini");

    match MemoryEngine::init(db_path, api_key, base_url, model) {
        Ok(engine) => {
            let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
            let mut map = match engines().lock() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("engine registry lock error: {}", e);
                    return -1;
                }
            };
            map.insert(handle, engine);
            handle as i64
        }
        Err(e) => {
            eprintln!("hindsight_init error: {}", e);
            -1
        }
    }
}

/// Initialize engine with separate embedding config.
/// Returns a positive handle on success, -1 on failure.
#[no_mangle]
pub extern "C" fn hindsight_init_with_embedding(
    db_path: *const c_char,
    api_key: *const c_char,
    base_url: *const c_char,
    model: *const c_char,
    embedding_model: *const c_char,
    embedding_base_url: *const c_char,
) -> i64 {
    let db_path = cstr_or_else!(db_path, "./hindsight.db");
    let api_key = cstr_or_else!(api_key, "");
    let base_url = cstr_or_else!(base_url, "https://api.openai.com");
    let model = cstr_or_else!(model, "gpt-4o-mini");
    let emb_model = cstr_or_else!(embedding_model, "text-embedding-3-small");
    let emb_url = cstr_or_else!(embedding_base_url, base_url);

    match MemoryEngine::init(db_path, api_key, base_url, model) {
        Ok(engine) => {
            let engine = engine.with_embedding(emb_model, emb_url);
            let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
            let mut map = match engines().lock() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("engine registry lock error: {}", e);
                    return -1;
                }
            };
            map.insert(handle, engine);
            handle as i64
        }
        Err(e) => {
            eprintln!("hindsight_init_with_embedding error: {}", e);
            -1
        }
    }
}

/// Destroy a tenant engine instance and release resources.
#[no_mangle]
pub extern "C" fn hindsight_destroy(handle: u64) -> i32 {
    let mut map = match engines().lock() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("engine registry lock error: {}", e);
            return -1;
        }
    };
    if map.remove(&handle).is_some() {
        0
    } else {
        -1
    }
}

/// List all active tenant handles as JSON array.
#[no_mangle]
pub extern "C" fn hindsight_list_tenants() -> *mut c_char {
    let map = match engines().lock() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("engine registry lock error: {}", e);
            return std::ptr::null_mut();
        }
    };
    let handles: Vec<u64> = map.keys().cloned().collect();
    json_to_cchar(handles)
}

// ──────────────────────────────────────────────────────────────────────
// Bank operations (all take handle as first arg)
// ──────────────────────────────────────────────────────────────────────

/// Create a new memory bank with disposition traits.
#[no_mangle]
pub extern "C" fn hindsight_create_bank(
    handle: u64,
    name: *const c_char,
    skepticism: u8,
    literalism: u8,
    empathy: u8,
) -> *mut c_char {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("default");
    let disposition = crate::models::Disposition {
        skepticism: skepticism.clamp(1, 5),
        literalism: literalism.clamp(1, 5),
        empathy: empathy.clamp(1, 5),
    };

    match engine.create_bank(name, Some(disposition)) {
        Ok(bank) => CString::new(bank.id).unwrap().into_raw(),
        Err(e) => {
            eprintln!("hindsight_create_bank error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Get bank info as JSON.
#[no_mangle]
pub extern "C" fn hindsight_get_bank(handle: u64, bank_id: *const c_char) -> *mut c_char {
    if bank_id.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    match engine.get_bank(bank_id) {
        Ok(Some(bank)) => json_to_cchar(bank),
        Ok(None) => std::ptr::null_mut(),
        Err(e) => {
            eprintln!("hindsight_get_bank error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// List all banks as JSON array.
#[no_mangle]
pub extern "C" fn hindsight_list_banks(handle: u64) -> *mut c_char {
    with_engine!(handle, engine, std::ptr::null_mut());
    match engine.list_banks() {
        Ok(banks) => json_to_cchar(banks),
        Err(e) => {
            eprintln!("hindsight_list_banks error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Delete a bank and all its data.
#[no_mangle]
pub extern "C" fn hindsight_delete_bank(handle: u64, bank_id: *const c_char) -> i32 {
    if bank_id.is_null() {
        return -1;
    }
    with_engine!(handle, eng, -1);
    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    match eng.delete_bank(bank_id) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hindsight_delete_bank error: {}", e);
            -1
        }
    }
}

/// Set bank background context.
#[no_mangle]
pub extern "C" fn hindsight_set_bank_context(
    handle: u64,
    bank_id: *const c_char,
    context: *const c_char,
) -> i32 {
    if bank_id.is_null() || context.is_null() {
        return -1;
    }
    with_engine!(handle, eng, -1);
    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    let context = unsafe { CStr::from_ptr(context) }.to_str().unwrap_or("");
    match eng.update_bank_context(bank_id, context) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hindsight_set_bank_context error: {}", e);
            -1
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Core memory operations
// ──────────────────────────────────────────────────────────────────────

/// Retain content into a bank.
#[no_mangle]
pub extern "C" fn hindsight_retain(
    handle: u64,
    bank_id: *const c_char,
    content: *const c_char,
) -> *mut c_char {
    if bank_id.is_null() || content.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    let content = unsafe { CStr::from_ptr(content) }.to_str().unwrap_or("");

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("tokio runtime error: {}", e);
            return std::ptr::null_mut();
        }
    };

    match rt.block_on(engine.retain(bank_id, content)) {
        Ok(result) => json_to_cchar(result),
        Err(e) => {
            eprintln!("hindsight_retain error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Recall memories relevant to a query.
#[no_mangle]
pub extern "C" fn hindsight_recall(
    handle: u64,
    bank_id: *const c_char,
    query: *const c_char,
    limit: u32,
) -> *mut c_char {
    if bank_id.is_null() || query.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    let query = unsafe { CStr::from_ptr(query) }.to_str().unwrap_or("");

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("tokio runtime error: {}", e);
            return std::ptr::null_mut();
        }
    };

    match rt.block_on(engine.recall(bank_id, query, limit as usize)) {
        Ok(results) => json_to_cchar(results),
        Err(e) => {
            eprintln!("hindsight_recall error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Reflect on memories with disposition-aware reasoning.
#[no_mangle]
pub extern "C" fn hindsight_reflect(
    handle: u64,
    bank_id: *const c_char,
    query: *const c_char,
) -> *mut c_char {
    if bank_id.is_null() || query.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    let query = unsafe { CStr::from_ptr(query) }.to_str().unwrap_or("");

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("tokio runtime error: {}", e);
            return std::ptr::null_mut();
        }
    };

    match rt.block_on(engine.reflect(bank_id, query)) {
        Ok(result) => json_to_cchar(result),
        Err(e) => {
            eprintln!("hindsight_reflect error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Delete a specific memory.
#[no_mangle]
pub extern "C" fn hindsight_delete_memory(handle: u64, memory_id: *const c_char) -> i32 {
    if memory_id.is_null() {
        return -1;
    }
    with_engine!(handle, eng, -1);
    let memory_id = unsafe { CStr::from_ptr(memory_id) }.to_str().unwrap_or("");
    match eng.delete_memory(memory_id) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("hindsight_delete_memory error: {}", e);
            -1
        }
    }
}

/// List memories in a bank as JSON array.
#[no_mangle]
pub extern "C" fn hindsight_list_memories(
    handle: u64,
    bank_id: *const c_char,
    limit: u32,
    offset: u32,
) -> *mut c_char {
    if bank_id.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    match engine.list_memories(bank_id, limit as usize, offset as usize) {
        Ok(memories) => json_to_cchar(memories),
        Err(e) => {
            eprintln!("hindsight_list_memories error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Consolidate similar memories into mental models.
#[no_mangle]
pub extern "C" fn hindsight_consolidate(
    handle: u64,
    bank_id: *const c_char,
    similarity_threshold: f32,
    max_groups: u32,
) -> *mut c_char {
    if bank_id.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    let threshold = if similarity_threshold <= 0.0 || similarity_threshold > 1.0 {
        0.8
    } else {
        similarity_threshold
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("tokio runtime error: {}", e);
            return std::ptr::null_mut();
        }
    };

    match rt.block_on(engine.consolidate(bank_id, threshold, max_groups as usize)) {
        Ok(result) => json_to_cchar(result),
        Err(e) => {
            eprintln!("hindsight_consolidate error: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Observation injection
// ──────────────────────────────────────────────────────────────────────

/// Inject a pre-structured observation directly (bypass LLM extraction).
#[no_mangle]
pub extern "C" fn hindsight_inject_observation(
    handle: u64,
    bank_id: *const c_char,
    content: *const c_char,
    fact_type: *const c_char,
    tags_json: *const c_char,
    confidence: f32,
) -> *mut c_char {
    if bank_id.is_null() || content.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    let content = unsafe { CStr::from_ptr(content) }.to_str().unwrap_or("");

    let ft = if fact_type.is_null() {
        None
    } else {
        let ft_str = unsafe { CStr::from_ptr(fact_type) }.to_str().unwrap_or("");
        match ft_str {
            "world" => Some(crate::models::FactType::World),
            "mental_model" => Some(crate::models::FactType::MentalModel),
            _ => Some(crate::models::FactType::Experience),
        }
    };

    let tags: Vec<String> = if tags_json.is_null() {
        Vec::new()
    } else {
        let tags_str = unsafe { CStr::from_ptr(tags_json) }.to_str().unwrap_or("[]");
        serde_json::from_str(tags_str).unwrap_or_default()
    };

    let conf = if confidence <= 0.0 {
        None
    } else {
        Some(confidence.clamp(0.0, 1.0))
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("tokio runtime error: {}", e);
            return std::ptr::null_mut();
        }
    };

    match rt.block_on(engine.inject_observation(bank_id, content, ft, tags, conf)) {
        Ok(id) => {
            let result = serde_json::json!({"id": id, "status": "ok"});
            json_to_cchar(result)
        }
        Err(e) => {
            eprintln!("hindsight_inject_observation error: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Graph query
// ──────────────────────────────────────────────────────────────────────

/// Query graph: traverse memory links from a starting node.
#[no_mangle]
pub extern "C" fn hindsight_query_graph(
    handle: u64,
    memory_id: *const c_char,
    max_hops: u32,
    link_types_json: *const c_char,
) -> *mut c_char {
    if memory_id.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let memory_id = unsafe { CStr::from_ptr(memory_id) }.to_str().unwrap_or("");

    let link_types: Vec<crate::models::LinkType> = if link_types_json.is_null() {
        Vec::new()
    } else {
        let lt_str = unsafe { CStr::from_ptr(link_types_json) }.to_str().unwrap_or("[]");
        let names: Vec<String> = serde_json::from_str(lt_str).unwrap_or_default();
        names
            .iter()
            .map(|n| match n.as_str() {
                "semantic" => crate::models::LinkType::Semantic,
                "temporal" => crate::models::LinkType::Temporal,
                "entity" => crate::models::LinkType::Entity,
                "causal" => crate::models::LinkType::Causal,
                _ => crate::models::LinkType::Semantic,
            })
            .collect()
    };

    match engine.query_graph(memory_id, max_hops as usize, link_types) {
        Ok(results) => {
            let json_results: Vec<serde_json::Value> = results
                .iter()
                .map(|(unit, hops)| {
                    serde_json::json!({
                        "unit": unit,
                        "hops": hops,
                    })
                })
                .collect();
            json_to_cchar(json_results)
        }
        Err(e) => {
            eprintln!("hindsight_query_graph error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Get all memories related to a named entity.
#[no_mangle]
pub extern "C" fn hindsight_get_related(
    handle: u64,
    bank_id: *const c_char,
    entity_name: *const c_char,
) -> *mut c_char {
    if bank_id.is_null() || entity_name.is_null() {
        return std::ptr::null_mut();
    }
    with_engine!(handle, engine, std::ptr::null_mut());

    let bank_id = unsafe { CStr::from_ptr(bank_id) }.to_str().unwrap_or("");
    let entity_name = unsafe { CStr::from_ptr(entity_name) }.to_str().unwrap_or("");

    match engine.get_related(bank_id, entity_name) {
        Ok(memories) => json_to_cchar(memories),
        Err(e) => {
            eprintln!("hindsight_get_related error: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Utilities
// ──────────────────────────────────────────────────────────────────────

/// Get library version string (static, do not free).
#[no_mangle]
pub extern "C" fn hindsight_version() -> *const c_char {
    static VERSION: &[u8] = b"0.2.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Free a string previously returned by this library (null is safe).
#[no_mangle]
pub extern "C" fn hindsight_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}
