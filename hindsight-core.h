#ifndef HINDSIGHT_CORE_H
#define HINDSIGHT_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Engine lifecycle (multi-tenant) ──────────────────────────────── */

/**
 * Initialize a new memory engine instance (tenant).
 * Each call creates an isolated engine with its own database.
 * @return Positive handle on success, -1 on failure
 */
int64_t hindsight_init(const char *db_path,
                       const char *api_key,
                       const char *base_url,
                       const char *model);

/**
 * Initialize engine with separate embedding config.
 * @return Positive handle on success, -1 on failure
 */
int64_t hindsight_init_with_embedding(const char *db_path,
                                       const char *api_key,
                                       const char *base_url,
                                       const char *model,
                                       const char *embedding_model,
                                       const char *embedding_base_url);

/**
 * Destroy a tenant engine and release resources.
 * @return 0 on success, -1 on failure
 */
int32_t hindsight_destroy(uint64_t handle);

/**
 * List all active tenant handles.
 * @return JSON array of handles (caller must free)
 */
char *hindsight_list_tenants(void);

/** Get library version string (static, do not free) */
const char *hindsight_version(void);

/* ── Bank management ──────────────────────────────────────────────── */

/**
 * Create a new memory bank with disposition traits (1-5 each).
 * @return Bank ID (caller must free), or null
 */
char *hindsight_create_bank(uint64_t handle,
                            const char *name,
                            uint8_t skepticism,
                            uint8_t literalism,
                            uint8_t empathy);

/**
 * Get bank info as JSON.
 * @return JSON string (caller must free), or null
 */
char *hindsight_get_bank(uint64_t handle, const char *bank_id);

/** List all banks as JSON array (caller must free) */
char *hindsight_list_banks(uint64_t handle);

/**
 * Delete a bank and all its data.
 * @return 0 on success, -1 on failure
 */
int32_t hindsight_delete_bank(uint64_t handle, const char *bank_id);

/**
 * Set bank background context.
 * @return 0 on success, -1 on failure
 */
int32_t hindsight_set_bank_context(uint64_t handle,
                                   const char *bank_id,
                                   const char *context);

/* ── Memory operations ────────────────────────────────────────────── */

/**
 * Retain (store) content into a memory bank.
 * @return JSON RetainResult (caller must free), or null on error
 */
char *hindsight_retain(uint64_t handle,
                       const char *bank_id,
                       const char *content);

/**
 * Recall memories relevant to a query.
 * @return JSON array of ScoredMemory (caller must free), or null on error
 */
char *hindsight_recall(uint64_t handle,
                       const char *bank_id,
                       const char *query,
                       uint32_t limit);

/**
 * Reflect on memories with disposition-aware reasoning.
 * @return JSON ReflectResult (caller must free), or null on error
 */
char *hindsight_reflect(uint64_t handle,
                        const char *bank_id,
                        const char *query);

/**
 * Delete a specific memory.
 * @return 0 on success, -1 on failure
 */
int32_t hindsight_delete_memory(uint64_t handle, const char *memory_id);

/**
 * List memories in a bank (paginated).
 * @return JSON array of MemoryUnit (caller must free), or null on error
 */
char *hindsight_list_memories(uint64_t handle,
                              const char *bank_id,
                              uint32_t limit,
                              uint32_t offset);

/**
 * Consolidate similar memories into mental models.
 * @return JSON ConsolidateResult (caller must free), or null on error
 */
char *hindsight_consolidate(uint64_t handle,
                            const char *bank_id,
                            float similarity_threshold,
                            uint32_t max_groups);

/* ── Observation injection ────────────────────────────────────────── */

/**
 * Inject a pre-structured observation directly, bypassing LLM extraction.
 * @param fact_type   "world"/"experience"/"mental_model", null for experience
 * @param tags_json   JSON array ["tag1","tag2"], null for []
 * @param confidence  0.0-1.0, 0.0 for default (1.0)
 * @return JSON {"id":"...","status":"ok"} (caller must free), or null on error
 */
char *hindsight_inject_observation(uint64_t handle,
                                    const char *bank_id,
                                    const char *content,
                                    const char *fact_type,
                                    const char *tags_json,
                                    float confidence);

/* ── Graph query ──────────────────────────────────────────────────── */

/**
 * Traverse memory graph from a starting node via BFS.
 * @return JSON array of {unit: MemoryUnit, hops: number} (caller must free)
 */
char *hindsight_query_graph(uint64_t handle,
                             const char *memory_id,
                             uint32_t max_hops,
                             const char *link_types_json);

/**
 * Get all memories related to a named entity.
 * @return JSON array of MemoryUnit (caller must free), or null on error
 */
char *hindsight_get_related(uint64_t handle,
                             const char *bank_id,
                             const char *entity_name);

/* ── Memory management ────────────────────────────────────────────── */

/** Free a string previously returned by this library (null is safe) */
void hindsight_free_string(char *s);

#ifdef __cplusplus
}
#endif

#endif /* HINDSIGHT_CORE_H */
