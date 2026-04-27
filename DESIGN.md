# Hindsight Core 设计分析与 Python 对比

## 一、项目概述

Hindsight 是一个面向 AI Agent 的长期记忆系统。原始版本使用 Python + FastAPI + PostgreSQL 实现（约 37K 行核心代码），运行在服务端。Rust 版本是其核心引擎的设备端移植，编译为 `libhindsight_core.so` 动态库，目标平台为 HarmonyOS NEXT、Android、Linux ARM64。

**设计哲学差异**：
- **Python 版**：服务端架构，PostgreSQL + pgvector 提供高性能向量检索，多租户通过数据库 schema 隔离
- **Rust 版**：设备端架构，SQLite + 暴力向量搜索（后续可换 LanceDB），多租户通过 bank_id 逻辑隔离

## 二、Python 版架构回顾

### 整体架构

```
                    ┌─────────────────────────────┐
                    │      REST API (FastAPI)      │
                    │  /v1/{tenant}/banks/{id}/...  │
                    ├─────────────────────────────┤
                    │     MemoryEngine 接口层      │
                    │     (interface.py ~588行)     │
                    ├──────┬──────┬──────┬────────┤
                    │Retain│Recall│Reflect│Consol. │
                    │引擎  │引擎  │引擎   │引擎    │
                    ├──────┴──────┴──────┴────────┤
                    │  LLM Wrapper (多provider)    │
                    │  Embeddings (本地/TEI)       │
                    │  Cross Encoder (本地/TEI)    │
                    │  Entity Resolver             │
                    │  Query Analyzer              │
                    ├─────────────────────────────┤
                    │  PostgreSQL + pgvector       │
                    │  Alembic 迁移管理            │
                    └─────────────────────────────┘
```

### Python 核心模块规模

| 模块 | 文件 | 行数 | 职责 |
|------|------|------|------|
| MemoryEngine | `memory_engine.py` | ~8,746 | 主引擎，协调所有操作 |
| Interface | `interface.py` | ~588 | 抽象接口定义 |
| Retain 编排器 | `retain/orchestrator.py` | ~1,595 | 多阶段记忆写入管线 |
| Retain 事实提取 | `retain/fact_extraction.py` | ~600 | LLM 事实提取 |
| Retain 类型 | `retain/types.py` | ~200 | ExtractedFact 等类型 |
| Recall 检索 | `search/retrieval.py` | ~800 | 四策略并行检索 |
| Recall 图检索 | `search/graph_retrieval.py` | ~300 | 图检索基类 |
| Recall 融合 | `search/fusion.py` | ~200 | RRF 融合 |
| Recall 重排 | `search/reranking.py` | ~300 | Cross-encoder 重排 |
| Reflect Agent | `reflect/agent.py` | ~800 | Agent 循环反思推理 |
| Reflect 工具 | `reflect/tools_schema.py` | ~300 | Agent 工具定义 |
| Reflect 观察 | `reflect/observations.py` | ~400 | 观察/趋势分析 |
| Reflect Delta | `reflect/delta_ops.py` | ~300 | 增量文档操作 |
| Reflect 结构化文档 | `reflect/structured_doc.py` | ~500 | 文档块结构 |
| LLM Wrapper | `llm_wrapper.py` | ~1,500 | 多 provider LLM 抽象 |
| Embeddings | `embeddings.py` | ~400 | 嵌入生成 |
| Entity Resolver | `entity_resolver.py` | ~400 | 实体解析与归一化 |
| Query Analyzer | `query_analyzer.py` | ~300 | 查询意图分析 |
| **合计** | | **~16,000+** | （核心引擎部分） |

加上 API 层、数据库迁移、配置管理、MCP server、集成层等，总计约 37,000 行。

## 三、Rust 版当前实现

### 架构

```
┌─────────────────────────────────────────────────┐
│               上层应用 (ArkTS / Kotlin)           │
├─────────────────────────────────────────────────┤
│              C FFI (19 函数)                      │
├─────────────────────────────────────────────────┤
│             MemoryEngine 核心 (~960行)            │
│  ┌──────────┬──────────┬───────────┬──────────┐ │
│  │ Retain   │ Recall   │ Reflect   │Consolidate│ │
│  │ 事实提取  │ 多策略   │ Agent循环  │ 相似整合  │ │
│  │ 实体识别  │ RRF融合  │ 6工具调用  │ 心智模型  │ │
│  │ 4类链接   │ 时间过滤  │ Done结束   │          │ │
│  └──────────┴──────────┴───────────┴──────────┘ │
├─────────────────────────────────────────────────┤
│  SqliteStore (~520行)  │  LlmClient (~300行)    │
│  SQLite + FTS5 + jieba │  OpenAI 兼容 HTTP API  │
└─────────────────────────────────────────────────┘
```

### Rust 模块规模

| 模块 | 文件 | 行数 | 职责 |
|------|------|------|------|
| MemoryEngine | `engine/memory_engine.rs` | ~960 | 主引擎，retain/recall/reflect/consolidate |
| SqliteStore | `storage/sqlite.rs` | ~520 | SQLite CRUD + 向量搜索 + FTS + 图遍历 |
| Schema | `storage/schema.rs` | ~100 | DDL 定义 |
| LlmClient | `llm/client.rs` | ~300 | HTTP LLM/Embedding/Rerank 客户端 |
| Models | `models/*.rs` | ~200 | 数据结构 |
| FFI | `ffi/mod.rs` | ~616 | 19 个 C API 导出 |
| **合计** | | **~2,700** | （核心引擎部分） |

Rust 版约 2,700 行实现了 Python 版 16,000+ 行的核心功能，代码密度比约 6:1。

## 四、逐功能对比分析

### 4.1 Retain（记忆写入）

| 功能点 | Python 实现 | Rust 实现 | 差距 |
|--------|------------|-----------|------|
| 文档存储 | ✅ 多格式、分块、去重 | ✅ 基础文本存储 | Python 有分块和去重逻辑 |
| 事实提取 | ✅ LLM 多轮提取，支持 world/experience/mental_model | ✅ 单轮 LLM 提取，同样 3 种类型 | **已完成** |
| 事实时间标注 | ✅ `occurred_start`/`occurred_end`/`where` | ✅ `occurred_start`/`occurred_end` | **已完成** |
| 实体提取 | ✅ LLM 提取 + 归一化 + 别名合并 | ✅ LLM 提取 + 按名合并 | Python 有别名合并（entity_resolver） |
| 实体类型 | ✅ person/location/org/concept/thing | ✅ 同 5 种类型 | **已完成** |
| 向量嵌入 | ✅ 本地 sentence-transformers 或 TEI | ✅ 远程 OpenAI 兼容 API | **已完成** |
| Temporal 链接 | ✅ 连续事实间自动创建 | ✅ 连续事实间自动创建 | **已完成** |
| Entity 链接 | ✅ 共享实体的事实间创建 | ✅ 共享实体的事实间创建 | **已完成** |
| Semantic 链接 | ✅ 嵌入余弦 > 0.75 | ✅ 嵌入余弦 > 0.75 | **已完成** |
| Causal 链接 | ✅ LLM 判断因果 | ✅ LLM 判断因果 | **已完成** |
| 链接权重 | ✅ 基于 confidence/相似度 | ✅ 基于 confidence/相似度 | **已完成** |
| 重复检测 | ✅ 嵌入相似度去重 | ❌ 未实现 | Python 有完整去重管线 |
| 增量更新 | ✅ 已有事实可更新 | ❌ 未实现 | Python 支持事实版本管理 |
| 多阶段管线 | ✅ Phase 1-4，每阶段可配置 | ✅ 单次顺序执行 | Python 有更细粒度的阶段控制 |

### 4.2 Recall（记忆检索）

| 功能点 | Python 实现 | Rust 实现 | 差距 |
|--------|------------|-----------|------|
| FTS 全文搜索 | ✅ pg_trgm (PostgreSQL) | ✅ FTS5 + jieba 中文分词 | Rust 用 FTS5 替代 pg_trgm，各有优势 |
| 向量搜索 | ✅ pgvector HNSW 索引 | ✅ 暴力余弦搜索 | **性能差距**：HNSW 亚线性 vs O(n) |
| 图扩展 | ✅ 从 top-K 沿链接扩展 | ✅ 同逻辑 | **已完成** |
| 时间衰减 | ✅ recency_boost | ✅ 时间衰减加权 | **已完成** |
| 时间过滤 | ✅ date_start/date_end | ✅ 中文关键词 → 日期范围 | **已完成** |
| RRF 融合 | ✅ k=60 | ✅ k=60 | **已完成** |
| Reranking | ✅ Cross-encoder (本地/TEI) | ✅ 远程 rerank API | **已完成** |
| 降级策略 | ✅ rerank 失败回退 | ✅ rerank 失败回退 | **已完成** |
| fact_type 过滤 | ✅ | ✅ | **已完成** |
| tags 过滤 | ✅ | ✅ | **已完成** |
| 时间范围过滤 | ✅ SQL WHERE | ✅ SQL WHERE | **已完成** |
| recall_filtered API | ✅ | ✅ | **已完成** |
| 查询意图分析 | ✅ query_analyzer.py | ❌ 未实现 | Python 分析查询类型（事实/关系/时间） |
| 多 bank 检索 | ✅ 跨 bank 聚合 | ❌ 单 bank | 设计差异：设备端不需要跨 bank |
| 自适应 top-K | ✅ 根据查询复杂度调整 | ❌ 固定 limit | 优化项 |

### 4.3 Reflect（反思推理）

| 功能点 | Python 实现 | Rust 实现 | 差距 |
|--------|------------|-----------|------|
| Agent 循环 | ✅ max 10 轮 | ✅ max 6 轮 | Rust 轮数较少，够用 |
| Agent 工具 | ✅ 5 个工具 | ✅ 6 个工具 | Rust 多了 search_observations 和 done |
| Done 工具 | ✅ 带证据验证 | ✅ 带证据返回 | **已完成** |
| Disposition 感知 | ✅ 三维性格影响推理 | ✅ 三维性格影响推理 | **已完成** |
| Background context | ✅ | ✅ | **已完成** |
| 观察分析 | ✅ Trend 枚举 (STABLE/STRENGTHENING/WEAKENING/NEW/STALE) | ❌ 未实现 | Python 有完整趋势追踪 |
| 增量文档操作 | ✅ 8 种 delta_ops | ❌ 未实现 | Python 支持 ADD/UPDATE/REMOVE 等 |
| 结构化文档 | ✅ StructuredDoc + Block | ❌ 未实现 | Python 有层级文档管理 |
| 心智模型合成 | ✅ | ✅ | **已完成** |

### 4.4 Consolidate（记忆整合）

| 功能点 | Python 实现 | Rust 实现 | 差距 |
|--------|------------|-----------|------|
| 相似度聚类 | ✅ 嵌入聚类 | ✅ 嵌入聚类 | **已完成** |
| LLM 合成 | ✅ 合成心智模型 | ✅ 合成心智模型 | **已完成** |
| 阈值控制 | ✅ | ✅ | **已完成** |
| 批量整合 | ✅ | ✅ | **已完成** |

### 4.5 存储层对比

| 功能点 | Python (PostgreSQL) | Rust (SQLite) | 说明 |
|--------|-------------------|---------------|------|
| 数据库 | PostgreSQL 16+ | SQLite 3 (bundled) | 设备端无 PG，SQLite 是唯一选择 |
| 向量索引 | pgvector HNSW | 暴力搜索 | **最大性能差距** |
| 全文搜索 | pg_trgm (trigram) | FTS5 + jieba | FTS5+jieba 对中文更友好 |
| 事务 | ✅ ACID | ✅ WAL 模式 | 都支持事务 |
| 并发 | ✅ 多连接 | ✅ WAL 允许并发读 | SQLite 单写 |
| Schema 迁移 | ✅ Alembic | ✅ 建表时 IF NOT EXISTS | Python 有版本化迁移 |
| 多租户 | ✅ Schema 隔离 | ✅ bank_id 逻辑隔离 | 不同隔离级别 |

### 4.6 LLM 调用层对比

| 功能点 | Python | Rust | 说明 |
|--------|--------|------|------|
| Provider 支持 | ✅ OpenAI/Anthropic/Gemini/Groq/MiniMax/Ollama/LM Studio/LiteLLM/Claude Code | ✅ OpenAI 兼容接口 | Rust 统一为 OpenAI 格式，覆盖主流 |
| Chat Completion | ✅ | ✅ | **已完成** |
| Embedding | ✅ 本地/TEI | ✅ 远程 API | **已完成** |
| Reranking | ✅ 本地/TEI | ✅ 远程 API | **已完成** |
| 流式输出 | ✅ SSE | ❌ 未实现 | 设备端暂不需要 |
| Token 计数 | ✅ tiktoken | ❌ 未实现 | 优化项 |
| 重试/超时 | ✅ 指数退避 | ✅ 固定超时 60s | Python 有更完善的错误处理 |

### 4.7 FFI 与上层接口

| 功能点 | Python | Rust | 说明 |
|--------|--------|------|------|
| REST API | ✅ FastAPI | ❌ | Python 独有 |
| MCP Server | ✅ Model Context Protocol | ❌ | Python 独有，Rust 不需要 |
| C FFI | ❌ | ✅ 19 个导出函数 | Rust 独有 |
| 多租户 Handle | ❌ | ✅ Handle 注册表 | Rust 独有，支持多实例 |
| 控制平面 | ✅ Next.js UI | ❌ | Python 独有 |

## 五、优先级排序的差距清单

### P0 — 核心功能（已完成 ✅）

- [x] Retain: 事实提取 + 实体 + 嵌入 + 4 类链接
- [x] Recall: FTS5 + 向量 + 图 + 时间，RRF 融合 + Rerank
- [x] Reflect: Agent 多轮工具调用 + Disposition 感知
- [x] Consolidate: 相似度聚类 + LLM 合成
- [x] Bank CRUD + Disposition + Background context
- [x] C FFI 导出
- [x] Graph API: query_graph / get_related / inject_observation
- [x] recall_filtered: fact_type / tags / 时间范围过滤
- [x] 中文时间解析
- [x] FTS5 + jieba 中文分词
- [x] 三平台编译 (Linux ARM64 / OHOS / Android)

### P1 — 重要优化（待实现）

- [ ] **向量搜索优化** — 当前 O(n) 暴力搜索。方案：LanceDB 或 HNSW 纯 Rust 实现。影响：当记忆条数超过 ~1000 时性能显著下降
- [ ] **记忆去重** — Retain 时检测嵌入相似度 > 阈值的已有事实，合并而非重复创建。Python 有完整管线
- [ ] **实体归一化/别名** — "张伟"/"老张"/"张总" 应合并为同一实体。Python 的 entity_resolver 做了此事
- [ ] **查询意图分析** — 分析用户查询是事实型/关系型/时间型，优化检索策略。Python 有 query_analyzer
- [ ] **观察趋势追踪** — Reflect 时追踪观察的 STABLE/STRENGTHENING/WEAKENING/NEW/STALE 趋势
- [ ] **增量文档操作** — Reflect 产出 8 种 delta_ops（ADD/UPDATE/REMOVE/CONSOLIDATE 等），而非简单覆写

### P2 — 锦上添花（可后续迭代）

- [ ] Token 计数（tiktoken 或简化估算）
- [ ] 流式 LLM 输出
- [ ] 自适应 top-K（根据查询复杂度调整检索数量）
- [ ] 事实版本管理（已有事实的更新追踪）
- [ ] Retain 多阶段管线（Python 有 Phase 1-4，每阶段可配置）
- [ ] LLM 重试 + 指数退避
- [ ] Schema 版本化迁移（替代当前的 IF NOT EXISTS）

## 六、设计决策与权衡

### 为什么 SQLite 而不是 PostgreSQL

| 维度 | PostgreSQL | SQLite |
|------|-----------|--------|
| 部署 | 需要服务端进程 | 单文件，零部署 |
| 大小 | ~50MB+ | ~1MB (bundled) |
| 向量索引 | pgvector HNSW | 暴力搜索（暂时） |
| 并发 | 多连接并行读写 | WAL 模式，单写多读 |
| 全文搜索 | pg_trgm | FTS5 + jieba |
| 适合场景 | 服务端，多用户 | 设备端，单用户 |

**结论**：设备端只能用 SQLite。向量搜索的性能差距通过后续引入 LanceDB 解决。

### 为什么统一为 OpenAI 兼容接口

Python 版支持 9+ 种 LLM provider，因为服务端需要灵活适配不同客户的 API key。设备端场景不同：
- 用户通过统一 API 代理（如 DeepSeek、One API）访问后端模型
- OpenAI 兼容格式已成为事实标准（DeepSeek、Moonshot、GLM 等均兼容）
- 减少代码复杂度，一个 client.rs 覆盖所有场景

### 为什么用 jieba 而不是 pg_trgm

pg_trgm 是 trigram 级别的模糊匹配，对中文支持差（需要额外分词插件）。FTS5 + jieba 是中文全文搜索的标准方案：
- jieba 预分词后存入 FTS5
- 支持中文短语精确匹配
- 搜索时同样 jieba 分词后查询

### 为什么 Reflect 用 6 轮而非 10 轮

设备端场景的反思查询通常比服务端简单。6 轮是经验值，覆盖绝大多数查询场景。如果不够，可通过 FFI 参数调整。

## 七、性能预估

### 存储规模

| 操作 | 100 条记忆 | 1,000 条记忆 | 10,000 条记忆 |
|------|-----------|-------------|--------------|
| FTS 搜索 | < 1ms | < 5ms | < 20ms |
| 向量搜索 | ~5ms | ~50ms | ~500ms |
| 图遍历 (3跳) | < 1ms | < 10ms | < 50ms |
| Recall 总计 | ~100ms | ~200ms | ~600ms |

注：Recall 包含远程 LLM 调用（rerank），实际时间主要取决于网络延迟。

### 内存占用

- SQLite 缓存：默认 2MB（可配置）
- 向量数据：1024 dims × 4 bytes × N 条 = ~4KB/条
- 引擎本身：~10MB
- **总计 1000 条记忆**：约 15MB

## 八、后续路线

### Phase 1（当前）
- [x] 核心引擎：retain / recall / reflect / consolidate
- [x] C FFI：19 个导出函数
- [x] 三平台编译

### Phase 2
- [ ] LanceDB 集成（向量索引）
- [ ] 记忆去重
- [ ] 实体归一化
- [ ] 观察趋势追踪

### Phase 3
- [ ] OpenClaw (ArkTS) 集成
- [ ] ZeroClaw (Kotlin) 集成
- [ ] 端到端测试

### Phase 4
- [ ] 查询意图分析
- [ ] 增量文档操作
- [ ] 事实版本管理
- [ ] Retain 多阶段管线
