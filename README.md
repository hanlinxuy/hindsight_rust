# Hindsight Core (Rust)

Hindsight AI Agent 记忆系统的设备端核心引擎，Rust 原生实现。

将 [Hindsight](https://github.com/hanlinxuy/hindsight) Python+PostgreSQL 服务端架构完整移植为设备端 SQLite + 远程LLM 的轻量方案，目标平台为 HarmonyOS NEXT、Android、Linux ARM64。

## 特性

- **全离线存储** — SQLite WAL 模式，FTS5 全文检索，jieba 中文分词
- **远程LLM** — 仅事实提取/反思推理调用远程 API（OpenAI 兼容接口）
- **多策略检索** — FTS5 + 向量语义 + 图扩展 + 时间感知，RRF 融合 + Rerank
- **Agent 反思** — 多轮工具调用循环（recall / search_observations / expand_context / get_entity_relations / done）
- **记忆整合** — 嵌入相似度聚类 + LLM 合成心智模型
- **多租户** — Handle 注册表，每个租户独立引擎和数据库
- **C FFI** — 19 个导出函数，cdylib/staticlib/lib 三种 crate 类型
- **三平台编译** — aarch64-unknown-linux-gnu / aarch64-unknown-linux-ohos / aarch64-linux-android

## 快速开始

```bash
# 构建
cargo build --release

# 运行测试（需要 LLM API）
HINDSIGHT_TEST_API_KEY=your-key \
HINDSIGHT_TEST_BASE_URL=https://api.deepseek.com \
HINDSIGHT_TEST_MODEL=deepseek-chat \
HINDSIGHT_TEST_EMBEDDING_BASE_URL=http://your-embedding-server:8001 \
HINDSIGHT_TEST_EMBEDDING_MODEL=bge-m3 \
cargo test --test integration_test

# 单元测试（不需要 API）
cargo test --test storage_test
```

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                   上层应用                        │
│         (HarmonyOS / Android / Linux)            │
├─────────────────────────────────────────────────┤
│                C FFI (19 函数)                    │
├─────────────────────────────────────────────────┤
│              MemoryEngine 核心                    │
│  ┌──────────┬──────────┬───────────┬──────────┐ │
│  │ Retain   │ Recall   │ Reflect   │Consolidate│ │
│  │ 事实提取  │ 多策略   │ Agent循环  │ 相似整合  │ │
│  │ 实体识别  │ RRF融合  │ 工具调用   │ 心智模型  │ │
│  │ 4类链接   │ 时间过滤  │ Done结束   │          │ │
│  └──────────┴──────────┴───────────┴──────────┘ │
├─────────────────────────────────────────────────┤
│  SqliteStore     │  LlmClient                   │
│  SQLite + FTS5   │  OpenAI 兼容 HTTP API        │
│  jieba 中文分词   │  (DeepSeek / OpenAI / etc)   │
└─────────────────────────────────────────────────┘
```

## 核心 API

| 操作 | 说明 |
|------|------|
| `retain(bank_id, content)` | 存储：LLM提取事实→实体→嵌入→4类链接 |
| `recall(bank_id, query, limit)` | 检索：FTS5+向量+图+时间 4策略RRF融合 |
| `recall_filtered(...)` | 过滤检索：支持 fact_type / tags / 时间范围 |
| `reflect(bank_id, query)` | 反思：6工具Agent循环，disposition感知 |
| `consolidate(bank_id, threshold)` | 整合：相似记忆聚类→LLM合成心智模型 |
| `inject_observation(...)` | 注入：绕过LLM直接存储结构化事实 |
| `query_graph(memory_id, hops)` | 图查询：BFS N跳遍历记忆链接 |
| `get_related(bank_id, entity)` | 实体查询：按实体名检索关联记忆 |

## 链接类型

| 类型 | 生成时机 | 说明 |
|------|----------|------|
| Temporal | retain 时连续事实间 | 时间顺序关系 |
| Entity | retain 时共享实体间 | 实体共现关系 |
| Semantic | retain 时嵌入余弦>0.75 | 语义相似关系 |
| Causal | retain 时 LLM 判断 | 因果推理关系 |

## 文件结构

```
hindsight-core/
├── src/
│   ├── engine/
│   │   └── memory_engine.rs    # 核心引擎 (~950行)
│   ├── storage/
│   │   ├── schema.rs           # SQLite 表结构 + FTS5 + 触发器
│   │   └── sqlite.rs           # 存储层 (CRUD + 搜索 + 图遍历)
│   ├── llm/
│   │   └── client.rs           # OpenAI 兼容 HTTP 客户端
│   ├── models/                 # 数据模型 (Bank, MemoryUnit, Entity, Link)
│   ├── ffi/
│   │   └── mod.rs              # C FFI 多租户导出层
│   ├── error.rs                # 错误类型
│   └── lib.rs
├── tests/
│   ├── storage_test.rs         # 15 个存储单元测试
│   ├── integration_test.rs     # 5 个 API 集成测试
│   └── profile_test.rs         # 50事实画像 + recall + reflect
├── build.rs                    # cbindgen 自动生成 .h
├── Cargo.toml
└── hindsight-core.h            # 生成的 C 头文件
```

## 数据库表结构

| 表 | 说明 |
|----|------|
| `banks` | 记忆库（含 disposition 三维性格 + 背景上下文）|
| `memory_units` | 记忆单元（事实/经验/心智模型 + 时间线 + 标签）|
| `embeddings` | 向量嵌入（BLOB 存储）|
| `memory_links` | 记忆间链接（4种类型 + 权重）|
| `entities` | 实体（人物/地点/概念/事物）|
| `unit_entities` | 记忆-实体关联 |
| `documents` | 原始文档 |
| `memory_units_fts` | FTS5 全文检索虚拟表（jieba 预分词）|

## 测试覆盖

| 测试套件 | 数量 | 说明 |
|----------|------|------|
| storage_test | 15 | 单元测试，不需要 API |
| integration_test | 5 | 真实 API 集成测试 |
| profile_test | 1 | 50条中文事实 → recall → reflect → consolidate |

## 跨平台编译

```bash
# 一键编译三平台
./scripts/build-all.sh release

# 或单独编译
cargo build --target aarch64-unknown-linux-ohos --release   # HarmonyOS
cargo build --target aarch64-linux-android --release         # Android
cargo build --target aarch64-unknown-linux-gnu --release     # Linux ARM64
```

## 与 Python 版差异

参见 [DESIGN.md](./DESIGN.md) 了解完整设计分析和差距对比。

## License

MIT
