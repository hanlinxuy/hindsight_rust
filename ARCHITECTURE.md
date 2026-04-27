# Hindsight Core - Rust Native Library

## 项目定位

Hindsight 是一个 AI agent 长期记忆系统。原始版本用 Python + PostgreSQL 实现（~37K 行）。
本仓库是其核心引擎的 Rust 移植版，编译为 `.so` 动态库，运行在鸿蒙 NEXT / Android 设备端。

**职责边界**：只做 core library，不做上层应用。上层调用方是 OpenClaw（鸿蒙 ArkTS）/ ZeroClaw（Android Kotlin）。

## 架构

```
OpenClaw (ArkTS)          ZeroClaw (Kotlin)
      |                        |
    NAPI                      JNI
      |                        |
      +---- C ABI (FFI) -------+
                |
    libhindsight_core.so (Rust)
         |
    ┌────┴────┐
    |         |
 远程 LLM   本地 SQLite
 (OpenAI/   (事实/向量/
  Anthropic)  实体/链接)
```

### 调用流程

1. **上层 App** 通过 NAPI/JNI 调用 C 风格函数（`hindsight_retain` / `hindsight_recall` / `hindsight_reflect`）
2. **FFI 层**（`ffi/mod.rs`）把 C 字符串/参数转成 Rust 类型，调引擎。使用 `OnceLock<Mutex<MemoryEngine>>` 保证线程安全。
3. **引擎**（`engine/memory_engine.rs`）编排三大操作
4. **LLM 客户端**（`llm/client.rs`）调远程 API 做 embedding / fact extraction / reranking / reasoning
5. **存储层**（`storage/sqlite.rs`）读写本地 SQLite 数据库

### 三大核心操作

**Retain（记忆写入）**
1. 存储原始文档到 documents 表
2. LLM 提取事实（world/experience/mental_model）
3. 存储事实到 memory_units 表
4. 调远程 embedding API → 存向量到 embeddings 表
5. LLM 提取实体 → 存实体表 + unit_entities 关联表
6. 构建时间顺序链接（consecutive facts）
7. 构建实体链接（仅链接实际共享实体的 fact 对）

**Recall（记忆检索）**
四路检索 + RRF 融合（k=60）：
1. FTS5 全文搜索
2. 向量余弦相似度搜索
3. 图扩展（从 top-5 结果沿 entity/temporal 链接扩展）
4. 远程 reranker 重排序（Cohere/Jina 兼容 API，失败时降级）

**Reflect（反思推理）**
1. Recall 检索相关记忆（top-10）
2. 带上 bank 的 disposition（skepticism/literalism/empathy 1-5）+ background_context
3. 调 LLM 生成 disposition-aware 的反思分析（JSON 格式）

## 文件结构

```
hindsight-core/src/
├── lib.rs                          # 模块导出
├── error.rs                        # 错误类型（7 变体）
│
├── models/                         # 数据结构
│   ├── mod.rs
│   ├── bank.rs                     # Bank, Disposition
│   ├── memory.rs                   # MemoryUnit, FactType, LinkType, MemoryLink
│   └── entity.rs                   # Entity
│
├── storage/                        # 存储层
│   ├── mod.rs
│   ├── schema.rs                   # SQLite DDL + FTS5 + 索引
│   └── sqlite.rs                   # SqliteStore（CRUD + 向量搜索 + FTS + 图遍历）
│
├── llm/                            # LLM 远程调用
│   ├── mod.rs
│   └── client.rs                   # reqwest + rustls：chat / embed / rerank
│
├── engine/                         # 核心引擎
│   ├── mod.rs
│   └── memory_engine.rs            # retain / recall / reflect + bank/memory 管理
│
└── ffi/                            # C FFI 导出
    └── mod.rs                      # 15 个 C API 函数（线程安全 + null 安全）

hindsight-core/tests/
└── storage_test.rs                 # 12 个存储层单元测试

hindsight-core.h                    # C 头文件
```

## C FFI 接口

| 函数 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `hindsight_init` | db_path, api_key, base_url, model | i32 | 初始化引擎 |
| `hindsight_create_bank` | name, sk, li, em | *mut c_char (bank_id) | 创建记忆银行 |
| `hindsight_get_bank` | bank_id | *mut c_char (JSON) | 获取银行信息 |
| `hindsight_list_banks` | - | *mut c_char (JSON array) | 列出所有银行 |
| `hindsight_delete_bank` | bank_id | i32 | 删除银行及全部数据 |
| `hindsight_set_bank_context` | bank_id, context | i32 | 设置银行背景上下文 |
| `hindsight_retain` | bank_id, content | *mut c_char (JSON) | 写入记忆 |
| `hindsight_recall` | bank_id, query, limit | *mut c_char (JSON) | 检索记忆 |
| `hindsight_reflect` | bank_id, query | *mut c_char (JSON) | 反思推理 |
| `hindsight_delete_memory` | memory_id | i32 | 删除单条记忆 |
| `hindsight_list_memories` | bank_id, limit, offset | *mut c_char (JSON) | 分页列出记忆 |
| `hindsight_version` | - | *const c_char | 版本号 |
| `hindsight_free_string` | *mut c_char | void | 释放库分配的字符串 |

所有返回 JSON 字符串的函数，调用方用完后必须调 `hindsight_free_string` 释放内存。

## 数据库 Schema（SQLite）

| 表 | 用途 |
|----|------|
| `banks` | 记忆银行（隔离的记忆空间 + disposition + background_context） |
| `memory_units` | 核心表：提取的事实 |
| `embeddings` | 向量嵌入（f32 BLOB） |
| `memory_links` | 记忆间链接（语义/时间/实体/因果） |
| `entities` | 命名实体（人/地/组织/概念） |
| `unit_entities` | 实体-记忆关联（多对多） |
| `documents` | 原始输入文档 |
| `memory_units_fts` | FTS5 全文搜索虚拟表 |

向量搜索：暴力余弦相似度（MVP），Recall 中 RRF 融合 FTS + vector + graph 三路分数。

## 编译状态

在 ARM64 Linux 服务器上验证：

| Target | 产出 | 大小 | 状态 |
|--------|------|------|------|
| `aarch64-unknown-linux-gnu` | `libhindsight_core.so` | 3.6 MB | ✅ 编译通过 + 12 测试通过 |
| `aarch64-unknown-linux-ohos` | `libhindsight_core.so` | 3.6 MB | ✅ 编译通过 |
| `aarch64-linux-android` | - | - | ⏳ target 下载中 |

OHOS 编译使用 stub 库（`libunwind.so`, `libtime_service_ndk.so`）绕过链接。这两个库在真实鸿蒙设备上有系统实现。

## 安全措施

- **FFI null safety**: 所有指针先判空再转 CStr
- **线程安全**: `OnceLock<Mutex<MemoryEngine>>` 全局引擎
- **HTTP timeout**: 60s 请求 + 10s 连接超时
- **TLS**: rustls（纯 Rust，无 OpenSSL 依赖）
- **SQLite**: bundled 模式（内嵌 SQLite，不依赖系统库）

## 与 Python 版本的功能对标

| 功能 | Python | Rust | 状态 |
|------|--------|------|------|
| 记忆银行 CRUD | ✅ | ✅ | 完成 |
| Disposition 特质 | ✅ | ✅ | 完成 |
| Background context | ✅ | ✅ | 完成 |
| 事实提取（LLM） | ✅ | ✅ | 完成 |
| 实体提取 | ✅ | ✅ | 完成 |
| 实体关联（unit_entities） | ✅ | ✅ | 完成 |
| 向量嵌入 | ✅ | ✅ | 完成 |
| FTS 全文搜索 | ✅ (pg_trgm) | ✅ (FTS5) | 完成 |
| 图遍历 | ✅ | ✅ | 完成 |
| RRF 融合 | ✅ | ✅ | 完成 |
| Reranking | ✅ | ✅ | 完成 |
| Reflect 推理 | ✅ | ✅ | 完成 |
| 原始文档存储 | ✅ | ✅ | 完成 |
| 记忆删除 | ✅ | ✅ | 完成 |
| 银行删除 | ✅ | ✅ | 完成 |
| C FFI 接口 | - | ✅ | Rust 新增 |
| 向量搜索优化 | pgvector | 暴力搜索 | 后续用 LanceDB |
| 记忆合并/巩固 | ✅ | - | 未移植 |
