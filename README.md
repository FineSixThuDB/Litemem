# LiteMem

> 一个仿照 [mem0 V3](https://github.com/mem0ai/mem0) 实现的轻量级 memory system。
> 每种 technique 拆到独立文件里，方便逐个理解；底层只保留一套 LLM/embedding 接口（OpenAI 兼容）和一个 vector store（[VexDB-Lite](https://github.com/VexDB-AI/vexdb-lite)）。

## 设计目标

- **逐文件可读**：write/read pipeline 的每一阶段一个文件，模块边界 = mem0 V3 的阶段边界。
- **精度对齐 mem0**：prompts、entity 抽取、md5 dedup、UUID→int 反幻觉映射、sigmoid BM25 归一化、entity boost 公式都按 mem0 V3 原样移植。
- **依赖收敛**：mem0 自带 20+ LLM、30+ vector store 接入；LiteMem 只保留 OpenAI-compatible LLM/embedder + 一个 VexDB-Lite vector store。
- **只做 sync**：不实现 async/AsyncMemory；保留 history、procedural memory、`linked_memory_ids` 软关联。

## 目录结构

```
litemem/
├── __init__.py
├── config.py                  # dataclass 配置（LLM / Embedder / VectorStore / LiteMemConfig）
├── data_models.py             # MemoryItem / VectorRecord / ScoredMemory / ExtractedFact
├── main.py                    # LiteMem 编排类（add / search / get / update / delete / history / reset）
│
├── write_pipeline/            # 写入流水线（对齐 mem0 V3 _add_to_vector_store 的 Phase 0–8）
│   ├── memory_extractor.py    # Phase 0–2：上下文收集 + LLM additive extraction
│   ├── deduplicator.py        # Phase 4–5：md5 hash 跨批/批内去重
│   ├── memory_writer.py       # Phase 3、6、8：批 embed + 批写 + 写消息缓冲
│   ├── entity_linker.py       # Phase 7：实体抽取 / 全局去重 / 批 search-then-upsert
│   └── procedural_memory.py   # 单独的 procedural memory 写入路径
│
├── read_pipeline/             # 检索流水线（对齐 mem0 V3 _search_vector_store 的 Step 1–9）
│   ├── query_preprocessor.py  # Step 1：lemmatize + entity extraction
│   ├── semantic_retriever.py  # Step 3：dense ANN（vector_store.search）
│   ├── keyword_retriever.py   # Step 4：按 session 即时建 rank_bm25 索引
│   ├── entity_retriever.py    # Step 6：实体 boost（mem0 同款 spread-attenuated 公式）
│   ├── rank_fusion.py         # Step 5、7、8：BM25 sigmoid 归一化 + 加性融合 + top-k
│   └── context_builder.py     # Step 9：MemoryItem 格式化
│
├── storage/
│   ├── memory_store.py        # SQLite history + recent messages buffer（mem0 SQLiteManager 移植）
│   ├── vector_store.py        # VexDB-Lite（DuckDB + VEX）向量库实现
│   └── entity_store.py        # 第二张 VexDB-Lite 表，专门放实体 + linked_memory_ids
│
├── utils/
│   ├── prompts.py             # ADDITIVE_EXTRACTION_PROMPT / PROCEDURAL_MEMORY_SYSTEM_PROMPT（mem0 原文）
│   ├── text_utils.py          # lemmatize_for_bm25 / extract_entities / extract_json / md5
│   ├── llm_client.py          # 单一 OpenAI-compatible chat completions 包装
│   └── embeddings.py          # 单一 OpenAI-compatible embeddings 包装
│
└── evaluation/
    ├── metrics.py             # token_count / recall@k / mrr
    └── benchmark_runner.py    # JSONL 输入的小型 benchmark
```

## 文件 ↔ mem0 V3 阶段对照表

### 写入：`LiteMem.add()` → `_add_to_vector_store`

| Phase | mem0 中的位置 | LiteMem 文件 |
|---|---|---|
| 0 | `db.get_last_messages` + `parse_messages` | `main.py` 调 `memory_store.get_last_messages` |
| 1 | `vector_store.search(top_k=10)` 拉 existing | `main.py` 中直接调 `vector_store.search` |
| 2 | LLM additive extraction + UUID→int 映射 | `write_pipeline/memory_extractor.py` |
| 3 | `embed_batch` | `write_pipeline/memory_writer.py::write_facts` |
| 4 | hash dedup（existing + within-batch） | `write_pipeline/deduplicator.py` |
| 5 | 同上 | 同上 |
| 6 | `vector_store.insert` + `batch_add_history` | `write_pipeline/memory_writer.py` |
| 7 | batch entity linking（global dedup + batch embed + batch search + split-and-upsert） | `write_pipeline/entity_linker.py::link_memories` |
| 8 | `db.save_messages` | `write_pipeline/memory_writer.py::write_facts` 末尾 |

### 检索：`LiteMem.search()` → `_search_vector_store`

| Step | mem0 行为 | LiteMem 文件 |
|---|---|---|
| 1 | `lemmatize_for_bm25` + `extract_entities` | `read_pipeline/query_preprocessor.py` |
| 2 | embed query | `read_pipeline/semantic_retriever.py::embed_query` |
| 3 | `vector_store.search`（over-fetch `max(top_k*4, 60)`） | `read_pipeline/semantic_retriever.py` |
| 4 | `vector_store.keyword_search`（BM25） | `read_pipeline/keyword_retriever.py`（rank_bm25 实现） |
| 5 | `get_bm25_params` + `normalize_bm25` | `read_pipeline/rank_fusion.py::normalize_bm25_scores` |
| 6 | entity boost（threshold=0.5、weight=0.5、spread attenuation） | `read_pipeline/entity_retriever.py` |
| 7 | 候选集构造 | `read_pipeline/rank_fusion.py::RankFusion.fuse` |
| 8 | additive score + max_possible 自适应分母 | 同上 |
| 9 | 格式化为 MemoryItem dict | `read_pipeline/context_builder.py` |

## VexDB-Lite 适配要点

VexDB-Lite 是 DuckDB 的扩展，靠 SQL 操作向量。LiteMem 的 schema：

```sql
CREATE TABLE litemem (
    id        VARCHAR PRIMARY KEY,
    embedding FLOAT[1536],
    -- 提升为独立列以便建索引/快速过滤
    user_id   VARCHAR,
    agent_id  VARCHAR,
    run_id    VARCHAR,
    actor_id  VARCHAR,
    role      VARCHAR,
    hash      VARCHAR,
    data      VARCHAR,
    text_lemmatized VARCHAR,
    created_at VARCHAR,
    updated_at VARCHAR,
    attributed_to VARCHAR,
    -- 其余 metadata 走 JSON
    payload   VARCHAR
);
CREATE INDEX litemem_vec_idx ON litemem USING GRAPH_INDEX(embedding)
  WITH (metric='cosine', m=16, ef_construction=200);
```

- 检索：`ORDER BY cosine_distance(embedding, ?::FLOAT[d]) LIMIT k`，再把 `1 - distance` 还原成 mem0 期望的相似度分数。
- 过滤：把 `eq/ne/gt/in/contains/AND/OR/NOT` 等 mem0 enhanced filter 翻译成 SQL `WHERE`；非 promoted 字段用 `json_extract_string(payload, '$.field')`。
- 持久化：`con = vexdb_lite.connect("path/to/litemem.db")`，`close()` 时 `CHECKPOINT`。
- BM25：VexDB-Lite 不支持，**改在 Python 用 `rank_bm25.BM25Okapi`**，每次 search 按 session 拉一次 `(id, text_lemmatized)` 重建索引（user/agent/run 三层过滤后通常只有几百到几万行，可接受）。

## 安装与运行

LiteMem 自身没有 `setup.py`，按 PYTHONPATH 用即可：

```bash
# 强烈推荐的依赖
pip install openai duckdb spacy rank_bm25
python -m spacy download en_core_web_sm
# VexDB-Lite：从 release 装 wheel
pip install /path/to/vexdb_lite-1.5.0-cp312-cp312-*.whl
```

```python
import os
os.environ.setdefault("OPENAI_API_KEY", "sk-...")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")  # 任意 OpenAI 兼容端点

from litemem import LiteMem, LiteMemConfig, LLMConfig, EmbedderConfig, VectorStoreConfig

config = LiteMemConfig(
    llm=LLMConfig(model="gpt-4o-mini"),
    embedder=EmbedderConfig(model="text-embedding-3-small", embedding_dims=1536),
    vector_store=VectorStoreConfig(
        collection_name="demo",
        embedding_dims=1536,
        db_path="./litemem_demo.db",
        distance_metric="cosine",
    ),
)
mem = LiteMem(config)

mem.add(
    [
        {"role": "user", "content": "我叫小明，住在上海，是一名 CS 研究生。"},
        {"role": "assistant", "content": "好的，我会记住的。"},
    ],
    user_id="xiaoming",
)

print(mem.search("用户在哪个城市？", filters={"user_id": "xiaoming"}))
mem.close()
```

## 与 mem0 的差异

| 维度 | mem0 V3 | LiteMem |
|---|---|---|
| LLM 接口 | 20+ provider（OpenAI、Anthropic、Bedrock、Gemini、…） | 1 个 OpenAI 兼容 |
| Embedding | 14+ provider | 1 个 OpenAI 兼容 |
| Vector Store | 30+（Qdrant、Pinecone、…） | 1 个 VexDB-Lite |
| BM25 | Qdrant 原生 sparse 向量 | rank_bm25（按 session 即时建索引） |
| Sync/Async | 都有 | 仅 sync |
| Telemetry | 启用 | 移除 |
| 配置系统 | Pydantic v2 | dataclass |
| 多模态 | 支持图像描述 | 暂不支持 |

`linked_memory_ids` / hash dedup / UUID→int 映射 / sigmoid BM25 归一化 / entity spread attenuation 等核心精度技巧全部对齐。

litemem/                                    （~/mem1/litemem/ — Python 包名小写）
├── config.py            dataclass 配置
├── data_models.py       MemoryItem / VectorRecord / ScoredMemory / ExtractedFact
├── main.py              LiteMem 编排类（add/search/get/update/delete/history/reset）
│
├── write_pipeline/      （对齐 _add_to_vector_store Phase 0-8）
│   ├── memory_extractor.py     ← Phase 0-2：上下文 + ADDITIVE_EXTRACTION_PROMPT + UUID→int 反幻觉
│   ├── deduplicator.py         ← Phase 4-5：md5 跨批/批内 hash dedup
│   ├── memory_writer.py        ← Phase 3、6、8：批 embed + 批写 + history + messages 缓冲
│   ├── entity_linker.py        ← Phase 7：global dedup + batch embed + batch search + split-and-upsert
│   └── procedural_memory.py    ← agent-only 写入路径
│
├── read_pipeline/       （对齐 _search_vector_store Step 1-9）
│   ├── query_preprocessor.py   ← lemmatize + extract_entities
│   ├── semantic_retriever.py   ← dense ANN（over-fetch max(top_k*4,60)）
│   ├── keyword_retriever.py    ← rank_bm25 按 session 即时建索引
│   ├── entity_retriever.py     ← entity boost（threshold=0.5, weight=0.5, spread attenuation）
│   ├── rank_fusion.py          ← BM25 sigmoid（query-length adaptive）+ 加性融合 + 自适应分母
│   └── context_builder.py      ← 格式化为 MemoryItem dict
│
├── storage/
│   ├── memory_store.py    SQLite：history + last 10 messages per session_scope
│   ├── vector_store.py    VexDB-Lite：SQL schema + GRAPH_INDEX(cosine) + WHERE 转译 + JSON payload
│   └── entity_store.py    VexDB-Lite 第二张表（{collection}_entities）
│
├── utils/
│   ├── prompts.py         ADDITIVE_EXTRACTION_PROMPT / AGENT_CONTEXT_SUFFIX / PROCEDURAL_PROMPT（mem0 原文）
│   ├── text_utils.py      lemmatize_for_bm25 / extract_entities（spaCy 优雅降级） / extract_json / md5
│   ├── llm_client.py      单一 OpenAI 兼容 chat 接口（含 o1/o3/gpt-5 reasoning 模型识别）
│   └── embeddings.py      单一 OpenAI 兼容 embeddings 接口
│
└── evaluation/
    ├── metrics.py         token_count / recall_at_k / mrr
    └── benchmark_runner.py JSONL 输入小型 benchmark

