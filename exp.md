# token-efficient experiment

## 成本建模

成本被建模为3类：
```shell
    A. Chat LLM token cost
    input tokens / output tokens / cached tokens / reasoning tokens

    B. Embedding token cost
    add embedding tokens / search query embedding tokens / entity embedding tokens

    C. Non-token cost
    BM25 构建耗时 / vector search 耗时 / entity search 耗时 / SQLite/VexDB IO
```
很多 technique 对 token 没影响，但对 latency 和 accuracy 有影响。比如：
```shell
    BM25: 不调用 LLM，不增加 chat token，但会增加 CPU 时间，可能提高关键词型 query accuracy

    Entity boost: 不调用 chat LLM，但 search 时会多做 entity embedding，所以增加 embedding token，可能提高实体型 query accuracy

    Last k messages: 会进入 add() 的 extraction prompt，增加 LLM input tokens，可能提高上下文一致性

    Existing memories retrieval: 本身是 embedding/vector search，不是 LLM，但检索出来的 existing memories 会塞进 extraction prompt，所以间接增加 LLM input tokens

    ContextBuilder: 只是格式化，不调用 LLM，不影响 token
```

## memory system techniques 建模
常规 LiteMem 路径里，真正直接调用 chat LLM 的核心 technique 其实只有一个：memory extraction。
但是，这个 LLM call 里面又包含多个可以单独消融的 LLM-prompt techniques。它们不会“额外调用一次 LLM”，但会改变这一次 LLM 调用的输入 prompt.

```shell
    A. 直接 chat LLM technique
    B. 改变 chat LLM prompt 的 technique
    C. 不调用 chat LLM 的 retrieval / postprocess technique
```
techniques编号为:

```shell
    L1: Additive memory extraction
        直接调用 chat LLM。
        这是 add() 常规路径里的核心 LLM technique。

    L2: Existing memories in extraction prompt
        不额外调用 chat LLM，但会把检索到的旧 memory 塞进 L1 的 prompt。
        影响 input tokens，也影响去重、链接、抽取质量。

    L3: Recent messages in extraction prompt
        不额外调用 chat LLM，但会把 last-k messages 塞进 L1 的 prompt。
        影响 input tokens，主要帮助指代消解、上下文补全、时间理解。

    L4: UUID anonymization inside extraction prompt
        不额外调用 chat LLM，但改变 L1 prompt 里的 memory id 表示。
        影响 token 很小，主要影响 linked_memory_ids 的稳定性和反幻觉。

    L5: JSON response_format
        不额外调用 chat LLM，但改变 L1 的 API 参数。
        影响输出可解析性、失败率、可能轻微影响 output tokens。

    R1: Semantic retrieval
        不调用 chat LLM，但调用 embedding API。
        是 search 候选池主路径。

    R2: BM25 rerank
        不调用 LLM，也不调用 embedding。
        本地 CPU technique，只影响重排。

    R3: Entity linking + entity boost
        不调用 chat LLM，但调用 embedding API。
        add 阶段建 entity store，search 阶段用 entity boost 加分。

    P1: Hash dedup
        不调用 LLM。
        是 LLM extraction 之后的 postprocess，会影响写入 memory 数量，进而间接影响后续检索成本。

    P2: ContextBuilder formatting
        不调用 LLM。
        只改变返回 JSON 结构，不建议作为 accuracy ablation technique。
```

## query workload 类型建模
在我们实际并不知道query workload当中，真正影响一个LLM的token消耗的“主要矛盾”是什么时，我们只能先通过，query workload本身的任务对query workload类型进行建模。
```shell
    按 query 类型分组：
    1. 语义泛化型 query
    2. 精确关键词型 query
    3. 实体指代型 query
    4. 时间/上下文型 query
    5. 多跳/关联型 query
```
同时为每条 query 保存可重新分组的 observable features：
```shell
    query_token_len
    query_char_len
    entity_count
    has_temporal_expression
    has_pronoun_or_coreference
    keyword_density
    gold_evidence_count
    expected_hop_count，如 dataset metadata 有。
    dataset_category/question_type/source
    retrieved_memory_count
    retrieved_context_tokens
    answer_prompt_tokens
    context_write_memory_count
    context_total_write_tokens
```
LoCoMo 优先用官方 category：multi-hop、temporal、open-domain、single-hop、adversarial；MemBench 优先用 slice/branch。没有 metadata 时再用 deterministic heuristic classifier。

## experiment metrics:

```shell
    某 technique 在某类 workload 上，
    带来了多少 accuracy gain，
    付出了多少 chat token / embedding token / latency cost？
```

```shell
    technique -> accuracy delta
    technique -> token delta
    technique -> latency delta
    technique -> cost-effectiveness
```

每个 config × query 输出：

* Retrieval metrics: recall@k、MRR、source-id recall。
* QA metrics: benchmark 原有 F1/exact_match/official_score。
* Cost metrics: C_chat_litemem、C_embed_litemem、C_non_token_litemem、C_answer_llm。
* Efficiency metrics:
    delta_accuracy = metric(config) - metric(C_FULL)
    delta_cost = cost(config) - cost(C_FULL)
    accuracy_per_1k_tokens
    accuracy_per_second
    Pareto status: cheaper-and-not-worse / more-expensive-and-better / dominated。


## experiment plan 

xxxxxx

```
最后得到表
```shell
config | accuracy | input_tokens | output_tokens | cached_tokens | embedding_tokens | latency
```

