"""Read pipeline modules — one file per signal/step of mem0 V3 search().

- query_preprocessor: lemmatize + entity extraction
- semantic_retriever: dense vector ANN
- keyword_retriever:  rank_bm25 over session-scoped corpus
- entity_retriever:   entity store boost
- rank_fusion:        additive score with adaptive divisor
- context_builder:    format scored hits into MemoryItem dicts
"""
