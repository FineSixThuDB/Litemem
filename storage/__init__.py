"""Storage layer — three persistent stores.

- memory_store: SQLite for history + recent messages buffer (mem0 SQLiteManager)
- vector_store: VexDB-Lite (DuckDB + VEX) for dense vectors + payload
- entity_store: a second VexDB-Lite collection scoped to entity records
"""
