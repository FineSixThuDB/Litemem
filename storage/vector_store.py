"""VexDB-Lite vector store — mem0's only ``VectorStoreFactory`` target in LiteMem.

VexDB-Lite is a DuckDB extension distributed as a Python wheel. We talk to
it through SQL:

    import vexdb_lite as vex
    con = vex.connect("path/to/litemem.db")
    con.execute("CREATE TABLE ... (..., embedding FLOAT[1536])")
    con.execute("CREATE INDEX idx ON tbl USING GRAPH_INDEX(embedding) "
                "WITH (metric='cosine')")
    con.execute("SELECT id FROM tbl ORDER BY cosine_distance(embedding, ?) LIMIT 10")

Mapping to mem0's Qdrant payload model:
- ``id``          → VARCHAR primary key (UUID string)
- ``vectors``     → ``embedding FLOAT[dim]`` column
- ``payload``     → split into:
  * promoted columns ``user_id`` / ``agent_id`` / ``run_id`` / ``actor_id`` /
    ``role`` / ``hash`` / ``data`` / ``text_lemmatized`` / ``created_at`` /
    ``updated_at`` — these have dedicated columns for fast filter + index
  * everything else stored as JSON in the ``payload`` TEXT column

What VexDB-Lite does NOT support natively:
- BM25 / full-text search → handled by ``read_pipeline.keyword_retriever``
  with rank_bm25 over a session-scoped corpus pulled from this store.

Distance → similarity conversion (so we match mem0's "higher = better"
score convention used by ``score_and_rank``):
- cosine: similarity = 1 - cosine_distance   (range [0, 1])
- l2:     similarity = 1 / (1 + l2_distance) (range (0, 1])
- ip:     similarity = inner_product         (unbounded; assume normalized)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from litemem.config import VectorStoreConfig
from litemem.data_models import VectorRecord

logger = logging.getLogger(__name__)

# Columns that are promoted to their own typed VARCHAR for indexed filter access.
# The remaining payload fields live in the JSON ``payload`` column.
_PROMOTED_COLUMNS = (
    "user_id", "agent_id", "run_id", "actor_id", "role",
    "hash", "data", "text_lemmatized", "created_at", "updated_at",
    "attributed_to",
)

# Operators supported by ``_build_where``. Mirrors mem0's enhanced filter syntax.
_SUPPORTED_OPS = frozenset(
    ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]
)
_OP_SQL = {
    "eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
}


def _connect_vex(db_path: str):
    """Open a VexDB-Lite connection (DuckDB + VEX extension auto-loaded)."""
    try:
        import vexdb_lite as vex
    except ImportError as e:
        raise ImportError(
            "vexdb_lite is required. Install the wheel from VexDB-Lite "
            "releases or run: pip install vexdb-lite"
        ) from e
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return vex.connect(db_path)


class _SQLiteVectorStore:
    """SQLite fallback with Python-side vector scoring.

    This keeps LiteMem runnable when the VEX DuckDB extension is unavailable.
    It intentionally favors compatibility over ANN performance: embeddings are
    stored as JSON and filtered rows are scored in Python.
    """

    def __init__(self, config: VectorStoreConfig, table_name: str):
        self.config = config
        self.table = table_name
        self.dim = config.embedding_dims
        self.distance = config.distance_metric.lower()
        self.db_path = config.db_path
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        try:
            self._create_table_and_index()
        except sqlite3.DatabaseError as e:
            if self.db_path == ":memory:" or "file is not a database" not in str(e).lower():
                raise
            self.con.close()
            self.db_path = self.db_path + ".sqlite"
            logger.warning(
                "Existing vector DB is not SQLite; using fallback file %s",
                self.db_path,
            )
            self.con = sqlite3.connect(self.db_path, check_same_thread=False)
            self.con.row_factory = sqlite3.Row
            self._create_table_and_index()

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _split_payload(payload: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        payload = payload or {}
        promoted = {c: payload.get(c) for c in _PROMOTED_COLUMNS}
        rest = {k: v for k, v in payload.items() if k not in _PROMOTED_COLUMNS}
        return promoted, json.dumps(rest, ensure_ascii=False)

    @staticmethod
    def _merge_payload(promoted_row: Dict[str, Any], json_payload: Optional[str]) -> Dict[str, Any]:
        rest = json.loads(json_payload) if json_payload else {}
        for k, v in promoted_row.items():
            if v is not None:
                rest[k] = v
        return rest

    def _create_table_and_index(self) -> None:
        table = self._quote_ident(self.table)
        col_defs = ",\n                ".join(
            f"{self._quote_ident(c)} TEXT" for c in _PROMOTED_COLUMNS
        )
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id        TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                {col_defs},
                payload   TEXT
            )
            """
        )
        for col in ("user_id", "agent_id", "run_id", "actor_id", "hash"):
            try:
                self.con.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._quote_ident(self.table + '_' + col + '_idx')} "
                    f"ON {table}({self._quote_ident(col)})"
                )
            except Exception as e:
                logger.debug(f"Could not create {col} SQLite index on {self.table}: {e}")
        self.con.commit()

    def _fetch_records(self, include_embedding: bool = False) -> List[Tuple[VectorRecord, Optional[List[float]]]]:
        cols = ["id"]
        if include_embedding:
            cols.append("embedding")
        cols.extend(_PROMOTED_COLUMNS)
        cols.append("payload")
        sql = f"SELECT {', '.join(self._quote_ident(c) for c in cols)} FROM {self._quote_ident(self.table)}"
        rows = self.con.execute(sql).fetchall()
        out: List[Tuple[VectorRecord, Optional[List[float]]]] = []
        for row in rows:
            offset = 1
            vector = None
            if include_embedding:
                try:
                    vector = [float(x) for x in json.loads(row["embedding"])]
                except Exception:
                    vector = []
                offset += 1
            promoted = {c: row[c] for c in _PROMOTED_COLUMNS}
            payload = self._merge_payload(promoted, row["payload"])
            out.append((VectorRecord(id=row["id"], score=None, payload=payload), vector))
        return out

    @staticmethod
    def _as_number(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _compare(cls, left: Any, op: str, right: Any) -> bool:
        if op in {"gt", "gte", "lt", "lte"}:
            left_num = cls._as_number(left)
            right_num = cls._as_number(right)
            if left_num is not None and right_num is not None:
                left_value, right_value = left_num, right_num
            else:
                left_value, right_value = str(left), str(right)
            if op == "gt":
                return left_value > right_value
            if op == "gte":
                return left_value >= right_value
            if op == "lt":
                return left_value < right_value
            return left_value <= right_value
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "in":
            return left in (right or [])
        if op == "nin":
            return left not in (right or [])
        if op == "contains":
            if isinstance(left, (list, tuple, set)):
                return right in left
            return str(right) in str(left or "")
        if op == "icontains":
            return str(right).lower() in str(left or "").lower()
        raise ValueError(f"Unsupported filter operator {op!r}")

    @classmethod
    def _matches_field(cls, payload: Dict[str, Any], key: str, expected: Any) -> bool:
        if expected == "*":
            return True
        actual = payload.get(key)
        if isinstance(expected, dict):
            return all(cls._compare(actual, op, value) for op, value in expected.items())
        if isinstance(expected, list):
            return actual in expected
        return actual == expected

    @classmethod
    def _matches_filters(cls, payload: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            norm_key = {"$or": "OR", "$not": "NOT", "$and": "AND"}.get(key, key)
            if norm_key == "AND":
                if not all(cls._matches_filters(payload, sub) for sub in (value or [])):
                    return False
            elif norm_key == "OR":
                if not any(cls._matches_filters(payload, sub) for sub in (value or [])):
                    return False
            elif norm_key == "NOT":
                if any(cls._matches_filters(payload, sub) for sub in (value or [])):
                    return False
            elif not cls._matches_field(payload, key, value):
                return False
        return True

    def _score(self, stored: List[float], query: List[float]) -> float:
        if not stored or not query:
            return 0.0
        n = min(len(stored), len(query))
        left = stored[:n]
        right = query[:n]
        if self.distance == "cosine":
            dot = sum(a * b for a, b in zip(left, right))
            left_norm = math.sqrt(sum(a * a for a in left))
            right_norm = math.sqrt(sum(b * b for b in right))
            if left_norm == 0.0 or right_norm == 0.0:
                return 0.0
            return max(0.0, min(1.0, dot / (left_norm * right_norm)))
        if self.distance == "l2":
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
            return 1.0 / (1.0 + dist)
        return sum(a * b for a, b in zip(left, right))

    def insert(self, vectors: List[List[float]], ids: List[str], payloads: List[Dict[str, Any]]) -> None:
        assert len(vectors) == len(ids) == len(payloads), "vectors/ids/payloads length mismatch"
        if not vectors:
            return
        rows = []
        for vec, vid, payload in zip(vectors, ids, payloads):
            promoted, json_payload = self._split_payload(payload)
            rows.append((
                vid,
                json.dumps([float(x) for x in vec]),
                *[promoted[c] for c in _PROMOTED_COLUMNS],
                json_payload,
            ))
        cols = ("id", "embedding", *_PROMOTED_COLUMNS, "payload")
        placeholders = ", ".join(["?"] * len(cols))
        self.con.executemany(
            f"INSERT INTO {self._quote_ident(self.table)} "
            f"({', '.join(self._quote_ident(c) for c in cols)}) VALUES ({placeholders})",
            rows,
        )
        self.con.commit()

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        sets = []
        params: List[Any] = []
        if vector is not None:
            sets.append(f"{self._quote_ident('embedding')} = ?")
            params.append(json.dumps([float(x) for x in vector]))
        if payload is not None:
            promoted, json_payload = self._split_payload(payload)
            for col in _PROMOTED_COLUMNS:
                sets.append(f"{self._quote_ident(col)} = ?")
                params.append(promoted[col])
            sets.append(f"{self._quote_ident('payload')} = ?")
            params.append(json_payload)
        if not sets:
            return
        params.append(vector_id)
        self.con.execute(
            f"UPDATE {self._quote_ident(self.table)} SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self.con.commit()

    def delete(self, vector_id: str) -> None:
        self.con.execute(f"DELETE FROM {self._quote_ident(self.table)} WHERE id = ?", [vector_id])
        self.con.commit()

    def get(self, vector_id: str) -> Optional[VectorRecord]:
        cols = ("id", *_PROMOTED_COLUMNS, "payload")
        row = self.con.execute(
            f"SELECT {', '.join(self._quote_ident(c) for c in cols)} "
            f"FROM {self._quote_ident(self.table)} WHERE id = ?",
            [vector_id],
        ).fetchone()
        if not row:
            return None
        promoted = {c: row[c] for c in _PROMOTED_COLUMNS}
        return VectorRecord(id=vector_id, score=None, payload=self._merge_payload(promoted, row["payload"]))

    def search(self, vectors: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, query: Optional[str] = None) -> List[VectorRecord]:
        scored: List[VectorRecord] = []
        for rec, stored_vector in self._fetch_records(include_embedding=True):
            if not self._matches_filters(rec.payload or {}, filters):
                continue
            rec.score = self._score(stored_vector or [], vectors)
            scored.append(rec)
        scored.sort(key=lambda item: float(item.score or 0.0), reverse=True)
        return scored[: int(top_k)]

    def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[VectorRecord]:
        records = [
            rec for rec, _ in self._fetch_records(include_embedding=False)
            if self._matches_filters(rec.payload or {}, filters)
        ]
        records.sort(
            key=lambda rec: (rec.payload or {}).get("updated_at") or (rec.payload or {}).get("created_at") or "",
            reverse=True,
        )
        return records[: int(top_k)] if top_k else records

    def list_pairs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        fields: Tuple[str, ...] = ("id", "text_lemmatized", "data"),
    ) -> List[Tuple[Any, ...]]:
        for f in fields:
            if f != "id" and f not in _PROMOTED_COLUMNS:
                raise ValueError(f"list_pairs only supports promoted columns; got {f!r}")
        rows = []
        for rec in self.list(filters=filters, top_k=top_k):
            row = []
            for field in fields:
                row.append(rec.id if field == "id" else (rec.payload or {}).get(field))
            rows.append(tuple(row))
        return rows

    def reset(self) -> None:
        self.con.execute(f"DROP TABLE IF EXISTS {self._quote_ident(self.table)}")
        self.con.commit()
        self._create_table_and_index()

    def delete_collection(self) -> None:
        self.con.execute(f"DROP TABLE IF EXISTS {self._quote_ident(self.table)}")
        self.con.commit()

    def checkpoint(self) -> None:
        self.con.commit()

    def close(self) -> None:
        if self.con is not None:
            try:
                self.con.commit()
                self.con.close()
            finally:
                self.con = None


class VexDBVectorStore:
    """VexDB-Lite-backed vector store with mem0-compatible interface."""

    def __init__(self, config: VectorStoreConfig, table_name: Optional[str] = None):
        self.config = config
        self.table = table_name or config.collection_name
        self.dim = config.embedding_dims
        self.distance = config.distance_metric.lower()
        if self.distance not in ("cosine", "l2", "ip"):
            raise ValueError(f"Unknown distance_metric: {self.distance!r}")
        self._fallback: Optional[_SQLiteVectorStore] = None
        try:
            self.con = _connect_vex(config.db_path)
            self.con.execute(
                f"SELECT {self._distance_fn()}([1,0]::FLOAT[2], [1,0]::FLOAT[2])"
            ).fetchone()
        except Exception as e:
            logger.warning(
                "VexDB-Lite unavailable or missing vector functions; "
                "falling back to SQLite brute-force vector store: %s",
                e,
            )
            self.con = None
            self._fallback = _SQLiteVectorStore(config, self.table)
            return
        # Bump ef_search for higher recall (default 40 is too low for LLM-style queries).
        try:
            self.con.execute(f"SET vex_ef_search = {int(config.ef_search)}")
        except Exception:
            pass  # not fatal if version doesn't support the SET
        self._create_table_and_index()

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _distance_fn(self) -> str:
        return {
            "cosine": "cosine_distance",
            "l2": "l2_distance",
            "ip": "inner_product",
        }[self.distance]

    def _distance_to_score(self, distance: float) -> float:
        if self.distance == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        if self.distance == "l2":
            return 1.0 / (1.0 + distance)
        # ip — already a similarity (assume normalized vectors)
        return float(distance)

    def _order_direction(self) -> str:
        # For ip we want largest first; for distance metrics smallest first.
        return "DESC" if self.distance == "ip" else "ASC"

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_table_and_index(self) -> None:
        col_defs = ",\n            ".join(
            f"{c} VARCHAR" for c in _PROMOTED_COLUMNS
        )
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id        VARCHAR PRIMARY KEY,
                embedding FLOAT[{self.dim}],
                {col_defs},
                payload   VARCHAR
            )
            """
        )
        # Lightweight scalar indexes for the most common filters.
        for col in ("user_id", "agent_id", "run_id", "actor_id", "hash"):
            try:
                self.con.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_{col}_idx "
                    f"ON {self.table}({col})"
                )
            except Exception as e:
                logger.debug(f"Could not create {col} index on {self.table}: {e}")

        # Vector ANN index (HNSW).
        try:
            self.con.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table}_vec_idx
                ON {self.table} USING GRAPH_INDEX(embedding)
                WITH (metric='{self.distance}',
                      m={int(self.config.hnsw_m)},
                      ef_construction={int(self.config.hnsw_ef_construction)})
                """
            )
        except Exception as e:
            logger.warning(f"Failed to create vector index on {self.table}: {e}")

    # ------------------------------------------------------------------
    # Payload split/merge
    # ------------------------------------------------------------------

    @staticmethod
    def _split_payload(payload: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
        """Return (promoted_columns, json_payload_string) from a full payload dict."""
        payload = payload or {}
        promoted = {c: payload.get(c) for c in _PROMOTED_COLUMNS}
        rest = {k: v for k, v in payload.items() if k not in _PROMOTED_COLUMNS}
        return promoted, json.dumps(rest, ensure_ascii=False)

    @staticmethod
    def _merge_payload(promoted_row: Dict[str, Any], json_payload: Optional[str]) -> Dict[str, Any]:
        rest = json.loads(json_payload) if json_payload else {}
        for k, v in promoted_row.items():
            if v is not None:
                rest[k] = v
        return rest

    # ------------------------------------------------------------------
    # Filter → SQL WHERE
    # ------------------------------------------------------------------

    def _column_expr(self, key: str) -> str:
        """SQL expression to read a payload key from a row."""
        if key in _PROMOTED_COLUMNS:
            return key
        return f"json_extract_string(payload, '$.{key}')"

    def _build_field_condition(
        self, key: str, value: Any, params: List[Any]
    ) -> Optional[str]:
        col = self._column_expr(key)
        if not isinstance(value, dict):
            if value == "*":
                return None  # wildcard → no constraint
            if isinstance(value, list):
                placeholders = ", ".join(["?"] * len(value))
                params.extend(value)
                return f"{col} IN ({placeholders})"
            params.append(value)
            return f"{col} = ?"

        # Operator dict
        clauses = []
        for op, v in value.items():
            if op in _OP_SQL:
                params.append(v)
                clauses.append(f"{col} {_OP_SQL[op]} ?")
            elif op == "in":
                if not v:
                    return "1=0"
                placeholders = ", ".join(["?"] * len(v))
                params.extend(v)
                clauses.append(f"{col} IN ({placeholders})")
            elif op == "nin":
                if not v:
                    continue
                placeholders = ", ".join(["?"] * len(v))
                params.extend(v)
                clauses.append(f"{col} NOT IN ({placeholders})")
            elif op == "contains":
                params.append(f"%{v}%")
                clauses.append(f"{col} LIKE ?")
            elif op == "icontains":
                params.append(f"%{v.lower()}%")
                clauses.append(f"LOWER({col}) LIKE ?")
            else:
                raise ValueError(f"Unsupported filter operator {op!r} for field {key!r}")
        return " AND ".join(clauses) if clauses else None

    def _build_where(self, filters: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """Translate a mem0-style filter dict to SQL ``WHERE`` + params list."""
        if not filters:
            return "", []
        params: List[Any] = []
        clauses: List[str] = []
        for key, value in filters.items():
            norm_key = {"$or": "OR", "$not": "NOT", "$and": "AND"}.get(key, key)
            if norm_key == "AND":
                sub_clauses = []
                for sub in value:
                    w, p = self._build_where(sub)
                    if w:
                        sub_clauses.append(f"({w})")
                        params.extend(p)
                if sub_clauses:
                    clauses.append("(" + " AND ".join(sub_clauses) + ")")
            elif norm_key == "OR":
                sub_clauses = []
                for sub in value:
                    w, p = self._build_where(sub)
                    if w:
                        sub_clauses.append(f"({w})")
                        params.extend(p)
                if sub_clauses:
                    clauses.append("(" + " OR ".join(sub_clauses) + ")")
            elif norm_key == "NOT":
                sub_clauses = []
                for sub in value:
                    w, p = self._build_where(sub)
                    if w:
                        sub_clauses.append(f"NOT ({w})")
                        params.extend(p)
                if sub_clauses:
                    clauses.append("(" + " AND ".join(sub_clauses) + ")")
            else:
                cond = self._build_field_condition(key, value, params)
                if cond:
                    clauses.append(cond)
        where = " AND ".join(clauses)
        return where, params

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        payloads: List[Dict[str, Any]],
    ) -> None:
        if self._fallback is not None:
            return self._fallback.insert(vectors, ids, payloads)
        assert len(vectors) == len(ids) == len(payloads), "vectors/ids/payloads length mismatch"
        if not vectors:
            return
        rows = []
        for vec, vid, payload in zip(vectors, ids, payloads):
            promoted, json_payload = self._split_payload(payload)
            rows.append((vid, vec, *[promoted[c] for c in _PROMOTED_COLUMNS], json_payload))
        col_list = ", ".join(("id", "embedding", *_PROMOTED_COLUMNS, "payload"))
        placeholders = ", ".join(["?"] * (2 + len(_PROMOTED_COLUMNS) + 1))
        self.con.executemany(
            f"INSERT INTO {self.table} ({col_list}) VALUES ({placeholders})",
            rows,
        )

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._fallback is not None:
            return self._fallback.update(vector_id, vector=vector, payload=payload)
        sets = []
        params: List[Any] = []
        if vector is not None:
            sets.append("embedding = ?")
            params.append(vector)
        if payload is not None:
            promoted, json_payload = self._split_payload(payload)
            for col in _PROMOTED_COLUMNS:
                sets.append(f"{col} = ?")
                params.append(promoted[col])
            sets.append("payload = ?")
            params.append(json_payload)
        if not sets:
            return
        params.append(vector_id)
        self.con.execute(
            f"UPDATE {self.table} SET {', '.join(sets)} WHERE id = ?",
            params,
        )

    def delete(self, vector_id: str) -> None:
        if self._fallback is not None:
            return self._fallback.delete(vector_id)
        self.con.execute(f"DELETE FROM {self.table} WHERE id = ?", [vector_id])

    def get(self, vector_id: str) -> Optional[VectorRecord]:
        if self._fallback is not None:
            return self._fallback.get(vector_id)
        row = self.con.execute(
            f"SELECT {', '.join(_PROMOTED_COLUMNS)}, payload FROM {self.table} WHERE id = ?",
            [vector_id],
        ).fetchone()
        if not row:
            return None
        promoted = dict(zip(_PROMOTED_COLUMNS, row[: len(_PROMOTED_COLUMNS)]))
        json_payload = row[len(_PROMOTED_COLUMNS)]
        return VectorRecord(
            id=vector_id,
            score=None,
            payload=self._merge_payload(promoted, json_payload),
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        vectors: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,  # unused; here for API parity with mem0
    ) -> List[VectorRecord]:
        if self._fallback is not None:
            return self._fallback.search(vectors, top_k=top_k, filters=filters, query=query)
        where, params = self._build_where(filters)
        where_clause = f"WHERE {where}" if where else ""
        dist_fn = self._distance_fn()
        order_dir = self._order_direction()
        # Bind the query vector last so its ? in ORDER BY matches.
        sql = (
            f"SELECT id, {', '.join(_PROMOTED_COLUMNS)}, payload, "
            f"       {dist_fn}(embedding, ?::FLOAT[{self.dim}]) AS dist "
            f"FROM {self.table} "
            f"{where_clause} "
            f"ORDER BY dist {order_dir} "
            f"LIMIT ?"
        )
        sql_params = [*params, vectors, int(top_k)]
        # Note: in DuckDB, parameters bind in order encountered. Both WHERE and
        # the SELECT use placeholders; we place WHERE params first since the
        # WHERE clause comes before the SELECT-list ? in execution but DuckDB
        # binds purely by position, so this layout is correct.

        # Actually DuckDB binds by order of '?' in the SQL string. The SELECT
        # list contains the ORDER vector reference; rewrite SQL so vector
        # binding precedes WHERE params... but since the SELECT list is parsed
        # first (in left-to-right '?' order), the vector ? actually comes
        # before any WHERE params. Re-order params accordingly:
        sql_params = [vectors, *params, int(top_k)]
        rows = self.con.execute(sql, sql_params).fetchall()
        results: List[VectorRecord] = []
        for row in rows:
            vid = row[0]
            promoted = dict(zip(_PROMOTED_COLUMNS, row[1: 1 + len(_PROMOTED_COLUMNS)]))
            json_payload = row[1 + len(_PROMOTED_COLUMNS)]
            dist = row[1 + len(_PROMOTED_COLUMNS) + 1]
            results.append(
                VectorRecord(
                    id=vid,
                    score=self._distance_to_score(float(dist)),
                    payload=self._merge_payload(promoted, json_payload),
                )
            )
        return results

    def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorRecord]:
        if self._fallback is not None:
            return self._fallback.list(filters=filters, top_k=top_k)
        where, params = self._build_where(filters)
        where_clause = f"WHERE {where}" if where else ""
        limit_clause = f"LIMIT {int(top_k)}" if top_k else ""
        sql = (
            f"SELECT id, {', '.join(_PROMOTED_COLUMNS)}, payload "
            f"FROM {self.table} {where_clause} "
            f"ORDER BY COALESCE(updated_at, created_at) DESC NULLS LAST "
            f"{limit_clause}"
        )
        rows = self.con.execute(sql, params).fetchall()
        out: List[VectorRecord] = []
        for row in rows:
            vid = row[0]
            promoted = dict(zip(_PROMOTED_COLUMNS, row[1: 1 + len(_PROMOTED_COLUMNS)]))
            json_payload = row[1 + len(_PROMOTED_COLUMNS)]
            out.append(VectorRecord(id=vid, score=None, payload=self._merge_payload(promoted, json_payload)))
        return out

    def list_pairs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        fields: Tuple[str, ...] = ("id", "text_lemmatized", "data"),
    ) -> List[Tuple[Any, ...]]:
        """Lightweight projection helper for the BM25 path — pulls just the
        requested columns to avoid loading full payload JSON.

        Only promoted columns may be requested via ``fields`` (raises if a
        caller asks for a JSON-only field).
        """
        if self._fallback is not None:
            return self._fallback.list_pairs(filters=filters, top_k=top_k, fields=fields)
        for f in fields:
            if f != "id" and f not in _PROMOTED_COLUMNS:
                raise ValueError(f"list_pairs only supports promoted columns; got {f!r}")
        where, params = self._build_where(filters)
        where_clause = f"WHERE {where}" if where else ""
        limit_clause = f"LIMIT {int(top_k)}" if top_k else ""
        sql = (
            f"SELECT {', '.join(fields)} FROM {self.table} {where_clause} {limit_clause}"
        )
        return self.con.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if self._fallback is not None:
            return self._fallback.reset()
        self.con.execute(f"DROP TABLE IF EXISTS {self.table}")
        self._create_table_and_index()

    def delete_collection(self) -> None:
        if self._fallback is not None:
            return self._fallback.delete_collection()
        self.con.execute(f"DROP TABLE IF EXISTS {self.table}")

    def checkpoint(self) -> None:
        if self._fallback is not None:
            return self._fallback.checkpoint()
        try:
            self.con.execute("CHECKPOINT")
        except Exception as e:
            logger.debug(f"CHECKPOINT failed (likely in-memory connection): {e}")

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
            return
        if self.con is not None:
            try:
                self.checkpoint()
                self.con.close()
            finally:
                self.con = None
