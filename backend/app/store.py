"""SQLite persistence: documents, chunks + embeddings, graph, metrics, chat history, query log."""
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER,
    extension TEXT,
    status TEXT NOT NULL,          -- uploaded | processing | ready | failed
    error TEXT,
    pages INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    ocr_pages INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    processed_ms INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,           -- "<Type>:<normalized name>"
    label TEXT NOT NULL,
    type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    page INTEGER,
    evidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_doc ON graph_edges(doc_id);
CREATE TABLE IF NOT EXISTS node_mentions (
    node_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    page INTEGER,
    evidence TEXT,
    PRIMARY KEY (node_id, doc_id, page, evidence)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    page INTEGER,
    evidence TEXT
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    occurred_on TEXT,
    description TEXT,
    counterparty TEXT,
    amount REAL,
    page INTEGER
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    query TEXT,
    latency_ms REAL,
    results INTEGER,
    insufficient INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

_lock = threading.RLock()
_conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
_conn.row_factory = sqlite3.Row
_conn.executescript(SCHEMA)
_chunk_cache: dict = {"version": 0, "data": None}
_version = 0


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _invalidate_cache():
    global _version
    _version += 1


def execute(sql: str, params=()):
    with _lock:
        cur = _conn.execute(sql, params)
        _conn.commit()
        return cur


def query(sql: str, params=()) -> list[dict]:
    with _lock:
        return [dict(r) for r in _conn.execute(sql, params).fetchall()]


def one(sql: str, params=()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


# ---------- documents ----------

def create_document(filename: str, path: str, size: int, extension: str) -> dict:
    doc_id = new_id()
    execute(
        "INSERT INTO documents (id, filename, path, size, extension, status, created_at) VALUES (?,?,?,?,?,?,?)",
        (doc_id, filename, path, size, extension, "uploaded", now()),
    )
    return get_document(doc_id)


def get_document(doc_id: str) -> dict | None:
    return one("SELECT * FROM documents WHERE id=?", (doc_id,))


def list_documents() -> list[dict]:
    return query("SELECT * FROM documents ORDER BY created_at DESC")


def set_document_status(doc_id: str, status: str, error: str | None = None, **extra):
    fields = {"status": status, "error": error, **extra}
    assignments = ", ".join(f"{k}=?" for k in fields)
    execute(f"UPDATE documents SET {assignments} WHERE id=?", (*fields.values(), doc_id))


def delete_document(doc_id: str) -> bool:
    doc = get_document(doc_id)
    if not doc:
        return False
    with _lock:
        for table in ("chunks", "graph_edges", "node_mentions", "metrics", "transactions"):
            _conn.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))
        _conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        _conn.commit()
    _remove_orphan_nodes()
    _invalidate_cache()
    try:
        from pathlib import Path

        Path(doc["path"]).unlink(missing_ok=True)
    except OSError:
        pass
    return True


def _remove_orphan_nodes():
    execute(
        "DELETE FROM graph_nodes WHERE id NOT IN (SELECT source FROM graph_edges UNION SELECT target FROM graph_edges)"
        " AND id NOT IN (SELECT node_id FROM node_mentions)"
    )


# ---------- chunks ----------

def save_chunks(doc_id: str, chunks: list[dict], vectors: np.ndarray, embed_model: str):
    with _lock:
        _conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        _conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, page, text, embed_model, embedding) VALUES (?,?,?,?,?,?)",
            [
                (doc_id, c["index"], c["page"], c["text"], embed_model, vectors[i].astype(np.float32).tobytes())
                for i, c in enumerate(chunks)
            ],
        )
        _conn.commit()
    _invalidate_cache()


def load_chunks(embed_model: str) -> list[dict]:
    """All chunks embedded with the active model, joined to their document. Cached until the corpus changes."""
    cache = _chunk_cache
    if cache["data"] is not None and cache["version"] == _version and cache.get("model") == embed_model:
        return cache["data"]
    rows = query(
        """SELECT c.id, c.doc_id, c.chunk_index, c.page, c.text, c.embedding, d.filename
           FROM chunks c JOIN documents d ON d.id=c.doc_id
           WHERE d.status='ready' AND c.embed_model=? ORDER BY c.id""",
        (embed_model,),
    )
    matrix = np.array([np.frombuffer(r.pop("embedding"), dtype=np.float32) for r in rows], dtype=np.float32) if rows else np.zeros((0, 1), dtype=np.float32)
    data = {"rows": rows, "matrix": matrix}
    cache.update(version=_version, data=data, model=embed_model)
    return data


def document_chunks(doc_id: str) -> list[dict]:
    return query("SELECT chunk_index, page, text FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,))


# ---------- graph / metrics / transactions ----------

def replace_doc_facts(doc_id: str, nodes: dict, edges: list[dict], mentions: list[dict], metrics: list[dict], transactions: list[dict]):
    with _lock:
        for table in ("graph_edges", "node_mentions", "metrics", "transactions"):
            _conn.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))
        for node_id, node in nodes.items():
            _conn.execute("INSERT OR IGNORE INTO graph_nodes (id, label, type) VALUES (?,?,?)", (node_id, node["label"], node["type"]))
        _conn.executemany(
            "INSERT INTO graph_edges (doc_id, source, target, relation, page, evidence) VALUES (?,?,?,?,?,?)",
            [(doc_id, e["source"], e["target"], e["relation"], e.get("page"), e.get("evidence")) for e in edges],
        )
        _conn.executemany(
            "INSERT OR IGNORE INTO node_mentions (node_id, doc_id, page, evidence) VALUES (?,?,?,?)",
            [(m["node_id"], doc_id, m.get("page"), m.get("evidence")) for m in mentions],
        )
        _conn.executemany(
            "INSERT INTO metrics (doc_id, name, value, page, evidence) VALUES (?,?,?,?,?)",
            [(doc_id, m["name"], m["value"], m.get("page"), m.get("evidence")) for m in metrics],
        )
        _conn.executemany(
            "INSERT INTO transactions (doc_id, occurred_on, description, counterparty, amount, page) VALUES (?,?,?,?,?,?)",
            [(doc_id, t.get("date"), t.get("description"), t.get("counterparty"), t.get("amount"), t.get("page")) for t in transactions],
        )
        _conn.commit()


# ---------- conversations ----------

def create_conversation(title: str) -> str:
    conv_id = new_id()
    execute("INSERT INTO conversations (id, title, created_at) VALUES (?,?,?)", (conv_id, title[:80], now()))
    return conv_id


def add_message(conv_id: str, role: str, content: str, citations: list | None = None):
    execute(
        "INSERT INTO messages (conversation_id, role, content, citations, created_at) VALUES (?,?,?,?,?)",
        (conv_id, role, content, json.dumps(citations) if citations is not None else None, now()),
    )


def conversation_messages(conv_id: str) -> list[dict]:
    rows = query("SELECT role, content, citations, created_at FROM messages WHERE conversation_id=? ORDER BY id", (conv_id,))
    for r in rows:
        r["citations"] = json.loads(r["citations"]) if r["citations"] else None
    return rows


def list_conversations() -> list[dict]:
    return query("SELECT * FROM conversations ORDER BY created_at DESC")


def delete_conversation(conv_id: str):
    execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
    execute("DELETE FROM conversations WHERE id=?", (conv_id,))


# ---------- monitoring ----------

def log_query(kind: str, query_text: str, latency_ms: float, results: int, insufficient: bool = False):
    execute(
        "INSERT INTO query_log (kind, query, latency_ms, results, insufficient, created_at) VALUES (?,?,?,?,?,?)",
        (kind, query_text[:300], round(latency_ms, 1), results, int(insufficient), now()),
    )


def timer():
    start = time.perf_counter()
    return lambda: (time.perf_counter() - start) * 1000
