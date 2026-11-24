"""FastAPI application: REST API plus the static single-page frontend."""
import csv
import io
import logging
import re
import time
import uuid
from pathlib import Path
from statistics import quantiles

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, extraction, llm, rag, store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="FinGraph AI", version="1.0.0")


# ---------- helpers ----------

def _doc_or_404(doc_id: str) -> dict:
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


def _ready_doc_ids(doc_ids: list[str] | None = None) -> list[str]:
    docs = store.query("SELECT id FROM documents WHERE status='ready' ORDER BY created_at")
    ids = [d["id"] for d in docs]
    return [i for i in ids if not doc_ids or i in doc_ids]


def _filenames() -> dict[str, str]:
    return {d["id"]: d["filename"] for d in store.list_documents()}


def _metrics_rows(doc_ids: list[str] | None = None, name: str | None = None) -> list[dict]:
    rows = store.query(
        """SELECT m.doc_id, m.name, m.value, m.page, m.evidence, d.filename, d.created_at
           FROM metrics m JOIN documents d ON d.id=m.doc_id WHERE d.status='ready' ORDER BY d.created_at, m.id""")
    if doc_ids:
        rows = [r for r in rows if r["doc_id"] in doc_ids]
    if name:
        rows = [r for r in rows if r["name"] == name.lower()]
    return rows


def _anomalies() -> list[dict]:
    documents = store.query("SELECT id, filename FROM documents WHERE status='ready'")
    invoices = store.query(
        """SELECT DISTINCT n.label, m.doc_id FROM node_mentions m
           JOIN graph_nodes n ON n.id=m.node_id WHERE n.type='Invoice'""")
    transactions = store.query("SELECT doc_id, occurred_on, description, amount FROM transactions")
    return extraction.find_anomalies(documents, _metrics_rows(), invoices, transactions)


HEADLINE_METRICS = ["total revenue", "revenue", "net income", "net profit", "gross profit", "total assets", "total"]


def _key_findings() -> list[str]:
    """Per-document headline figures. Values are never merged across documents, so nothing is overwritten."""
    per_doc: dict[str, list[str]] = {}
    for row in _metrics_rows():  # ordered by upload time
        seen = per_doc.setdefault(row["doc_id"], [])
        if row["name"] in HEADLINE_METRICS and row["name"] not in [s.split(":")[0].lower() for s in seen]:
            seen.append(f"{row['name'].title()}: {row['value']:,.2f}")
    findings = [f"{_filenames().get(doc_id, doc_id)}: {', '.join(items)}" for doc_id, items in per_doc.items() if items]
    txns = store.one("SELECT COUNT(*) AS n, COALESCE(SUM(amount),0) AS total FROM transactions")
    if txns and txns["n"]:
        findings.append(f"{txns['n']} transactions extracted, totalling {txns['total']:,.2f}.")
    return findings


# ---------- health and dashboard ----------

@app.get("/api/health")
def health():
    provider = llm.get_provider()
    return {"status": "ok", "llm": provider.name, "llm_available": llm.llm_available(), "chat_model": getattr(provider, "chat_model", None)}


@app.get("/api/dashboard")
def dashboard():
    docs = store.list_documents()
    counts = {s: sum(1 for d in docs if d["status"] == s) for s in ("uploaded", "processing", "ready", "failed")}
    graph = store.one("SELECT (SELECT COUNT(*) FROM graph_nodes) AS nodes, (SELECT COUNT(*) FROM graph_edges) AS edges")
    anomalies = _anomalies()
    return {
        "documents": len(docs), "status_counts": counts,
        "chunks": store.one("SELECT COUNT(*) AS n FROM chunks")["n"],
        "graph": graph, "anomalies": len(anomalies),
        "key_findings": _key_findings(),
        "recent_anomalies": anomalies[:5],
    }


# ---------- documents ----------

@app.post("/api/documents", status_code=202)
async def upload_document(file: UploadFile, background: BackgroundTasks):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: PDF, DOCX, JPG, PNG.")
    content = await file.read()
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB limit.")
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    stored = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    stored.write_bytes(content)
    doc = store.create_document(Path(file.filename).name, str(stored), len(content), ext)
    background.add_task(rag.process_document, doc["id"])
    return doc


@app.get("/api/documents")
def list_documents():
    return store.list_documents()


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    return _doc_or_404(doc_id)


@app.get("/api/documents/{doc_id}/chunks")
def document_chunks(doc_id: str):
    _doc_or_404(doc_id)
    return store.document_chunks(doc_id)


@app.post("/api/documents/{doc_id}/reprocess", status_code=202)
def reprocess(doc_id: str, background: BackgroundTasks):
    _doc_or_404(doc_id)
    background.add_task(rag.process_document, doc_id)
    return {"status": "queued"}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if not store.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


# ---------- retrieval, chat ----------

class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=config.TOP_K, ge=1, le=20)
    doc_ids: list[str] | None = None


@app.post("/api/search")
def search(req: SearchRequest):
    """RAG debugging playground: returns every retrieved chunk with its vector, BM25, RRF and rerank scores."""
    start = time.perf_counter()
    results = rag.hybrid_search(req.query, req.k, req.doc_ids)
    latency = (time.perf_counter() - start) * 1000
    store.log_query("search", req.query, latency, len(results), rag.is_insufficient(results))
    return {"query": req.query, "latency_ms": round(latency, 1), "insufficient": rag.is_insufficient(results), "results": results}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.conversation_id:
        if not store.one("SELECT id FROM conversations WHERE id=?", (req.conversation_id,)):
            raise HTTPException(404, "Conversation not found")
        conv_id = req.conversation_id
    else:
        conv_id = store.create_conversation(req.message)
    history = store.conversation_messages(conv_id)
    result = rag.answer(req.message, history)
    store.add_message(conv_id, "user", req.message)
    store.add_message(conv_id, "assistant", result["answer"], result["citations"])
    return {"conversation_id": conv_id, **result}


@app.get("/api/conversations")
def conversations():
    return store.list_conversations()


@app.get("/api/conversations/{conv_id}")
def conversation(conv_id: str):
    if not store.one("SELECT id FROM conversations WHERE id=?", (conv_id,)):
        raise HTTPException(404, "Conversation not found")
    return {"id": conv_id, "messages": store.conversation_messages(conv_id)}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    store.delete_conversation(conv_id)
    return {"deleted": conv_id}


# ---------- knowledge graph ----------

@app.get("/api/graph")
def graph(doc_id: str | None = None):
    where, params = "", ()
    if doc_id:
        where, params = "WHERE e.doc_id=?", (doc_id,)
    edges = store.query(
        f"""SELECT e.source, e.target, e.relation, e.doc_id, e.page, e.evidence, d.filename
            FROM graph_edges e JOIN documents d ON d.id=e.doc_id {where}""", params)
    node_ids = {e["source"] for e in edges} | {e["target"] for e in edges}
    if doc_id:
        node_ids |= {m["node_id"] for m in store.query("SELECT node_id FROM node_mentions WHERE doc_id=?", (doc_id,))}
    nodes = []
    for nid in sorted(node_ids):
        row = store.one("SELECT * FROM graph_nodes WHERE id=?", (nid,))
        if row:
            docs = store.query("SELECT DISTINCT doc_id FROM node_mentions WHERE node_id=?", (nid,))
            nodes.append({**row, "documents": [d["doc_id"] for d in docs]})
    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph/nodes/{node_id:path}")
def graph_node(node_id: str):
    """Explore one entity: its connections, and the document excerpts that support it."""
    node = store.one("SELECT * FROM graph_nodes WHERE id=?", (node_id,))
    if not node:
        raise HTTPException(404, "Entity not found")
    edges = store.query(
        """SELECT e.source, e.target, e.relation, e.page, e.evidence, d.id AS doc_id, d.filename
           FROM graph_edges e JOIN documents d ON d.id=e.doc_id WHERE e.source=? OR e.target=?""", (node_id, node_id))
    mentions = store.query(
        """SELECT m.page, m.evidence, d.id AS doc_id, d.filename FROM node_mentions m
           JOIN documents d ON d.id=m.doc_id WHERE m.node_id=? ORDER BY d.filename""", (node_id,))
    return {"node": node, "connections": edges, "supporting_documents": mentions}


# ---------- financial intelligence ----------

@app.get("/api/metrics")
def metrics(doc_id: str | None = None, name: str | None = None):
    return _metrics_rows([doc_id] if doc_id else None, name)


@app.get("/api/transactions")
def transactions(doc_id: str | None = None):
    if doc_id:
        return store.query("SELECT t.*, d.filename FROM transactions t JOIN documents d ON d.id=t.doc_id WHERE t.doc_id=? ORDER BY occurred_on", (doc_id,))
    return store.query("SELECT t.*, d.filename FROM transactions t JOIN documents d ON d.id=t.doc_id ORDER BY occurred_on")


@app.get("/api/compare")
def compare(metric: str, doc_ids: str | None = None):
    """Compare one metric across documents, in upload order. Adds change vs first and vs previous document."""
    wanted = [d.strip() for d in doc_ids.split(",") if d.strip()] if doc_ids else None
    rows = _metrics_rows(wanted, metric)
    seen, unique = set(), []
    for r in rows:  # keep the first occurrence per document
        if r["doc_id"] not in seen:
            seen.add(r["doc_id"])
            unique.append({"doc_id": r["doc_id"], "value": r["value"]})
    if not unique:
        raise HTTPException(404, f"No documents contain the metric '{metric}'")
    return {"metric": metric.lower(), "rows": extraction.compare_metric(metric.lower(), unique, _filenames())}


class CalcRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


@app.post("/api/calculate")
def calculate(req: CalcRequest):
    """Basic arithmetic only: numbers, + - * / ( ) and %. Anything else is rejected."""
    expr = req.expression.replace(",", "")
    if not re.fullmatch(r"[0-9\.\s\+\-\*/\(\)%]+", expr):
        raise HTTPException(400, "Only numbers and + - * / ( ) % are allowed.")
    try:
        value = eval(expr.replace("%", "/100"), {"__builtins__": {}}, {})  # restricted: validated characters only
    except Exception:
        raise HTTPException(400, "Could not evaluate the expression.")
    return {"expression": req.expression, "result": round(float(value), 4)}


@app.get("/api/anomalies")
def anomalies():
    return _anomalies()


@app.get("/api/summary")
def summary():
    findings = _key_findings()
    anomalies_list = _anomalies()
    narrative = None
    if llm.llm_available() and findings:
        try:
            narrative = llm.get_provider().chat([{"role": "user", "content": (
                "Write a short summary (max 120 words) of these key financial findings. Use only these figures, do not add others.\n\n"
                + "\n".join(findings) + "\n\nAnomalies:\n" + "\n".join(a["detail"] for a in anomalies_list))}])
        except Exception as exc:
            logging.getLogger("fingraph").warning("Summary generation failed: %s", exc)
    return {"key_findings": findings, "anomalies": anomalies_list, "narrative": narrative}


@app.get("/api/report.md", response_class=PlainTextResponse)
def report_markdown():
    s = summary()
    lines = ["# FinGraph financial report", "", "## Key findings"]
    lines += [f"- {f}" for f in s["key_findings"]] or ["- No metrics extracted yet."]
    lines += ["", "## Anomalies"]
    lines += [f"- **{a['severity'].upper()}** {a['type']}: {a['detail']} ({', '.join(a['documents'])})" for a in s["anomalies"]] or ["- None detected."]
    if s.get("narrative"):
        lines += ["", "## Summary", s["narrative"]]
    lines += ["", "## Documents"] + [f"- {d['filename']} ({d['status']}, {d['pages']} pages)" for d in store.list_documents()]
    return "\n".join(lines)


@app.get("/api/report.csv")
def report_csv():
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["document", "metric", "value", "page", "evidence"])
    for r in _metrics_rows():
        writer.writerow([r["filename"], r["name"], r["value"], r["page"], r["evidence"]])
    buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=fingraph_metrics.csv"})


# ---------- evaluation and monitoring ----------

class EvalRequest(BaseModel):
    cases: list[dict]
    k: int = Field(default=config.TOP_K, ge=1, le=20)


@app.post("/api/eval")
def run_eval(req: EvalRequest):
    if not req.cases or len(req.cases) > 200:
        raise HTTPException(400, "Provide between 1 and 200 evaluation cases.")
    return rag.evaluate(req.cases, req.k)


@app.get("/api/monitoring")
def monitoring():
    rows = store.query("SELECT kind, latency_ms, results, insufficient, query, created_at FROM query_log ORDER BY id DESC LIMIT 200")
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    p95 = round(quantiles(latencies, n=20)[18], 1) if len(latencies) >= 2 else (latencies[0] if latencies else None)
    return {
        "queries_recorded": len(rows),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "p95_latency_ms": p95,
        "insufficient_rate": round(sum(r["insufficient"] for r in rows) / len(rows), 3) if rows else None,
        "avg_results": round(sum(r["results"] for r in rows) / len(rows), 2) if rows else None,
        "recent": rows[:20],
    }


# ---------- frontend ----------

FRONTEND_DIR = config.BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
