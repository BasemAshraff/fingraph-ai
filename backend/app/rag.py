"""RAG pipeline: ingestion, hybrid retrieval, reranking, grounded answers and evaluation."""
import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import chunking, config, extraction, llm, parsing, store

log = logging.getLogger("fingraph.rag")

INSUFFICIENT_MESSAGE = (
    "I couldn't find enough information in the uploaded documents to answer this. "
    "Try uploading a document that covers it, or rephrase the question."
)
STOPWORDS = set("a an the of in on for to and or is are was were be by with what which who how much many does do did from at as it this that".split())

SYSTEM_PROMPT = """You are FinGraph, an assistant that answers questions using ONLY the numbered sources provided.
Rules:
- Cite sources inline with their labels, for example [S2], after every claim that uses them.
- Never invent figures. When you compare or calculate, show the numbers and the arithmetic.
- When facts come from different documents, say which document each fact comes from.
- If the sources do not contain the answer, reply with exactly: INSUFFICIENT_CONTEXT"""


# ---------- ingestion ----------

def process_document(doc_id: str):
    """Run the full pipeline for one document. Errors are stored on the document, never raised to the caller."""
    doc = store.get_document(doc_id)
    if not doc:
        return
    elapsed = time.perf_counter()
    store.set_document_status(doc_id, "processing", error=None)
    try:
        pages = parsing.extract_pages(Path(doc["path"]))
        if sum(len(p["text"]) for p in pages) == 0:
            raise parsing.ParsingError("No readable text was found in this file.")
        chunks = chunking.chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            raise parsing.ParsingError("The document produced no text chunks.")
        provider = llm.get_provider()
        vectors = llm.embed_texts([c["text"] for c in chunks])
        store.save_chunks(doc_id, chunks, vectors, provider.name)

        facts = extraction.extract_facts(doc_id, doc["filename"], pages)
        store.replace_doc_facts(doc_id, facts["nodes"], facts["edges"], facts["mentions"], facts["metrics"], facts["transactions"])

        store.set_document_status(
            doc_id, "ready", error=None,
            pages=len(pages), chunk_count=len(chunks),
            ocr_pages=sum(1 for p in pages if p["method"] == "ocr"),
            processed_ms=int((time.perf_counter() - elapsed) * 1000),
        )
        log.info("Processed %s: %d pages, %d chunks", doc["filename"], len(pages), len(chunks))
    except Exception as exc:
        log.exception("Failed to process %s", doc["filename"])
        store.set_document_status(doc_id, "failed", error=str(exc)[:500])


# ---------- retrieval ----------

def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9.,]*", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.k1, self.b = k1, b
        self.avgdl = (sum(len(d) for d in documents) / len(documents)) if documents else 0.0
        self.df: Counter = Counter()
        for d in documents:
            self.df.update(set(d))
        self.n = len(documents)
        self.tf = [Counter(d) for d in documents]

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            for term in set(query_tokens):
                if term not in tf:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                freq = tf[term]
                out[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return out


def _rerank(query: str, candidates: list[dict]) -> None:
    """Adds rerank_score in [0, 1] to each candidate. Uses the LLM when available, otherwise query-term coverage."""
    if llm.llm_available() and candidates:
        try:
            listing = "\n".join(f"[{i}] {c['text'][:500]}" for i, c in enumerate(candidates))
            raw = llm.get_provider().chat(
                [{"role": "user", "content": (
                    f"Question: {query}\n\nRate how useful each passage is for answering the question, 0 (irrelevant) to 10 (fully answers it). "
                    f'Reply as JSON: {{"scores":[{{"id":0,"score":7}}]}}\n\n{listing}')}],
                json_mode=True,
            )
            for item in json.loads(raw).get("scores", []):
                idx = int(item["id"])
                if 0 <= idx < len(candidates):
                    candidates[idx]["rerank_score"] = max(0.0, min(1.0, float(item["score"]) / 10))
            for c in candidates:
                c.setdefault("rerank_score", 0.0)
            return
        except Exception as exc:
            log.warning("LLM rerank failed, using lexical rerank: %s", exc)
    q_terms = set(_tokens(query))
    for c in candidates:
        terms = set(_tokens(c["text"]))
        c["rerank_score"] = len(q_terms & terms) / len(q_terms) if q_terms else 0.0


def hybrid_search(query: str, k: int = config.TOP_K, doc_ids: list[str] | None = None) -> list[dict]:
    """Vector search + BM25 fused with reciprocal rank fusion, then reranked and diversified across documents."""
    provider = llm.get_provider()
    data = store.load_chunks(provider.name)
    rows = data["rows"]
    if doc_ids:
        keep = [i for i, r in enumerate(rows) if r["doc_id"] in set(doc_ids)]
        rows = [rows[i] for i in keep]
        matrix = data["matrix"][keep] if keep else np.zeros((0, 1), dtype=np.float32)
    else:
        matrix = data["matrix"]
    if not rows:
        return []

    q_vec = llm.embed_texts([query])[0]
    vector_scores = matrix @ q_vec if matrix.shape[1] == q_vec.shape[0] else np.zeros(len(rows))
    bm25_scores = BM25([_tokens(r["text"]) for r in rows]).scores(_tokens(query))

    vector_rank = np.argsort(-vector_scores)[:50]
    bm25_rank = [i for i in np.argsort(-bm25_scores)[:50] if bm25_scores[i] > 0]
    rrf = np.zeros(len(rows))
    for rank, idx in enumerate(vector_rank):
        rrf[idx] += 1 / (60 + rank + 1)
    for rank, idx in enumerate(bm25_rank):
        rrf[idx] += 1 / (60 + rank + 1)

    pool = [int(i) for i in np.argsort(-rrf)[: max(k * 3, 12)] if rrf[i] > 0]
    candidates = [{
        "chunk_id": rows[i]["id"], "doc_id": rows[i]["doc_id"], "filename": rows[i]["filename"],
        "page": rows[i]["page"], "chunk_index": rows[i]["chunk_index"], "text": rows[i]["text"],
        "vector_score": round(float(vector_scores[i]), 4), "bm25_score": round(float(bm25_scores[i]), 4),
        "rrf_score": round(float(rrf[i]), 5),
    } for i in pool]
    _rerank(query, candidates)
    max_rrf = max((c["rrf_score"] for c in candidates), default=1) or 1
    for c in candidates:
        c["final_score"] = round(0.7 * c["rerank_score"] + 0.3 * (c["rrf_score"] / max_rrf), 4)
    candidates.sort(key=lambda c: -c["final_score"])

    # Keep several documents in the answer when more than one is relevant (multi-document questions).
    per_doc_cap = k if len({c["doc_id"] for c in candidates}) <= 1 else max(2, math.ceil(k / 2))
    picked, per_doc = [], Counter()
    for c in candidates:
        if per_doc[c["doc_id"]] < per_doc_cap:
            picked.append(c)
            per_doc[c["doc_id"]] += 1
        if len(picked) == k:
            break
    for rank, c in enumerate(picked, 1):
        c["rank"] = rank
    return picked


def is_insufficient(results: list[dict]) -> bool:
    if not results:
        return True
    best_vector = max(r["vector_score"] for r in results)
    best_lexical = max(r["bm25_score"] for r in results)
    return best_vector < config.MIN_VECTOR_SCORE and best_lexical == 0


# ---------- answers ----------

def _excerpt(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def answer(question: str, history: list[dict] | None = None) -> dict:
    elapsed = time.perf_counter()
    results = hybrid_search(question, config.TOP_K)
    insufficient = is_insufficient(results)
    citations = [{
        "ref": f"S{r['rank']}", "doc_id": r["doc_id"], "filename": r["filename"], "page": r["page"],
        "excerpt": _excerpt(r["text"]), "score": r["final_score"],
    } for r in results]

    if insufficient:
        text = INSUFFICIENT_MESSAGE
        citations = []
    elif not llm.llm_available():
        # Offline mode: no generation, so return the most relevant excerpts with citations.
        text = "Offline mode (no OPENAI_API_KEY): these are the most relevant passages from your documents.\n\n" + "\n\n".join(
            f"[{c['ref']}] {c['filename']}, page {c['page']}: {c['excerpt']}" for c in citations)
    else:
        context = "\n\n".join(
            f"[{c['ref']}] Document: {c['filename']} (page {c['page']})\n{results[i]['text']}"
            for i, c in enumerate(citations))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or [])[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"})
        reply = llm.get_provider().chat(messages).strip()
        if reply.startswith("INSUFFICIENT_CONTEXT") or not reply:
            text, citations, insufficient = INSUFFICIENT_MESSAGE, [], True
        else:
            text = reply
            cited = set(re.findall(r"\[S(\d+)\]", reply))
            citations = [c for c in citations if c["ref"][1:] in cited] or citations

    latency = (time.perf_counter() - elapsed) * 1000
    store.log_query("chat", question, latency, len(results), insufficient)
    return {"answer": text, "citations": citations, "insufficient": insufficient, "latency_ms": round(latency, 1)}


# ---------- evaluation ----------

def evaluate(cases: list[dict], k: int = config.TOP_K) -> dict:
    """Each case: {"question": str, "expected_document": filename or null, "expected_text": optional substring}.
    expected_document null means the system should abstain (no relevant document exists)."""
    report, hits, reciprocal, abstain_ok, abstain_total = [], 0, 0.0, 0, 0
    retrieval_cases = 0
    for case in cases:
        results = hybrid_search(case["question"], k)
        expected = case.get("expected_document")
        if expected is None:
            abstain_total += 1
            correct = is_insufficient(results)
            abstain_ok += int(correct)
            report.append({"question": case["question"], "expected": None, "abstained": correct, "passed": correct})
            continue
        retrieval_cases += 1
        rank = None
        for i, r in enumerate(results, 1):
            if r["filename"] == expected and (not case.get("expected_text") or case["expected_text"].lower() in r["text"].lower()):
                rank = i
                break
        if rank:
            hits += 1
            reciprocal += 1 / rank
        report.append({"question": case["question"], "expected": expected, "rank": rank, "passed": rank is not None,
                       "top_document": results[0]["filename"] if results else None})
    return {
        "k": k,
        "hit_rate_at_k": round(hits / retrieval_cases, 3) if retrieval_cases else None,
        "mrr": round(reciprocal / retrieval_cases, 3) if retrieval_cases else None,
        "abstention_accuracy": round(abstain_ok / abstain_total, 3) if abstain_total else None,
        "cases": report,
    }
