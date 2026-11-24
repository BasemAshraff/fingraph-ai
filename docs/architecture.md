# FinGraph AI: system architecture

```mermaid
flowchart LR
    subgraph Browser["Web UI (frontend/index.html)"]
        V1[Dashboard]
        V2[Document library]
        V3[Knowledge graph]
        V4[Wiki Ask chat]
        V5[Financials]
        V6[RAG debug & monitoring]
    end

    Browser -- REST/JSON --> API

    subgraph API["FastAPI (backend/app/main.py)"]
        R1[Upload + status + delete]
        R2[Hybrid search / chat]
        R3[Graph + node detail]
        R4[Metrics, compare, anomalies, report]
        R5[Eval + monitoring]
    end

    R1 -- background task --> PIPE
    subgraph PIPE["Ingestion pipeline (rag.py)"]
        P1[Parse: pypdf / python-docx]
        P2[OCR: pdfium render + Tesseract]
        P3[Chunk: paragraph-aware, overlap, page kept]
        P4[Embed: OpenAI or offline hashing]
        P5[Extract: rules + optional LLM]
        P1 --> P2 --> P3 --> P4
        P1 --> P5
    end

    R2 --> RET
    subgraph RET["Retrieval (rag.py)"]
        Q1[Vector cosine] 
        Q2[BM25 lexical]
        Q3[Reciprocal rank fusion]
        Q4[Rerank: LLM or lexical]
        Q5[Per-document diversity cap]
        Q1 --> Q3
        Q2 --> Q3 --> Q4 --> Q5
    end

    RET --> GEN[Grounded answer with [S#] citations / abstain]
    GEN --> OPENAI[(OpenAI API or OpenAI-compatible server)]
    P4 --> OPENAI
    Q1 -. query embedding .-> OPENAI

    P3 --> DB
    P4 --> DB
    P5 --> DB[(SQLite: documents, chunks + embeddings, graph, metrics, transactions, chats, query log)]
    R3 --> DB
    R4 --> DB
    R5 --> DB
```

## Components

| Layer | Files | Notes |
|---|---|---|
| API | `backend/app/main.py` | FastAPI routes, input validation, static frontend |
| Ingestion | `backend/app/parsing.py`, `chunking.py`, `rag.process_document` | Errors stored on the document as `failed` with a message |
| Retrieval and answers | `backend/app/rag.py` | BM25, vector search, RRF, reranking, grounded answers, evaluation |
| Financial logic | `backend/app/extraction.py` | Entities, relations, metrics, transactions, anomaly rules, comparisons |
| Models | `backend/app/llm.py` | Provider abstraction: OpenAI / OpenAI-compatible, or offline fallback |
| Storage | `backend/app/store.py` | SQLite with WAL-free single connection and a lock |
| UI | `frontend/index.html` | Single file, no build step |
