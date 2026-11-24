# RAG pipeline: decisions, limitations and improvements

## Pipeline

1. **Parse.** PDFs use their text layer (pypdf). A page with fewer than 20 characters of text is treated as scanned: it is rendered with pypdfium2 and OCR'd with Tesseract. JPG and PNG go straight to OCR. DOCX text includes table cells. Each page keeps its number.
2. **Chunk.** Paragraph-aware packing up to 800 characters, with 120 characters of overlap carried into the next chunk. Chunks never cross a page, so every citation has a page number. Long paragraphs are split by length.
3. **Embed.** OpenAI `text-embedding-3-small` by default, in batches of 96. Vectors are normalised and stored as float32 blobs. Each chunk records the embedding model, so offline and OpenAI vectors are never mixed.
4. **Extract.** Rules run on every document: vendor, customer, invoice number, reporting period, labelled financial metrics (including negatives written in parentheses), and pipe-delimited transaction rows. If a key is set, the LLM is also asked for entities and relations in JSON mode. The rule output is kept if the model call fails.
5. **Retrieve.** For a question, run vector search (cosine) and BM25 over the chunks, fuse the two rankings with reciprocal rank fusion (k = 60), take the top candidates, and rerank them. The LLM rerank returns a 0 to 10 relevance score per candidate in one JSON call. Without a key, the rerank is query-term coverage.
6. **Diversify.** When more than one document is relevant, no document may supply more than half of the top-k chunks. This stops one long document from crowding out the others on multi-document questions.
7. **Abstain.** If no chunk has a vector score of at least 0.25 and none matches a query keyword, the app returns an "insufficient information" reply without calling the model. If the model replies with the sentinel `INSUFFICIENT_CONTEXT`, the same reply is returned.
8. **Answer.** The model receives the numbered sources and the last six turns of the conversation. It must cite sources as `[S#]` and show arithmetic. Only the sources it cites are returned as citations, falling back to all retrieved sources when it cites none.
9. **Evaluate and monitor.** `POST /api/eval` reports hit rate @k, MRR and abstention accuracy. Every chat and search is logged with latency, result count and whether the answer abstained.

## Key technical decisions

- **SQLite with in-process vector search.** A corpus of a few thousand chunks fits in memory, so numpy matrix multiplication is faster and simpler than running a vector database. The store is cached and invalidated on every write. The dependency list stays short and the project runs with no extra services. This is the first thing to change at scale.
- **Hybrid retrieval.** Embeddings handle paraphrase. BM25 handles exact identifiers such as `INV-2026-014` and specific amounts, which embeddings often blur. Fusion by rank avoids having to calibrate the two score scales.
- **Rules first, LLM second, for extraction.** Financial documents follow recognisable patterns, so rules are cheap, deterministic and explainable. The LLM adds coverage and does not block ingestion when it fails.
- **Every fact carries evidence.** Metrics, graph edges and node mentions store the page and the source line. The UI shows this, so a reviewer can trace each number to the document.
- **Provider abstraction.** One interface covers OpenAI, OpenAI-compatible local servers and the offline fallback. Switching is configuration only.
- **Anomaly checks use robust statistics.** The median absolute deviation is not distorted by the outlier it is looking for, unlike a mean and standard deviation. Cross-document metric swings are not flagged automatically, because documents may cover different companies and periods. Those comparisons are left to the explicit compare endpoint.
- **Single-file frontend.** No build step, so the app is easy to run and review. The trade-off is a hand-written force layout instead of a graph library.

## Limitations

- **Offline mode is not a real retrieval test.** The hashed embedder matches words but not meaning. Retrieval quality claims should be measured with OpenAI embeddings, using `/api/eval`.
- **Rule extraction assumes a label format.** It recognises labels such as "Total revenue:" and pipe-delimited transaction rows. Real statements use many layouts, so recall on real documents will be lower than on the samples. The LLM pass helps, but its output has not been measured against labelled data.
- **Tables in PDFs are read as text.** Column alignment is lost in pypdf output, so a PDF table is only as clear as its text order. Table extraction with pdfplumber or a layout model would be the next step.
- **OCR quality depends on the scan.** Tesseract works well on clean, high-resolution scans and poorly on skewed, low-resolution or handwritten input. No pre-processing (deskew, denoise, contrast) is applied.
- **Chunk-level metric values are not reconciled.** If two documents report different totals for the same period, the app shows both and flags the difference only in the cases the rules cover.
- **Single process.** Ingestion runs as a background task inside the web server. A crash during processing leaves the document in `processing`, which can be fixed by reprocessing it. A job queue would solve this.
- **No authentication, no multi-user isolation.** The data model has no tenant or owner.
- **Anomaly rules are heuristics.** They can produce false positives. They are a triage aid, not an audit.
- **Conversation context is limited.** The last six turns are sent to the model. Questions are not rewritten into standalone queries, so a follow-up such as "and net income?" relies on the model to combine the history with its retrieval results.

## Potential improvements

1. Use a layout-aware parser (for example pdfplumber, or a document AI service) to keep table structure, and store tables as rows.
2. Add a cross-encoder reranker (local or hosted) in place of the LLM rerank, to cut cost and latency.
3. Rewrite follow-up questions into standalone queries before retrieval.
4. Build a labelled evaluation set of real-world layouts and track recall of the extraction rules and retrieval metrics over time.
5. Replace the in-memory vector scan with a vector index (sqlite-vec, pgvector or Qdrant) once the corpus grows.
6. Move ingestion to a job queue with retries and progress reporting.
7. Add reconciliation rules across documents for the same company and period, and show the agreed figure with its sources.
8. Add authentication, per-user workspaces and audit logging.
