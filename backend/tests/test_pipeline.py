from app import chunking, extraction, rag, store


def test_all_samples_ready_and_ocr_used(ingested):
    assert all(d["status"] == "ready" for d in ingested.values()), {k: v["error"] for k, v in ingested.items()}
    assert ingested["scanned_invoice_sahara_mar2026.png"]["ocr_pages"] == 1
    assert ingested["scanned_statement_sahara_q1.pdf"]["ocr_pages"] == 1
    assert ingested["annual_statement_delta_fy2025.pdf"]["ocr_pages"] == 0


def test_chunking_keeps_pages_and_overlaps():
    text = " ".join(f"sentence{i} with some words" for i in range(200))
    chunks = chunking.chunk_pages([{"page": 3, "text": text}], size=300, overlap=60)
    assert len(chunks) > 1
    assert all(c["page"] == 3 for c in chunks)
    assert all(len(c["text"]) <= 300 for c in chunks)
    assert chunks[0]["text"][-30:].split()[-1] in chunks[1]["text"]  # overlap carried forward


def test_ocr_text_is_readable(ingested):
    chunks = store.document_chunks(ingested["scanned_invoice_sahara_mar2026.png"]["id"])
    joined = " ".join(c["text"] for c in chunks)
    assert "INV-2026-014" in joined and "2,350.00" in joined


def test_retrieval_finds_the_right_document(ingested):
    top = rag.hybrid_search("which vendor issued invoice INV-2026-014", 3)
    assert top[0]["filename"] == "scanned_invoice_sahara_mar2026.png"
    top = rag.hybrid_search("what is the payment term for late payments", 3)
    assert top[0]["filename"] == "vendor_agreement_nile_cloud.docx"


def test_insufficient_context_is_detected(ingested):
    results = rag.hybrid_search("what is the weather in Cairo", 3)
    assert rag.is_insufficient(results)
    result = rag.answer("what is the weather in Cairo")
    assert result["insufficient"] and result["citations"] == []


def test_multi_document_question_returns_several_documents(ingested):
    results = rag.hybrid_search("total revenue and net income", 6)
    assert len({r["doc_id"] for r in results}) >= 2


def test_metrics_and_transactions_extracted(ingested):
    metrics = store.query("SELECT name, value FROM metrics m JOIN documents d ON d.id=m.doc_id WHERE d.filename='annual_statement_delta_fy2025.pdf'")
    values = {m["name"]: m["value"] for m in metrics}
    assert values["total revenue"] == 250000.0
    assert values["net income"] == 32500.0
    txns = store.query("SELECT amount FROM transactions t JOIN documents d ON d.id=t.doc_id WHERE d.filename='invoice_nile_cloud_jan2026.pdf'")
    assert len(txns) == 6


def test_graph_links_invoice_to_vendor_and_period(ingested):
    edges = store.query("SELECT source, target, relation FROM graph_edges WHERE relation='ISSUED_BY' AND source='Invoice:inv-2026-001'")
    assert edges and edges[0]["target"] == "Vendor:nile cloud services"
    rels = {e["relation"] for e in store.query("SELECT relation FROM graph_edges")}
    assert {"ISSUED_BY", "BILLED_TO", "IN_PERIOD", "WITH"} <= rels


def test_anomalies_flagged(ingested):
    from app.main import _anomalies

    kinds = {a["type"] for a in _anomalies()}
    assert "total_mismatch" in kinds  # management report: 12,000 + 1,800 != 15,000
    assert "outlier_transaction" in kinds  # 95,000 emergency restore vs 4,500 median
    assert "duplicate_invoice" in kinds  # INV-2026-001 appears in invoice and management report


def test_invoice_regex_ignores_plain_words():
    facts = extraction.extract_facts("x", "f.pdf", [{"page": 1, "text": "Use this invoice template for billing.", "method": "text"}])
    assert not [n for n in facts["nodes"].values() if n["type"] == "Invoice"]


def test_rule_extraction_handles_negative_parentheses():
    facts = extraction.extract_facts("x", "f.pdf", [{"page": 1, "text": "Net income: (4,200.00)", "method": "text"}])
    assert facts["metrics"][0]["value"] == -4200.0


def test_compare_metric_changes():
    rows = extraction.compare_metric("total", [{"doc_id": "a", "value": 100}, {"doc_id": "b", "value": 150}, {"doc_id": "c", "value": 120}], {})
    assert rows[1]["change_vs_first_pct"] == 50.0
    assert rows[2]["change_vs_previous_pct"] == -20.0
