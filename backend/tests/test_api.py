def test_health_reports_offline_mode(client):
    body = client.get("/api/health").json()
    assert body["llm_available"] is False


def test_upload_rejects_unsupported_type(client):
    res = client.post("/api/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_upload_rejects_empty_file(client):
    res = client.post("/api/documents", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert res.status_code == 400


def test_upload_process_and_delete(client, ingested):
    content = (__import__("pathlib").Path(__file__).resolve().parents[2] / "samples" / "vendor_agreement_nile_cloud.docx").read_bytes()
    res = client.post("/api/documents", files={"file": ("temp_copy.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert res.status_code == 202
    doc_id = res.json()["id"]
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "ready"  # background task runs before the test client returns
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get(f"/api/documents/{doc_id}").status_code == 404


def test_chat_returns_grounded_citations(client, ingested):
    body = client.post("/api/chat", json={"message": "Which vendor issued invoice INV-2026-014?"}).json()
    assert body["conversation_id"]
    assert body["citations"][0]["filename"] == "scanned_invoice_sahara_mar2026.png"
    assert body["citations"][0]["excerpt"]


def test_chat_abstains_when_no_information(client, ingested):
    body = client.post("/api/chat", json={"message": "What is the weather in Cairo tomorrow?"}).json()
    assert body["insufficient"] is True
    assert "couldn't find enough information" in body["answer"]


def test_conversation_history_is_kept(client, ingested):
    first = client.post("/api/chat", json={"message": "What was total revenue in FY 2025?"}).json()
    second = client.post("/api/chat", json={"message": "And net income?", "conversation_id": first["conversation_id"]}).json()
    msgs = client.get(f"/api/conversations/{first['conversation_id']}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert second["conversation_id"] == first["conversation_id"]


def test_graph_and_node_detail(client, ingested):
    graph = client.get("/api/graph").json()
    assert len(graph["nodes"]) > 5 and len(graph["edges"]) > 5
    detail = client.get("/api/graph/nodes/Vendor:nile cloud services").json()
    assert detail["connections"] and detail["supporting_documents"]


def test_compare_and_calculate(client, ingested):
    rows = client.get("/api/compare", params={"metric": "net income"}).json()["rows"]
    assert len(rows) >= 2
    assert client.post("/api/calculate", json={"expression": "(250000 - 190000) / 250000 * 100"}).json()["result"] == 24.0
    assert client.post("/api/calculate", json={"expression": "__import__('os')"}).status_code == 400


def test_anomalies_and_report(client, ingested):
    assert any(a["type"] == "total_mismatch" for a in client.get("/api/anomalies").json())
    md = client.get("/api/report.md").text
    assert "# FinGraph financial report" in md and "Anomalies" in md
    csv_text = client.get("/api/report.csv").text
    assert csv_text.startswith("document,metric,value")


def test_retrieval_evaluation(client, ingested):
    cases = [
        {"question": "What was total revenue in FY 2025?", "expected_document": "annual_statement_delta_fy2025.pdf"},
        {"question": "Which vendor issued invoice INV-2026-014?", "expected_document": "scanned_invoice_sahara_mar2026.png"},
        {"question": "What is the weather in Cairo?", "expected_document": None},
    ]
    report = client.post("/api/eval", json={"cases": cases}).json()
    assert report["hit_rate_at_k"] >= 0.5
    assert report["abstention_accuracy"] == 1.0


def test_monitoring_records_queries(client, ingested):
    client.post("/api/search", json={"query": "net income"})
    mon = client.get("/api/monitoring").json()
    assert mon["queries_recorded"] > 0 and mon["avg_latency_ms"] is not None


def test_search_debug_returns_scores(client, ingested):
    body = client.post("/api/search", json={"query": "total revenue", "k": 3}).json()
    first = body["results"][0]
    assert {"vector_score", "bm25_score", "rrf_score", "rerank_score", "final_score"} <= first.keys()


def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "FinGraph" in res.text
