"""Exercises the LLM-backed code paths (grounded answers, LLM rerank, LLM extraction) with a scripted fake
provider, so they are tested without network access or an API key."""
import json
import re
import shutil

import pytest

from app import config, llm, rag, store, extraction
from app.llm import HashingEmbedder

SAMPLES = __import__("pathlib").Path(__file__).resolve().parents[2] / "samples"


class FakeProvider:
    name = "fake:test-embed"
    chat_model = "fake-chat"

    def __init__(self):
        self.embedder = HashingEmbedder()
        self.calls = []

    def embed(self, texts):
        return self.embedder.embed(texts)

    def chat(self, messages, json_mode=False, temperature=0.1):
        prompt = messages[-1]["content"]
        self.calls.append((json_mode, prompt[:60]))
        if json_mode and "Rate how useful" in prompt:
            n = len(re.findall(r"^\[(\d+)\] ", prompt, re.M))
            return json.dumps({"scores": [{"id": i, "score": 9 if i == 0 else 2} for i in range(n)]})
        if json_mode and "entities" in prompt:
            return json.dumps({
                "entities": [{"type": "Company", "name": "Delta Retail Ltd"}, {"type": "Vendor", "name": "Nile Cloud Services"}],
                "relations": [{"source": "Vendor:Nile Cloud Services", "target": "Company:Delta Retail Ltd", "relation": "SERVES", "evidence": "hosting"}],
            })
        if "weather" in prompt.lower():
            return "INSUFFICIENT_CONTEXT"
        return "Total revenue was 250,000.00 for FY 2025 [S1]."


@pytest.fixture
def fake(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(llm, "_provider", provider)
    return provider


def test_llm_ingestion_and_grounded_answer(fake):
    src = SAMPLES / "annual_statement_delta_fy2025.pdf"
    dst = config.UPLOAD_DIR / "llm_path_copy.pdf"
    shutil.copy(src, dst)
    doc = store.create_document("llm_path_copy.pdf", str(dst), dst.stat().st_size, ".pdf")
    rag.process_document(doc["id"])
    doc = store.get_document(doc["id"])
    assert doc["status"] == "ready", doc["error"]

    # LLM extraction pass added an entity and a relation.
    assert store.one("SELECT id FROM graph_nodes WHERE id='Vendor:nile cloud services'")

    result = rag.answer("What was total revenue in FY 2025?")
    assert not result["insufficient"]
    assert result["citations"] and result["citations"][0]["ref"] == "S1"
    assert "250,000.00" in result["answer"]
    assert any(call[0] is True for call in fake.calls), "LLM rerank should have run in JSON mode"


def test_llm_insufficient_sentinel_is_mapped(fake):
    result = rag.answer("What is the weather in Cairo?")
    assert result["insufficient"] is True
    assert result["citations"] == []


def test_history_is_passed_to_model(fake, monkeypatch):
    seen = {}

    original = fake.chat

    def spy(messages, json_mode=False, temperature=0.1):
        if not json_mode:  # the answer call, not the rerank call
            seen.setdefault("messages", messages)
        return original(messages, json_mode, temperature)

    monkeypatch.setattr(fake, "chat", spy)
    history = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"}]
    rag.answer("What was total revenue in FY 2025?", history)
    roles = [m["role"] for m in seen["messages"]]
    assert roles[:3] == ["system", "user", "assistant"]
