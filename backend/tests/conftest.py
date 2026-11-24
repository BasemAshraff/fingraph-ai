"""Tests run fully offline: a temporary data directory and no OpenAI key."""
import os
import shutil
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="fingraph_test_")
os.environ["FINGRAPH_DATA_DIR"] = _tmp
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

import pytest  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture(scope="session")
def ingested():
    """Upload and process every sample once for the whole test session."""
    from app import config, rag, store

    for sample in sorted(SAMPLES.iterdir()):
        dst = config.UPLOAD_DIR / sample.name
        shutil.copy(sample, dst)
        doc = store.create_document(sample.name, str(dst), dst.stat().st_size, sample.suffix)
        rag.process_document(doc["id"])
    return {d["filename"]: d for d in store.list_documents()}


@pytest.fixture(scope="session")
def client(ingested):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
