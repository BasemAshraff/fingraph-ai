"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("FINGRAPH_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "fingraph.db"

# LLM provider. If OPENAI_API_KEY is empty the app runs in offline mode:
# a hashing embedder for retrieval and extractive (non-generative) answers.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # optional: any OpenAI-compatible server (e.g. Ollama at http://localhost:11434/v1)

# RAG tuning
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # characters carried into the next chunk
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_VECTOR_SCORE = float(os.getenv("MIN_VECTOR_SCORE", "0.25"))  # below this, context is treated as irrelevant

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
