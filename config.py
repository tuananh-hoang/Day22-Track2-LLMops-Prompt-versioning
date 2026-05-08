"""
config.py — Shared configuration helpers for Day 22 Lab
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── LangSmith settings ─────────────────────────────────────────────────────
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT  = os.getenv("LANGSMITH_PROJECT", "day22-langsmith-lab")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# ── OpenAI / LLM settings ──────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "text-embedding-3-small")

# ── Enable LangSmith tracing (must be set before importing LangChain) ──────
def enable_tracing(project: str = LANGSMITH_PROJECT) -> None:
    """Set LangSmith environment variables to activate tracing."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = project
    os.environ["LANGCHAIN_ENDPOINT"]   = LANGSMITH_ENDPOINT


# ── Sanity check ───────────────────────────────────────────────────────────
def check_config() -> None:
    """Print a summary of loaded configuration values."""
    ok = True
    if not LANGSMITH_API_KEY:
        print("❌ LANGSMITH_API_KEY not set")
        ok = False
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set")
        ok = False

    if ok:
        print("✅ Config loaded successfully")
        print(f"   LangSmith project : {LANGSMITH_PROJECT}")
        print(f"   OpenAI endpoint   : {OPENAI_BASE_URL}")
        print(f"   Default LLM model : {DEFAULT_LLM_MODEL}")
        print(f"   Embedding model   : {EMBEDDING_MODEL}")
    else:
        print("\nPlease copy .env.example to .env and fill in the required values.")


if __name__ == "__main__":
    check_config()