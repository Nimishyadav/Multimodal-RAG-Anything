"""Shared local model and ingestion settings for all entry points."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TEXT_MODEL_NAME = os.environ.get("RAG_TEXT_MODEL", "qwen3:4b").strip() or "qwen3:4b"
VISION_MODEL_NAME = os.environ.get("RAG_VISION_MODEL", "qwen2.5vl:3b").strip() or "qwen2.5vl:3b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be true or false.")
    return value in {"1", "true", "yes", "on"}
