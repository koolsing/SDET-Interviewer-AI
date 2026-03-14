"""
llm.py — Ollama LLM client with model switching and streaming support.
All conversation history is kept in-memory only — nothing is persisted.

Qwen3 reasoning tags (<think>…</think>) are stripped at this layer so
no caller ever receives raw thinking output.
"""
import re
import json
import requests
from typing import List, Dict, Iterator

from config import OLLAMA_BASE_URL, DEFAULT_MODEL


def _strip_think(text: str) -> str:
    """Remove Qwen3's <think>…</think> internal reasoning blocks."""
    # Remove complete <think>...</think> blocks (multi-line)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If the model started thinking without an opening tag, remove everything up to </think>
    text = re.sub(r"^.*?</think>\n*", "", text, flags=re.DOTALL)
    # Remove any remaining stray </think> (just in case)
    text = text.replace("</think>", "")
    return text.strip()


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.base_url = OLLAMA_BASE_URL

    # ── Model management ──────────────────────────────────────────────────────

    def list_models(self) -> List[str]:
        """Return list of model names available in local Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def is_running(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def switch_model(self, model: str):
        self.model = model

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, messages: List[Dict], stream: bool = False) -> str:
        """
        Send a list of messages to Ollama and return the full response text.
        messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_ctx": 8192,
            },
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
                stream=stream,
            )
            r.raise_for_status()

            if not stream:
                return _strip_think(r.json()["message"]["content"])

            # streaming — accumulate chunks then strip think tags from full text
            full = ""
            for line in r.iter_lines():
                if line:
                    obj = json.loads(line)
                    chunk = obj.get("message", {}).get("content", "")
                    full += chunk
            return _strip_think(full)

        except requests.exceptions.ConnectionError:
            return "[Error: Could not connect to Ollama. Is it running? (ollama serve)]"
        except Exception as e:
            return f"[Error: {e}]"

    def chat_stream(self, messages: List[Dict]) -> Iterator[str]:
        """
        Streaming version — yields text chunks as they arrive from the model.
        Useful for printing the response character-by-character.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.7, "num_ctx": 8192},
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
                stream=True,
            ) as r:
                r.raise_for_status()
                full = ""
                for line in r.iter_lines():
                    if line:
                        obj = json.loads(line)
                        chunk = obj.get("message", {}).get("content", "")
                        if chunk:
                            full += chunk
                        if obj.get("done"):
                            break
                # Strip think tags from the fully accumulated response
                cleaned = _strip_think(full)
                yield cleaned
        except Exception as e:
            yield f"[Error: {e}]"


# Convenience module-level instance
_client: OllamaClient | None = None

def get_client(model: str = DEFAULT_MODEL) -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient(model)
    return _client
