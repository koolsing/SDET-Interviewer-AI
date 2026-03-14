"""
knowledge_base.py — Document ingestion, local embedding & retrieval (RAG)

Drop .pdf / .txt / .md files into resources/<topic-slug>/ then run:
    python main.py --index

At interview time each question turn will retrieve the top-k most relevant
chunks from your documents and inject them into the LLM prompt.
"""
import os
import json
import hashlib
import re
from typing import List, Dict, Tuple

import numpy as np

from config import RESOURCES_DIR, INDEX_DIR, TOPICS

# ── Optional heavy imports (only needed at index time) ─────────────────────
def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def _parse_pdf(path: str) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(path)

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 400       # approximate tokens (words)
CHUNK_OVERLAP = 80

def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ── Index persistence ──────────────────────────────────────────────────────

def _index_path(topic_slug: str) -> Tuple[str, str]:
    os.makedirs(INDEX_DIR, exist_ok=True)
    base = os.path.join(INDEX_DIR, topic_slug)
    return base + ".npy", base + ".json"

def _load_index(topic_slug: str) -> Tuple[np.ndarray | None, List[Dict]]:
    npy_path, json_path = _index_path(topic_slug)
    if os.path.exists(npy_path) and os.path.exists(json_path):
        vectors = np.load(npy_path)
        with open(json_path, "r") as f:
            meta = json.load(f)
        return vectors, meta
    return None, []

def _save_index(topic_slug: str, vectors: np.ndarray, meta: List[Dict]):
    npy_path, json_path = _index_path(topic_slug)
    np.save(npy_path, vectors)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

# ── Public API ─────────────────────────────────────────────────────────────

def index_topic(topic_slug: str, verbose: bool = True) -> int:
    """
    Parse all docs in resources/<topic_slug>/, chunk them, embed, and save index.
    Returns the number of chunks indexed.
    """
    topic_dir = os.path.join(RESOURCES_DIR, topic_slug)
    if not os.path.isdir(topic_dir):
        return 0

    supported = (".pdf", ".txt", ".md")
    files = [
        os.path.join(topic_dir, f)
        for f in os.listdir(topic_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]
    if not files:
        return 0

    if verbose:
        print(f"    Indexing {len(files)} file(s) for [{topic_slug}]...")

    model = _get_sentence_transformer()
    all_chunks: List[str] = []
    meta: List[Dict] = []

    for fpath in files:
        try:
            text = _read_file(fpath)
            chunks = _chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                meta.append({
                    "source": os.path.basename(fpath),
                    "topic": topic_slug,
                    "hash": _file_hash(fpath),
                    "text": chunk,
                })
        except Exception as e:
            if verbose:
                print(f"    ⚠ Could not read {fpath}: {e}")

    if not all_chunks:
        return 0

    vectors = model.encode(all_chunks, show_progress_bar=False, normalize_embeddings=True)
    _save_index(topic_slug, vectors, meta)

    if verbose:
        print(f"    ✓ {len(all_chunks)} chunks indexed for [{topic_slug}]")
    return len(all_chunks)


def index_all(verbose: bool = True) -> Dict[str, int]:
    """Index every topic that has documents. Returns {slug: chunk_count}."""
    results = {}
    for slug in TOPICS:
        count = index_topic(slug, verbose=verbose)
        if count:
            results[slug] = count
    return results


def retrieve(topic_slug: str, query: str, top_k: int = 3) -> List[str]:
    """
    Return the top_k most relevant text chunks for a query within a topic.
    Returns [] if no index exists for the topic.
    """
    vectors, meta = _load_index(topic_slug)
    if vectors is None or len(meta) == 0:
        return []

    model = _get_sentence_transformer()
    q_vec = model.encode([query], normalize_embeddings=True)[0]

    scores = vectors @ q_vec          # cosine similarity (vectors are normalised)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [meta[i]["text"] for i in top_indices]


def list_docs() -> Dict[str, List[str]]:
    """Return {slug: [filename, ...]} for every topic that has indexed docs."""
    result = {}
    for slug in TOPICS:
        _, meta = _load_index(slug)
        if meta:
            seen = []
            for m in meta:
                if m["source"] not in seen:
                    seen.append(m["source"])
            result[slug] = seen
    return result


def topic_has_docs(topic_slug: str) -> bool:
    vectors, meta = _load_index(topic_slug)
    return vectors is not None and len(meta) > 0
