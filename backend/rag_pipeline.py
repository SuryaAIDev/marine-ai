"""
RAG Pipeline (Chat_RAG version integrated)
Uses FAISS + sentence-transformers + Ollama Phi3
STRICT grounded RAG (no hallucination allowed)
"""

import os
import pickle
import requests
from pathlib import Path
from typing import List

import numpy as np

# ── Paths ─────────────────────────────────────────────────────
_RAG_DIR = Path(__file__).parent.parent / "rag_data" / "claude_rag"
_INDEX_PATH = _RAG_DIR / "fish.index"
_RECORDS_PATH = _RAG_DIR / "records.pkl"

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

TOP_K = 1  # same as chat_rag

_faiss_index = None
_records = []
_embedder = None


# ─────────────────────────────────────────────────────────────
# Load resources (FAISS + embeddings)
# ─────────────────────────────────────────────────────────────
def _load_resources():
    global _faiss_index, _records, _embedder

    if _faiss_index is not None:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    print("[RAG] Loading FAISS index...")
    _faiss_index = faiss.read_index(str(_INDEX_PATH))

    print("[RAG] Loading records...")
    with open(_RECORDS_PATH, "rb") as f:
        _records = pickle.load(f)

    print("[RAG] Loading embedding model...")
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("[RAG] RAG ready ✓")


# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────
def _embed(text: str) -> np.ndarray:
    vec = _embedder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Retrieval (Chat_RAG logic)
# ─────────────────────────────────────────────────────────────
def _retrieve(query: str) -> List[str]:
    _load_resources()

    query_vec = _embed(query)
    _, indices = _faiss_index.search(query_vec, TOP_K)

    hits = []
    for idx in indices[0]:
        if idx >= len(_records):
            continue

        record = _records[idx]
        if isinstance(record, dict):
            text = record.get("Description") or record.get("description") or str(record)
        else:
            text = str(record)

        hits.append(text)

    return hits


# ─────────────────────────────────────────────────────────────
# Ollama call
# ─────────────────────────────────────────────────────────────
def _call_ollama(prompt: str) -> str:
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        r = requests.post(_OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"


# ─────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION (used by backend)
# ─────────────────────────────────────────────────────────────
def generate_species_answer(species: str, user_query: str) -> str:
    """
    STRICT GROUNDED RAG PIPELINE

    Image → detection → species →
    Retrieve → Build grounded prompt → Phi3 → Answer
    """

    _load_resources()

    # 🔥 IMPORTANT: Retrieval uses BOTH species + question
    retrieval_query = f"{species}. {user_query}"
    contexts = _retrieve(retrieval_query)

    if not contexts:
        return "Species not found in database."

    # Format context EXACTLY like chat_rag
    context_block = "\n\n".join(
        f"[Species {i+1}]\nDescription: {ctx}" for i, ctx in enumerate(contexts)
    )

    # STRICT grounded prompt (ported from chat_rag)
    prompt = (
        "You are a marine biology assistant. "
        "Answer the question using ONLY the species information provided below. "
        "If the answer cannot be found in the provided information, "
        'respond with exactly: "Species not found in database."\n\n'
        f"=== Species Database ===\n{context_block}\n\n"
        f"=== Question ===\n{user_query}\n\n"
        "=== Answer ==="
    )

    answer = _call_ollama(prompt)
    return answer



'''

"""
RAG Pipeline Integration
Reuses the existing claude_rag/ FAISS index and sentence-transformer embeddings.
Calls Ollama (phi3) locally for answer generation.
"""

import os
import sys
import json
import pickle
import requests
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_RAG_DIR    = Path(__file__).parent.parent / "rag_data" / "claude_rag"
_INDEX_PATH = _RAG_DIR / "fish.index"
_RECORDS_PATH = _RAG_DIR / "records.pkl"

# Ollama endpoint
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_faiss_index = None
_records: List[dict] = []
_embedder = None

TOP_K = 3  # Number of RAG chunks to retrieve


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_resources():
    """Load FAISS index, records, and sentence-transformer model (once)."""
    global _faiss_index, _records, _embedder

    if _faiss_index is not None:
        return  # Already loaded

    # ── FAISS ──────────────────────────────────────────────────────────────
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu") from e

    if not _INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {_INDEX_PATH}. "
            "Make sure rag_data/claude_rag/fish.index exists."
        )
    print(f"[RAG] Loading FAISS index from {_INDEX_PATH} …")
    _faiss_index = faiss.read_index(str(_INDEX_PATH))
    print(f"[RAG] FAISS index loaded ({_faiss_index.ntotal} vectors) ✓")

    # ── Records ────────────────────────────────────────────────────────────
    if not _RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Records not found at {_RECORDS_PATH}. "
            "Make sure rag_data/claude_rag/records.pkl exists."
        )
    with open(_RECORDS_PATH, "rb") as f:
        _records = pickle.load(f)
    print(f"[RAG] Loaded {len(_records)} records ✓")

    # ── Sentence-Transformer ───────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        ) from e

    model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    print(f"[RAG] Loading embedder '{model_name}' …")
    _embedder = SentenceTransformer(model_name)
    print("[RAG] Embedder loaded ✓")


def _embed(text: str) -> np.ndarray:
    """Return a (1, dim) float32 numpy array."""
    vec = _embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")


def _retrieve(query: str, k: int = TOP_K) -> List[str]:
    """Return up to *k* context strings from FAISS."""
    _load_resources()
    query_vec = _embed(query)
    distances, indices = _faiss_index.search(query_vec, k)

    contexts: List[str] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(_records):
            continue
        record = _records[idx]
        # records.pkl may store dicts or plain strings
        if isinstance(record, dict):
            # Try common key names
            text = (
                record.get("description")
                or record.get("text")
                or record.get("content")
                or str(record)
            )
        else:
            text = str(record)
        contexts.append(text.strip())

    return contexts


def _call_ollama(prompt: str) -> str:
    """Send *prompt* to Ollama phi3 and return the generated text."""
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] Cannot connect to Ollama. "
            "Make sure Ollama is running: `ollama serve` and model is pulled: `ollama pull phi3`"
        )
    except requests.exceptions.Timeout:
        return "[ERROR] Ollama request timed out. Try a shorter query or restart Ollama."
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_species_answer(species: str, user_query: str) -> str:
    """
    End-to-end RAG + Phi-3 answer for a detected *species* and a *user_query*.

    Steps:
      1. Build retrieval query from species name
      2. Retrieve relevant context chunks from FAISS
      3. Build a grounded prompt
      4. Call Ollama phi3
      5. Return the generated answer
    """
    _load_resources()

    # Step 3 — Retrieve context
    retrieval_query = f"{species} marine species"
    contexts = _retrieve(retrieval_query, k=TOP_K)

    if contexts:
        context_block = "\n\n".join(
            f"[Source {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
    else:
        context_block = f"No detailed records found for '{species}'."

    # Step 4 — Build prompt
    prompt = f"""You are a marine biology expert assistant. Answer the user's question using the provided species information.

DETECTED SPECIES: {species}

RETRIEVED SPECIES INFORMATION:
{context_block}

USER QUESTION: {user_query}

Instructions:
- Answer directly and informatively based on the retrieved information.
- If the retrieved context is insufficient, use your general knowledge about marine species.
- Keep the answer concise but complete (3–5 sentences).
- Do not mention that you are an AI or reference "retrieved information" explicitly.

ANSWER:"""

    # Step 5 — Generate answer
    answer = _call_ollama(prompt)
    return answer

'''