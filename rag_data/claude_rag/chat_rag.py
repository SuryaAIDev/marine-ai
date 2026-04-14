"""
chat_rag.py
-----------
Interactive RAG chat loop.
Loads fish.index + records.pkl, embeds user queries, retrieves the
top-k most relevant species, and sends a grounded prompt to Ollama.
"""

import pickle
import json
from typing import List, Dict

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────
INDEX_PATH   = "fish.index"
RECORDS_PATH = "records.pkl"
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"
TOP_K        = 1
# ─────────────────────────────────────────────────────────────────────────────


def load_artifacts(
    index_path: str,
    records_path: str,
) -> tuple[faiss.IndexFlatIP, List[Dict[str, str]]]:
    """Load the FAISS index and records list from disk."""
    index = faiss.read_index(index_path)
    with open(records_path, "rb") as fh:
        records: List[Dict[str, str]] = pickle.load(fh)
    print(f"[chat_rag] Loaded {index.ntotal} vectors and {len(records)} records.")
    return index, records


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Encode and L2-normalise a single query string."""
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.astype(np.float32)


def retrieve(
    query_vec: np.ndarray,
    index: faiss.IndexFlatIP,
    records: List[Dict[str, str]],
    top_k: int,
) -> List[Dict[str, str]]:
    """Return the top_k most similar records (no score threshold)."""
    _, indices = index.search(query_vec, top_k)
    return [records[i] for i in indices[0] if i < len(records)]


def build_context(hits: List[Dict[str, str]]) -> str:
    """Format retrieved records into a readable context block."""
    blocks = []
    for idx, hit in enumerate(hits, start=1):
        block = (
            f"[Species {idx}]\n"
            f"Name: {hit['Species Name']}\n"
            f"Description: {hit['Description']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_prompt(context: str, question: str) -> str:
    """Construct the final prompt sent to Ollama."""
    return (
        "You are a marine biology assistant. "
        "Answer the question using ONLY the species information provided below. "
        "If the answer cannot be found in the provided information, "
        'respond with exactly: "Species not found in database."\n\n'
        f"=== Species Database ===\n{context}\n\n"
        f"=== Question ===\n{question}\n\n"
        "=== Answer ==="
    )


def query_ollama(prompt: str) -> str:
    """Send a non-streaming request to Ollama and return the response text."""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[Error] Could not connect to Ollama. Make sure it is running on localhost:11434."
    except requests.exceptions.HTTPError as exc:
        return f"[Error] Ollama returned HTTP {exc.response.status_code}."
    except Exception as exc:  # noqa: BLE001
        return f"[Error] Unexpected error: {exc}"


def main() -> None:
    # ── Bootstrap ────────────────────────────────────────────────────────────
    print(f"[chat_rag] Loading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    index, records = load_artifacts(INDEX_PATH, RECORDS_PATH)
    print("[chat_rag] Ready. Type 'exit' or 'quit' to stop.\n")

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat_rag] Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("[chat_rag] Goodbye.")
            break

        # 1. Embed query
        query_vec = embed_query(user_input, model)

        # 2. Retrieve top-k species
        hits = retrieve(query_vec, index, records, TOP_K)

        if not hits:
            print("Assistant: Species not found in database.\n")
            continue

        # 3. Build context + prompt
        context = build_context(hits)
        prompt  = build_prompt(context, user_input)

        # 4. Query Ollama
        answer = query_ollama(prompt)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()