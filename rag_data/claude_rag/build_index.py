"""
build_index.py
--------------
Reads fish_summary.csv, embeds Species Name + Description,
builds a FAISS IndexFlatIP, and saves fish.index + records.pkl.
"""

import csv
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────
CSV_PATH    = "claude_descriptions.csv"
INDEX_PATH  = "fish.index"
RECORDS_PATH = "records.pkl"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────


def load_records(csv_path: str) -> List[Dict[str, str]]:
    """Read fish_summary.csv and return a list of clean record dicts."""
    records: List[Dict[str, str]] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            species_name = row["Species Name"].strip()
            description  = row["Description"].strip()

            # Skip rows that are missing either field
            if not species_name or not description:
                continue

            records.append({
                "Species Name": species_name,
                "Description":  description,
            })

    print(f"[build_index] Loaded {len(records)} records from '{csv_path}'.")
    return records


def build_texts(records: List[Dict[str, str]]) -> List[str]:
    """Combine Species Name and Description into a single embedding string."""
    return [
        f"{r['Species Name']}: {r['Description']}"
        for r in records
    ]


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Encode texts into L2-normalised float32 embeddings."""
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS inner-product index from the embedding matrix."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[build_index] FAISS index built — {index.ntotal} vectors, dim={dim}.")
    return index


def save_artifacts(
    index: faiss.IndexFlatIP,
    records: List[Dict[str, str]],
    index_path: str,
    records_path: str,
) -> None:
    """Persist the FAISS index and records list to disk."""
    faiss.write_index(index, index_path)
    print(f"[build_index] Saved FAISS index → '{index_path}'.")

    with open(records_path, "wb") as fh:
        pickle.dump(records, fh)
    print(f"[build_index] Saved records    → '{records_path}'.")


def main() -> None:
    # 1. Load records from CSV
    records = load_records(CSV_PATH)
    if not records:
        raise ValueError("No valid records found. Check your CSV file.")

    # 2. Build embedding texts
    texts = build_texts(records)

    # 3. Load model and embed
    print(f"[build_index] Loading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_texts(texts, model)

    # 4. Build FAISS index
    index = build_faiss_index(embeddings)

    # 5. Save artefacts
    save_artifacts(index, records, INDEX_PATH, RECORDS_PATH)
    print("[build_index] Done ✓")


if __name__ == "__main__":
    main()