# 🐠 Marine Species AI — Complete Setup Guide

End-to-end pipeline: **YOLOv8 detection → FAISS RAG → Phi-3 (Ollama) answer generation**  
100% local. No paid APIs. Runs on 8–16 GB RAM, CPU-only laptop.

---

## 📁 Final Project Structure

```
marine_ai/
│
├── backend/
│   ├── __init__.py          ← Python package marker
│   ├── app.py               ← FastAPI server (POST /analyze)
│   ├── yolo_detector.py     ← loads best.pt, runs inference
│   ├── rag_pipeline.py      ← FAISS retrieval + Ollama phi3 call
│   └── models/
│       └── best.pt          ← ⚠️  YOU PLACE YOUR MODEL HERE
│
├── rag_data/
│   └── claude_rag/          ← ⚠️  COPY YOUR EXISTING claude_rag/ HERE
│       ├── fish.index
│       ├── records.pkl
│       └── claude_descriptions.csv
│
├── frontend/
│   └── streamlit_app.py     ← Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Step 1 — Place Your Existing Files

Before anything else, copy your existing assets into the project:

```bash
# 1a. Copy your YOLO model
cp /path/to/your/best.pt marine_ai/backend/models/best.pt

# 1b. Copy your entire claude_rag folder
cp -r /path/to/your/claude_rag  marine_ai/rag_data/claude_rag
```

After this, verify:
```
marine_ai/backend/models/best.pt          ✓
marine_ai/rag_data/claude_rag/fish.index  ✓
marine_ai/rag_data/claude_rag/records.pkl ✓
```

---

## 🐍 Step 2 — Python Environment

```bash
cd marine_ai

# Create virtual environment
python -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note on PyTorch:** `ultralytics` will automatically install CPU-only PyTorch
> if no GPU is detected. If it pulls a GPU version and you want to save disk space:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install ultralytics
> ```

---

## 🦙 Step 3 — Start Ollama + Pull Phi-3

Install Ollama from https://ollama.com/download (one-time setup).

```bash
# Pull the Phi-3 model (~2 GB, downloaded once)
ollama pull phi3

# Start the Ollama server (keep this terminal open)
ollama serve
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
# Should return a JSON list of models including phi3
```

---

## 🚀 Step 4 — Run the FastAPI Backend

Open a **new terminal** (keep Ollama running in the first one):

```bash
cd marine_ai
source venv/bin/activate   # or venv\Scripts\activate on Windows

uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## 🖥️ Step 5 — Run the Streamlit Frontend

Open a **third terminal**:

```bash
cd marine_ai
source venv/bin/activate

streamlit run frontend/streamlit_app.py
```

Streamlit will open automatically at **http://localhost:8501**

---

## 🧪 Step 6 — Test the API with cURL

```bash
# Basic test — replace fish.jpg with your actual image path
curl -X POST http://localhost:8000/analyze \
  -F "image=@/path/to/fish.jpg" \
  -F "query=Explain this fish and is it dangerous?"
```

Expected JSON response:
```json
{
  "detected_labels": ["Labridae"],
  "answer": "The Labridae, commonly known as wrasses, are a diverse family of marine fish found primarily in tropical and subtropical oceans. They inhabit coral reef environments and are generally not dangerous to humans. Most species feed on small invertebrates and are important for reef ecosystem balance."
}
```

### More cURL examples:

```bash
# Question about habitat
curl -X POST http://localhost:8000/analyze \
  -F "image=@fish.jpg" \
  -F "query=Where is this species found?"

# Question about danger
curl -X POST http://localhost:8000/analyze \
  -F "image=@fish.jpg" \
  -F "query=Is this fish dangerous to humans?"

# General description
curl -X POST http://localhost:8000/analyze \
  -F "image=@fish.jpg" \
  -F "query=Give me a detailed description of this marine species."
```

---

## 🔄 Full Data Flow (What Happens Inside)

```
User uploads image + types question
        │
        ▼
[FastAPI /analyze endpoint]
        │
        ▼
[STEP 1] yolo_detector.py
  → Loads best.pt (once, cached globally)
  → Runs YOLOv8 inference on uploaded image
  → Returns: ["Labridae", "Pomacanthidae"]
        │
        ▼
[STEP 2] Extract top detected class label
  → top_species = "Labridae"
        │
        ▼
[STEP 3] rag_pipeline.py — FAISS retrieval
  → Embeds "Labridae marine species" with sentence-transformers
  → Searches fish.index (FAISS) for top-3 nearest neighbors
  → Loads matching records from records.pkl
        │
        ▼
[STEP 4] Build grounded prompt
  → Combines: detected species + retrieved context + user question
        │
        ▼
[STEP 5] Call Ollama phi3
  → POST http://localhost:11434/api/generate
  → Phi-3 reads the context and generates a natural language answer
        │
        ▼
[STEP 6] Return JSON
  → { "detected_labels": [...], "answer": "..." }
        │
        ▼
Streamlit displays result to user
```

---

## 🔧 Environment Variables (Optional Overrides)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `OLLAMA_MODEL` | `phi3` | Model name (try `phi3:mini` for less RAM) |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model name |

Example:
```bash
OLLAMA_MODEL=phi3:mini uvicorn backend.app:app --reload
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: best.pt` | Copy your model to `backend/models/best.pt` |
| `FileNotFoundError: fish.index` | Copy your `claude_rag/` folder to `rag_data/claude_rag/` |
| `Cannot connect to Ollama` | Run `ollama serve` in a separate terminal |
| `Model phi3 not found` | Run `ollama pull phi3` first |
| Streamlit shows backend error | Make sure `uvicorn` is running on port 8000 |
| Slow inference | Normal on CPU — phi3 takes ~10–30s per query |
| Out of memory | Use `phi3:mini` instead of `phi3` |

---

## 📦 Dependencies Summary

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | REST API server |
| `streamlit` | Web UI |
| `ultralytics` | YOLOv8 inference |
| `sentence-transformers` | Text embeddings for RAG |
| `faiss-cpu` | Vector similarity search |
| `requests` | HTTP calls to Ollama |
| `Pillow` | Image handling in Streamlit |

---

*All tools are free, open-source, and run 100% locally.*
