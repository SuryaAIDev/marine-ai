"""
Marine Species AI - FastAPI Backend
POST /analyze → YOLO detection → RAG retrieval → Phi-3 answer
"""

import os
import uuid
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .yolo_detector import detect_species
from .rag_pipeline import generate_species_answer

app = FastAPI(
    title="Marine Species AI API",
    description="Upload a fish image + ask a question → get AI-powered species info",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(tempfile.gettempdir()) / "marine_ai_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {"status": "Marine Species AI is running 🐟"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(..., description="Fish image (jpg/png)"),
    query: str = Form(..., description="Your question about the species"),
):
    # ── 1. Save uploaded image to temp file ──────────────────────────────────
    ext = Path(image.filename).suffix or ".jpg"
    tmp_path = TEMP_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        contents = await image.read()
        tmp_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    try:
        # ── 2. YOLO detection ────────────────────────────────────────────────
        detected_labels = detect_species(str(tmp_path))

        if not detected_labels:
            return JSONResponse(
                status_code=200,
                content={
                    "detected_labels": [],
                    "answer": (
                        "No marine species were detected in the image. "
                        "Please upload a clearer photo of a fish or marine animal."
                    ),
                },
            )

        # ── 3 → 6. RAG + Phi-3 answer ────────────────────────────────────────
        top_species = detected_labels[0]
        answer = generate_species_answer(
            species=top_species,
            user_query=query,
        )

        return JSONResponse(
            content={
                "detected_labels": detected_labels,
                "answer": answer,
            }
        )

    finally:
        # Always clean up temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
