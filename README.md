# 🔍 TruthLens — AI Fact Verification (Phase 8 · Deployable)

Text · Audio · Image → verdict with Tree-of-Thought + Reflection + Legal RAG +
Forensics + SQLite/Metrics/RAGAS/LangSmith — **containerized for Azure**.

## 🚀 Run locally
```bash
cd truthlens
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r req_phase7.txt
cp .env.example .env       # add 6 keys (GEMINI_API_KEY_1..6)
python run.py              # http://localhost:8000  ·  /metrics dashboard
```

## ☁️ Phase 8 — Deploy to Azure Container Apps
Full guide: **deploy/AZURE_DEPLOY.md**. Quick version:
```bash
az login
bash deploy/azure_deploy.sh      # builds image on ACR, deploys, sets secrets
```
Files:
- **Dockerfile** — python:3.11-slim + tesseract-ocr + ffmpeg + TensorFlow (full features)
- **.dockerignore** — skips .venv, db, chroma, dataset
- **requirements.txt** — pinned deps for the image (tensorflow-cpu)
- **deploy/azure_deploy.sh** — one-shot ACR build + Container App create + secrets
- **deploy/AZURE_DEPLOY.md** — manual steps, persistence (Azure Files), cost notes

Key points:
- Container reads **$PORT** (uvicorn 0.0.0.0). Scale-to-zero (min 0 replicas = no idle cost).
- Gemini/LangSmith keys → **Container App secrets** (never in the image).
- Persistence: mount **Azure Files** at /app/data (SQLite + Chroma + model), or use Azure PostgreSQL.

## Features (Phases 1-7)
6-key rotation · Intent Router · Claim · Pre-Grounder · Web Search · Context ·
🌳 Tree of Thought · Reflection (LLM-as-judge) · Legal RAG (BNS+IPC+Constitution) ·
Audio (Gemini+Whisper) · Image (OCR+Vision) · Forensics (ELA+EXIF+CNN) ·
SQLite (5 tables) · Metrics (/metrics) · RAGAS (evaluation/evaluate.py) · LangSmith tracing


## 🎙️ Audio input — two ways (business-standard)
The mic button opens a dropdown with two options:
- **Record with mic** — speak your claim; a live recording bar shows time + Stop/Cancel (uses the browser MediaRecorder API, sent as webm).
- **Upload audio file** — pick an existing mp3/wav/m4a/ogg voice note.
Both paths feed the same audio → transcription → verification pipeline.

## Roadmap: 1-8 ✅ COMPLETE
Config: backend/config.py
# True_len_Capstone
