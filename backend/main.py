import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from backend.routers import chat
from backend import config

app = FastAPI(title="TruthLens API", version="0.7.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(chat.router)


def _ragas_summary():
    """Read the latest RAGAS evaluation summary (written by evaluation/evaluate.py)."""
    import json
    path = getattr(config, "RAGAS_SUMMARY_PATH", "evaluation/results/summary_latest.json")
    if not os.path.exists(path):
        return {"available": False,
                "reason": "No evaluation run yet. Run: python evaluation/evaluate.py"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True
        return data
    except Exception as e:
        return {"available": False, "reason": f"Could not read summary: {str(e)[:100]}"}


@app.get("/health")
def health():
    return {"status": "ok", "phase": 7, "llm_configured": config.llm_ready(),
            "keys": len(config.GEMINI_API_KEYS),
            "reflection": config.REFLECTION_ENABLED,
            "forgery_cnn": os.path.exists(config.FORGERY_MODEL_PATH),
            "db": config.DB_ENABLED, "langsmith": config.LANGSMITH_ENABLED}


@app.get("/api/v1/metrics")
def metrics():
    """Single dashboard endpoint: SQLite metrics + RAGAS summary in one payload."""
    from backend import db
    data = db.get_metrics()
    data["ragas"] = _ragas_summary()
    return data


@app.get("/metrics")
def metrics_page():
    p = os.path.join(config.FRONTEND_DIR, "metrics.html")
    return FileResponse(p) if os.path.exists(p) else {"error": "metrics.html not found"}


@app.get("/")
def index():
    return FileResponse(os.path.join(config.FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
