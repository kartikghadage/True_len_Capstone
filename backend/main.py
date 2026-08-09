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


@app.get("/health")
def health():
    import os
    return {"status": "ok", "phase": 7, "llm_configured": config.llm_ready(),
            "keys": len(config.GEMINI_API_KEYS),
            "reflection": config.REFLECTION_ENABLED,
            "forgery_cnn": os.path.exists(config.FORGERY_MODEL_PATH),
            "db": config.DB_ENABLED, "langsmith": config.LANGSMITH_ENABLED}


@app.get("/api/v1/metrics")
def metrics():
    from backend import db
    return db.get_metrics()


@app.get("/api/v1/ragas")
def ragas_summary():
    """Latest RAGAS evaluation scores for the dashboard (reads summary_latest.json)."""
    import os
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


@app.get("/metrics")
def metrics_page():
    import os
    p = os.path.join(config.FRONTEND_DIR, "metrics.html")
    return FileResponse(p) if os.path.exists(p) else {"error": "metrics.html not found"}


@app.get("/")
def index():
    return FileResponse(os.path.join(config.FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
