from __future__ import annotations
import os, sys, csv, json, re, time
from pathlib import Path
from datetime import datetime

# --- make project root importable (so `backend` is found from anywhere) ---
_here = Path(__file__).resolve()
for _p in [_here.parent] + list(_here.parents):
    if (_p / "backend").is_dir():
        sys.path.insert(0, str(_p)); break
else:
    sys.path.insert(0, str(_here.parent.parent))

EVAL_DIR = _here.parent
DATASET_PATH = EVAL_DIR / "evaluation_samples.jsonl"
RESULTS_DIR = EVAL_DIR / "results"; RESULTS_DIR.mkdir(exist_ok=True)
EVAL_DELAY = float(os.getenv("EVAL_DELAY", "0.0"))

# ---------------- Gemini client (no fallback) ----------------
def _key():
    for k in ("GEMINI_API_KEY_1","GEMINI_API_KEY_2","GEMINI_API_KEY_3","GEMINI_API_KEY_4",
              "GEMINI_API_KEY_5","GEMINI_API_KEY_6","GEMINI_API_KEY"):
        v = os.getenv(k)
        if v and "paste_your" not in v: return v
    try:
        from dotenv import load_dotenv; load_dotenv(_here.parent.parent / ".env")
        for k in ("GEMINI_API_KEY_1","GEMINI_API_KEY"):
            v = os.getenv(k)
            if v and "paste_your" not in v: return v
    except Exception: pass
    return ""

def _models():
    c = []
    if os.getenv("EVAL_MODEL"): c.append(os.getenv("EVAL_MODEL"))
    try:
        from backend import config as cfg
        for m in (getattr(cfg,"MODEL_FAST",None), getattr(cfg,"MODEL_REASONING",None)):
            if m and m not in c: c.append(m)
    except Exception: pass
    for m in ("gemini-2.0-flash","gemini-1.5-flash-latest","gemini-1.5-flash-002",
              "gemini-2.5-flash","gemini-flash-latest","gemini-1.5-flash"):
        if m not in c: c.append(m)
    return c

_client = None
def _get_client():
    """Return a working Gemini client, or raise (no fallback)."""
    global _client
    if _client is not None: return _client
    key = _key()
    if not key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY_1..6 in .env")
    from langchain_google_genai import ChatGoogleGenerativeAI
    for model in _models():
        try:
            cli = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
            cli.invoke("Reply with: ok")
            print(f"[OK] Using Gemini model: {model}"); _client = cli; return _client
        except Exception as e:
            print(f"[..] model '{model}' not usable ({str(e)[:60]}); next")
    raise RuntimeError("No working Gemini model found for this API key.")

def _ask(prompt):
    """Ask Gemini; raise on failure (no silent fallback)."""
    return getattr(_get_client().invoke(prompt), "content", "") or ""

def _score(text):
    for pat in (r'"?score"?\s*[:=]\s*([01](?:\.\d+)?)', r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b'):
        m = re.search(pat, text or "")
        if m:
            try: return max(0.0, min(1.0, float(m.group(1))))
            except Exception: pass
    raise ValueError(f"Could not parse a 0..1 score from Gemini reply: {text[:80]!r}")

# ---------------- RAGAS metrics (Gemini-judged) ----------------
def m_faithfulness(a, cs):
    ctx = "\n".join(f"- {c}" for c in cs) or "(none)"
    return _score(_ask(f"Evaluate FAITHFULNESS. Is the ANSWER fully supported by the CONTEXT "
        f"(no invented facts)?\nCONTEXT:\n{ctx}\nANSWER: {a}\n"
        f'Reply ONLY JSON: {{"score": <0..1>}} (1=fully supported, 0=not).'))

def m_answer_correctness(a, g):
    return _score(_ask(f"Evaluate ANSWER CORRECTNESS. How well does the ANSWER match the "
        f"GROUND TRUTH in meaning?\nGROUND TRUTH: {g}\nANSWER: {a}\n"
        f'Reply ONLY JSON: {{"score": <0..1>}} (1=same meaning, 0=wrong).'))

def m_context_precision(q, a, cs):
    if not cs: return 0.0
    ctx = "\n".join(f"[{i}] {c}" for i, c in enumerate(cs))
    return _score(_ask(f"Evaluate CONTEXT PRECISION (Precision@k). How many retrieved CONTEXTS "
        f"are relevant to the QUESTION?\nQUESTION: {q}\nANSWER: {a}\nCONTEXTS:\n{ctx}\n"
        f'Reply ONLY JSON: {{"score": <0..1>}} = relevant/total.'))

def m_context_recall(g, cs):
    if not cs: return 0.0
    ctx = "\n".join(f"- {c}" for c in cs)
    return _score(_ask(f"Evaluate CONTEXT RECALL (Recall@k). Is every claim in the GROUND TRUTH "
        f"supported by the CONTEXTS?\nGROUND TRUTH: {g}\nCONTEXTS:\n{ctx}\n"
        f'Reply ONLY JSON: {{"score": <0..1>}} = covered/total.'))

# ---------------- dataset ----------------
_DEMO = [
    {"question":"Is BNS 2023 replacing IPC 1860 in India?","answer":"Yes, Bharatiya Nyaya Sanhita 2023 replaced the Indian Penal Code from 1 July 2024.","contexts":["Bharatiya Nyaya Sanhita, 2023 came into force from 1 July 2024 and replaced the Indian Penal Code, 1860."],"ground_truth":"BNS 2023 replaced IPC 1860 from 1 July 2024."},
    {"question":"Does EXIF metadata alone prove an image is fake?","answer":"No, EXIF metadata alone cannot prove forgery. It is only one supporting signal with ELA and CNN classification.","contexts":["EXIF metadata can show camera details, timestamps, GPS, or editing software, but it should be treated as a supporting signal, not final proof."],"ground_truth":"Metadata is a supporting forgery signal, not absolute proof."},
    {"question":"Was J.P. Morgan Indian?","answer":"No. J.P. Morgan was an American financier and banker, not Indian.","contexts":["J. P. Morgan was an American financier and investment banker born in Hartford, Connecticut, USA."],"ground_truth":"J.P. Morgan was American, not Indian."},
]
def load_samples():
    if not DATASET_PATH.exists():
        with DATASET_PATH.open("w", encoding="utf-8") as f:
            for r in _DEMO: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[INFO] Created demo dataset: {DATASET_PATH}")
    out = []
    for ln, line in enumerate(DATASET_PATH.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line: continue
        try: row = json.loads(line)
        except json.JSONDecodeError as e: print(f"[WARN] bad JSON line {ln}: {e}"); continue
        c = row.get("contexts") or row.get("retrieved_contexts") or []
        if isinstance(c, str): c = [c]
        if not isinstance(c, list): c = []
        out.append({"question": str(row.get("question") or row.get("user_input") or ""),
                    "answer": str(row.get("answer") or row.get("response") or ""),
                    "contexts": [str(x) for x in c],
                    "ground_truth": str(row.get("ground_truth") or row.get("reference") or "")})
    return out

# ---------------- run ----------------
def evaluate_samples(samples):
    _get_client()  # fail fast if Gemini not available
    print("[INFO] Evaluation engine: gemini-judge\n")
    rows = []
    for i, s in enumerate(samples, 1):
        q, a, c, g = s["question"], s["answer"], s["contexts"], s["ground_truth"]
        faith = m_faithfulness(a, c); corr = m_answer_correctness(a, g)
        prec = m_context_precision(q, a, c); rec = m_context_recall(g, c)
        avg = round((faith + corr + prec + rec) / 4, 4)
        status = "pass" if avg >= 0.7 else ("partial" if avg >= 0.45 else "fail")
        print(f"  [{i}] {q[:44]:44s} faith={faith:.2f} corr={corr:.2f} prec={prec:.2f} rec={rec:.2f}")
        rows.append({"id": i, "question": q, "answer": a, "ground_truth": g, "contexts_count": len(c),
                     "faithfulness": round(faith,4), "answer_correctness": round(corr,4),
                     "context_precision": round(prec,4), "context_recall": round(rec,4),
                     "avg_score": avg, "status": status, "engine": "gemini-judge"})
        if EVAL_DELAY > 0: time.sleep(EVAL_DELAY)
    return rows

def _avg(rows, k): return round(sum(float(r[k]) for r in rows)/len(rows), 4)

def save_results(rows):
    out = RESULTS_DIR / f"evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    fields = ["id","question","answer","ground_truth","contexts_count","faithfulness",
              "answer_correctness","context_precision","context_recall","avg_score","status","engine"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return out

def save_summary(rows):
    if not rows: return None
    sc = {}
    for r in rows: sc[r["status"]] = sc.get(r["status"], 0) + 1
    summary = {"timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "engine": "gemini-judge",
               "samples": len(rows), "faithfulness": _avg(rows,"faithfulness"),
               "answer_correctness": _avg(rows,"answer_correctness"),
               "context_precision": _avg(rows,"context_precision"),
               "context_recall": _avg(rows,"context_recall"),
               "overall": _avg(rows,"avg_score"), "status_breakdown": sc}
    for path in (RESULTS_DIR / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json",
                 RESULTS_DIR / "summary_latest.json"):
        with path.open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    return RESULTS_DIR / "summary_latest.json"

def print_summary(rows):
    if not rows: print("[ERROR] No rows."); return
    sc = {}
    for r in rows: sc[r["status"]] = sc.get(r["status"], 0) + 1
    print("\n===== TruthLens — RAGAS-style Evaluation =====")
    print(f"Engine              : gemini-judge")
    print(f"Samples             : {len(rows)}")
    print(f"Faithfulness        : {_avg(rows,'faithfulness')}")
    print(f"Answer Correctness  : {_avg(rows,'answer_correctness')}")
    print(f"Context Precision@k : {_avg(rows,'context_precision')}")
    print(f"Context Recall@k    : {_avg(rows,'context_recall')}")
    print(f"Overall             : {_avg(rows,'avg_score')}")
    print(f"Status              : {sc}")
    print("==============================================\n")

def main():
    samples = load_samples()
    if not samples: print("[ERROR] No samples."); return
    rows = evaluate_samples(samples)   # raises if Gemini not available
    out = save_results(rows); summ = save_summary(rows); print_summary(rows)
    print(f"[OK] Results saved to: {out}")
    if summ: print(f"[OK] Summary saved to: {summ}")

if __name__ == "__main__":
    main()
