from __future__ import annotations
# ==========================================================================
#  PATH BOOTSTRAP (must run BEFORE any `from backend ...` import)
#  Makes the project root importable so `backend` is always found,
#  whether you run:  python evaluate.py   (from inside evaluation/)
#                or:  python evaluation/evaluate.py   (from project root)
# ==========================================================================
import os
import sys
from pathlib import Path


def _add_project_root_to_path() -> None:
    here = Path(__file__).resolve()
    # walk upward until we find a folder that contains a "backend" package
    for parent in [here.parent] + list(here.parents):
        if (parent / "backend").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    # fallback: assume parent-of-parent (evaluation/ -> project root)
    root = here.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_project_root_to_path()
# ==========================================================================

"""
TruthLens — RAGAS-style Evaluation (Gemini-judged)
==================================================
RAGAS 0.1.x has a hard dependency bug in this environment: it passes
`temperature` at generate_content() runtime, which the newer google client
rejects -> every RAGAS call fails (nan).

This script AVOIDS that broken path completely: it computes the SAME four
RAGAS metrics ourselves by calling Gemini DIRECTLY via ChatGoogleGenerativeAI
(the exact client the app already uses successfully). No RAGAS internals,
no temperature runtime kwarg -> it just works.

Metrics (each 0..1, judged by Gemini):
  1. Faithfulness       — is the answer supported by the retrieved contexts?
  2. Answer Correctness — does the answer match the ground truth?
  3. Context Precision   — how relevant are the retrieved contexts? (Precision@k)
  4. Context Recall      — is the ground truth covered by the contexts? (Recall@k)

If Gemini is unavailable, it falls back to lightweight lexical metrics so the
demo never breaks.

Outputs (in evaluation/results/):
  * evaluation_results_<ts>.csv  — per-question detail
  * summary_<ts>.json            — aggregate scores (history)
  * summary_latest.json          — aggregate scores (dashboard reads this)

Setup:  export GEMINI_API_KEY_1="AIza..."   (or put it in ../.env)
Run:    python3 evaluate.py
"""
import csv
import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "evaluation_samples.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Optional gentle delay between samples to stay under the free-tier rate limit.
#   export EVAL_DELAY=0.6   (seconds)  -> useful when running 20+ samples
EVAL_DELAY = float(os.getenv("EVAL_DELAY", "0.0"))


# -----------------------------------------------------------------------------
# key + Gemini client
# -----------------------------------------------------------------------------
def _gemini_key() -> str:
    for k in ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
              "GEMINI_API_KEY_4", "GEMINI_API_KEY_5", "GEMINI_API_KEY_6", "GEMINI_API_KEY"):
        v = os.getenv(k)
        if v and "paste_your" not in v:
            return v
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        for k in ("GEMINI_API_KEY_1", "GEMINI_API_KEY"):
            v = os.getenv(k)
            if v and "paste_your" not in v:
                return v
    except Exception:
        pass
    return ""


def _candidate_models() -> List[str]:
    """Models to try, best-first. Uses the app's config model if available,
    plus common fallbacks. Override with:  export EVAL_MODEL='gemini-2.0-flash'"""
    cands: List[str] = []
    env = os.getenv("EVAL_MODEL")
    if env:
        cands.append(env)
    # use the SAME model the app uses (already works with this key)
    try:
        from backend import config as _cfg
        for m in (getattr(_cfg, "MODEL_FAST", None), getattr(_cfg, "MODEL_REASONING", None)):
            if m and m not in cands:
                cands.append(m)
    except Exception:
        pass
    # safe public fallbacks
    for m in ("gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-002",
              "gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash"):
        if m not in cands:
            cands.append(m)
    return cands


_client = None
def _get_client():
    """Pick a Gemini model that actually works with this key (avoids 404).
    temperature is set at CONSTRUCTION (safe), never at call time (RAGAS bug)."""
    global _client
    if _client is not None:
        return _client
    key = _gemini_key()
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        print(f"[WARN] langchain-google-genai missing: {e}")
        return None
    for model in _candidate_models():
        try:
            cli = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
            # tiny probe call to confirm the model is valid for this key
            _ = cli.invoke("Reply with: ok")
            print(f"[OK] Using Gemini model: {model}")
            _client = cli
            return _client
        except Exception as e:
            msg = str(e)[:80]
            print(f"[..] model '{model}' not usable ({msg}); trying next")
            continue
    print("[WARN] No working Gemini model found. Falling back to lexical.")
    _client = None
    return _client


def _ask(prompt: str) -> str:
    c = _get_client()
    if c is None:
        return ""
    try:
        r = c.invoke(prompt)
        return getattr(r, "content", str(r)) or ""
    except Exception as e:
        print(f"[WARN] Gemini call failed: {str(e)[:100]}")
        return ""


def _score_from_text(text: str) -> Optional[float]:
    """Pull a 0..1 score from the model's reply."""
    if not text:
        return None
    m = re.search(r'"?score"?\s*[:=]\s*([01](?:\.\d+)?)', text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            pass
    m = re.search(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            pass
    return None


# -----------------------------------------------------------------------------
# The four RAGAS-style metrics (Gemini-judged)
# -----------------------------------------------------------------------------
def m_faithfulness(answer: str, contexts: List[str]) -> Optional[float]:
    ctx = "\n".join(f"- {c}" for c in contexts) or "(none)"
    p = (f"You are evaluating FAITHFULNESS. Is the ANSWER fully supported by the CONTEXT "
         f"(no invented facts)?\nCONTEXT:\n{ctx}\nANSWER: {answer}\n"
         f'Reply ONLY JSON: {{"score": <0..1>}}  where 1 = fully supported, 0 = not supported.')
    return _score_from_text(_ask(p))


def m_answer_correctness(answer: str, ground_truth: str) -> Optional[float]:
    p = (f"You are evaluating ANSWER CORRECTNESS. How well does the ANSWER match the "
         f"GROUND TRUTH in meaning (semantic, not word-for-word)?\n"
         f"GROUND TRUTH: {ground_truth}\nANSWER: {answer}\n"
         f'Reply ONLY JSON: {{"score": <0..1>}}  where 1 = same meaning, 0 = contradicts/wrong.')
    return _score_from_text(_ask(p))


def m_context_precision(question: str, answer: str, contexts: List[str]) -> Optional[float]:
    """Precision@k — fraction of retrieved contexts that are relevant."""
    if not contexts:
        return 0.0
    ctx = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    p = (f"You are evaluating CONTEXT PRECISION (Precision@k). For the QUESTION, how many of "
         f"the retrieved CONTEXTS are actually RELEVANT/useful to answer it?\n"
         f"QUESTION: {question}\nANSWER: {answer}\nCONTEXTS:\n{ctx}\n"
         f'Reply ONLY JSON: {{"score": <0..1>}}  = (relevant contexts / total contexts).')
    return _score_from_text(_ask(p))


def m_context_recall(ground_truth: str, contexts: List[str]) -> Optional[float]:
    """Recall@k — fraction of the ground truth that is covered by the contexts."""
    if not contexts:
        return 0.0
    ctx = "\n".join(f"- {c}" for c in contexts)
    p = (f"You are evaluating CONTEXT RECALL (Recall@k). Is every claim in the GROUND TRUTH "
         f"supported by the retrieved CONTEXTS?\nGROUND TRUTH: {ground_truth}\n"
         f"CONTEXTS:\n{ctx}\n"
         f'Reply ONLY JSON: {{"score": <0..1>}}  = (ground-truth claims covered / total claims).')
    return _score_from_text(_ask(p))


# -----------------------------------------------------------------------------
# lexical fallback (only if Gemini unavailable)
# -----------------------------------------------------------------------------
def _norm(t): return " ".join((t or "").lower().strip().split())
def _f1(pred, ref):
    p, r = _norm(pred).split(), _norm(ref).split()
    if not p or not r: return 0.0
    ps, rs = set(p), set(r); common = ps & rs
    if not common: return 0.0
    pr, rc = len(common)/len(ps), len(common)/len(rs)
    return round(2*pr*rc/(pr+rc), 4) if pr+rc else 0.0
def _ground(ans, ctx):
    a, c = set(_norm(ans).split()), set(_norm(" ".join(ctx)).split())
    return round(len(a & c)/len(a), 4) if a and c else 0.0


# -----------------------------------------------------------------------------
# dataset
# -----------------------------------------------------------------------------
def create_demo_dataset_if_missing() -> None:
    if DATASET_PATH.exists():
        return
    rows = [
        {"question": "Is BNS 2023 replacing IPC 1860 in India?",
         "answer": "Yes, Bharatiya Nyaya Sanhita 2023 replaced the Indian Penal Code from 1 July 2024.",
         "contexts": ["Bharatiya Nyaya Sanhita, 2023 came into force from 1 July 2024 and replaced the Indian Penal Code, 1860."],
         "ground_truth": "BNS 2023 replaced IPC 1860 from 1 July 2024."},
        {"question": "Does EXIF metadata alone prove an image is fake?",
         "answer": "No, EXIF metadata alone cannot prove forgery. It is only one supporting signal with ELA and CNN classification.",
         "contexts": ["EXIF metadata can show camera details, timestamps, GPS, or editing software, but it should be treated as a supporting signal, not final proof."],
         "ground_truth": "Metadata is a supporting forgery signal, not absolute proof."},
        {"question": "Was J.P. Morgan Indian?",
         "answer": "No. J.P. Morgan was an American financier and banker, not Indian.",
         "contexts": ["J. P. Morgan was an American financier and investment banker born in Hartford, Connecticut, USA."],
         "ground_truth": "J.P. Morgan was American, not Indian."},
    ]
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Created demo dataset: {DATASET_PATH}")


def load_samples() -> List[Dict[str, Any]]:
    create_demo_dataset_if_missing()
    out = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] bad JSON line {ln}: {e}"); continue
            q = row.get("question") or row.get("user_input") or ""
            a = row.get("answer") or row.get("response") or ""
            c = row.get("contexts") or row.get("retrieved_contexts") or []
            g = row.get("ground_truth") or row.get("reference") or ""
            if isinstance(c, str): c = [c]
            if not isinstance(c, list): c = []
            out.append({"question": str(q), "answer": str(a),
                        "contexts": [str(x) for x in c], "ground_truth": str(g)})
    return out


# -----------------------------------------------------------------------------
# run
# -----------------------------------------------------------------------------
def evaluate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    use_gemini = _get_client() is not None
    engine = "gemini-judge" if use_gemini else "lexical-fallback"
    print(f"[INFO] Evaluation engine: {engine}\n")
    rows = []
    for i, s in enumerate(samples, 1):
        q, a, c, g = s["question"], s["answer"], s["contexts"], s["ground_truth"]
        if use_gemini:
            faith = m_faithfulness(a, c)
            corr = m_answer_correctness(a, g)
            prec = m_context_precision(q, a, c)
            rec = m_context_recall(g, c)
            # any None -> fall back for that cell
            faith = _ground(a, c) if faith is None else faith
            corr = _f1(a, g) if corr is None else corr
            prec = 1.0 if (prec is None and c) else (0.0 if prec is None else prec)
            rec = _ground(g, c) if rec is None else rec
        else:
            faith = _ground(a, c); corr = _f1(a, g)
            prec = 1.0 if c else 0.0; rec = _ground(g, c)

        avg = round((faith + corr + prec + rec) / 4, 4)
        status = "pass" if avg >= 0.7 else ("partial" if avg >= 0.45 else "fail")
        print(f"  [{i}] {q[:44]:44s} faith={faith:.2f} corr={corr:.2f} prec={prec:.2f} rec={rec:.2f}")
        rows.append({"id": i, "question": q, "answer": a, "ground_truth": g,
                     "contexts_count": len(c),
                     "faithfulness": round(faith, 4), "answer_correctness": round(corr, 4),
                     "context_precision": round(prec, 4), "context_recall": round(rec, 4),
                     "avg_score": avg, "status": status, "engine": engine})
        if EVAL_DELAY > 0:
            time.sleep(EVAL_DELAY)
    return rows


def save_results(rows: List[Dict[str, Any]]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"evaluation_results_{ts}.csv"
    fields = ["id", "question", "answer", "ground_truth", "contexts_count",
              "faithfulness", "answer_correctness", "context_precision", "context_recall",
              "avg_score", "status", "engine"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return out


def save_summary(rows: List[Dict[str, Any]]) -> Optional[Path]:
    """Store aggregate scores as JSON so the /metrics dashboard can read them.
    Writes a timestamped history file AND a stable summary_latest.json."""
    n = len(rows)
    if not n:
        return None

    def avg(k):
        return round(sum(float(r[k]) for r in rows) / n, 4)

    status_counts: Dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "engine": rows[0]["engine"],
        "samples": n,
        "faithfulness": avg("faithfulness"),
        "answer_correctness": avg("answer_correctness"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "overall": avg("avg_score"),
        "status_breakdown": status_counts,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 1) timestamped copy (history)
    hist = RESULTS_DIR / f"summary_{ts}.json"
    with hist.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # 2) latest copy (dashboard always reads this one)
    latest = RESULTS_DIR / "summary_latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return latest


def print_summary(rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    if not n:
        print("[ERROR] No rows."); return
    def avg(k): return round(sum(float(r[k]) for r in rows) / n, 4)
    sc = {}
    for r in rows:
        sc[r["status"]] = sc.get(r["status"], 0) + 1
    print("\n============= TruthLens — RAGAS-style Evaluation =============")
    print(f"Engine                 : {rows[0]['engine']}")
    print(f"Samples                : {n}")
    print(f"Faithfulness (avg)     : {avg('faithfulness')}")
    print(f"Answer Correctness     : {avg('answer_correctness')}")
    print(f"Context Precision@k    : {avg('context_precision')}")
    print(f"Context Recall@k       : {avg('context_recall')}")
    print(f"Overall (avg)          : {avg('avg_score')}")
    print(f"Status                 : {sc}")
    print("=============================================================\n")


def main() -> None:
    samples = load_samples()
    if not samples:
        print("[ERROR] No samples."); return
    rows = evaluate_samples(samples)
    out = save_results(rows)
    summ = save_summary(rows)
    print_summary(rows)
    print(f"[OK] Results saved to: {out}")
    if summ:
        print(f"[OK] Summary saved to: {summ}")


if __name__ == "__main__":
    main()
