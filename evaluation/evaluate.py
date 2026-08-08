from __future__ import annotations
"""
TruthLens Phase 7 Evaluation Script
-----------------------------------
Runs even when RAGAS 0.4.x has the ChatVertexAI import bug.
1) Patches ragas/llms/base.py if ChatVertexAI is missing.
2) Tries RAGAS evaluation.
3) Falls back to a lightweight local metric so the demo never breaks.

Run:  python3 evaluate.py
"""
import csv
import json
import os
import site
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# make project root importable (so `backend` works if ever needed)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "evaluation_samples.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# 1. RAGAS ChatVertexAI bug patch
# -----------------------------------------------------------------------------
def find_ragas_base_file() -> Optional[Path]:
    roots: List[Path] = []
    try:
        for p in site.getsitepackages():
            roots.append(Path(p))
    except Exception:
        pass
    us = site.getusersitepackages()
    if us:
        roots.append(Path(us))
    for p in sys.path:
        if p:
            roots.append(Path(p))
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        f = root / "ragas" / "llms" / "base.py"
        if f.exists():
            return f
    return None


def patch_ragas_chatvertexai_bug() -> bool:
    base_file = find_ragas_base_file()
    if base_file is None:
        print("[WARN] RAGAS base.py not found. RAGAS may not be installed.")
        return False
    try:
        text = base_file.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Could not read RAGAS base.py: {exc}")
        return False
    if "ChatVertexAI" not in text:
        print("[OK] RAGAS base.py does not reference ChatVertexAI. No patch needed.")
        return True
    if "class ChatVertexAI:" in text or "from langchain_google_vertexai import ChatVertexAI" in text:
        print("[OK] ChatVertexAI patch/import already present.")
        return True
    marker = "from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    patch = (
        "try:\n"
        "    from langchain_google_vertexai import ChatVertexAI\n"
        "except Exception:\n"
        "    class ChatVertexAI:\n"
        "        pass\n"
    )
    if marker not in text:
        # try a looser match
        import re
        text2 = re.sub(r"from langchain_community\.chat_models\.vertexai import ChatVertexAI",
                       "try:\n    from langchain_google_vertexai import ChatVertexAI\nexcept Exception:\n    class ChatVertexAI:\n        pass",
                       text, count=1)
        if text2 == text:
            print(f"[WARN] Could not find ChatVertexAI import marker in: {base_file}")
            return False
        try:
            base_file.write_text(text2, encoding="utf-8")
            print(f"[OK] Patched RAGAS ChatVertexAI bug in: {base_file}")
            return True
        except Exception as exc:
            print(f"[WARN] Could not patch RAGAS base.py: {exc}")
            return False
    try:
        base_file.write_text(text.replace(marker, patch, 1), encoding="utf-8")
        print(f"[OK] Patched RAGAS ChatVertexAI bug in: {base_file}")
        return True
    except PermissionError:
        print(f"[WARN] Permission denied while patching: {base_file}")
        return False
    except Exception as exc:
        print(f"[WARN] Could not patch RAGAS base.py: {exc}")
        return False


# -----------------------------------------------------------------------------
# 2. Dataset helpers
# -----------------------------------------------------------------------------
def create_demo_dataset_if_missing() -> None:
    if DATASET_PATH.exists():
        return
    demo_rows = [
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
        for row in demo_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[INFO] Created demo dataset: {DATASET_PATH}")


def load_samples() -> List[Dict[str, Any]]:
    create_demo_dataset_if_missing()
    samples: List[Dict[str, Any]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping invalid JSON at line {line_no}: {exc}")
                continue
            question = row.get("question") or row.get("user_input") or ""
            answer = row.get("answer") or row.get("response") or ""
            contexts = row.get("contexts") or row.get("retrieved_contexts") or []
            ground_truth = row.get("ground_truth") or row.get("reference") or ""
            if isinstance(contexts, str):
                contexts = [contexts]
            if not isinstance(contexts, list):
                contexts = []
            samples.append({"question": str(question), "answer": str(answer),
                            "contexts": [str(c) for c in contexts], "ground_truth": str(ground_truth)})
    return samples


# -----------------------------------------------------------------------------
# 3. Lightweight fallback metrics
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def token_f1(prediction: str, reference: str) -> float:
    p = normalize_text(prediction).split()
    r = normalize_text(reference).split()
    if not p or not r:
        return 0.0
    ps, rs = set(p), set(r)
    common = ps & rs
    if not common:
        return 0.0
    precision = len(common) / len(ps)
    recall = len(common) / len(rs)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def context_grounding(answer: str, contexts: List[str]) -> float:
    a = set(normalize_text(answer).split())
    c = set(normalize_text(" ".join(contexts)).split())
    if not a or not c:
        return 0.0
    return round(len(a & c) / len(a), 4)


def run_lightweight_eval(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for idx, s in enumerate(samples, start=1):
        correctness = token_f1(s["answer"], s["ground_truth"])
        grounding = context_grounding(s["answer"], s["contexts"])
        if correctness >= 0.65 and grounding >= 0.45:
            status = "pass"
        elif correctness >= 0.35 or grounding >= 0.30:
            status = "partial"
        else:
            status = "fail"
        rows.append({"id": idx, "question": s["question"], "answer": s["answer"],
                     "ground_truth": s["ground_truth"], "contexts_count": len(s["contexts"]),
                     "answer_correctness": correctness, "context_grounding": grounding,
                     "status": status, "engine": "lightweight_fallback"})
    return rows


# -----------------------------------------------------------------------------
# 4. Optional RAGAS runner
# -----------------------------------------------------------------------------
def run_ragas_eval(samples: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    try:
        patch_ragas_chatvertexai_bug()
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, faithfulness
    except Exception as exc:
        print(f"[WARN] RAGAS import failed. Falling back. Reason: {exc}")
        return None
    try:
        data = {"question": [s["question"] for s in samples],
                "answer": [s["answer"] for s in samples],
                "contexts": [s["contexts"] for s in samples],
                "ground_truth": [s["ground_truth"] for s in samples]}
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[answer_correctness, faithfulness], raise_exceptions=False)
        df = result.to_pandas()
        rows = []
        for idx, row in df.iterrows():
            rows.append({"id": int(idx) + 1, "question": samples[idx]["question"],
                         "answer": samples[idx]["answer"], "ground_truth": samples[idx]["ground_truth"],
                         "contexts_count": len(samples[idx]["contexts"]),
                         "answer_correctness": float(row.get("answer_correctness", 0) or 0),
                         "context_grounding": float(row.get("faithfulness", 0) or 0),
                         "status": "ragas_scored", "engine": "ragas"})
        return rows
    except Exception as exc:
        print(f"[WARN] RAGAS evaluation failed. Falling back. Reason: {exc}")
        return None


# -----------------------------------------------------------------------------
# 5. Save + summary
# -----------------------------------------------------------------------------
def save_results(rows: List[Dict[str, Any]]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"evaluation_results_{ts}.csv"
    fields = ["id", "question", "answer", "ground_truth", "contexts_count",
              "answer_correctness", "context_grounding", "status", "engine"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def print_summary(rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    if total == 0:
        print("[ERROR] No rows evaluated.")
        return
    avg_c = round(sum(float(r["answer_correctness"]) for r in rows) / total, 4)
    avg_g = round(sum(float(r["context_grounding"]) for r in rows) / total, 4)
    engine = rows[0].get("engine", "unknown")
    sc: Dict[str, int] = {}
    for r in rows:
        st = str(r.get("status", "unknown"))
        sc[st] = sc.get(st, 0) + 1
    print("\n================ TruthLens Phase 7 Evaluation ================")
    print(f"Evaluation engine    : {engine}")
    print(f"Total samples        : {total}")
    print(f"Average correctness  : {avg_c}")
    print(f"Average grounding    : {avg_g}")
    print(f"Status counts        : {sc}")
    print("==============================================================\n")


def main() -> None:
    samples = load_samples()
    if not samples:
        print("[ERROR] No evaluation samples found.")
        return
    rows = run_ragas_eval(samples)
    if rows is None:
        rows = run_lightweight_eval(samples)
    out = save_results(rows)
    print_summary(rows)
    print(f"[OK] Results saved to: {out}")


if __name__ == "__main__":
    main()
