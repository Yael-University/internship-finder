"""
Offline evaluation harness for the internship-finder scoring pipeline.

Metrics computed:
  1. Fit-score precision/recall  — how well fit_score >= threshold predicts
     jobs you'd actually apply to, using hand-labeled data in eval_labels.json
  2. Retrieval recall@k          — how many expected jobs a query returns in
     the top-k results, using an ephemeral ChromaDB built from report data

Usage:
    python scripts/eval.py                    # runs both evals, k=5
    python scripts/eval.py --fit-only
    python scripts/eval.py --retrieval-only
    python scripts/eval.py --k 3
    python scripts/eval.py --labels data_folder/eval_labels.json

Prerequisites:
    pip install chromadb sentence-transformers
    Label data_folder/eval_labels.json (set would_apply: true/false)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root without installing the package
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_labels(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _labeled_jobs(labels: dict) -> list[dict]:
    return [j for j in labels["jobs"] if j["would_apply"] is not None]


# ── fit-score eval ────────────────────────────────────────────────────────────

def eval_fit_scores(labels: dict) -> dict:
    threshold = labels.get("fit_score_threshold", 7)
    jobs = _labeled_jobs(labels)

    if not jobs:
        return {"error": "No labeled jobs found. Set would_apply: true/false in eval_labels.json."}

    tp = fp = fn = tn = 0
    for job in jobs:
        predicted_match = job["fit_score"] >= threshold
        actual_match    = bool(job["would_apply"])
        if predicted_match and actual_match:
            tp += 1
        elif predicted_match and not actual_match:
            fp += 1
        elif not predicted_match and actual_match:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(jobs) if jobs else 0.0

    false_negatives = [j for j in jobs if j["fit_score"] < threshold and j["would_apply"]]
    false_positives = [j for j in jobs if j["fit_score"] >= threshold and not j["would_apply"]]

    return {
        "n_labeled":       len(jobs),
        "threshold":       threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":       round(precision, 3),
        "recall":          round(recall, 3),
        "f1":              round(f1, 3),
        "accuracy":        round(accuracy, 3),
        "false_negatives": [{"role": j["role"], "company": j["company"], "score": j["fit_score"]} for j in false_negatives],
        "false_positives": [{"role": j["role"], "company": j["company"], "score": j["fit_score"]} for j in false_positives],
    }


# ── retrieval eval ────────────────────────────────────────────────────────────

def _build_ephemeral_store(jobs: list[dict], model):
    """
    Build an in-memory ChromaDB collection from all jobs in eval_labels.json.
    Uses 'Role: X at Company Y in Location Z' as the document text since
    full descriptions aren't stored in report files.
    """
    import chromadb

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        name="eval_jobs",
        metadata={"hnsw:space": "cosine"},
    )

    doc_texts = [
        f"Role: {j['role']}\nCompany: {j['company']}\nLocation: {j['location']}"
        for j in jobs
    ]
    embeddings = model.encode(doc_texts, convert_to_tensor=False).tolist()

    col.upsert(
        ids=[j["id"] for j in jobs],
        documents=doc_texts,
        metadatas=[{
            "role":      j["role"],
            "company":   j["company"],
            "fit_score": j["fit_score"],
        } for j in jobs],
        embeddings=embeddings,
    )
    return col


def eval_retrieval(labels: dict, k: int = 5) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {"error": "sentence-transformers not installed. Run: pip install sentence-transformers"}

    try:
        import chromadb  # noqa: F401
    except ImportError:
        return {"error": "chromadb not installed. Run: pip install chromadb"}

    queries = labels.get("retrieval_queries", [])
    if not queries:
        return {"error": "No retrieval queries found in eval_labels.json."}

    all_jobs = labels["jobs"]
    if not all_jobs:
        return {"error": "No jobs in eval_labels.json."}

    print("  Loading sentence-transformers model (all-MiniLM-L6-v2)…", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  Building ephemeral ChromaDB from report data…", flush=True)
    col = _build_ephemeral_store(all_jobs, model)

    results_per_query = []
    total_recall = 0.0

    for q in queries:
        query_text  = q["query"]
        expected    = set(q["expected_ids"])
        n_expected  = len(expected)

        qvec = model.encode(query_text, convert_to_tensor=False).tolist()
        hits = col.query(
            query_embeddings=[qvec],
            n_results=min(k, col.count()),
            include=["metadatas"],
        )
        returned_ids = set(hits["ids"][0])
        hits_in_top_k = expected & returned_ids
        recall_at_k   = len(hits_in_top_k) / n_expected if n_expected > 0 else 0.0
        total_recall  += recall_at_k

        results_per_query.append({
            "query":       query_text,
            "expected":    n_expected,
            "hits_at_k":   len(hits_in_top_k),
            "recall_at_k": round(recall_at_k, 3),
            "notes":       q.get("notes", ""),
            "missed": [
                eid for eid in expected - returned_ids
            ],
        })

    mean_recall = total_recall / len(queries) if queries else 0.0

    return {
        "k":             k,
        "n_queries":     len(queries),
        "mean_recall":   round(mean_recall, 3),
        "per_query":     results_per_query,
    }


# ── formatting ────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "#" * filled + "-" * (width - filled)


def print_fit_results(r: dict) -> None:
    if "error" in r:
        print(f"\n[fit-score] ERROR: {r['error']}")
        return

    print("\n" + "=" * 58)
    print("  FIT-SCORE EVAL")
    print("=" * 58)
    print(f"  Labeled jobs    : {r['n_labeled']}  (threshold: {r['threshold']})")
    print(f"  TP / FP / FN / TN: {r['tp']} / {r['fp']} / {r['fn']} / {r['tn']}")
    print()
    print(f"  Precision  [{_bar(r['precision'])}]  {r['precision']:.3f}")
    print(f"  Recall     [{_bar(r['recall'])}]  {r['recall']:.3f}")
    print(f"  F1         [{_bar(r['f1'])}]  {r['f1']:.3f}")
    print(f"  Accuracy   [{_bar(r['accuracy'])}]  {r['accuracy']:.3f}")

    if r["false_negatives"]:
        print(f"\n  False negatives (you'd apply but score < {r['threshold']}):")
        for j in r["false_negatives"]:
            print(f"    [{j['score']}] {j['role']} @ {j['company']}")

    if r["false_positives"]:
        print(f"\n  False positives (score >= {r['threshold']} but wouldn't apply):")
        for j in r["false_positives"]:
            print(f"    [{j['score']}] {j['role']} @ {j['company']}")


def print_retrieval_results(r: dict) -> None:
    if "error" in r:
        print(f"\n[retrieval] ERROR: {r['error']}")
        return

    print("\n" + "=" * 58)
    print(f"  RETRIEVAL EVAL  (recall@{r['k']})")
    print("=" * 58)
    print(f"  Queries tested  : {r['n_queries']}")
    print(f"  Mean recall@{r['k']}  : {r['mean_recall']:.3f}  [{_bar(r['mean_recall'])}]")
    print()

    for q in r["per_query"]:
        bar = _bar(q["recall_at_k"])
        print(f"  [{bar}] {q['recall_at_k']:.2f}  {q['query'][:48]}")
        if q["missed"]:
            print(f"           missed: {', '.join(q['missed'])}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Eval harness for internship-finder")
    parser.add_argument("--labels",          default="data_folder/eval_labels.json")
    parser.add_argument("--fit-only",        action="store_true")
    parser.add_argument("--retrieval-only",  action="store_true")
    parser.add_argument("--k",               type=int, default=5, help="top-k for retrieval recall")
    parser.add_argument("--json",            action="store_true", help="output raw JSON instead of formatted report")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    labels = _load_labels(labels_path)

    run_fit        = not args.retrieval_only
    run_retrieval  = not args.fit_only

    fit_results        = eval_fit_scores(labels)           if run_fit       else None
    retrieval_results  = eval_retrieval(labels, k=args.k)  if run_retrieval else None

    if args.json:
        output = {}
        if fit_results:
            output["fit_scores"] = fit_results
        if retrieval_results:
            output["retrieval"] = retrieval_results
        print(json.dumps(output, indent=2))
        return

    if fit_results:
        print_fit_results(fit_results)
    if retrieval_results:
        print_retrieval_results(retrieval_results)

    print()


if __name__ == "__main__":
    os.chdir(_ROOT)  # run from project root so relative paths work
    main()
