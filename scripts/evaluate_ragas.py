"""RAGAS evaluation with four metrics:
  - Context Precision   (are the useful chunks ranked at the top?)
  - Context Recall      (do the chunks contain all facts of the reference answer?)
  - Faithfulness        (is every claim in the answer supported by the chunks?)
  - Answer Relevancy    (does the answer address the question?)
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from rag import pipeline

RESULTS_DIR = config.DATA_DIR / "eval_results"
SUMMARY_FILE = RESULTS_DIR / "summary.jsonl"
INPUT_COLS = {"user_input", "retrieved_contexts", "response", "reference"}


def load_dataset(path: Path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    urls = [u for u in spec.get("urls", []) if u.startswith("http")]
    questions = [
        {"question": q["question"].strip(), "reference": q["reference"].strip()}
        for q in spec.get("questions", [])
        if q.get("question") and q.get("reference")
    ]
    return urls, questions

def collect_rows(store, questions):
    """Run every question through the real pipeline and capture what RAGAS needs."""
    rows = []
    for i, q in enumerate(questions, start=1):
        result = pipeline.answer(store, q["question"])
        rows.append({
            "user_input": q["question"],
            "retrieved_contexts": [p.page_content for p in result.passages],  # chunks sent to the LLM
            "response": result.answer,
            "reference": q["reference"],
        })
        print(f"  [{i}/{len(questions)}] {q['question'][:70]}")
    return rows


def _patch_vertexai():
    """Some ragas versions import modules that newer langchain-community removed. Stub them."""
    import importlib
    import types

    for mod, cls in [
        ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
        ("langchain_community.llms.vertexai", "VertexAI"),
    ]:
        try:
            importlib.import_module(mod)
        except ImportError:
            stub = types.ModuleType(mod)
            setattr(stub, cls, type(cls, (), {}))
            sys.modules[mod] = stub


def _import_ragas():
    _patch_vertexai()
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        print(f"Could not import RAGAS: {exc!r}")
        print("Check versions with:  uv pip list | findstr /i ragas")
        sys.exit(2)
    return (EvaluationDataset, evaluate, LangchainEmbeddingsWrapper, LangchainLLMWrapper,
            Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy)


def run_ragas(rows):
    (EvaluationDataset, evaluate, LangchainEmbeddingsWrapper, LangchainLLMWrapper,
     Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy) = _import_ragas()
    from ragas.run_config import RunConfig  # imported after the patch has run

    from rag.embeddings import get_embeddings
    from rag.llm import get_llm

    judge = os.getenv("RAGAS_JUDGE_MODEL", config.LLM_MODEL)
    print(f"Scoring with judge model: {judge}")
    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
        ],
        llm=LangchainLLMWrapper(get_llm(judge)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        show_progress=True,
        run_config=RunConfig(timeout=300, max_workers=8),
    )
    return result.to_pandas()


def settings_snapshot() -> dict:
    return {
        "llm": config.LLM_MODEL,
        "embedding": config.EMBEDDING_MODEL,
        "use_rerank": config.USE_RERANK,
        "retrieve_k": config.RETRIEVE_K,
        "rerank_top_n": config.RERANK_TOP_N,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


SHORT_NAMES = {
    "llm_context_precision_with_reference": "ctx_precision",
    "context_recall": "ctx_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "ans_relevancy",
}


def print_history():
    if not SUMMARY_FILE.exists():
        print("No saved runs yet.")
        return
    runs = [json.loads(line) for line in SUMMARY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    metric_names = sorted({m for r in runs for m in r["scores"]})
    header = f"{'label':<18}{'n':>4}  " + "  ".join(f"{SHORT_NAMES.get(m, m)[:14]:>14}" for m in metric_names) + "  note"
    print(header)
    print("-" * (len(header) + 20))
    for r in runs:
        cells = "  ".join(f"{r['scores'].get(m, float('nan')):>14.3f}" for m in metric_names)
        print(f"{r['label'][:17]:<18}{r['n_questions']:>4}  {cells}  {r.get('note', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the real estate RAG app")
    parser.add_argument("--dataset", default=str(ROOT / "tests" / "ragas_dataset.json"))
    parser.add_argument("--label", default="run", help="name for this run, e.g. baseline, hybrid, rerank")
    parser.add_argument("--note", default="", help="short description of what changed in this run")
    parser.add_argument("--no-ingest", action="store_true", help="reuse the articles already in Chroma")
    parser.add_argument("--history", action="store_true", help="show all saved runs and exit")
    args = parser.parse_args()

    if args.history:
        print_history()
        return 0

    if not config.api_key():
        print("OPENAI_API_KEY is missing (.env).")
        return 2

    _import_ragas()  # fail fast, before spending money on answers

    urls, questions = load_dataset(Path(args.dataset))
    if not questions:
        print("No questions with a 'reference' answer found in the dataset file.")
        return 2

    if args.no_ingest:
        store = pipeline.load_existing_store()
        if store is None:
            print("Chroma is empty. Run without --no-ingest.")
            return 2
    else:
        if not urls:
            print("No valid URLs in the dataset file (replace the PASTE_... placeholders).")
            return 2
        print(f"Ingesting {len(urls)} article(s)...")
        report = pipeline.ingest(urls, progress=lambda m: print("  " + m))
        if report.store is None:
            print("No articles could be loaded.")
            return 2
        store = report.store

    print(f"Answering {len(questions)} question(s)...")
    rows = collect_rows(store, questions)

    print("Running RAGAS...")
    df = run_ragas(rows)

    score_cols = [c for c in df.columns if c not in INPUT_COLS]
    means = df[score_cols].mean(numeric_only=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = RESULTS_DIR / f"{args.label}_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    with SUMMARY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "label": args.label,
            "note": args.note,
            "time": stamp,
            "n_questions": len(rows),
            "scores": {k: round(float(v), 4) for k, v in means.items()},
            "settings": settings_snapshot(),
        }) + "\n")

    print("\n=== RAGAS results (average over questions) ===")
    for name, value in means.items():
        print(f"  {name:<28}{value:.3f}")
    nan_count = int(df[score_cols].isna().sum().sum())
    if nan_count:
        print(f"\nWarning: {nan_count} score(s) could not be computed (shown as NaN) and were ignored in the averages.")
    print(f"\nPer-question scores: {csv_path}")
    print("Compare runs with:  uv run python scripts/evaluate_ragas.py --history")
    return 0


if __name__ == "__main__":
    sys.exit(main())

    