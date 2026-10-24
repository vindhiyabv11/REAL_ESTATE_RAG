"""Quality check: ingest the URLs in tests/eval_questions.json, ask each question and
check that the expected keywords appear in the answer.

Run from the project root:   python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from rag import pipeline


def main() -> int:
    if not config.api_key():
        print("OPENAI_API_KEY is missing (.env).")
        return 2
    spec = json.loads((ROOT / "tests" / "eval_questions.json").read_text(encoding="utf-8"))

    print(f"Ingesting {len(spec['urls'])} article(s)...")
    report = pipeline.ingest(spec["urls"], progress=print)
    if report.store is None:
        print("No articles could be loaded - aborting.")
        return 2

    passed = 0
    for i, case in enumerate(spec["questions"], start=1):
        result = pipeline.answer(report.store, case["question"])
        text = result.answer.lower()
        missing = [k for k in case["expected_keywords"] if k.lower() not in text]
        ok = not missing
        passed += ok
        print(f"\n[{i}] {'PASS' if ok else 'FAIL'}  {case['question']}")
        print(f"    answer : {result.answer[:300]}")
        if missing:
            print(f"    missing: {missing}")

    total = len(spec["questions"])
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())