"""RAGAS evaluation harness for RAG pipeline quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.retrieval.rag_chain import build_rag_chain


@dataclass
class EvalCase:
    question: str
    ground_truth: str


DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        question="What do customers say about delivery times?",
        ground_truth="Customers frequently complain about slow delivery, with many reporting waits of 2-4 weeks.",
    ),
    EvalCase(
        question="What are the most common product complaints?",
        ground_truth="Common complaints include product quality issues, items not matching descriptions, and damaged packaging.",
    ),
    EvalCase(
        question="How do customers feel about electronics products?",
        ground_truth="Electronics receive mixed reviews with praise for functionality but complaints about setup difficulty.",
    ),
    EvalCase(
        question="What positive feedback do customers give?",
        ground_truth="Positive feedback highlights good product quality, fast shipping when it works, and value for money.",
    ),
    EvalCase(
        question="What issues do customers report with furniture?",
        ground_truth="Furniture complaints include assembly difficulty, size discrepancies, and delivery damage.",
    ),
]


def run_evaluation(
    cases: list[EvalCase] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run RAGAS evaluation on the RAG pipeline.
    
    Args:
        cases: Test cases to evaluate. Defaults to DEFAULT_CASES.
        output_path: Optional path to save results JSON.
    
    Returns:
        Dictionary with metric scores and per-question breakdown.
    """
    cases = cases or DEFAULT_CASES
    chain = build_rag_chain()

    dataset: list[dict[str, Any]] = []
    for case in cases:
        result = chain.invoke(case.question)
        dataset.append({
            "question": case.question,
            "answer": result["answer"],
            "contexts": [doc["text"] for doc in result["sources"]],
            "ground_truth": case.ground_truth,
        })

    metrics = [faithfulness, answer_relevancy, context_precision]
    scores = evaluate(dataset, metrics=metrics)

    output = {
        "aggregate": {
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
        },
        "cases": dataset,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved evaluation results to {output_path}")

    return output


def print_report(results: dict[str, Any]) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 50)
    print("RAGAS Evaluation Report")
    print("=" * 50)
    
    agg = results["aggregate"]
    print(f"\nFaithfulness:      {agg['faithfulness']:.2%}")
    print(f"Answer Relevancy:  {agg['answer_relevancy']:.2%}")
    print(f"Context Precision: {agg['context_precision']:.2%}")
    
    avg = sum(agg.values()) / len(agg)
    print(f"\nOverall Score:     {avg:.2%}")
    
    target = 0.80
    if avg >= target:
        print(f"✅ Meets target ({target:.0%})")
    else:
        print(f"❌ Below target ({target:.0%})")
    
    print("=" * 50)


if __name__ == "__main__":
    results = run_evaluation(output_path=Path("scripts/ragas_results.json"))
    print_report(results)

