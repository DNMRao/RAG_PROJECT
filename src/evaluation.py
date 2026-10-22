"""Run question sets through the full pipeline, score them, and write the report."""

import json
import re
from pathlib import Path

from src.generation import CITATION_PATTERN, answer_question
from src.ingestion import build_chunks, load_documents
from src.retrieval import MIN_SCORE, MODEL_NAME, MODEL_REVISION, TOP_K, Retriever

CALIBRATION_PATH = Path("data/evaluation/calibration.json")
QUESTIONS_PATH = Path("data/evaluation/questions.json")
REPORT_PATH = Path("reports/evaluation_report.md")
GENERATION_MODE = "deterministic extractive (quotes evidence paragraphs; no LLM)"

BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"


def normalize(text: str) -> str:
    """Lower-case and collapse whitespace so comparisons ignore formatting."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------- metrics


def recall_at_k(expected: list[str], retrieved_docs: list[str]) -> float | None:
    """Fraction of expected documents present in the top-k (None if nothing expected)."""
    if not expected:
        return None
    return len(set(expected) & set(retrieved_docs)) / len(expected)


def reciprocal_rank(expected: list[str], retrieved_docs: list[str]) -> float | None:
    """1/rank of the first hit from an expected document; 0 if none in top-k."""
    if not expected:
        return None
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected:
            return 1 / rank
    return 0.0


def answer_relevance(required_terms: list[str], text: str, answered: bool) -> float | None:
    """Rubric: fraction of required terms that appear in the answer (0 if it abstained)."""
    if not required_terms:
        return None
    if not answered:
        return 0.0
    answer = normalize(text)
    return sum(normalize(term) in answer for term in required_terms) / len(required_terms)


def faithfulness(text: str, evidence_texts: list[str]) -> float:
    """Rubric: fraction of answer sentences found verbatim in the evidence it was given."""
    stripped = CITATION_PATTERN.sub("", text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
    evidence = normalize(" ".join(evidence_texts))
    return sum(normalize(s) in evidence for s in sentences) / len(sentences)


# ---------------------------------------------------------------- evaluation


def evaluate_case(retriever: Retriever, case: dict) -> dict:
    expected = case["expected_sources"]
    raw_hits = retriever.search(case["question"], k=TOP_K)  # retrieval metrics: no threshold
    retrieved_docs = [h.chunk.document_id for h in raw_hits]

    ans = answer_question(retriever, case["question"])  # the same end-to-end pipeline
    answered = ans.status == "answered"
    return {
        "id": case["id"],
        "question": case["question"],
        "expected": expected,
        "should_abstain": case["should_abstain"],
        "top_score": raw_hits[0].score,
        "retrieved": [h.chunk.chunk_id for h in raw_hits],
        "status": ans.status,
        "citations": list(ans.citations),
        "decision_correct": answered != case["should_abstain"],
        "recall": recall_at_k(expected, retrieved_docs),
        "rr": reciprocal_rank(expected, retrieved_docs),
        "relevance": answer_relevance(case["required_terms"], ans.text, answered),
        "faithfulness": faithfulness(ans.text, [h.chunk.text for h in ans.evidence]) if answered else None,
        "safety_events": list(ans.safety_events),
        "answer": ans.text,
    }


def mean(values: list[float | None]) -> float | None:
    """Average of the defined values; None when no question had this metric."""
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        f"recall@{TOP_K}": mean([r["recall"] for r in rows]),
        "MRR": mean([r["rr"] for r in rows]),
        "answer relevance": mean([r["relevance"] for r in rows]),
        "faithfulness": mean([r["faithfulness"] for r in rows]),
        "decision accuracy": mean([float(r["decision_correct"]) for r in rows]),
    }


# ---------------------------------------------------------------- report


def fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def summary_table(summaries: dict[str, dict]) -> list[str]:
    names = [k for k in next(iter(summaries.values())) if k != "n"]
    lines = ["| Set | n | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for set_name, s in summaries.items():
        lines.append(f"| {set_name} | {s['n']} | " + " | ".join(fmt(s[k]) for k in names) + " |")
    return lines


def results_table(rows: list[dict]) -> list[str]:
    lines = [
        "| id | expected | should | status | top score | top-3 retrieved | citations "
        f"| recall@{TOP_K} | RR | relevance | faithful | decision |",
        "|---" * 12 + "|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {', '.join(r['expected']) or '–'} "
            f"| {'abstain' if r['should_abstain'] else 'answer'} | {r['status']} "
            f"| {r['top_score']:.3f} | {', '.join(r['retrieved'])} | {', '.join(r['citations']) or '–'} "
            f"| {fmt(r['recall'])} | {fmt(r['rr'])} | {fmt(r['relevance'])} | {fmt(r['faithfulness'])} "
            f"| {'✓' if r['decision_correct'] else '✗'} |"
        )
    return lines


def generated_section(cal_rows: list[dict], final_rows: list[dict]) -> str:
    lines = [
        BEGIN,
        "_This section is generated by `uv run python -m src.evaluation`. Do not edit it by hand._",
        "",
        "## Configuration",
        "",
        "- Chunking: one chunk per paragraph, title prepended, no overlap",
        f"- Embedding model: `{MODEL_NAME}` at revision `{MODEL_REVISION}` (CPU, normalised vectors)",
        "- Vector store: Qdrant local in-memory collection, cosine distance, rebuilt from "
        "`data/knowledge_base/` on every run",
        f"- Retrieval: top-k = {TOP_K}; answer only if top score ≥ {MIN_SCORE} (chosen on calibration set)",
        "- Evidence: chunks containing injected instructions are removed; the document with the "
        "highest summed score is used",
        f"- Generation mode: {GENERATION_MODE}",
        "",
        "## Metric rubrics",
        "",
        "| Metric | Rubric | Questions scored |",
        "|---|---|---|",
        f"| recall@{TOP_K} | fraction of expected source documents present in the raw top-{TOP_K} hits "
        "(before threshold) | supported |",
        "| MRR | mean of 1/rank of the first hit from an expected document (0 if absent) | supported |",
        "| answer relevance | fraction of the labelled `required_terms` that appear in the answer "
        "(case-insensitive); 0 if the system abstained | supported |",
        "| faithfulness | fraction of answer sentences (citations removed) found verbatim in the "
        "evidence passed to the generator | answered |",
        "| decision accuracy | 1 if it answered a supported question or abstained on an unsupported one |"
        " all |",
        "",
        "Faithfulness is 1.0 by construction for extractive answers; it shows that no unsupported "
        "text was added, not that the quoted passage answers the question (that is relevance).",
        "",
        "## Summary",
        "",
        *summary_table({"calibration": summarize(cal_rows), "final (questions.json)": summarize(final_rows)}),
        "",
        "## Threshold calibration evidence (calibration.json)",
        "",
        *results_table(cal_rows),
        "",
        "## Final per-question results (questions.json)",
        "",
        *results_table(final_rows),
        "",
    ]
    events = [(r["id"], e) for r in final_rows for e in r["safety_events"]]
    if events:
        lines += ["### Safety events", "", *[f"- {qid}: {e}" for qid, e in events], ""]
    lines.append(END)
    return "\n".join(lines)


REPORT_TEMPLATE = """# Northstar RAG Evaluation Report

{generated}

## Failure analysis

_TODO: analyse at least two failures or fragile successes, using the retrieved chunks above._

## Threshold decision

_TODO: explain how {min_score} was chosen from the calibration table and how much margin it has._

## Recommendations (prioritised)

1. _TODO_
2. _TODO_
3. _TODO_

## Submission questions

1. Which retrieval failures matter most to the support scenario, and why?
2. How did you select your no-answer threshold?
3. What evidence suggests an answer is faithful to retrieved context?
4. What would you change before allowing live support colleagues to use the system?
"""


def write_report(generated: str, path: Path = REPORT_PATH) -> None:
    """Create the report, or replace only the generated block so hand-written text survives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        start, end = old.find(BEGIN), old.find(END)
        if start != -1 and end != -1:
            path.write_text(old[:start] + generated + old[end + len(END):], encoding="utf-8")
            return
    path.write_text(REPORT_TEMPLATE.format(generated=generated, min_score=MIN_SCORE), encoding="utf-8")


def main() -> None:
    retriever = Retriever(build_chunks(load_documents()))
    cal_rows = [evaluate_case(retriever, c) for c in json.loads(CALIBRATION_PATH.read_text())]
    final_rows = [evaluate_case(retriever, c) for c in json.loads(QUESTIONS_PATH.read_text())]

    write_report(generated_section(cal_rows, final_rows))
    for name, rows in [("calibration", cal_rows), ("final", final_rows)]:
        print(name, {k: fmt(v) if isinstance(v, float) else v for k, v in summarize(rows).items()})
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
