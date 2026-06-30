"""Evaluation harness: runs the real Anthropic API against eval/golden_set.json,
scores classification accuracy, and flags likely hallucinations in drafted replies.

This is the one place in the repo that makes real LLM calls -- it costs money and
is intentionally NOT part of the default CI workflow. Run it manually:

    python -m eval.run_eval
    python -m eval.run_eval --output-file eval/report.md

Requires ANTHROPIC_API_KEY to be set (via .env or the environment).
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.llm.anthropic_client import AnthropicLLMClient
from app.llm.base import LLMClient, LLMError, TicketClassification
from app.services.ticket_pipeline import derive_confidence_flag
from eval.faithfulness import find_unsupported_claims

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"


@dataclass
class TicketResult:
    id: str
    subject: str
    expected_category: str
    expected_urgency: str
    latency_ms: float
    predicted: TicketClassification | None = None
    error: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)

    @property
    def category_match(self) -> bool | None:
        return None if self.predicted is None else self.predicted.category == self.expected_category

    @property
    def urgency_match(self) -> bool | None:
        return None if self.predicted is None else self.predicted.urgency == self.expected_urgency


async def evaluate_ticket(llm_client: LLMClient, item: dict) -> TicketResult:
    start = time.perf_counter()
    try:
        classification = await llm_client.classify_ticket(item["subject"], item["body"])
    except LLMError as exc:
        return TicketResult(
            id=item["id"],
            subject=item["subject"],
            expected_category=item["expected_category"],
            expected_urgency=item["expected_urgency"],
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            error=str(exc),
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    source_text = f"{item['subject']}\n{item['body']}"
    unsupported = find_unsupported_claims(classification.draft_reply, source_text)

    return TicketResult(
        id=item["id"],
        subject=item["subject"],
        expected_category=item["expected_category"],
        expected_urgency=item["expected_urgency"],
        latency_ms=latency_ms,
        predicted=classification,
        unsupported_claims=unsupported,
    )


def render_report(results: list[TicketResult], model: str) -> str:
    completed = [r for r in results if r.predicted is not None]
    errored = [r for r in results if r.error is not None]

    category_correct = sum(1 for r in completed if r.category_match)
    urgency_correct = sum(1 for r in completed if r.urgency_match)
    low_confidence = sum(
        1
        for r in completed
        if derive_confidence_flag(r.predicted.confidence).value == "low_confidence"
    )
    flagged_replies = [r for r in completed if r.unsupported_claims]
    total_input_tokens = sum(r.predicted.input_tokens for r in completed)
    total_output_tokens = sum(r.predicted.output_tokens for r in completed)
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0.0

    n = len(results)
    lines = [
        "# Evaluation Report",
        "",
        f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Model: {model}",
        f"- Golden set size: {n}",
        f"- Completed: {len(completed)}  |  Errored: {len(errored)}",
        "",
        "## Accuracy",
        "",
        f"- Category accuracy: {category_correct}/{len(completed)} "
        f"({_pct(category_correct, len(completed))}%)",
        f"- Urgency accuracy: {urgency_correct}/{len(completed)} "
        f"({_pct(urgency_correct, len(completed))}%)",
        f"- Flagged low-confidence: {low_confidence}/{len(completed)}",
        f"- Flagged possible hallucination: {len(flagged_replies)}/{len(completed)}",
        f"- Avg latency: {avg_latency:.0f} ms",
        f"- Total tokens: {total_input_tokens} in / {total_output_tokens} out",
        "",
        "## Per-ticket results",
        "",
        "| id | category (exp/got) | urgency (exp/got) | confidence | faithfulness | notes |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        if r.error is not None:
            lines.append(f"| {r.id} | - | - | - | - | ERROR: {r.error} |")
            continue
        cat_mark = "OK" if r.category_match else "MISMATCH"
        urg_mark = "OK" if r.urgency_match else "MISMATCH"
        faith = "OK" if not r.unsupported_claims else f"FLAGGED: {', '.join(r.unsupported_claims)}"
        lines.append(
            f"| {r.id} | {r.expected_category}/{r.predicted.category} ({cat_mark}) "
            f"| {r.expected_urgency}/{r.predicted.urgency} ({urg_mark}) "
            f"| {r.predicted.confidence:.2f} | {faith} | |"
        )

    return "\n".join(lines) + "\n"


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.0f}"


async def main(output_file: str | None) -> None:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    settings = get_settings()
    llm_client = AnthropicLLMClient(settings=settings)

    results = []
    for item in golden_set:
        result = await evaluate_ticket(llm_client, item)
        results.append(result)
        status = "ERROR" if result.error else ("OK" if result.category_match else "MISMATCH")
        print(f"[{result.id}] {status} - {result.subject[:60]}")

    report = render_report(results, model=settings.anthropic_model)
    print("\n" + report)

    if output_file:
        Path(output_file).write_text(report)
        print(f"Report written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-file", default=None, help="Optional path to also write the report as markdown"
    )
    args = parser.parse_args()
    asyncio.run(main(args.output_file))
