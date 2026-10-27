"""Explicit, bounded real-provider probe; never imported/run by the test suite.

Example after installing this repository:
    python scripts/probe_jev.py --limit 1 --output private-data/jev-probe.json

Supply TYPESAFE_API_KEY through the environment. This script never prints/writes it,
response bodies, exception messages or tracebacks. Fixture expectations are local only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from yuki_participation.models import (
    CandidateKind,
    HostUnitOption,
    Scope,
    ScopedEvent,
    Snapshot,
    SourceRef,
)
from yuki_participation.observer import PROJECTION_REVISION, InputTooLarge, JevObserver
from yuki_participation.rubric import CRITERIA, REVISION

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = "developer-authored synthetic"


def load_cases(path: Path) -> list[dict]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("provenance") != PROVENANCE:
        raise ValueError("fixture_requires_explicit_synthetic_provenance")
    cases = dataset["cases"]
    if not 1 <= len(cases) <= 8 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("fixture_requires_one_to_eight_unique_cases")
    for case in cases:
        for dimension, expected in case["expected"].items():
            criteria = CRITERIA.get(dimension, {})
            if dimension == "unit_selection":
                criteria = {
                    **{o["key"]: "" for o in case["focus"].get("unit_options", [])},
                    "unknown": "",
                }
            if expected not in criteria:
                raise ValueError("fixture_expected_label_not_in_rubric")
        make_snapshot(case, sequence=1, now=100)
    return cases


def make_snapshot(case: dict, *, sequence: int, now: float) -> Snapshot:
    scope = Scope(conversation_id="synthetic-smoke", generation=1)
    context = case.get("context", [])
    visible = [*context, case["focus"]]
    neutral_refs = {item["id"]: f"e{index}" for index, item in enumerate(visible, start=1)}

    def source(item: dict, at: float) -> ScopedEvent:
        return ScopedEvent(
            scope=scope,
            ref=SourceRef(event_id=neutral_refs[item["id"]], revision=1),
            thread=item.get("thread", "thread-1"),
            unit_ambiguous=bool(item.get("unit_options")),
            unit_options=tuple(HostUnitOption(**o) for o in item.get("unit_options", [])),
            author=item["author"],
            target=item["target"],
            text=item["text"],
            kind=item.get("kind", "human"),
            at=at,
            reply_to=SourceRef(
                event_id=neutral_refs.get(item["reply_to"], "unavailable"), revision=1
            )
            if item.get("reply_to")
            else None,
        )

    return Snapshot(
        scope=scope,
        focus=source(case["focus"], now - 1),
        context=tuple(
            source(item, now - len(context) - 1 + index) for index, item in enumerate(context)
        ),
        sequence=sequence,
        issued_at=now,
        kind=CandidateKind(case.get("kind", "conversation")),
    )


async def probe(cases: list[dict], observer: JevObserver) -> dict:
    """Sequential, one attempt per distinct case. No automatic retries or concurrency."""
    rows = []
    for sequence, case in enumerate(cases[:8], start=1):
        start = time.perf_counter()
        row = {"case_id": case["id"], "expected": case["expected"], "attempt_count": 1}
        try:
            snapshot = make_snapshot(case, sequence=sequence, now=time.time())
            row["http_attempt_count"] = 1
            result = await observer.evaluate(snapshot)
            row["request_bytes"] = result.request_bytes
            comparisons = {}
            for dimension, expected in case["expected"].items():
                answer = result.answers.get(dimension)
                comparisons[dimension] = {
                    "expected": expected,
                    "actual": answer.choice if answer else None,
                    "matches": answer is not None and answer.choice == expected,
                    "expected_probability": answer.p(expected) if answer else None,
                }
            row.update(
                {
                    "status": "observed",
                    "model_revision": result.model_revision,
                    "rubric_revision": result.rubric_revision,
                    "request_bytes": result.request_bytes,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "omitted_context": result.snapshot.omitted_context,
                    "answers": {k: v.model_dump(mode="json") for k, v in result.answers.items()},
                    "invalid_dimensions": list(result.invalid_dimensions),
                    "comparisons": comparisons,
                }
            )
        except InputTooLarge as exc:
            row.update(
                {
                    "status": "local_input_too_large",
                    "attempt_count": 0,
                    "http_attempt_count": 0,
                    "required_request_bytes": exc.request_bytes,
                    "limit_bytes": exc.limit_bytes,
                }
            )
        except Exception as exc:
            # Deliberately exclude str(exc), repr(exc), response bodies and request headers.
            row.update(
                {
                    "status": "request_failed",
                    "error_type": type(exc).__name__,
                    "http_status": exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None,
                }
            )
        row["latency_seconds"] = round(time.perf_counter() - start, 4)
        rows.append(row)
    comparisons = [comparison for row in rows for comparison in row.get("comparisons", {}).values()]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": PROVENANCE,
        "limitations": "Synthetic developer expectations; not independent human labels or "
        "a claim of real-chat accuracy. No provider retries.",
        "configured_model": observer.model,
        "rubric_revision": REVISION,
        "projection_revision": PROJECTION_REVISION,
        "request_byte_ceiling": observer.max_request_bytes,
        "cases_requested": len(rows),
        "observed_count": sum(row["status"] == "observed" for row in rows),
        "compared_dimensions": len(comparisons),
        "matching_dimensions": sum(item["matches"] for item in comparisons),
        "input_tokens": sum(row.get("input_tokens") or 0 for row in rows),
        "output_tokens": sum(row.get("output_tokens") or 0 for row in rows),
        "usage_complete": all(
            row.get("input_tokens") is not None and row.get("output_tokens") is not None
            for row in rows
            if row.get("http_attempt_count")
        ),
        "rows": rows,
    }


async def run(args: argparse.Namespace, key: str) -> int:
    cases = load_cases(args.fixtures)
    endpoint = os.environ.get("SEMANTIC_PARTICIPATION_ENDPOINT", "")
    model = os.environ.get("SEMANTIC_PARTICIPATION_MODEL", "")
    if not endpoint or not model:
        raise ValueError("semantic endpoint and model are not configured in this process")
    observer = JevObserver(
        key,
        identity=lambda: {"display_name": "Yuki", "aliases": ()},
        endpoint=endpoint,
        request_model=model,
        response_model=os.environ.get("SEMANTIC_PARTICIPATION_RESPONSE_MODEL") or None,
    )
    try:
        report = await probe(cases[: args.limit], observer)
    finally:
        await observer.aclose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases_requested": report["cases_requested"],
                "observed_count": report["observed_count"],
                "input_tokens": report["input_tokens"],
                "matching_dimensions": report["matching_dimensions"],
                "compared_dimensions": report["compared_dimensions"],
            }
        )
    )
    return int(any(row["status"] != "observed" for row in report["rows"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "fixtures" / "semantic-smoke.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        parser.error("TYPESAFE_API_KEY is required; provide it through the environment.")
    try:
        return asyncio.run(run(args, key))
    except Exception as exc:
        # Even setup/output failures omit exception text rather than risking secret leakage.
        print(
            json.dumps(
                {
                    "status": "probe_setup_or_output_failed",
                    "error_type": type(exc).__name__,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
