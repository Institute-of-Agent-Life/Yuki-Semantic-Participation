"""Probe a fixed synthetic subset with real Jev; never sends QQ messages.

Run only in an environment where SEMANTIC_PARTICIPATION_API_KEY is already set.
The fixture contains developer-authored Chinese messages, not user conversations.
The output excludes request bodies, message text, credentials and provider bodies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from yuki_participation.models import Scope, ScopedEvent, Snapshot, SourceRef
from yuki_participation.observer import JevObserver

SELECTED = {"a2", "a3", "a4", "c2", "c3", "c5", "e3", "e5", "e6", "e8", "f2", "f4"}


def make_event(row: dict[str, object], scope: Scope) -> ScopedEvent:
    reply = row.get("reply_to")
    return ScopedEvent(
        scope=scope,
        ref=SourceRef(event_id=str(row["id"]), revision=1),
        thread=str(row["thread"]),
        author=str(row["author"]),
        target=str(row["target"]),
        text=str(row["text"]),
        at=float(row["at"]),
        kind=str(row["kind"]),
        reply_to=SourceRef(event_id=str(reply), revision=1) if reply else None,
    )


async def probe(fixture: dict[str, object]) -> dict[str, object]:
    key = os.environ.get("SEMANTIC_PARTICIPATION_API_KEY", "")
    if not key:
        raise ValueError("Jev key is not configured in this process")
    model = os.environ.get("SEMANTIC_PARTICIPATION_MODEL", "jev-1.13.0")
    observer = JevObserver(
        key, identity=lambda: {"display_name": "Yuki", "aliases": ()}, model=model
    )
    results: list[dict[str, object]] = []
    try:
        for scene in fixture["scenes"]:
            scope = Scope(conversation_id=f"synthetic:{scene['id']}", generation=1)
            prior: list[ScopedEvent] = []
            for row in scene["events"]:
                focus = make_event(row, scope)
                if row["id"] in SELECTED:
                    context = prior[-6:]
                    if focus.reply_to is not None:
                        anchor = next((item for item in prior if item.ref == focus.reply_to), None)
                        if anchor and anchor not in context:
                            context = [anchor, *context[-5:]]
                    snapshot = Snapshot(
                        scope=scope,
                        focus=focus,
                        context=tuple(context),
                        sequence=1,
                        issued_at=focus.at + 1,
                    )
                    started = time.perf_counter()
                    result: dict[str, object] = {
                        "case_id": row["id"],
                        "scene": scene["id"],
                        "gold_interaction": row["act"],
                        "gold_floor": row["floor"],
                        "mentions_bot": row["mentions_bot"],
                    }
                    try:
                        observation = await observer.evaluate(snapshot)
                        result.update(
                            {
                                "status": "observed",
                                "interaction": observation.answers.get("interaction_mark").choice
                                if observation.answers.get("interaction_mark")
                                else None,
                                "floor": observation.answers.get("floor_state").choice
                                if observation.answers.get("floor_state")
                                else None,
                                "invalid_dimensions": observation.invalid_dimensions,
                                "answers": {
                                    name: {
                                        "choice": answer.choice,
                                        "probabilities": answer.probabilities,
                                    }
                                    for name, answer in observation.answers.items()
                                },
                                "input_tokens": observation.input_tokens,
                                "output_tokens": observation.output_tokens,
                                "request_bytes": observation.request_bytes,
                            }
                        )
                    except (httpx.HTTPError, ValueError) as exc:
                        result.update(
                            {
                                "status": "error",
                                "error_category": type(exc).__name__,
                                "http_status": exc.response.status_code
                                if isinstance(exc, httpx.HTTPStatusError)
                                else None,
                            }
                        )
                    result["latency_seconds"] = round(time.perf_counter() - started, 3)
                    results.append(result)
                prior.append(focus)
    finally:
        await observer.aclose()
    observed = [row for row in results if row["status"] == "observed"]
    return {
        "kind": "isolated_real_jev_on_developer_synthetic_messages",
        "model": model,
        "requests": len(results),
        "observed": len(observed),
        "interaction_matches_gold": sum(
            row["interaction"] == row["gold_interaction"] for row in observed
        ),
        "floor_matches_gold": sum(row["floor"] == row["gold_floor"] for row in observed),
        "input_tokens_known_sum": sum(
            int(row["input_tokens"]) for row in observed if row["input_tokens"] is not None
        ),
        "output_tokens_known_sum": sum(
            int(row["output_tokens"]) for row in observed if row["output_tokens"] is not None
        ),
        "rows": results,
        "limitations": [
            "Gold labels were authored with the fixture and are not independent blind annotation.",
            "These requests do not use live Yuki admission, Main Agent, tools, or QQ delivery.",
            "No accuracy claim about real group chats follows from these synthetic cases.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8-sig"))
    print(json.dumps(asyncio.run(probe(fixture)), ensure_ascii=False))


if __name__ == "__main__":
    main()
