"""Explicit fixture connection. Production code does not supply these values."""

from yuki_participation.observer import JevObserver

ENDPOINT = "https://semantic.example/systemone"
MODEL = "jev-1.13.0"


def jev_observer(
    api_key: str = "synthetic-key",
    *,
    identity,
    client=None,
    request_model: str = MODEL,
    response_model: str | None = None,
    endpoint: str = ENDPOINT,
    name: str = "fixture",
    revision: str = "fixture",
    model: str | None = None,
):
    return JevObserver(
        api_key,
        identity=identity,
        endpoint=endpoint,
        request_model=model or request_model,
        response_model=response_model,
        client=client,
        name=name,
        revision=revision,
    )
