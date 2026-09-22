import pytest

pytest.importorskip("openai")

from examples.openai_daily_client import build_request


def test_daily_client_only_adds_gateway_context_when_reference_exists():
    messages = [{"role": "user", "content": "question"}]

    assert build_request(messages, None) == {"model": "adaptive", "messages": messages}

    request = build_request(messages, "reference text")
    envelope = request["extra_body"]["gateway_context"]
    assert set(envelope) == {"blocks"}
    assert envelope["blocks"] == [{
        "id": "daily-reference",
        "source": "daily-client",
        "kind": "document",
        "content": "reference text",
        "optional": True,
    }]
    assert "optimize" not in envelope
