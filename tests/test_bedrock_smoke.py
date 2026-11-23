import json
import logging
import sys

import pytest

from agentops_lab.cli.bedrock_smoke import main

REGION = "eu-central-1"
MODEL_ID = "eu.amazon.nova-micro-v1:0"
EXPECTED_REQUEST = {
    "modelId": MODEL_ID,
    "messages": [{"role": "user", "content": [{"text": "Reply with OK."}]}],
    "system": [{"text": "Reply briefly with plain text."}],
    "inferenceConfig": {"maxTokens": 16, "temperature": 0},
}


def test_smoke_succeeds_with_model_text(bedrock_stub, caplog, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agentops-bedrock-smoke"])
    bedrock_stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 12, "outputTokens": 1, "totalTokens": 13},
            "metrics": {"latencyMs": 10},
        },
        EXPECTED_REQUEST,
    )

    with caplog.at_level(logging.INFO, logger="agentops_lab.cli.bedrock_smoke"):
        assert main() == 0

    assert json.loads(caplog.records[-1].message) == {
        "event": "bedrock_smoke_succeeded",
        "model_id": MODEL_ID,
        "region": REGION,
    }


@pytest.mark.parametrize("error_code", ["AccessDeniedException", "ThrottlingException"])
def test_smoke_reports_bedrock_failure(bedrock_stub, caplog, monkeypatch, error_code):
    monkeypatch.setattr(sys, "argv", ["agentops-bedrock-smoke"])
    bedrock_stub.add_client_error("converse", error_code, expected_params=EXPECTED_REQUEST)

    assert main() == 1

    event = json.loads(caplog.records[-1].message)
    assert event["event"] == "bedrock_smoke_failed"
    assert error_code in event["error"]


def test_smoke_requires_model_id(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", ["agentops-bedrock-smoke"])
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)

    assert main() == 1

    assert json.loads(caplog.records[-1].message)["error"] == "Set BEDROCK_MODEL_ID."
