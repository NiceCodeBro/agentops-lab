import json
import sys
from pathlib import Path

import pytest
from botocore.stub import ANY
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from agentops_lab.research import build_research_graph
from agentops_lab.research_cli import main
from agentops_lab.research_models import ResearchBrief, ResearchRequest

FIXTURE = Path("examples/nvda.synthetic.json")
ANALYSIS = {
    "key_findings": [
        {"text": "Hypothetical data center revenue grew 12% YoY.", "source_ids": ["demo-results"]}
    ],
    "opportunities": [
        {"text": "Cloud accelerator demand could support growth.", "source_ids": ["demo-results"]}
    ],
    "risks": [
        {"text": "Advanced packaging supply may be concentrated.", "source_ids": ["demo-risks"]}
    ],
    "missing_information": ["What is the full-year margin outlook?"],
}
EXPECTED_REQUEST = {
    "modelId": "eu.amazon.nova-micro-v1:0",
    "messages": ANY,
    "system": ANY,
    "inferenceConfig": {"maxTokens": 768, "temperature": 0},
}


@pytest.fixture
def request_with_evidence():
    return ResearchRequest.model_validate_json(FIXTURE.read_bytes())


@pytest.mark.parametrize("output_mode", ["tool", "json"])
def test_cli_returns_structured_brief(bedrock_stub, monkeypatch, capsys, output_mode):
    monkeypatch.setattr(
        sys, "argv", ["agentops-research", "--ticker", "NVDA", "--output-mode", output_mode]
    )
    if output_mode == "tool":
        content = {
            "toolUse": {"toolUseId": "research-1", "name": "ResearchAnalysis", "input": ANALYSIS}
        }
        params = {**EXPECTED_REQUEST, "toolConfig": ANY}
    else:
        content = {"text": json.dumps(ANALYSIS)}
        params = EXPECTED_REQUEST

    bedrock_stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [content]}},
            "stopReason": "tool_use" if output_mode == "tool" else "end_turn",
            "usage": {"inputTokens": 20, "outputTokens": 20, "totalTokens": 40},
            "metrics": {"latencyMs": 10},
        },
        params,
    )

    assert main() == 0
    brief = ResearchBrief.model_validate_json(capsys.readouterr().out)
    assert brief.company == "NVIDIA Corporation"
    assert brief.ticker == "NVDA"
    assert brief.synthetic is True
    assert brief.key_findings[0].source_ids == ["demo-results"]
    assert brief.opportunities[0].source_ids == ["demo-results"]
    assert brief.risks[0].source_ids == ["demo-risks"]
    assert [source.source_id for source in brief.sources] == ["demo-results", "demo-risks"]
    assert brief.sources[0].url is None


def test_graph_preserves_source_metadata(request_with_evidence):
    payload = request_with_evidence.model_dump(mode="json")
    source = payload["excerpts"][0]
    source["url"] = "https://example.com/report"
    request = ResearchRequest.model_validate(payload)

    def analyze(messages):
        evidence = json.loads(messages[-1].text)
        assert evidence["excerpts"][0]["text"] == source["text"]
        assert "url" not in evidence["excerpts"][0]
        return ANALYSIS

    brief = build_research_graph(RunnableLambda(analyze)).invoke({"request": request})["brief"]
    assert str(brief.sources[0].url) == source["url"]
    assert brief.sources[0].document_date.isoformat() == source["document_date"]


def test_empty_evidence_skips_model(request_with_evidence):
    def unexpected_call(messages):
        pytest.fail("A request without excerpts must not call the model")

    request = request_with_evidence.model_copy(update={"excerpts": []})
    brief = build_research_graph(RunnableLambda(unexpected_call)).invoke({"request": request})[
        "brief"
    ]
    assert (
        not brief.key_findings and not brief.opportunities and not brief.risks and not brief.sources
    )
    assert "Insufficient evidence" in brief.missing_information[0]


def test_model_can_report_insufficient_evidence(request_with_evidence):
    analysis = {**ANALYSIS, "key_findings": [], "opportunities": [], "risks": []}
    result = build_research_graph(RunnableLambda(lambda messages: analysis)).invoke(
        {"request": request_with_evidence}
    )
    assert not result["brief"].sources
    assert "Insufficient evidence" in result["brief"].missing_information[0]


@pytest.mark.parametrize(
    ("source_ids", "error", "match"),
    [(["demo-results", "invented"], ValueError, "absent"), ([], ValidationError, "source_ids")],
)
def test_rejects_unsupported_citations(request_with_evidence, source_ids, error, match):
    analysis = {**ANALYSIS, "key_findings": [{"text": "A claim", "source_ids": source_ids}]}
    graph = build_research_graph(RunnableLambda(lambda messages: analysis))
    with pytest.raises(error, match=match):
        graph.invoke({"request": request_with_evidence})


def test_request_rejects_duplicate_ids(request_with_evidence):
    payload = request_with_evidence.model_dump()
    payload["excerpts"].append(payload["excerpts"][0])
    with pytest.raises(ValidationError, match="unique source_id"):
        ResearchRequest.model_validate(payload)


def test_cli_reports_failed_invocation(bedrock_stub, monkeypatch, capsys, caplog):
    monkeypatch.setattr(sys, "argv", ["agentops-research", "--ticker", "NVDA"])
    bedrock_stub.add_client_error(
        "converse", "AccessDeniedException", expected_params={**EXPECTED_REQUEST, "toolConfig": ANY}
    )
    assert main() == 1
    assert capsys.readouterr().out == ""
    assert "AccessDeniedException" in caplog.records[-1].message
