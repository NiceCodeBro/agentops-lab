import json
import sys
from unittest.mock import Mock

from botocore.stub import ANY
from langchain_core.runnables import RunnableLambda

from agentops_lab.filing_store import FilingHit, FilingStore
from agentops_lab.research import build_research_graph
from agentops_lab.research_cli import main
from agentops_lab.sec import PILOT_FILING, create_chunks
from agentops_lab.sec_cli import smoke
from agentops_lab.sec_research import EVIDENCE_TOPICS, build_sec_request


def make_filing_hits():
    sections = {
        "1": ["The two business segments sell different products."],
        "1A": ["A supplier outage or export restrictions could disrupt operations."],
        "7": ["This paragraph discusses an unrelated list of office locations."],
    }
    excerpts = create_chunks(PILOT_FILING, sections)
    return [FilingHit(excerpt, 0.4) for excerpt in excerpts]


def test_sec_request_deduplicates_and_excludes_off_topic_hits():
    business, risk, irrelevant = make_filing_hits()
    store = Mock(spec=FilingStore)
    store.search_filings.side_effect = [[business, irrelevant], [risk, irrelevant], [risk]]

    request = build_sec_request(store, " nvda ")

    searches = [(call.args, call.kwargs) for call in store.search_filings.call_args_list]
    assert searches == [((topic.query, "NVDA"), {"top_k": 3}) for topic in EVIDENCE_TOPICS]
    assert request.excerpts == [business.excerpt, risk.excerpt]
    assert request.synthetic is False


def test_sec_cli_runs_existing_research_graph_with_nova_pro_default(
    bedrock_stub,
    monkeypatch,
    capsys,
):
    business, risk, irrelevant = make_filing_hits()
    store = Mock(spec=FilingStore)
    store.search_filings.side_effect = [[business, irrelevant], [risk], [risk]]
    monkeypatch.setattr("agentops_lab.research_cli.load_filing_store", lambda: store)
    monkeypatch.delenv("BEDROCK_MODEL_ID")
    monkeypatch.setattr(sys, "argv", ["agentops-research", "--sec", "--ticker", "NVDA"])

    analysis = {
        "key_findings": [
            {"text": business.excerpt.text, "source_ids": [business.excerpt.source_id]}
        ],
        "risks": [{"text": risk.excerpt.text, "source_ids": [risk.excerpt.source_id]}],
        "opportunities": [],
        "missing_information": [],
    }
    message = {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": "research-1", "name": "ResearchAnalysis", "input": analysis}}
        ],
    }
    bedrock_stub.add_response(
        "converse",
        {
            "output": {"message": message},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 20, "outputTokens": 20, "totalTokens": 40},
            "metrics": {"latencyMs": 10},
        },
        {
            "modelId": "eu.amazon.nova-pro-v1:0",
            "messages": ANY,
            "system": ANY,
            "inferenceConfig": {"maxTokens": 768, "temperature": 0},
            "toolConfig": ANY,
        },
    )

    assert main() == 0
    brief = json.loads(capsys.readouterr().out)
    reference = brief["sources"][1]
    assert reference["source_id"] == risk.excerpt.source_id
    assert reference["document_date"] == "2026-02-25"
    assert reference["url"] == PILOT_FILING.url
    assert reference["filing"]["section"] == "Risk Factors"


def test_no_relevant_sec_matches_skips_reasoning():
    store = Mock(spec=FilingStore)
    irrelevant = make_filing_hits()[2]
    store.search_filings.return_value = [irrelevant]
    request = build_sec_request(store, "NVDA")
    analyze = Mock()
    graph = build_research_graph(RunnableLambda(lambda messages: analyze(messages)))
    output = graph.invoke({"request": request})

    analyze.assert_not_called()
    assert not output["brief"].sources
    assert "Insufficient evidence" in output["brief"].missing_information[0]


def test_smoke_flags_wrong_topic_in_expected_section(monkeypatch, capsys):
    business, risk, irrelevant = make_filing_hits()
    store = Mock(spec=FilingStore)
    store.search_filings.side_effect = [[business], [irrelevant], [risk]]
    monkeypatch.setattr("agentops_lab.sec_cli.load_filing_store", lambda: store)
    monkeypatch.setattr(sys, "argv", ["agentops-sec-smoke"])

    assert smoke() == 1
    output = capsys.readouterr().out
    assert "MISS | Supply chain" in output
    assert "2/3 queries returned a topical excerpt" in output
