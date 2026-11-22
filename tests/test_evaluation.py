import json
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import TypeAdapter

from agentops_lab.evaluation import EvaluationCase, evaluate_brief, run_evaluations
from agentops_lab.research import build_research_graph
from agentops_lab.research_models import CitedClaim, ResearchRequest


@pytest.fixture
def cases():
    raw_cases = Path("examples/evaluation.synthetic.json").read_bytes()
    return TypeAdapter(list[EvaluationCase]).validate_json(raw_cases)


def expected_analysis(case):
    evidence = {excerpt.source_id: excerpt.text for excerpt in case.request.excerpts}
    return {
        "key_findings": [
            {"text": topic.description, "source_ids": topic.source_ids}
            for topic in case.key_findings
        ],
        "risks": [
            {"text": topic.description, "source_ids": topic.source_ids} for topic in case.risks
        ],
        "opportunities": [
            {
                "text": topic.description,
                "source_ids": topic.source_ids,
                "evidence_quote": evidence[topic.source_ids[0]],
            }
            for topic in case.opportunities
        ],
        "missing_information": case.known_gaps,
    }


def test_runner_scores_all_five_cases_offline(cases, capsys):
    demo = ResearchRequest.model_validate_json(Path("examples/nvda.synthetic.json").read_bytes())
    assert len(cases) == len({case.id for case in cases}) == 5
    assert cases[0].request == demo
    cases_by_ticker = {case.request.ticker: case for case in cases}

    def analyze(messages):
        request = json.loads(messages[-1].text)
        return expected_analysis(cases_by_ticker[request["ticker"]])

    graph = build_research_graph(RunnableLambda(analyze))
    assert run_evaluations(cases, graph) == 0
    output = capsys.readouterr().out
    assert "5/5 cases without agent or citation errors" in output
    assert "key_findings: 6/6 expected topics across scored cases" in output
    assert "risks: 5/5 expected topics across scored cases" in output
    assert "opportunities: 5/5 expected topics across scored cases" in output
    assert "Claim to review" not in output
    assert "Potentially contradicted gap" not in output


def test_flags_unsupported_numbers_and_gaps_despite_valid_ids(cases):
    case = next(case for case in cases if case.id == "margin-guidance")
    response = {
        "key_findings": [
            {
                "text": "Margin guidance was 9% to 11%, with a 30% upside target.",
                "source_ids": ["aurora-outlook"],
            },
            {"text": "The company completed an acquisition.", "source_ids": ["aurora-outlook"]},
        ],
        "risks": [],
        "opportunities": [],
        "missing_information": ["What is the full-year operating margin guidance?"],
    }
    graph = build_research_graph(RunnableLambda(lambda messages: response))
    brief = graph.invoke({"request": case.request})["brief"]
    result = evaluate_brief(case, brief)

    assert len(result.coverage["key_findings"].covered) == 1
    assert len(result.coverage["risks"].missed) == 1
    assert not result.citation_errors
    assert "numbers absent from cited text: 30" in result.claim_reviews[0]
    assert "no expected topic matched" in result.claim_reviews[1]
    assert "9% to 11% [aurora-outlook]" in result.gap_reviews[0]
    assert result.omitted_gaps == ["bookings forecast"]


def test_audits_cited_ids_and_expected_source_mismatches(cases):
    case = cases[0]
    analysis = expected_analysis(case)
    graph = build_research_graph(RunnableLambda(lambda messages: analysis))
    brief = graph.invoke({"request": case.request})["brief"]
    brief.risks = [
        CitedClaim(text=case.risks[0].description, source_ids=["demo-results", "invented"])
    ]
    result = evaluate_brief(case, brief)

    assert result.citation_errors == [
        "Unknown source IDs: invented",
        "Cited IDs missing from the source list: invented",
    ]
    assert "cites different sources" in result.claim_reviews[0]


def test_review_flags_disclosed_nvda_growth_rate_and_accepts_genuine_gap(cases):
    case = cases[0]
    response = expected_analysis(case)
    response["missing_information"] = [
        "missing specific financial values and units for revenue growth",
        "missing full-year margin outlook",
    ]
    graph = build_research_graph(RunnableLambda(lambda messages: response))
    brief = graph.invoke({"request": case.request})["brief"]
    result = evaluate_brief(case, brief)

    assert len(result.gap_reviews) == 1
    assert "12% year-over-year growth rate" in result.gap_reviews[0]
    assert not result.omitted_gaps


def test_runner_continues_after_graph_rejects_invalid_citation(cases, capsys):
    first = expected_analysis(cases[0])
    first["risks"][0]["source_ids"] = ["invented"]
    responses = iter([first, expected_analysis(cases[1])])
    graph = build_research_graph(RunnableLambda(lambda messages: next(responses)))

    assert run_evaluations(cases[:2], graph) == 1
    output = capsys.readouterr().out
    assert "Research output cites source IDs absent" in output
    assert "1/2 cases without agent or citation errors" in output
