import json
import sys
from threading import Barrier
from unittest.mock import Mock

import pytest

from agentops_lab.agents.judge import match_quote
from agentops_lab.cli import investment_analysis as cli
from agentops_lab.models.workflow import (
    Arguments,
    EvidenceQuote,
    ExecutionEvent,
    InvestmentReport,
    InvestmentRequest,
    JudgeReview,
    ReasoningReply,
    ResearchFindings,
    ResearchPlan,
    WorkflowLimits,
)
from agentops_lab.retrieval.filing_store import FilingHit
from agentops_lab.retrieval.sec import PILOT_FILING, create_chunks
from agentops_lab.workflows.investment_analysis import build_investment_graph

GROWTH = "Data Center revenue grew 55 percent year over year in fiscal 2026."
EXPORT = "Export restrictions reduced sales of the affected products in China."
LICENSE = "The filing reports a charge of $4.5 billion related to H20 export controls."
TASKS = [
    {
        "question": "How did Data Center perform?",
        "query": "Data Center fiscal 2026 revenue year over year growth",
        "reason": "Data Center is material to operating performance.",
    },
    {
        "question": "How did export controls affect sales?",
        "query": "China export restrictions reduced product sales",
        "reason": "Licensing changes can constrain addressable markets.",
    },
]
FOLLOW_UP = {
    "question": "Did export controls create an H20 inventory charge?",
    "query": "H20 export licensing inventory purchase obligation charge",
    "reason": "Quantifies the operating impact of export controls.",
    "evidence_needed": "The charge and amount disclosed in the filing.",
    "affected_claim_ids": ["bear-1-argument-1"],
}
REQUEST = InvestmentRequest(company="NVIDIA Corporation", ticker="NVDA")


def claim(text, source, quote, *, classification="fact"):
    return {
        "text": text,
        "classification": classification,
        "source_ids": [source],
        "quotes": [{"source_id": source, "excerpt": quote}],
        "assumptions": [],
    }


class Scenario:
    def __init__(self):
        self.license, self.export, self.growth = create_chunks(
            PILOT_FILING,
            {"1": [LICENSE], "1A": [EXPORT], "7": [GROWTH]},
        )
        self.store = Mock()
        self.store.search_filings_batch.side_effect = self.search
        self.tasks = TASKS
        self.follow_ups = []
        self.second_follow_ups = []
        self.new_evidence = True
        self.bad_citation = None
        self.include_reversed_claim = True
        self.fail_schema = None
        self.missing_review = False
        self.resolve = True
        self.barrier = None
        self.payloads = {}

    def search(self, queries, ticker, top_k, *, on_embedding):
        assert ticker == "NVDA"
        assert top_k == 3
        indexed = {
            TASKS[0]["query"]: [self.growth, self.growth],
            TASKS[1]["query"]: [self.export],
            FOLLOW_UP["query"]: [self.export, self.license] if self.new_evidence else [self.export],
        }
        results = []
        for query in queries:
            on_embedding(12, 10, True)
            results.append([FilingHit(excerpt, 0.4) for excerpt in indexed[query]])
        return results

    def reasoner(self, schema, messages):
        payload = json.loads(messages[-1].text)
        name = schema.__name__
        if schema is Arguments:
            name = "bull" if "distinct bullish" in messages[0].text else "bear"
        self.payloads.setdefault(name, []).append(payload)
        if schema is self.fail_schema:
            raise RuntimeError("private request data")
        growth_id = self.growth.source_id
        export_id = self.export.source_id
        license_id = self.license.source_id
        match schema:
            case _ if schema is ResearchPlan:
                output = {"tasks": self.tasks}
            case _ if schema is ResearchFindings:
                if len(self.payloads[name]) == 1:
                    output = {
                        "findings": [claim(GROWTH, growth_id, GROWTH)],
                        "risks": [claim(EXPORT, export_id, EXPORT)],
                    }
                else:
                    output = {"findings": [], "risks": [claim(LICENSE, license_id, LICENSE)]}
            case _ if schema is Arguments:
                if self.barrier:
                    self.barrier.wait(timeout=5)
                if name == "bull":
                    claims = [claim(GROWTH, growth_id, GROWTH)]
                    if self.include_reversed_claim:
                        claims += [
                            claim(
                                "Export restrictions create a sales growth opportunity in China.",
                                export_id,
                                EXPORT,
                                classification="interpretation",
                            ),
                            claim("Annual revenue doubled.", growth_id, GROWTH),
                        ]
                    if self.bad_citation == "id":
                        claims[0]["source_ids"] = ["unknown"]
                    if self.bad_citation == "quote":
                        claims[0]["quotes"][0]["excerpt"] = (
                            "Made up financial language for testing."
                        )
                else:
                    claims = [claim(EXPORT, export_id, EXPORT)]
                output = {"arguments": claims}
            case _ if schema is JudgeReview:
                reviews = []
                for item in payload["claims"]:
                    text = item["text"]
                    if text.startswith("Export restrictions create"):
                        support = "risk_as_upside"
                        reason = "The cited restriction reduced sales and supplies no upside."
                    elif text == "Annual revenue doubled.":
                        support = "unsupported"
                        reason = "The excerpt gives segment growth, not companywide revenue growth."
                    else:
                        support = "supported"
                        reason = "The filing excerpt directly documents the claim."
                    reviews.append(
                        {"claim_id": item["claim_id"], "support": support, "reason": reason}
                    )
                if self.missing_review:
                    reviews.pop()
                second = bool(payload["prior_gaps"])
                resolutions = (
                    [
                        {
                            "gap_id": payload["prior_gaps"][0]["gap_id"],
                            "resolved": self.resolve,
                            "reason": (
                                "The filing identifies the size of the H20 charge."
                                if self.resolve
                                else "The excerpt did not resolve the question."
                            ),
                            "quotes": [{"source_id": license_id, "excerpt": LICENSE}],
                        }
                    ]
                    if second
                    else []
                )
                output = {
                    "reviews": reviews,
                    "follow_ups": self.second_follow_ups if second else self.follow_ups,
                    "resolutions": resolutions,
                }
            case _:
                raise AssertionError(schema)
        return ReasoningReply(output=output, input_tokens=100, output_tokens=25)

    def graph(self, limits=None):
        return build_investment_graph(self.reasoner, self.store, limits=limits)

    def run(self, limits=None):
        return self.graph(limits).invoke({"request": REQUEST})["final_report"]


@pytest.fixture
def scenario():
    return Scenario()


def test_prioritized_planner_queries_drive_vector_search_and_preserve_provenance(scenario):
    scenario.tasks = [
        TASKS[1],
        {**TASKS[1], "query": f"{TASKS[1]['query'].upper()}?"},
        TASKS[0],
    ]
    report = scenario.run()
    queries = [task["query"] for task in reversed(TASKS)]
    assert [task.query for task in report.plan.tasks] == queries
    assert all(task.question and task.reason for task in report.plan.tasks)
    assert scenario.store.search_filings_batch.call_args.args == (queries, "NVDA", 3)
    assert [trace.query for trace in report.retrieval] == queries
    assert [source.source_id for source in report.sources] == [
        scenario.export.source_id,
        scenario.growth.source_id,
    ]
    for source in report.sources:
        assert source.document_date == PILOT_FILING.filing_date
        assert "sec.gov" in str(source.url)
        assert source.filing.accession_number == PILOT_FILING.accession_number
        assert source.filing.section in {"MD&A", "Risk Factors"}
    evidence = scenario.payloads["ResearchFindings"][0]["evidence"]
    assert [item["source_id"] for item in evidence] == [
        source.source_id for source in report.sources
    ]


def test_parallel_branches_wait_at_fan_in_and_merge_report_and_stream_events(scenario):
    scenario.barrier = Barrier(2)
    events = []
    for part in scenario.graph().stream(
        {"request": REQUEST}, stream_mode=["custom", "values"], version="v2"
    ):
        if part["type"] == "custom":
            events.append(ExecutionEvent.model_validate(part["data"]))
        elif "final_report" in part["data"]:
            report = part["data"]["final_report"]
    assert report.status == "completed"
    assert report.bullish_arguments and report.bearish_arguments
    assert report.material_risks and report.key_findings
    assert events[0].event_type == "workflow_started"
    assert events[-1].event_type == "workflow_completed"
    assert {event.run_id for event in events} == {report.run_id}
    assert set(event.model_dump_json() for event in events) == set(
        event.model_dump_json() for event in report.events
    )
    completed = [event.agent for event in events if event.event_type == "agent_completed"]
    assert completed[:2] == ["planner", "research"]
    assert set(completed[2:4]) == {"bull", "bear"}
    assert completed[-1] == "judge"
    judge_started = next(
        index
        for index, event in enumerate(events)
        if event.agent == "judge" and event.event_type == "agent_started"
    )
    assert {
        event.agent for event in events[:judge_started] if event.event_type == "agent_completed"
    }
    assert all(event.agent != "judge" for event in events[:judge_started])
    assert scenario.payloads["bull"][0]["evidence"] == scenario.payloads["bear"][0]["evidence"]
    assert InvestmentReport.model_validate_json(report.model_dump_json()) == report


def test_judge_accepts_sourced_claims_and_rejects_unsupported_and_reversed_claims(scenario):
    report = scenario.run()
    assert report.bullish_arguments[0].decision.support == "supported"
    rejected = {item.decision.support: item for item in report.rejected_or_uncertain_claims}
    assert set(rejected) == {"risk_as_upside", "unsupported"}
    reversal = rejected["risk_as_upside"]
    assert reversal.decision.status == "rejected"
    assert reversal.claim.source_ids == [scenario.export.source_id]
    assert reversal.claim.quotes[0].excerpt == EXPORT
    assert any(event.event_type == "claim_rejected" for event in report.events)
    assert report.research_rounds == 1
    assert len(report.model_calls) == 7
    assert scenario.store.search_filings_batch.call_count == 1


@pytest.mark.parametrize("invalid", ["id", "quote"])
def test_invalid_ids_and_fabricated_quotes_are_rejected_before_semantic_review(scenario, invalid):
    scenario.bad_citation = invalid
    report = scenario.run()
    assert not report.bullish_arguments
    rejected = report.rejected_or_uncertain_claims
    assert rejected[0].decision.support == "invalid_citation"
    assert all(
        claim["claim_id"] != rejected[0].claim.claim_id
        for claim in scenario.payloads["JudgeReview"][0]["claims"]
    )


def test_judge_follow_up_searches_targeted_query_and_reassesses_only_affected_claims(scenario):
    scenario.follow_ups = [FOLLOW_UP]
    report = scenario.run()
    assert report.status == "completed"
    assert report.research_rounds == 2
    follow_up_call = scenario.store.search_filings_batch.call_args_list[1]
    assert follow_up_call.args[0] == [FOLLOW_UP["query"]]
    assert report.retrieval[-1].selected_source_ids == [scenario.license.source_id]
    assert report.evidence_gaps[0].status == "resolved"
    assert report.evidence_gaps[0].resolution_quotes[0].source_id == scenario.license.source_id
    assert {"bear-1-argument-1", "research-2-risk-1"} == {
        claim["claim_id"] for claim in scenario.payloads["JudgeReview"][1]["claims"]
    }
    assert report.material_risks[-1].claim.text == LICENSE
    assert report.bullish_arguments[0].decision.status == "accepted"
    assert len(scenario.payloads["bull"]) == len(scenario.payloads["bear"]) == 1
    assert len(report.model_calls) == 10
    assert sum(event.event_type == "evidence_requested" for event in report.events) == 1


def test_follow_up_cap_leaves_unresolved_gaps_and_disallows_a_third_round(scenario):
    scenario.follow_ups = [FOLLOW_UP]
    scenario.second_follow_ups = [
        {
            **FOLLOW_UP,
            "question": "Did the company later reverse the charge?",
            "query": "H20 inventory charge reversal",
        }
    ]
    scenario.resolve = False
    report = scenario.run()
    assert report.research_rounds == scenario.store.search_filings_batch.call_count == 2
    assert [gap.status for gap in report.evidence_gaps] == ["unresolved", "unresolved"]
    assert "2 evidence gaps remain unresolved" in report.judge_assessment


@pytest.mark.parametrize(
    "limits",
    [
        WorkflowLimits(max_research_rounds=1),
        WorkflowLimits(max_model_calls=7),
        WorkflowLimits(max_queries=2),
        WorkflowLimits(max_chunks=2),
    ],
)
def test_configured_limits_leave_unexecuted_follow_up_explicit(scenario, limits):
    scenario.follow_ups = [FOLLOW_UP]
    report = scenario.run(limits)
    assert report.research_rounds == 1
    assert len(report.model_calls) <= limits.max_model_calls
    assert len(report.sources) <= limits.max_chunks
    assert len(report.retrieval) <= limits.max_queries
    assert report.evidence_gaps[0].status == "unresolved"
    assert not any(event.event_type == "evidence_requested" for event in report.events)


def test_follow_up_with_only_previous_hits_skips_duplicate_reasoning(scenario):
    scenario.follow_ups = [FOLLOW_UP]
    scenario.new_evidence = False
    report = scenario.run()
    assert report.research_rounds == 2
    assert report.evidence_gaps[0].status == "unresolved"
    assert len(scenario.payloads["JudgeReview"]) == len(scenario.payloads["ResearchFindings"]) == 1
    assert len(report.model_calls) == 8


def test_duplicate_and_unanswerable_follow_up_queries_are_not_executed(scenario):
    scenario.follow_ups = [
        {**FOLLOW_UP, "query": TASKS[0]["query"]},
        {**FOLLOW_UP, "question": "What counterparties were affected?", "query": None},
    ]
    report = scenario.run()
    assert report.research_rounds == 1
    assert [gap.status for gap in report.evidence_gaps] == ["unresolved", "unresolved"]


def test_second_review_does_not_duplicate_prior_gap(scenario):
    scenario.follow_ups = [FOLLOW_UP]
    scenario.second_follow_ups = [FOLLOW_UP]
    scenario.resolve = False
    report = scenario.run()
    assert [gap.status for gap in report.evidence_gaps] == ["unresolved"]


def test_quote_verification_removes_a_terminal_period_without_rewriting_evidence(scenario):
    source = scenario.growth
    quote = EvidenceQuote(
        source_id=source.source_id, excerpt="Data Center revenue grew 55 percent."
    )
    sources = {source.source_id: source}
    assert match_quote(quote, sources).excerpt == "Data Center revenue grew 55 percent"
    inaccurate = quote.model_copy(update={"excerpt": "Data Center revenue grew 50 percent."})
    assert match_quote(inaccurate, sources) is None


def test_empty_evidence_only_runs_planner_and_retrieval(scenario):
    scenario.store.search_filings_batch.side_effect = None
    scenario.store.search_filings_batch.return_value = [[], []]
    report = scenario.run()
    assert report.status == "insufficient_evidence"
    assert scenario.payloads.keys() == {"ResearchPlan"}
    assert not report.bullish_arguments
    assert not report.sources
    assert [event.agent for event in report.events if event.event_type == "agent_started"] == [
        "planner",
        "research",
    ]


@pytest.mark.parametrize("schema", [ResearchPlan, ResearchFindings, Arguments, JudgeReview])
def test_agent_model_failure_never_presents_a_completed_report(scenario, schema):
    scenario.fail_schema = schema
    report = scenario.run()
    assert report.status == "failed"
    assert report.events[-1].event_type == "workflow_failed"
    assert not any(event.event_type == "workflow_completed" for event in report.events)
    assert report.errors
    assert any(not call.succeeded for call in report.model_calls)
    assert "private request data" not in report.model_dump_json()


def test_judge_omission_yields_uncertain_claim_and_failed_run(scenario):
    scenario.missing_review = True
    report = scenario.run()
    assert report.status == "failed"
    assert report.rejected_or_uncertain_claims[-1].decision.status == "uncertain"


def test_cli_streams_public_progress_and_writes_valid_report(scenario, monkeypatch, capsys):
    model = Mock(model_id="eu.amazon.nova-pro-v1:0", region_name="eu-central-1")
    load_model = Mock(return_value=model)
    load_store = Mock(return_value=scenario.store)
    monkeypatch.setattr(cli, "load_bedrock_model", load_model)
    monkeypatch.setattr(cli, "load_filing_store", load_store)
    monkeypatch.setattr(cli, "build_agent_reasoner", lambda _: scenario.reasoner)
    monkeypatch.setattr(sys, "argv", ["agentops-analyze", "--ticker", "nvda"])
    assert cli.main() == 0
    result = capsys.readouterr()
    assert "[planner]" in result.err and "[judge]" in result.err
    assert InvestmentReport.model_validate_json(result.out).status == "completed"
    load_model.assert_called_once_with(max_tokens=1600, default_model_id=model.model_id)
    load_store.assert_called_once_with(embedding_attempts=1)
