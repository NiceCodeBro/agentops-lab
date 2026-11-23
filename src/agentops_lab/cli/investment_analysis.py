"""Run the five-agent investment analysis on the indexed SEC filing."""

import argparse
import sys

from agentops_lab.bedrock import build_agent_reasoner, load_bedrock_model
from agentops_lab.models.workflow import ExecutionEvent, InvestmentRequest, WorkflowLimits
from agentops_lab.retrieval.filing_store import load_filing_store
from agentops_lab.retrieval.sec import COMPANY, TICKER
from agentops_lab.workflows.investment_analysis import build_investment_graph

DEFAULT_MODEL = "eu.amazon.nova-pro-v1:0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, type=str.upper, choices=[TICKER])
    parser.add_argument("--question", help="Focus the analysis on a question about the SEC filing.")
    defaults = WorkflowLimits()
    parser.add_argument("--max-model-calls", type=int, default=defaults.max_model_calls)
    parser.add_argument("--max-research-rounds", type=int, default=defaults.max_research_rounds)
    parser.add_argument("--max-queries", type=int, default=defaults.max_queries)
    parser.add_argument("--max-chunks", type=int, default=defaults.max_chunks)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--max-output-tokens", type=int, default=defaults.max_output_tokens)
    args = vars(parser.parse_args())
    ticker = args.pop("ticker")
    question = args.pop("question")
    try:
        limits = WorkflowLimits.model_validate(args)
        request = InvestmentRequest(company=COMPANY, ticker=ticker)
        if question is not None:
            request = InvestmentRequest(company=COMPANY, ticker=ticker, question=question)
        model = load_bedrock_model(
            max_tokens=limits.max_output_tokens, default_model_id=DEFAULT_MODEL
        )
        store = load_filing_store(embedding_attempts=1)
        graph = build_investment_graph(build_agent_reasoner(model), store, limits=limits)
    except Exception as error:
        print(
            f"Investment analysis configuration failed ({type(error).__name__}).", file=sys.stderr
        )
        return 1

    print(f"Model {model.model_id} in {model.region_name}", file=sys.stderr)
    report = None
    try:
        for chunk in graph.stream(
            {"request": request}, stream_mode=["custom", "values"], version="v2"
        ):
            if chunk["type"] == "custom":
                event = ExecutionEvent.model_validate(chunk["data"])
                print(f"[{event.agent}] {event.summary}", file=sys.stderr, flush=True)
            elif "final_report" in chunk["data"]:
                report = chunk["data"]["final_report"]
    except Exception as error:
        print(f"Investment analysis failed ({type(error).__name__}).", file=sys.stderr)
        return 1
    if report is None:
        print("Investment analysis ended without a report.", file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2))
    return int(report.status == "failed")
