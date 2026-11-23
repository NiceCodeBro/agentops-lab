"""Evaluate standalone research against synthetic cases."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from agentops_lab.bedrock import build_structured_model, load_bedrock_model
from agentops_lab.evaluation.research import EvaluationCase, run_evaluations
from agentops_lab.models.research import ResearchAnalysis
from agentops_lab.workflows.research import build_research_graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("examples/evaluation.synthetic.json"))
    parser.add_argument("--case", action="append", help="Select a case ID. Repeat to select more.")
    args = parser.parse_args()

    raw_cases = args.cases.read_bytes()
    cases = TypeAdapter(list[EvaluationCase]).validate_json(raw_cases)
    if args.case:
        unknown = set(args.case) - {case.id for case in cases}
        if unknown:
            parser.error(f"Unknown case IDs: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in args.case]

    model = load_bedrock_model(max_tokens=768)
    analyze = build_structured_model(model, ResearchAnalysis, output_mode="tool")
    graph = build_research_graph(analyze)
    print(f"Model: {model.model_id} ({model.region_name})")
    return run_evaluations(cases, graph)
