"""Evaluate synthetic research cases with deterministic, review-oriented checks."""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.exceptions import OutputParserException
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, TypeAdapter, model_validator

from agentops_lab.bedrock import build_structured_model, load_bedrock_model
from agentops_lab.research import build_research_graph
from agentops_lab.research_models import (
    NonEmptyText,
    ResearchAnalysis,
    ResearchBrief,
    ResearchRequest,
)

SECTIONS = ("key_findings", "risks", "opportunities")


class ExpectedTopic(BaseModel):
    description: NonEmptyText
    # All groups must match. Within a group, any phrase is sufficient.
    phrases: list[list[NonEmptyText]] = Field(min_length=1)
    source_ids: list[NonEmptyText] = Field(min_length=1)


class EvaluationCase(BaseModel):
    id: NonEmptyText
    request: ResearchRequest
    key_findings: list[ExpectedTopic]
    risks: list[ExpectedTopic]
    opportunities: list[ExpectedTopic]
    disclosed: list[ExpectedTopic] = Field(default_factory=list)
    known_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_fixture(self) -> Self:
        if not self.request.synthetic:
            raise ValueError("Evaluation cases must be labeled synthetic.")
        source_ids = {excerpt.source_id for excerpt in self.request.excerpts}
        for topic in self.key_findings + self.risks + self.opportunities + self.disclosed:
            if set(topic.source_ids) - source_ids:
                raise ValueError(f"Unknown expected source ID in case {self.id}.")
            if any(not alternatives for alternatives in topic.phrases):
                raise ValueError(f"Empty phrase group in case {self.id}.")
        return self


@dataclass
class SectionCoverage:
    covered: list[str]
    missed: list[str]


@dataclass
class CaseResult:
    coverage: dict[str, SectionCoverage] = field(default_factory=dict)
    citation_errors: list[str] = field(default_factory=list)
    claim_reviews: list[str] = field(default_factory=list)
    gap_reviews: list[str] = field(default_factory=list)
    omitted_gaps: list[str] = field(default_factory=list)


def matches_topic(text: str, topic: ExpectedTopic) -> bool:
    words = " " + " ".join(re.findall(r"\w+", text.casefold())) + " "
    for alternatives in topic.phrases:
        phrases = [" ".join(re.findall(r"\w+", phrase.casefold())) for phrase in alternatives]
        if not any(f" {phrase} " in words for phrase in phrases):
            return False
    return True


def evaluate_brief(case: EvaluationCase, brief: ResearchBrief) -> CaseResult:
    result = CaseResult()
    excerpts = {excerpt.source_id: excerpt.text for excerpt in case.request.excerpts}
    reported_ids = {source.source_id for source in brief.sources}
    cited_ids: set[str] = set()

    sections = (
        ("key_findings", case.key_findings, brief.key_findings),
        ("risks", case.risks, brief.risks),
        ("opportunities", case.opportunities, brief.opportunities),
    )
    for name, expected, claims in sections:
        covered = set()
        for claim in claims:
            claim_ids = set(claim.source_ids)
            cited_ids.update(claim_ids)
            matching = [topic for topic in expected if matches_topic(claim.text, topic)]
            covered.update(topic.description for topic in matching)

            reasons = []
            if not matching:
                reasons.append("no expected topic matched")
            elif not any(set(topic.source_ids) <= claim_ids for topic in matching):
                reasons.append("cites different sources from the expected topic")

            cited_text = " ".join(excerpts[source_id] for source_id in claim_ids & excerpts.keys())
            claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim.text.replace(",", "")))
            cited_numbers = set(re.findall(r"\d+(?:\.\d+)?", cited_text.replace(",", "")))
            extra_numbers = claim_numbers - cited_numbers
            if extra_numbers:
                reasons.append(
                    f"numbers absent from cited text: {', '.join(sorted(extra_numbers))}"
                )
            if reasons:
                reason_text = ", ".join(reasons)
                refs = ", ".join(claim.source_ids)
                result.claim_reviews.append(f"{name}: {claim.text} [{refs}] ({reason_text})")

        result.coverage[name] = SectionCoverage(
            covered=[topic.description for topic in expected if topic.description in covered],
            missed=[topic.description for topic in expected if topic.description not in covered],
        )

    invalid_ids = (cited_ids | reported_ids) - excerpts.keys()
    if invalid_ids:
        result.citation_errors.append(f"Unknown source IDs: {', '.join(sorted(invalid_ids))}")
    absent_references = cited_ids - reported_ids
    if absent_references:
        missing = ", ".join(sorted(absent_references))
        result.citation_errors.append(f"Cited IDs missing from the source list: {missing}")

    for gap in brief.missing_information:
        for topic in case.disclosed:
            if matches_topic(gap, topic):
                refs = ", ".join(topic.source_ids)
                result.gap_reviews.append(f"{gap} | Evidence: {topic.description} [{refs}]")

    gaps_text = " ".join(brief.missing_information).casefold()
    result.omitted_gaps = [gap for gap in case.known_gaps if gap.casefold() not in gaps_text]
    return result


def run_evaluations(cases: list[EvaluationCase], graph: CompiledStateGraph) -> int:
    totals = {section: [0, 0] for section in SECTIONS}
    failures = 0
    print("Synthetic research evaluation. Topic coverage and flags are heuristics.")
    print(
        "Valid citation IDs establish identity only. Review factual support against the excerpts."
    )

    for case in cases:
        print(f"\n{case.id} ({case.request.ticker}, synthetic)")
        try:
            output = graph.invoke({"request": case.request})
        except (BotoCoreError, ClientError, OutputParserException, ValueError) as error:
            print(f"  Agent failed ({type(error).__name__}): {error}")
            failures += 1
            continue

        brief = output["brief"]
        result = evaluate_brief(case, brief)
        for section, coverage in result.coverage.items():
            covered = len(coverage.covered)
            expected = covered + len(coverage.missed)
            totals[section][0] += covered
            totals[section][1] += expected
            print(f"  {section}: {covered}/{expected} expected topics")
            for topic in coverage.covered:
                print(f"    Covered: {topic}")
            for topic in coverage.missed:
                print(f"    Missed: {topic}")
            for claim in getattr(brief, section):
                refs = ", ".join(claim.source_ids)
                print(f"    Agent: {claim.text} [{refs}]")

        print(f"  Citation ID errors: {len(result.citation_errors)}")
        if result.citation_errors:
            failures += 1
        for error in result.citation_errors:
            print(f"    {error}")
        for claim in result.claim_reviews:
            print(f"  Claim to review: {claim}")
        for gap in result.gap_reviews:
            print(f"  Potentially contradicted gap: {gap}")
        for gap in result.omitted_gaps:
            print(f"  Expected gap not matched: {gap}")
        for gap in brief.missing_information:
            print(f"  Reported gap: {gap}")

    print(f"\nSummary: {len(cases) - failures}/{len(cases)} cases without agent or citation errors")
    for section, (covered, expected) in totals.items():
        print(f"  {section}: {covered}/{expected} expected topics across scored cases")
    return int(failures > 0)


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
