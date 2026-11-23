"""Schemas for investment analysis, its public events, and its shared state."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agentops_lab.models.research import DocumentExcerpt, NonEmptyText, SourceReference

type ShortText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)
]
type Agent = Literal["planner", "research", "bull", "bear", "judge", "workflow"]
type ClaimRole = Literal["research", "bull", "bear"]
type EventType = Literal[
    "workflow_started",
    "agent_started",
    "research_completed",
    "agent_completed",
    "agent_failed",
    "handoff",
    "evidence_requested",
    "claim_accepted",
    "claim_rejected",
    "workflow_completed",
    "workflow_failed",
]
type Support = Literal[
    "supported", "partial", "unsupported", "contradicted", "risk_as_upside", "invalid_citation"
]


@dataclass(frozen=True)
class ReasoningReply:
    output: BaseModel | dict | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    validation_codes: tuple[str, ...] = ()


type Reasoner = Callable[[type[BaseModel], list[BaseMessage]], ReasoningReply]


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class InvestmentRequest(WorkflowModel):
    company: NonEmptyText
    ticker: NonEmptyText
    question: ShortText = (
        "Assess business performance and investment strengths and risks, including supply, "
        "customer concentration, and export controls."
    )


class WorkflowLimits(WorkflowModel):
    max_model_calls: int = Field(default=13, ge=6)
    max_research_rounds: int = Field(default=2, ge=1, le=2)
    max_queries: int = Field(default=6, ge=1, le=10)
    max_chunks: int = Field(default=8, ge=1, le=16)
    top_k: int = Field(default=3, ge=1, le=10)
    max_output_tokens: int = Field(default=1600, ge=256, le=4096)


class ResearchTask(WorkflowModel):
    question: ShortText
    query: ShortText = Field(description="An SEC filing semantic-search query, not a URL.")
    reason: ShortText


class ResearchPlan(WorkflowModel):
    tasks: list[ResearchTask] = Field(min_length=1, max_length=10)


class EvidenceQuote(WorkflowModel):
    source_id: NonEmptyText
    excerpt: ShortText = Field(description="A verbatim excerpt from the specified source.")


class Claim(WorkflowModel):
    text: ShortText
    classification: Literal["fact", "interpretation", "uncertainty"]
    source_ids: list[NonEmptyText] = Field(min_length=1, max_length=3)
    quotes: list[EvidenceQuote] = Field(min_length=1, max_length=3)
    assumptions: list[ShortText] = Field(default_factory=list, max_length=3)


class ResearchFindings(WorkflowModel):
    findings: list[Claim] = Field(max_length=3)
    risks: list[Claim] = Field(max_length=3)


class Arguments(WorkflowModel):
    arguments: list[Claim] = Field(max_length=3)


class CandidateClaim(Claim):
    claim_id: NonEmptyText
    agent: ClaimRole
    category: Literal["finding", "risk", "argument"]
    research_round: int


class ClaimReview(WorkflowModel):
    claim_id: NonEmptyText
    support: Literal["supported", "partial", "unsupported", "contradicted", "risk_as_upside"]
    reason: ShortText


class ClaimDecision(WorkflowModel):
    claim_id: NonEmptyText
    status: Literal["accepted", "rejected", "uncertain"]
    support: Support
    reason: ShortText


class FollowUpQuestion(WorkflowModel):
    question: ShortText
    reason: ShortText
    evidence_needed: ShortText
    query: ShortText | None
    affected_claim_ids: list[NonEmptyText] = Field(max_length=6)


class GapResolution(WorkflowModel):
    gap_id: NonEmptyText
    resolved: bool
    reason: ShortText
    quotes: list[EvidenceQuote] = Field(max_length=2)


class JudgeReview(WorkflowModel):
    reviews: list[ClaimReview] = Field(max_length=24)
    follow_ups: list[FollowUpQuestion] = Field(default_factory=list, max_length=2)
    resolutions: list[GapResolution] = Field(default_factory=list, max_length=2)


class EvidenceGap(FollowUpQuestion):
    gap_id: NonEmptyText
    status: Literal["pending", "resolved", "unresolved"]
    note: ShortText
    resolution_quotes: list[EvidenceQuote] = Field(default_factory=list)


class RetrievalTrace(WorkflowModel):
    research_round: int
    query: NonEmptyText
    candidate_source_ids: list[str]
    selected_source_ids: list[str]


class ModelCall(WorkflowModel):
    agent: Agent
    service: Literal["reasoning", "embedding"]
    elapsed_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    succeeded: bool


class ExecutionEvent(WorkflowModel):
    run_id: NonEmptyText
    agent: Agent
    event_type: EventType
    timestamp: datetime
    summary: ShortText
    source_ids: list[str] = Field(default_factory=list)


class WorkflowIssue(WorkflowModel):
    agent: Agent
    message: ShortText


class ReviewedClaim(WorkflowModel):
    claim: CandidateClaim
    decision: ClaimDecision


class InvestmentReport(WorkflowModel):
    run_id: NonEmptyText
    company: NonEmptyText
    ticker: NonEmptyText
    question: ShortText
    status: Literal["completed", "insufficient_evidence", "failed"]
    research_summary: ShortText
    key_findings: list[ReviewedClaim]
    bullish_arguments: list[ReviewedClaim]
    bearish_arguments: list[ReviewedClaim]
    material_risks: list[ReviewedClaim]
    judge_assessment: ShortText
    rejected_or_uncertain_claims: list[ReviewedClaim]
    evidence_gaps: list[EvidenceGap]
    sources: list[SourceReference]
    plan: ResearchPlan | None
    research_rounds: int
    retrieval: list[RetrievalTrace]
    model_calls: list[ModelCall]
    elapsed_ms: int
    events: list[ExecutionEvent]
    errors: list[WorkflowIssue]


class InvestmentInput(TypedDict):
    request: InvestmentRequest


class InvestmentUpdates(TypedDict, total=False):
    run_id: str
    started_at: float
    plan: ResearchPlan
    research_rounds: int
    evidence: list[DocumentExcerpt]
    research_claims: list[CandidateClaim]
    bull_claims: list[CandidateClaim]
    bear_claims: list[CandidateClaim]
    decisions: dict[str, ClaimDecision]
    gaps: list[EvidenceGap]
    retrieval: Annotated[list[RetrievalTrace], add]
    model_calls: Annotated[list[ModelCall], add]
    events: Annotated[list[ExecutionEvent], add]
    errors: Annotated[list[WorkflowIssue], add]
    final_report: InvestmentReport


class InvestmentState(InvestmentInput, InvestmentUpdates):
    pass


class InvestmentOutput(TypedDict):
    final_report: InvestmentReport
