"""Bounded, evidence-based investment analysis with five LangGraph agent roles."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from time import perf_counter
from typing import Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agentops_lab.agents.bear import bear_messages
from agentops_lab.agents.bull import bull_messages
from agentops_lab.agents.judge import (
    decide,
    invalid_citation,
    judge_messages,
    match_quote,
    resolve_gap,
)
from agentops_lab.agents.planner import bound_plan, planner_messages, query_key
from agentops_lab.agents.research import investment_research_messages
from agentops_lab.models.research import DocumentExcerpt, SourceReference
from agentops_lab.models.workflow import (
    Agent,
    Arguments,
    CandidateClaim,
    Claim,
    ClaimRole,
    EventType,
    EvidenceGap,
    ExecutionEvent,
    InvestmentInput,
    InvestmentOutput,
    InvestmentReport,
    InvestmentState,
    InvestmentUpdates,
    JudgeReview,
    ModelCall,
    Reasoner,
    ResearchFindings,
    ResearchPlan,
    RetrievalTrace,
    ReviewedClaim,
    WorkflowIssue,
    WorkflowLimits,
)
from agentops_lab.retrieval.filing_store import FilingStore
from agentops_lab.retrieval.sec_research import select_distinct_excerpts


@dataclass
class NodeActivity:
    """Collect this node's public events and usage for LangGraph's list reducers."""

    run_id: str
    agent: Agent
    events: list[ExecutionEvent] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    errors: list[WorkflowIssue] = field(default_factory=list)

    def emit(self, kind: EventType, summary: str, source_ids: list[str] | None = None) -> None:
        event = ExecutionEvent(
            run_id=self.run_id,
            agent=self.agent,
            event_type=kind,
            timestamp=datetime.now(UTC),
            summary=summary,
            source_ids=source_ids or [],
        )
        self.events.append(event)
        get_stream_writer()(event.model_dump(mode="json"))

    def fail(self, message: str) -> None:
        self.errors.append(WorkflowIssue(agent=self.agent, message=message))
        self.emit("agent_failed", message)

    def ask[Output: BaseModel](
        self, reasoner: Reasoner, schema: type[Output], messages: list[BaseMessage]
    ) -> Output | None:
        started = perf_counter()
        reply = None
        output = None
        try:
            reply = reasoner(schema, messages)
            if reply.stop_reason == "max_tokens":
                self.fail("The model reached the output token limit.")
            elif reply.validation_codes:
                codes = ", ".join(reply.validation_codes)
                self.fail(f"Structured model output failed validation ({codes}).")
            else:
                output = schema.model_validate(reply.output)
        except Exception as error:
            self.fail(f"Model call or structured output failed ({type(error).__name__}).")
        self.model_calls.append(
            ModelCall(
                agent=self.agent,
                service="reasoning",
                elapsed_ms=elapsed_ms(started),
                input_tokens=reply.input_tokens if reply else None,
                output_tokens=reply.output_tokens if reply else None,
                succeeded=output is not None,
            )
        )
        return output

    def record_embedding(self, duration: int, tokens: int | None, succeeded: bool) -> None:
        self.model_calls.append(
            ModelCall(
                agent=self.agent,
                service="embedding",
                elapsed_ms=duration,
                input_tokens=tokens,
                output_tokens=0,
                succeeded=succeeded,
            )
        )

    def updates(self) -> InvestmentUpdates:
        return {"events": self.events, "model_calls": self.model_calls, "errors": self.errors}


def elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


def candidates(
    claims: list[Claim],
    agent: ClaimRole,
    category: Literal["finding", "risk", "argument"],
    research_round: int,
    sources: dict[str, DocumentExcerpt],
) -> list[CandidateClaim]:
    return [
        CandidateClaim(
            **claim.model_dump(exclude={"quotes"}),
            quotes=[match_quote(quote, sources) or quote for quote in claim.quotes],
            claim_id=f"{agent}-{research_round}-{category}-{number}",
            agent=agent,
            category=category,
            research_round=research_round,
        )
        for number, claim in enumerate(claims, start=1)
    ]


def build_investment_graph(
    reasoner: Reasoner, store: FilingStore, *, limits: WorkflowLimits | None = None
) -> CompiledStateGraph:
    limits = limits or WorkflowLimits()
    initial_queries = min(4, limits.max_queries, limits.max_model_calls - 5)

    def planner(state: InvestmentInput) -> InvestmentUpdates:
        started_at = perf_counter()
        activity = NodeActivity(uuid4().hex, "planner")
        activity.emit("workflow_started", "Started investment analysis.")
        activity.emit("agent_started", "Planning prioritized SEC research.")
        plan = activity.ask(
            reasoner, ResearchPlan, planner_messages(state["request"], initial_queries)
        )
        updates: InvestmentUpdates = {"run_id": activity.run_id, "started_at": started_at}
        if plan is not None:
            try:
                plan = bound_plan(plan, initial_queries)
            except ValueError:
                activity.fail("The Planner returned no executable search query.")
            else:
                updates["plan"] = plan
                activity.emit(
                    "agent_completed", f"Prioritized {len(plan.tasks)} research questions."
                )
                activity.emit("handoff", "Sent the planned queries to Research.")
        return {**updates, **activity.updates()}

    def research(state: InvestmentState, *, follow_up: bool = False) -> InvestmentUpdates:
        activity = NodeActivity(state["run_id"], "research")
        round_number = 2 if follow_up else 1
        activity.emit("agent_started", f"Started SEC research round {round_number}.")
        prior = state.get("evidence", [])
        prior_ids = {excerpt.source_id for excerpt in prior}
        if follow_up:
            gaps = [gap for gap in state["gaps"] if gap.status == "pending"]
            questions = [gap.model_dump(mode="json") for gap in gaps]
            queries = [gap.query for gap in gaps if gap.query is not None]
            chunk_limit = limits.max_chunks - len(prior)
        else:
            tasks = state["plan"].tasks
            questions = [task.model_dump(mode="json") for task in tasks]
            queries = [task.query for task in tasks]
            # Leave room for distinct follow-up excerpts when the budgets permit another round.
            reserve = 0
            if (
                limits.max_research_rounds == 2
                and len(queries) < limits.max_queries
                and 5 + len(queries) + 3 <= limits.max_model_calls
            ):
                reserve = min(2, limits.max_chunks - min(limits.max_chunks, len(queries)))
            chunk_limit = limits.max_chunks - reserve

        try:
            results = store.search_filings_batch(
                queries,
                state["request"].ticker,
                limits.top_k,
                on_embedding=activity.record_embedding,
            )
        except Exception as error:
            activity.fail(f"SEC evidence retrieval failed ({type(error).__name__}).")
            return {"research_rounds": round_number, **activity.updates()}

        ranked = [[hit.excerpt for hit in hits] for hits in results]
        new_evidence = select_distinct_excerpts(ranked, limit=chunk_limit, excluded=prior_ids)
        new_ids = [excerpt.source_id for excerpt in new_evidence]
        selected = set(new_ids)
        traces = [
            RetrievalTrace(
                research_round=round_number,
                query=query,
                candidate_source_ids=[excerpt.source_id for excerpt in hits],
                selected_source_ids=[
                    excerpt.source_id for excerpt in hits if excerpt.source_id in selected
                ],
            )
            for query, hits in zip(queries, ranked, strict=True)
        ]
        claims = list(state.get("research_claims", []))
        if new_evidence:
            findings = activity.ask(
                reasoner,
                ResearchFindings,
                investment_research_messages(
                    request=state["request"].model_dump(mode="json"),
                    assignments=questions,
                    evidence=[excerpt.model_dump(mode="json") for excerpt in new_evidence],
                ),
            )
            if findings is not None:
                sources = {excerpt.source_id: excerpt for excerpt in new_evidence}
                claims += candidates(
                    findings.findings, "research", "finding", round_number, sources
                )
                claims += candidates(findings.risks, "research", "risk", round_number, sources)

        activity.emit("research_completed", f"Retrieved {len(new_ids)} new SEC excerpts.", new_ids)
        if not activity.errors:
            activity.emit("agent_completed", f"Finished SEC research round {round_number}.")
            if new_evidence:
                target = "Judge" if follow_up else "Bull and Bear"
                activity.emit("handoff", f"Sent research evidence to {target}.", new_ids)
        return {
            "research_rounds": round_number,
            "evidence": [*prior, *new_evidence],
            "research_claims": claims,
            "retrieval": traces,
            **activity.updates(),
        }

    def debate(state: InvestmentState, *, agent: Literal["bull", "bear"]) -> InvestmentUpdates:
        activity = NodeActivity(state["run_id"], agent)
        evidence = state["evidence"]
        sources = {excerpt.source_id: excerpt for excerpt in evidence}
        ids = [excerpt.source_id for excerpt in evidence]
        activity.emit("agent_started", f"Evaluating the {agent} case.", ids)
        build_messages = bull_messages if agent == "bull" else bear_messages
        arguments = activity.ask(
            reasoner,
            Arguments,
            build_messages(
                request=state["request"].model_dump(mode="json"),
                research=[
                    claim.model_dump(mode="json")
                    for claim in state["research_claims"]
                    if invalid_citation(claim, sources) is None
                ],
                evidence=[excerpt.model_dump(mode="json") for excerpt in evidence],
            ),
        )
        if arguments is None:
            return activity.updates()
        claims = candidates(arguments.arguments, agent, "argument", 1, sources)
        activity.emit("agent_completed", f"Proposed {len(claims)} {agent} arguments.")
        activity.emit("handoff", "Sent arguments to Judge.")
        if agent == "bull":
            return {"bull_claims": claims, **activity.updates()}
        return {"bear_claims": claims, **activity.updates()}

    def judge(state: InvestmentState) -> InvestmentUpdates:
        if state.get("errors"):
            return {}
        activity = NodeActivity(state["run_id"], "judge")
        round_number = state["research_rounds"]
        sources = {excerpt.source_id: excerpt for excerpt in state["evidence"]}
        activity.emit("agent_started", f"Reviewing claims for research round {round_number}.")
        new_ids = {
            source_id
            for trace in state["retrieval"]
            if trace.research_round == 2
            for source_id in trace.selected_source_ids
        }
        pending = [gap for gap in state.get("gaps", []) if gap.status == "pending"]
        affected = {key for gap in pending for key in gap.affected_claim_ids}
        all_claims = collect_claims(state)
        claims = (
            all_claims
            if round_number == 1
            else [
                claim
                for claim in all_claims
                if claim.claim_id in affected or claim.research_round == 2
            ]
        )
        decisions = dict(state.get("decisions", {}))
        valid_claims = []
        for claim in claims:
            if decision := invalid_citation(claim, sources):
                decisions[claim.claim_id] = decision
            else:
                valid_claims.append(claim)
        # Reserve two reasoning calls for follow-up Research and Judge, plus one per new query.
        capacity = min(
            2,
            limits.max_queries - len(state["retrieval"]),
            limits.max_model_calls - len(state["model_calls"]) - 3,
        )
        if round_number >= limits.max_research_rounds or len(sources) >= limits.max_chunks:
            capacity = 0
        review_ids = (
            new_ids | {key for claim in valid_claims for key in claim.source_ids}
            if round_number == 2
            else sources.keys()
        )
        review = activity.ask(
            reasoner,
            JudgeReview,
            judge_messages(
                request=state["request"].model_dump(mode="json"),
                claims=[claim.model_dump(mode="json") for claim in valid_claims],
                evidence=[
                    excerpt.model_dump(mode="json")
                    for key, excerpt in sources.items()
                    if key in review_ids
                ],
                follow_up_capacity=max(0, capacity),
                searched_queries=[trace.query for trace in state["retrieval"]],
                prior_gaps=[gap.model_dump(mode="json") for gap in pending],
                new_source_ids=sorted(new_ids),
            ),
        )
        eligible = {claim.claim_id for claim in valid_claims}
        counts = Counter(item.claim_id for item in review.reviews) if review else Counter()
        reviews = (
            {item.claim_id: item for item in review.reviews if counts[item.claim_id] == 1}
            if review
            else {}
        )
        if review is not None and (
            counts.keys() != eligible or any(n != 1 for n in counts.values())
        ):
            activity.fail("The Judge did not return exactly one assessment per eligible claim.")
        for claim in valid_claims:
            decisions[claim.claim_id] = decide(claim, reviews.get(claim.claim_id))
        for claim in claims:
            decision = decisions[claim.claim_id]
            if decision.status in {"accepted", "rejected"}:
                event: EventType = (
                    "claim_accepted" if decision.status == "accepted" else "claim_rejected"
                )
                activity.emit(
                    event,
                    f"{claim.claim_id} {decision.status}.",
                    [key for key in claim.source_ids if key in sources],
                )

        gaps = list(state.get("gaps", []))
        if review is not None and not activity.errors:
            resolutions = {item.gap_id: item for item in review.resolutions}
            new_sources = {key: sources[key] for key in new_ids}
            gaps = [
                resolve_gap(gap, resolutions.get(gap.gap_id), new_sources)
                if gap.status == "pending"
                else gap
                for gap in gaps
            ]
            seen = {query_key(trace.query) for trace in state["retrieval"]}
            gap_questions = {query_key(gap.question) for gap in gaps}
            known_claim_ids = {claim.claim_id for claim in all_claims}
            for item in review.follow_ups:
                question = query_key(item.question)
                if question in gap_questions:
                    continue
                gap_questions.add(question)
                query = query_key(item.query or "")
                status: Literal["pending", "unresolved"] = "unresolved"
                if not query:
                    note = "The Judge identified no actionable query for the indexed filing."
                elif query in seen:
                    note = "This search was already used or assigned."
                elif capacity <= 0:
                    note = "The remaining research budget does not allow this search."
                else:
                    status = "pending"
                    note = "Selected for targeted follow-up research."
                    seen.add(query)
                    capacity -= 1
                gap = EvidenceGap(
                    **item.model_dump(exclude={"affected_claim_ids"}),
                    affected_claim_ids=[
                        key for key in item.affected_claim_ids if key in known_claim_ids
                    ],
                    gap_id=f"gap-{len(gaps) + 1}",
                    status=status,
                    note=note,
                )
                gaps.append(gap)
                if status == "pending":
                    activity.emit("evidence_requested", f"{gap.gap_id}: {gap.question}"[:400])
            activity.emit("agent_completed", f"Assessed {len(claims)} claims.")
        return {"decisions": decisions, "gaps": gaps, **activity.updates()}

    def after_research(state: InvestmentState) -> list[str]:
        return ["bull", "bear"] if state.get("evidence") and not state.get("errors") else ["report"]

    def after_judge(state: InvestmentState) -> Literal["research_follow_up", "report"]:
        if (
            not state.get("errors")
            and state["research_rounds"] < limits.max_research_rounds
            and any(gap.status == "pending" for gap in state.get("gaps", []))
        ):
            return "research_follow_up"
        return "report"

    def after_follow_up(state: InvestmentState) -> Literal["judge", "report"]:
        if not state.get("errors") and any(
            trace.research_round == 2 and trace.selected_source_ids for trace in state["retrieval"]
        ):
            return "judge"
        return "report"

    graph = StateGraph(
        InvestmentState, input_schema=InvestmentInput, output_schema=InvestmentOutput
    )
    graph.add_node("planner", planner)
    graph.add_node("research", research)
    graph.add_node("bull", partial(debate, agent="bull"))
    graph.add_node("bear", partial(debate, agent="bear"))
    graph.add_node("judge", judge)
    graph.add_node("research_follow_up", partial(research, follow_up=True))
    graph.add_node("report", final_report)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: "report" if state.get("errors") else "research",
        ["report", "research"],
    )
    graph.add_conditional_edges("research", after_research, ["bull", "bear", "report"])
    graph.add_edge(["bull", "bear"], "judge")
    graph.add_conditional_edges("judge", after_judge)
    graph.add_conditional_edges("research_follow_up", after_follow_up)
    graph.add_edge("report", END)
    return graph.compile()


def collect_claims(state: InvestmentState) -> list[CandidateClaim]:
    return [
        *state.get("research_claims", []),
        *state.get("bull_claims", []),
        *state.get("bear_claims", []),
    ]


def final_report(state: InvestmentState) -> InvestmentOutput:
    request = state["request"]
    errors = state.get("errors", [])
    sources = [
        SourceReference.model_validate(excerpt.model_dump(exclude={"text"}))
        for excerpt in state.get("evidence", [])
    ]
    activity = NodeActivity(state["run_id"], "workflow")
    decisions = state.get("decisions", {})
    reviewed = [
        ReviewedClaim(claim=claim, decision=decisions.get(claim.claim_id) or decide(claim, None))
        for claim in collect_claims(state)
    ]
    accepted = [item for item in reviewed if item.decision.status == "accepted"]
    excluded = [item for item in reviewed if item.decision.status != "accepted"]
    gaps = [
        gap.model_copy(
            update={
                "status": "unresolved",
                "note": "Follow-up did not yield a completed assessment of new evidence.",
            }
        )
        if gap.status == "pending"
        else gap
        for gap in state.get("gaps", [])
    ]
    rounds = state.get("research_rounds", 0)
    status: Literal["completed", "insufficient_evidence", "failed"]
    if errors:
        status = "failed"
        summary = "Analysis stopped after an execution failure. Any findings below are partial."
    elif not accepted:
        status = "insufficient_evidence"
        summary = "The available filing evidence yielded no accepted investment research claims."
    else:
        status = "completed"
        summary = f"Reviewed {len(sources)} filing excerpts and accepted {len(accepted)} claims."
    unresolved = sum(gap.status == "unresolved" for gap in gaps)
    rejected = sum(item.decision.status == "rejected" for item in excluded)
    assessment = (
        f"Accepted {len(accepted)}, rejected {rejected}, and marked {len(excluded) - rejected} "
        f"claims uncertain. {unresolved} evidence gaps remain unresolved."
    )
    activity.emit("workflow_failed" if errors else "workflow_completed", summary)
    events = sorted([*state["events"], *activity.events], key=lambda event: event.timestamp)
    return {
        "final_report": InvestmentReport(
            run_id=state["run_id"],
            company=request.company,
            ticker=request.ticker,
            question=request.question,
            status=status,
            research_summary=summary,
            key_findings=[item for item in accepted if item.claim.category == "finding"],
            bullish_arguments=[item for item in accepted if item.claim.agent == "bull"],
            bearish_arguments=[item for item in accepted if item.claim.agent == "bear"],
            material_risks=[item for item in accepted if item.claim.category == "risk"],
            judge_assessment=assessment,
            rejected_or_uncertain_claims=excluded,
            evidence_gaps=gaps,
            sources=sources,
            plan=state.get("plan"),
            research_rounds=rounds,
            retrieval=state.get("retrieval", []),
            model_calls=state.get("model_calls", []),
            elapsed_ms=elapsed_ms(state["started_at"]),
            events=events,
            errors=errors,
        )
    }
