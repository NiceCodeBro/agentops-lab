"""Assess whether the filing supports each claim's wording and direction."""

import re

from langchain_core.messages import BaseMessage

from agentops_lab.agents._messages import EVIDENCE_RULES, agent_messages
from agentops_lab.models.research import DocumentExcerpt
from agentops_lab.models.workflow import (
    CandidateClaim,
    ClaimDecision,
    ClaimReview,
    EvidenceGap,
    EvidenceQuote,
    GapResolution,
)


def judge_messages(**payload: object) -> list[BaseMessage]:
    return agent_messages(
        f"""Review every supplied claim ID once. Compare the complete claim against its cited
passages in context, including figures, periods, entities, assumptions, and the direction of
the argument. A valid ID or copied quote does not establish support. Mark supported only if the
claim as written is justified. Mark partial for incomplete support, unsupported for absent or
irrelevant evidence, and contradicted for a direct conflict. If a bullish argument turns a
negative disclosure, such as a loss of sales from export restrictions, into upside without
affirmative evidence, mark risk_as_upside even when its citation exists. A mitigation alone
does not establish incremental upside. Give a concise public rationale per decision.
Identify at most two material evidence gaps even when no follow-up budget remains. Give a
focused query only when the indexed 10-K could plausibly resolve the gap; otherwise use null.
Before naming a gap, look for the answer in all supplied excerpts, even if no claim quoted it.
Use affected_claim_ids to identify claims the evidence could
change. An empty list is appropriate for a missing topic with no existing claim. Use a new
query and describe the question, its materiality, and the evidence needed. If prior gaps are
provided, review each named gap. Mark it resolved only with a verbatim quote from a new source
that actually answers the question. Use the new sources to reconsider affected claims, but
mark a claim supported only when its own citations support it. Return an empty resolutions
list on the first pass. Use resolutions to update prior gaps without listing them again.
Provide only brief, public-facing rationales.
{EVIDENCE_RULES}""",
        **payload,
    )


def match_quote(quote: EvidenceQuote, sources: dict[str, DocumentExcerpt]) -> EvidenceQuote | None:
    """Find exact wording, allowing only whitespace and an added terminal period."""
    source = sources.get(quote.source_id)
    normalized = " ".join(quote.excerpt.split())
    if source is None or len(normalized.split()) < 5:
        return None
    text = " ".join(source.text.split())
    if re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text):
        return quote.model_copy(update={"excerpt": normalized})
    if normalized.endswith("."):
        fragment = normalized[:-1]
        # The model may add a sentence period to an exact fragment within a sentence.
        if re.search(rf"(?<!\w){re.escape(fragment)}(?![\w.])", text):
            return quote.model_copy(update={"excerpt": fragment})
    return None


def invalid_citation(
    claim: CandidateClaim, sources: dict[str, DocumentExcerpt]
) -> ClaimDecision | None:
    cited = set(claim.source_ids)
    if cited == {quote.source_id for quote in claim.quotes} and all(
        match_quote(quote, sources) is not None for quote in claim.quotes
    ):
        return None
    return ClaimDecision(
        claim_id=claim.claim_id,
        status="rejected",
        support="invalid_citation",
        reason="A cited source was unavailable or a quote did not match the cited excerpt.",
    )


def decide(claim: CandidateClaim, review: ClaimReview | None) -> ClaimDecision:
    if review is None:
        return ClaimDecision(
            claim_id=claim.claim_id,
            status="uncertain",
            support="partial",
            reason="The Judge did not return a unique assessment of this claim.",
        )
    if review.support == "supported" and claim.classification != "uncertainty":
        status = "accepted"
    elif review.support in {"supported", "partial"}:
        status = "uncertain"
    else:
        status = "rejected"
    return ClaimDecision(
        claim_id=claim.claim_id, status=status, support=review.support, reason=review.reason
    )


def resolve_gap(
    gap: EvidenceGap,
    resolution: GapResolution | None,
    new_sources: dict[str, DocumentExcerpt],
) -> EvidenceGap:
    if resolution is not None and resolution.resolved and resolution.quotes:
        matches = [match_quote(quote, new_sources) for quote in resolution.quotes]
        if None not in matches:
            return gap.model_copy(
                update={
                    "status": "resolved",
                    "note": resolution.reason,
                    "resolution_quotes": matches,
                }
            )
    note = (
        resolution.reason
        if resolution is not None and not resolution.resolved
        else "Follow-up did not provide verified new evidence that closed this gap."
    )
    return gap.model_copy(update={"status": "unresolved", "note": note})
