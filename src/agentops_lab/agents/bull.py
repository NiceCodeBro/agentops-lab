"""Evaluate evidence for the positive investment case."""

from langchain_core.messages import BaseMessage

from agentops_lab.agents._messages import CLAIM_RULES, EVIDENCE_RULES, agent_messages
from agentops_lab.models.research import DocumentExcerpt
from agentops_lab.models.workflow import CandidateClaim, InvestmentRequest


def bull_messages(
    *,
    request: InvestmentRequest,
    research: list[CandidateClaim],
    evidence: list[DocumentExcerpt],
) -> list[BaseMessage]:
    return agent_messages(
        f"""Develop up to three distinct bullish arguments anchored in reported results or
documented business strengths. A growth opportunity requires affirmative evidence of upside.
A risk, loss, restriction, or warning is never evidence of an opportunity by itself.
A mitigation may limit downside if directly documented. It does not establish incremental
growth. State the assumptions for any conditional interpretation. Return an empty thesis if
there is insufficient affirmative evidence.
{EVIDENCE_RULES}
{CLAIM_RULES}""",
        request=request.model_dump(mode="json"),
        research=[item.model_dump(mode="json") for item in research],
        evidence=[item.model_dump(mode="json") for item in evidence],
    )
