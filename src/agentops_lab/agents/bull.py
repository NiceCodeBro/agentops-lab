"""Evaluate evidence for the positive investment case."""

from langchain_core.messages import BaseMessage

from agentops_lab.agents._messages import CLAIM_RULES, EVIDENCE_RULES, agent_messages


def bull_messages(**payload: object) -> list[BaseMessage]:
    return agent_messages(
        f"""Develop up to three distinct bullish arguments anchored in reported results or
documented business strengths. A growth opportunity requires affirmative evidence of upside.
A risk, loss, restriction, or warning is never evidence of an opportunity by itself.
A mitigation may limit downside if directly documented. It does not establish incremental
growth. State the assumptions for any conditional interpretation. Return an empty thesis if
there is insufficient affirmative evidence.
{EVIDENCE_RULES}
{CLAIM_RULES}""",
        **payload,
    )
