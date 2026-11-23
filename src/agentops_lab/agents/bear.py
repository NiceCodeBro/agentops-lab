"""Evaluate documented exposures and evidence-based downside scenarios."""

from langchain_core.messages import BaseMessage

from agentops_lab.agents._messages import CLAIM_RULES, EVIDENCE_RULES, agent_messages


def bear_messages(**payload: object) -> list[BaseMessage]:
    return agent_messages(
        f"""Develop up to three distinct bearish arguments informed by the supplied research.
Challenge the assumptions a bullish reading of the evidence would depend on, including whether
reported growth can persist. Separate documented exposures from conditional downside scenarios
and give the assumptions for each scenario. Avoid predicting an adverse outcome from a risk
disclosure alone. Leave the thesis empty if evidence is insufficient.
{EVIDENCE_RULES}
{CLAIM_RULES}""",
        **payload,
    )
