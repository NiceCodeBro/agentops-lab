"""Single-node LangGraph workflow for excerpt-based research."""

from typing import NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentops_lab.agents.research import research_excerpts
from agentops_lab.models.research import ResearchAnalysis, ResearchBrief, ResearchRequest


class ResearchInput(TypedDict):
    request: ResearchRequest


class ResearchOutput(TypedDict):
    brief: ResearchBrief


class ResearchState(ResearchInput):
    brief: NotRequired[ResearchBrief]


def build_research_graph(
    analyze: Runnable[list[BaseMessage], ResearchAnalysis],
) -> CompiledStateGraph:
    def research(state: ResearchInput) -> ResearchOutput:
        return {"brief": research_excerpts(state["request"], analyze)}

    graph = StateGraph(ResearchState, input_schema=ResearchInput, output_schema=ResearchOutput)
    graph.add_node("research", research)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    return graph.compile()
