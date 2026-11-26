"""Track events, failures and model usage for a workflow node."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter

from langchain_core.messages import BaseMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from agentops_lab.models.workflow import (
    Agent,
    EventType,
    ExecutionEvent,
    InvestmentUpdates,
    ModelCall,
    Reasoner,
    WorkflowIssue,
)


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
