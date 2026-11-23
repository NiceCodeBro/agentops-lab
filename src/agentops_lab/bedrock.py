"""Configure LangChain's Bedrock chat model for local or hosted use."""

import json
import os
from typing import Literal

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

from agentops_lab.models.workflow import Reasoner, ReasoningReply


def load_bedrock_model(*, max_tokens: int, default_model_id: str = "") -> ChatBedrockConverse:
    model_id = os.getenv("BEDROCK_MODEL_ID", "").strip() or default_model_id
    if not model_id:
        raise ValueError("Set BEDROCK_MODEL_ID.")

    session = boto3.Session(region_name=os.getenv("AWS_REGION") or None)
    if not session.region_name:
        raise ValueError("Set AWS_REGION or configure a region in your AWS profile.")

    config = Config(
        connect_timeout=5,
        read_timeout=30,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    runtime_client = session.client("bedrock-runtime", config=config)
    control_client = session.client("bedrock", config=config)
    return ChatBedrockConverse(
        model=model_id,
        region_name=session.region_name,
        client=runtime_client,
        bedrock_client=control_client,
        max_tokens=max_tokens,
        temperature=0,
        disable_streaming=True,
    )


def build_structured_model[Output: BaseModel](
    model: ChatBedrockConverse,
    response_schema: type[Output],
    *,
    output_mode: Literal["tool", "json"],
) -> Runnable[list[BaseMessage], Output]:
    """Prepare Bedrock to return a validated response using the caller's schema."""
    if output_mode == "tool":
        return model.with_structured_output(response_schema, method="function_calling")

    def parse_json(message: BaseMessage) -> Output:
        return response_schema.model_validate_json(message.text)

    def add_json_instruction(messages: list[BaseMessage]) -> list[BaseMessage]:
        return json_messages(response_schema, messages)

    prompt_builder = RunnableLambda(add_json_instruction)
    response_parser = RunnableLambda(parse_json)
    return prompt_builder | model | response_parser


def json_messages(schema: type[BaseModel], messages: list[BaseMessage]) -> list[BaseMessage]:
    schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    instruction = f"Return only one JSON object matching this schema: {schema_json}"
    if messages and isinstance(messages[0], SystemMessage):
        return [SystemMessage(f"{messages[0].text}\n{instruction}"), *messages[1:]]
    return [SystemMessage(instruction), *messages]


def build_agent_reasoner(model: ChatBedrockConverse) -> Reasoner:
    """Parse a single structured JSON response and retain only usage and safe diagnostics."""

    def invoke(schema: type[BaseModel], messages: list[BaseMessage]) -> ReasoningReply:
        raw = model.invoke(json_messages(schema, messages))
        usage = raw.usage_metadata or {}
        stop_reason = raw.response_metadata.get("stopReason")
        text = raw.text.strip()
        if text.startswith("```json\n") and text.endswith("\n```"):
            text = text.removeprefix("```json\n").removesuffix("\n```")
        output = None
        validation_codes = ()
        if stop_reason != "max_tokens":
            try:
                output = schema.model_validate_json(text)
            except ValidationError as error:
                issues = error.errors(include_input=False, include_context=False, include_url=False)
                validation_codes = tuple(sorted({item["type"] for item in issues}))
        return ReasoningReply(
            output=output,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            stop_reason=stop_reason,
            validation_codes=validation_codes,
        )

    return invoke
