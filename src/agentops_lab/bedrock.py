"""Configure LangChain's Bedrock chat model for local or hosted use."""

import json
import os
from typing import Literal

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


def load_bedrock_model(*, max_tokens: int) -> ChatBedrockConverse:
    model_id = os.getenv("BEDROCK_MODEL_ID", "").strip()
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

    schema = response_schema.model_json_schema()
    schema_json = json.dumps(schema, separators=(",", ":"))
    json_instruction = f"Return only a JSON object matching this schema: {schema_json}"

    def add_json_instruction(messages: list[BaseMessage]) -> list[BaseMessage]:
        if messages and isinstance(messages[0], SystemMessage):
            system = f"{messages[0].text}\n{json_instruction}"
            return [SystemMessage(system), *messages[1:]]
        return [SystemMessage(json_instruction), *messages]

    def parse_json(message: BaseMessage) -> Output:
        return response_schema.model_validate_json(message.text)

    prompt_builder = RunnableLambda(add_json_instruction)
    response_parser = RunnableLambda(parse_json)
    return prompt_builder | model | response_parser
