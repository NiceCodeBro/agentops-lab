"""Configure LangChain's Bedrock chat model for local or hosted use."""

import json
import os

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agentops_lab.research_models import ResearchAnalysis


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


def research_model(
    model: ChatBedrockConverse, *, output_mode: str
) -> Runnable[list[BaseMessage], ResearchAnalysis]:
    if output_mode == "tool":
        return model.with_structured_output(ResearchAnalysis, method="function_calling")

    schema = ResearchAnalysis.model_json_schema()
    schema_json = json.dumps(schema, separators=(",", ":"))
    json_instruction = f"Return only a JSON object matching this schema: {schema_json}"

    def request_json(messages: list[BaseMessage]) -> list[BaseMessage]:
        system = f"{messages[0].text}\n{json_instruction}"
        return [SystemMessage(system), *messages[1:]]

    def parse_json(message: BaseMessage) -> ResearchAnalysis:
        return ResearchAnalysis.model_validate_json(message.text)

    prompt_builder = RunnableLambda(request_json)
    response_parser = RunnableLambda(parse_json)
    return prompt_builder | model | response_parser
