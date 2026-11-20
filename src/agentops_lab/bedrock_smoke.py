"""Run a short text inference through Strands and Amazon Bedrock."""

import argparse
import json
import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from strands import Agent
from strands.models import BedrockModel
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
    ModelThrottledException,
)

logger = logging.getLogger(__name__)


def check_bedrock(session: boto3.Session, model_id: str) -> None:
    model = BedrockModel(
        boto_session=session,
        boto_client_config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "total_max_attempts": 1},
        ),
        model_id=model_id,
        max_tokens=16,
        temperature=0,
        streaming=False,
    )
    agent = Agent(
        model=model,
        callback_handler=None,
        retry_strategy=None,
        system_prompt="Reply briefly with plain text.",
    )
    result = agent("Reply with OK.")
    if not any(block.get("text", "").strip() for block in result.message["content"]):
        raise RuntimeError("Bedrock returned no text.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_id = os.getenv("BEDROCK_MODEL_ID", "").strip()
    if not model_id:
        logger.error(
            json.dumps({"event": "bedrock_smoke_failed", "error": "Set BEDROCK_MODEL_ID."})
        )
        return 1

    context = {"model_id": model_id}
    try:
        session = boto3.Session(region_name=os.getenv("AWS_REGION") or None)
        if not session.region_name:
            raise RuntimeError("Set AWS_REGION or configure a region in your AWS profile.")
        context["region"] = session.region_name
        check_bedrock(session, model_id)
    except (
        BotoCoreError,
        ClientError,
        ContextWindowOverflowException,
        MaxTokensReachedException,
        ModelThrottledException,
        RuntimeError,
    ) as error:
        logger.error(json.dumps({"event": "bedrock_smoke_failed", **context, "error": str(error)}))
        return 1

    logger.info(json.dumps({"event": "bedrock_smoke_succeeded", **context}))
    return 0
