"""Run a short text inference through LangChain and Amazon Bedrock."""

import argparse
import json
import logging

from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.messages import HumanMessage, SystemMessage

from agentops_lab.bedrock import load_bedrock_model

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    context = {}
    try:
        model = load_bedrock_model(max_tokens=16)
        context = {"model_id": model.model_id, "region": model.region_name}
        messages = [SystemMessage("Reply briefly with plain text."), HumanMessage("Reply with OK.")]
        response = model.invoke(messages)
        if not response.text.strip():
            raise ValueError("Bedrock returned no text.")
    except (BotoCoreError, ClientError, ValueError) as error:
        logger.error(json.dumps({"event": "bedrock_smoke_failed", **context, "error": str(error)}))
        return 1

    logger.info(json.dumps({"event": "bedrock_smoke_succeeded", **context}))
    return 0
