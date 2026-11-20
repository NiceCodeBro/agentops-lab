"""Run excerpt-based company research using Amazon Bedrock."""

import argparse
import json
import logging
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.exceptions import OutputParserException

from agentops_lab.bedrock import load_bedrock_model, research_model
from agentops_lab.research import build_research_graph
from agentops_lab.research_models import ResearchRequest

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/nvda.synthetic.json"),
        help="Research input JSON",
    )
    parser.add_argument("--ticker", help="Expected ticker in the input file")
    parser.add_argument("--output-mode", choices=("tool", "json"), default="tool")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        input_json = args.input.read_bytes()
        request = ResearchRequest.model_validate_json(input_json)
        if args.ticker and request.ticker.upper() != args.ticker.strip().upper():
            raise ValueError("--ticker does not match the ticker in the input file.")

        model = load_bedrock_model(max_tokens=768)
        analyze = research_model(model, output_mode=args.output_mode)
        graph = build_research_graph(analyze)
        result = graph.invoke({"request": request})
    except OutputParserException:
        logger.error(
            json.dumps({"event": "research_failed", "error": "Invalid structured model output."})
        )
        return 1
    except (BotoCoreError, ClientError, OSError, ValueError) as error:
        logger.error(json.dumps({"event": "research_failed", "error": str(error)}))
        return 1

    brief = result["brief"]
    print(brief.model_dump_json(indent=2))
    return 0
