"""Run excerpt-based company research using Amazon Bedrock."""

import argparse
import json
import logging
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.exceptions import OutputParserException

from agentops_lab.bedrock import build_structured_model, load_bedrock_model
from agentops_lab.filing_store import load_filing_store
from agentops_lab.research import build_research_graph
from agentops_lab.research_models import ResearchAnalysis, ResearchRequest
from agentops_lab.sec import TICKER
from agentops_lab.sec_research import build_sec_request

logger = logging.getLogger(__name__)
SEC_DEFAULT_MODEL = "eu.amazon.nova-pro-v1:0"
DEMO_INPUT = Path("examples/nvda.synthetic.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Research input JSON")
    source.add_argument("--sec", action="store_true", help="Retrieve SEC filing excerpts")
    parser.add_argument("--ticker", help="Expected ticker in the input file or SEC retrieval")
    parser.add_argument("--output-mode", choices=("tool", "json"), default="tool")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        if args.sec:
            store = load_filing_store()
            request = build_sec_request(store, args.ticker or TICKER)
            model = load_bedrock_model(max_tokens=768, default_model_id=SEC_DEFAULT_MODEL)
        else:
            input_path = args.input or DEMO_INPUT
            input_json = input_path.read_bytes()
            request = ResearchRequest.model_validate_json(input_json)
            if args.ticker and request.ticker.upper() != args.ticker.strip().upper():
                raise ValueError("--ticker does not match the ticker in the input file.")
            model = load_bedrock_model(max_tokens=768)
        analyze = build_structured_model(model, ResearchAnalysis, output_mode=args.output_mode)
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
