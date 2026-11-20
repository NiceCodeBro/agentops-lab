# AgentOps Lab

AgentOps Lab is a portfolio project for multi-agent stock research on AWS. The goal is to let users follow how agents gather financial evidence, evaluate bullish and bearish arguments, and produce a sourced report. The same Python research logic is intended to run on Amazon Bedrock AgentCore Runtime and Amazon EKS.

The first milestone includes a Bedrock smoke command and a Research Agent that turns supplied financial excerpts into a cited brief. LangGraph runs the research workflow, LangChain's `ChatBedrockConverse` handles inference, and Pydantic validates inputs and outputs.

## Project layout

```text
src/agentops_lab/          Python application package
  bedrock.py              Bedrock model and structured-output configuration
  bedrock_smoke.py        Local AWS connectivity command
  research.py             LangGraph workflow
  research_models.py      Pydantic input and output schemas
  research_cli.py         Local research command
examples/                 Synthetic research input
tests/                    Offline tests
pyproject.toml            Package, dependencies, pytest, and Ruff configuration
uv.lock                   Resolved dependencies for reproducible installs
.env.example              Example local AWS and Bedrock settings
```

## Local development

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the project root, run:

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

The pytest suite uses a stubbed Bedrock API and runs offline without AWS credentials.

## Bedrock connectivity

Copy `.env.example` to `.env` and replace the placeholders. For Amazon Nova Micro via the EU inference profile from Frankfurt, use `AWS_REGION=eu-central-1` and `BEDROCK_MODEL_ID=eu.amazon.nova-micro-v1:0`. When `AWS_REGION` is unset, Boto3 resolves the Region from `AWS_DEFAULT_REGION` or the selected AWS profile. For an existing AWS IAM Identity Center profile, set `AWS_PROFILE` in `.env` and sign in with AWS CLI v2. The AWS SDK can also discover credentials through its normal credential chain. [Nova Micro availability](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-micro.html) · [Boto3 configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html) · [Boto3 credential sources](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)

```sh
cp .env.example .env
# Replace the Region and model placeholders in .env.
# For IAM Identity Center, set AWS_PROFILE in .env and run:
# aws sso login --profile your-sso-profile
uv run --locked --env-file .env agentops-bedrock-smoke
```

The AWS identity needs `bedrock:InvokeModel` for the selected model. For cross-Region inference, grant it on the inference profile and the underlying foundation model in every destination Region. Any applicable service control policies must allow the invocation across those Regions. The selected model must be available to the account. [Bedrock inference permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-prereq.html) · [Inference profile permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html)

The command sends a short, non-streaming prompt with a 16-token output limit. A live request incurs Bedrock usage. It logs a JSON event and exits with status `0` when Bedrock returns text or `1` on a configuration or inference failure. [LangChain Bedrock integration](https://docs.langchain.com/oss/python/integrations/chat/bedrock)

## Local research

From the repository root, run the bundled NVDA demo with the same AWS configuration:

```sh
uv run --locked --env-file .env agentops-research --ticker NVDA
```

The bundled excerpts are fictional and the output is marked `"synthetic": true`. To supply your own excerpts, use `--input path/to/request.json`. The JSON file contains `company`, `ticker`, and `excerpts`. Each excerpt needs a unique `source_id`, `title`, ISO `document_date`, and `text`. Set `url` when available and `synthetic` on the request when using fictional data. If you pass `--ticker`, it must match the input ticker. The command writes the validated brief as JSON to stdout.

The workflow runs `START → research → END`. The research node receives a typed request, passes the supplied text to the model, and validates a structured analysis. Each finding, opportunity, and risk must cite at least one existing source ID. The application copies the cited source metadata from the request and preserves the given company and ticker. If there are no excerpts, the node returns an insufficient-evidence brief without model inference. Citation checks verify that each ID exists in the input. Whether the excerpt supports the wording of a claim still requires review. [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

The default output mode uses LangChain's tool-based structured output, which works with Nova Micro through Converse. Nova Micro does not support Bedrock's native JSON-schema output. For models without compatible tool calling, add `--output-mode json` to request a plain JSON object and validate it with Pydantic. Both modes use the configured model, make at most one inference call, and cap the output at 768 tokens. Invalid output fails the command. [Nova Micro capabilities](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-micro.html) · [Nova tool choice](https://docs.aws.amazon.com/nova/latest/userguide/tool-choice.html)
