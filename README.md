# AgentOps Lab

AgentOps Lab is a portfolio project for multi-agent stock research on AWS. The goal is to let users follow how agents gather financial evidence, evaluate bullish and bearish arguments, and produce a sourced report. The same Python research logic is intended to run on Amazon Bedrock AgentCore Runtime and Amazon EKS.

Phase 1 provides a local Python package, offline checks, and a Bedrock connectivity command using the Strands Agents SDK.

## Project layout

```text
src/agentops_lab/          Python application package
  bedrock_smoke.py        Local AWS connectivity command
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

The command sends a short, non-streaming prompt with a 16-token output limit. The request may incur a small Bedrock charge. It logs a JSON event and exits with status `0` when Bedrock returns text or `1` on a configuration or inference failure. [Strands Bedrock provider](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/amazon-bedrock/)
