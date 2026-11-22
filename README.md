# AgentOps Lab

AgentOps Lab is a portfolio project for multi-agent stock research on AWS. The goal is to let users follow how agents gather financial evidence, evaluate bullish and bearish arguments, and produce a sourced report. The same Python research logic is intended to run on Amazon Bedrock AgentCore Runtime and Amazon EKS.

The Research Agent turns supplied financial excerpts or retrieved SEC filing excerpts into a cited brief. LangGraph runs the research workflow, LangChain's `ChatBedrockConverse` handles inference, and Pydantic validates inputs and outputs.

## Project layout

```text
src/agentops_lab/          Python application package
  bedrock.py              Bedrock model and structured-output configuration
  bedrock_smoke.py        Local AWS connectivity command
  research.py             LangGraph workflow
  research_models.py      Pydantic input and output schemas
  research_cli.py         Local research command
  sec.py                  SEC discovery, primary HTML parsing, and chunking
  filing_store.py         Titan embeddings, S3 source storage, and S3 Vectors retrieval
  sec_research.py         Build a research request from SEC search results
  sec_cli.py              Manual ingestion and retrieval smoke commands
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

## Research quality evaluation

The five cases in `examples/evaluation.synthetic.json` are fictional. Each records expected findings, risks, opportunities, and the source IDs that support them. Some cases also record facts that the agent could incorrectly report as missing, and gaps it should surface.

Run `uv run --locked pytest` for deterministic offline checks. To run the evaluation with Bedrock, use the same AWS configuration as the research CLI. The evaluator makes one model call per selected case and runs all five cases by default.

```sh
uv run --locked --env-file .env agentops-evaluate --case nvda-supplier-risk --case margin-guidance
```

The report shows which expected topics matched, the actual claims and citations, citation ID errors, and gaps or claims to review. Coverage uses case-specific phrase matching. A fact listed as missing is flagged when it matches a topic the fixture marks as disclosed. Numbers absent from cited text, unexpected topics, and unexpected sources also generate review flags. Paraphrases, qualifications, and misleading claims containing the expected words can evade these checks. Read claims against their cited excerpts to judge factual support. The command returns a nonzero exit status for agent failures or citation ID errors. Coverage misses and review flags are diagnostic.

## NVIDIA SEC retrieval pilot

The pilot uses NVIDIA's Form 10-K filed February 25, 2026, accession `0001045810-26-000021`, covering the fiscal year ended January 25, 2026. The ingestion command discovers that accession in NVIDIA's EDGAR submissions and downloads the primary filing. Set `SEC_USER_AGENT` in your environment to your organization or project name and an administrative contact email. The command waits at least one second between requests and retries transient failures with backoff. If EDGAR denies the client, you can pass a locally saved copy of the primary filing with `--html path/to/nvda-20260125.htm`. The command checks the copy's inline XBRL issuer, form, and reporting period. [SEC filing detail](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/0001045810-26-000021-index.htm) · [SEC fair access guidance](https://www.sec.gov/about/webmaster-frequently-asked-questions)

With Python dependencies installed, check the SEC download and parsing without AWS writes:

```sh
uv run --locked --env-file .env agentops-ingest-sec --dry-run
```

Live use requires an existing private S3 Standard source bucket, a private S3 vector bucket, and a cosine vector index in the AWS Region. Set `SEC_SOURCE_BUCKET`, `S3_VECTOR_BUCKET`, and `S3_VECTOR_INDEX` in `.env`. The index dimension must match `SEC_EMBEDDING_DIMENSIONS`, which defaults to 256. Titan supports 256, 512, and 1,024 dimensions. Both ingestion and queries verify the index before embedding. [Titan embedding parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html) · [S3 Vectors indexes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-indexes.html)

- The ingestion identity needs `s3vectors:GetIndex`, `s3vectors:PutVectors`, `s3:PutObject`, `s3:GetObject`, and `bedrock:InvokeModel` for Titan.
- The retrieval identity needs `s3vectors:GetIndex`, `s3vectors:QueryVectors`, `s3vectors:GetVectors` for filtered results and metadata, `s3:GetObject`, and `bedrock:InvokeModel` for Titan and the configured reasoning profile and its destination models. [S3 Vectors access management](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-access-management.html) · [Bedrock inference profile permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html)

```sh
uv run --locked --env-file .env agentops-ingest-sec
uv run --locked --env-file .env agentops-sec-smoke --show-text
uv run --locked --env-file .env agentops-research --sec --ticker NVDA
```

SEC research defaults to the Nova Pro EU profile when `BEDROCK_MODEL_ID` is unset. An existing value overrides the default. You can set `BEDROCK_MODEL_ID=eu.amazon.nova-pro-v1:0` explicitly for the pilot. Research with `--input` and the synthetic demo continue to use the configured model.

Ingestion extracts narrative from Business, Risk Factors, and MD&A. It strips HTML layout, hidden XBRL, and tables. The source bucket holds complete extracted chunks and their provenance. The index holds embeddings, stable keys, and small filterable metadata fields. Reingesting an identical corpus upserts the same keys. A changed corpus at the same accession is rejected so it cannot mix with previously indexed chunks.

The research command makes three fixed semantic queries filtered to NVDA, drops distant and off-topic hits, and passes up to six unique chunks to the existing LangGraph agent. `SEC_MAX_DISTANCE` controls the cosine-distance cutoff and defaults to `0.70`. A report's cited sources include the filing date, SEC URL, accession number, section, and chunk number. If retrieval supplies no usable evidence, the graph returns an insufficient-evidence brief without invoking the reasoning model. The smoke test checks topic and section matches, and citation validation checks source IDs. Evaluate the wording of each financial claim against the actual chunk text.
