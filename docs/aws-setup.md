# AWS sandbox setup

Use this guide to configure Bedrock, ingest the NVIDIA filing, or manage the Terraform sandbox. [Back to the project overview](../README.md)

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

## NVIDIA SEC retrieval pilot

The pilot uses NVIDIA's Form 10-K filed February 25, 2026, accession `0001045810-26-000021`, covering the fiscal year ended January 25, 2026. The ingestion command discovers that accession in NVIDIA's EDGAR submissions and downloads the primary filing. Set `SEC_USER_AGENT` in your environment to your organization or project name and an administrative contact email. The command waits at least one second between requests and retries transient failures with backoff. If EDGAR denies the client, you can pass a locally saved copy of the primary filing with `--html path/to/nvda-20260125.htm`. The command checks the copy's inline XBRL issuer, form, and reporting period. [SEC filing detail](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/0001045810-26-000021-index.htm) · [SEC fair access guidance](https://www.sec.gov/about/webmaster-frequently-asked-questions)

With Python dependencies installed, check the SEC download and parsing without AWS writes:

```sh
uv run --locked --env-file .env agentops-ingest-sec --dry-run
```

Live use requires an existing private S3 Standard source bucket, a private S3 vector bucket, and a cosine vector index in the AWS Region. The [Terraform sandbox](#terraform-setup-and-lifecycle) manages those resources. Set `SEC_SOURCE_BUCKET`, `S3_VECTOR_BUCKET`, and `S3_VECTOR_INDEX` in `.env` from its outputs. The index dimension must match `SEC_EMBEDDING_DIMENSIONS`, which defaults to 256. Both ingestion and queries verify the index before embedding. [Titan embedding parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html) · [S3 Vectors indexes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-indexes.html)

- The ingestion identity needs `s3vectors:GetIndex`, `s3vectors:PutVectors`, `s3:PutObject`, `s3:GetObject`, and `bedrock:InvokeModel` for Titan.
- The retrieval identity needs `s3vectors:GetIndex`, `s3vectors:QueryVectors`, `s3vectors:GetVectors` for filtered results and metadata, `s3:GetObject`, and `bedrock:InvokeModel` for Titan and the configured reasoning profile and its destination models. [S3 Vectors access management](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-access-management.html) · [Bedrock inference profile permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html)

```sh
uv run --locked --env-file .env agentops-ingest-sec
uv run --locked --env-file .env agentops-sec-smoke --show-text
uv run --locked --env-file .env agentops-research --sec --ticker NVDA
```

SEC research defaults to the Nova Pro EU profile when `BEDROCK_MODEL_ID` is unset. An existing value overrides the default. You can set `BEDROCK_MODEL_ID=eu.amazon.nova-pro-v1:0` explicitly for the pilot. Research with `--input` and the synthetic demo continue to use the configured model.

## Terraform setup and lifecycle

The [dev environment](../infra/terraform/environments/dev/) uses the official HashiCorp AWS provider and is scoped to `eu-central-1`. It manages one S3 source bucket with all four public access blocks and SSE-S3, plus one S3 vector bucket and its 256-dimensional cosine index. Vector buckets always block public access and the index inherits SSE-S3 from its bucket. Terraform uses local state for this sandbox. Keep the state, saved plans, and actual variable values private. Keep `.terraform.lock.hcl` in Git. [Vector bucket access](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html) · [Vector encryption](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-data-encryption.html)

From the repository root, configure the sandbox account and names:

```sh
cd infra/terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
export AWS_PROFILE=your-sandbox-sso-profile
aws sso login --profile "$AWS_PROFILE"
aws sts get-caller-identity --region eu-central-1 --query Account --output text
```

Compare the STS result with the expected sandbox account before writing it to `terraform.tfvars`. Terraform's `allowed_account_ids` check enforces that account during planning and apply. With a current AWS CLI, check the exact source bucket using `aws s3api head-bucket` with `--expected-bucket-owner`, and the vector bucket and index using `aws s3vectors get-vector-bucket` and `get-index`, all against `eu-central-1`. Investigate permission errors before proceeding. If any pilot resource already exists and is unmanaged, import it into this state using its existing name or ARN. Check for an owning Terraform state before importing. [AWS provider account guard](https://registry.terraform.io/providers/hashicorp/aws/latest/docs) · [Terraform import](https://developer.hashicorp.com/terraform/cli/commands/import)

Initialize the locked provider, then validate and review a saved plan:

```sh
terraform init -lockfile=readonly
terraform fmt -check
terraform validate
terraform plan -out=sec-dev.tfplan
terraform show sec-dev.tfplan
```

Apply only after explicit deployment approval, using `terraform apply sec-dev.tfplan`. Review a fresh plan if the configuration or state has changed. After apply, run `terraform output -json ingestion_environment` and copy the values into the Python project's `.env`. The Python commands read these identifiers and manage filing data only. Before any approved teardown, review `terraform plan -destroy` and preserve needed evidence. Deleting the vector index deletes its vectors.
