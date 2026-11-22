variable "aws_region" {
  description = "AWS region for the SEC pilot."
  type        = string
  default     = "eu-central-1"

  validation {
    condition     = var.aws_region == "eu-central-1"
    error_message = "The dev SEC pilot must use eu-central-1."
  }
}

variable "aws_profile" {
  description = "Local AWS SSO profile. If omitted, the provider uses the AWS credential chain."
  type        = string
  default     = null
}

variable "aws_account_id" {
  description = "Expected AWS sandbox account ID, verified independently before planning or applying."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "Supply the 12-digit sandbox AWS account ID."
  }
}

variable "source_bucket_name" {
  description = "Unique name for the private S3 bucket that stores SEC filing excerpts."
  type        = string
  nullable    = false
}

variable "vector_bucket_name" {
  description = "Name of the private S3 Vectors bucket for the SEC pilot."
  type        = string
  nullable    = false
}

variable "vector_index_name" {
  description = "Name of the vector index within the vector bucket."
  type        = string
  default     = "nvda-10k-v1"
}

variable "embedding_dimensions" {
  description = "Titan Text Embeddings V2 dimensions. A change requires a replacement index."
  type        = number
  default     = 256

  validation {
    condition     = contains([256, 512, 1024], var.embedding_dimensions)
    error_message = "Titan Text Embeddings V2 requires 256, 512, or 1024 dimensions."
  }
}
