output "ingestion_environment" {
  description = "Copy these values into the local Python .env after the approved apply."
  value = {
    AWS_REGION               = var.aws_region
    SEC_SOURCE_BUCKET        = aws_s3_bucket.source.bucket
    S3_VECTOR_BUCKET         = aws_s3vectors_vector_bucket.filings.vector_bucket_name
    S3_VECTOR_INDEX          = aws_s3vectors_index.filings.index_name
    SEC_EMBEDDING_DIMENSIONS = tostring(aws_s3vectors_index.filings.dimension)
  }
}

output "vector_bucket_arn" {
  description = "ARN of the private S3 Vectors bucket."
  value       = aws_s3vectors_vector_bucket.filings.vector_bucket_arn
}

output "vector_index_arn" {
  description = "ARN of the SEC filing vector index."
  value       = aws_s3vectors_index.filings.index_arn
}
