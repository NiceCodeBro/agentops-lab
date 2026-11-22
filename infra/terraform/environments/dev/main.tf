resource "aws_s3_bucket" "source" {
  bucket        = var.source_bucket_name
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "source" {
  bucket = aws_s3_bucket.source.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "source" {
  bucket = aws_s3_bucket.source.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3vectors_vector_bucket" "filings" {
  vector_bucket_name = var.vector_bucket_name
  force_destroy      = false

  encryption_configuration {
    sse_type = "AES256"
  }
}

resource "aws_s3vectors_index" "filings" {
  index_name         = var.vector_index_name
  vector_bucket_name = aws_s3vectors_vector_bucket.filings.vector_bucket_name
  data_type          = "float32"
  dimension          = var.embedding_dimensions
  distance_metric    = "cosine"
}
