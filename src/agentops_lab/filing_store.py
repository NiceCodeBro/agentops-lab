"""Titan embedding and S3 Vectors search for SEC filing excerpts."""

import json
import math
import os
from contextlib import closing
from dataclasses import dataclass
from itertools import batched

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from agentops_lab.research_models import DocumentExcerpt, NonEmptyText
from agentops_lab.sec import FilingCorpus

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
SUPPORTED_DIMENSIONS = (256, 512, 1024)


class FilingStoreSettings(BaseModel):
    source_bucket: NonEmptyText
    vector_bucket: NonEmptyText
    index_name: NonEmptyText
    dimensions: int = 256
    max_distance: float = Field(default=0.70, ge=0, le=2)


@dataclass(frozen=True)
class FilingHit:
    excerpt: DocumentExcerpt
    distance: float


def embed_text(runtime: BaseClient, text: str, dimensions: int) -> list[float]:
    if dimensions not in SUPPORTED_DIMENSIONS:
        raise ValueError("Titan Text Embeddings V2 requires 256, 512, or 1024 dimensions.")
    if not text.strip():
        raise ValueError("Embedding input cannot be empty.")

    payload = {"inputText": text, "dimensions": dimensions, "normalize": True}
    response = runtime.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )
    with closing(response["body"]) as body:
        data = json.load(body)
    vector = data["embedding"]
    if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"Titan did not return a finite {dimensions}-dimensional vector.")
    if not any(vector):
        raise ValueError("Titan returned a zero vector for a cosine index.")
    return vector


class FilingStore:
    def __init__(
        self,
        settings: FilingStoreSettings,
        *,
        runtime: BaseClient,
        vectors: BaseClient,
        source: BaseClient,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.vectors = vectors
        self.source = source

    def validate_index(self) -> None:
        dimensions = self.settings.dimensions
        if dimensions not in SUPPORTED_DIMENSIONS:
            raise ValueError("Titan Text Embeddings V2 requires 256, 512, or 1024 dimensions.")
        response = self.vectors.get_index(
            vectorBucketName=self.settings.vector_bucket, indexName=self.settings.index_name
        )
        index = response["index"]
        if (
            index["dimension"] != dimensions
            or index["distanceMetric"] != "cosine"
            or index["dataType"] != "float32"
        ):
            raise ValueError(
                f"Expected a float32 cosine S3 vector index with {dimensions} dimensions."
            )

    def read_corpus(self, key: str) -> FilingCorpus:
        response = self.source.get_object(Bucket=self.settings.source_bucket, Key=key)
        with closing(response["Body"]) as body:
            data = body.read()
        return FilingCorpus.model_validate_json(data)

    def ingest(self, corpus: FilingCorpus) -> int:
        self.validate_index()
        filings = [excerpt.filing for excerpt in corpus.chunks]
        first = filings[0]
        if first is None or any(
            filing is None
            or filing.accession_number != first.accession_number
            or filing.ticker != first.ticker
            for filing in filings
        ):
            raise ValueError("A corpus must contain excerpts from one identified SEC filing.")

        source_key = f"sec/{first.ticker}/{first.accession_number}.json"
        body = corpus.model_dump_json()
        try:
            self.source.put_object(
                Bucket=self.settings.source_bucket,
                Key=source_key,
                Body=body.encode(),
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
                raise
            existing = self.read_corpus(source_key)
            if existing != corpus:
                raise ValueError(
                    "Existing source differs from this corpus. Use a new source location and index."
                ) from error

        for batch in batched(corpus.chunks, 100):
            vectors = []
            for excerpt in batch:
                filing = excerpt.filing
                assert filing is not None  # Ingestion verified that every excerpt has provenance.
                embedding_text = f"{excerpt.title}\n\n{excerpt.text}"
                embedding = embed_text(self.runtime, embedding_text, self.settings.dimensions)
                vectors.append(
                    {
                        "key": excerpt.source_id,
                        "data": {"float32": embedding},
                        "metadata": {
                            "ticker": first.ticker,
                            "source_key": source_key,
                            "accession_number": filing.accession_number,
                            "section": filing.section,
                            "filing_date": excerpt.document_date.isoformat(),
                        },
                    }
                )
            self.vectors.put_vectors(
                vectorBucketName=self.settings.vector_bucket,
                indexName=self.settings.index_name,
                vectors=vectors,
            )
        return len(corpus.chunks)

    def search_filings(self, query: str, ticker: str, top_k: int) -> list[FilingHit]:
        ticker = ticker.strip().upper()
        if not ticker or not query.strip() or not 1 <= top_k <= 10:
            raise ValueError("Provide a query, a ticker, and a top_k between 1 and 10.")
        self.validate_index()
        embedding = embed_text(self.runtime, query, self.settings.dimensions)
        response = self.vectors.query_vectors(
            vectorBucketName=self.settings.vector_bucket,
            indexName=self.settings.index_name,
            topK=top_k,
            queryVector={"float32": embedding},
            filter={"ticker": {"$eq": ticker}},
            returnMetadata=True,
            returnDistance=True,
        )

        hits = []
        corpora: dict[str, dict[str, DocumentExcerpt]] = {}
        for vector in sorted(response["vectors"], key=lambda item: (item["distance"], item["key"])):
            distance = vector["distance"]
            metadata = vector["metadata"]
            if metadata["ticker"] != ticker or not 0 <= distance <= self.settings.max_distance:
                continue
            source_key = metadata["source_key"]
            if source_key not in corpora:
                corpus = self.read_corpus(source_key)
                corpora[source_key] = {excerpt.source_id: excerpt for excerpt in corpus.chunks}
            excerpt = corpora[source_key].get(vector["key"])
            if excerpt is None or excerpt.filing is None or excerpt.filing.ticker != ticker:
                raise ValueError(f"Vector {vector['key']} has no matching filing provenance in S3.")
            hits.append(FilingHit(excerpt=excerpt, distance=distance))
        return hits


def load_filing_store() -> FilingStore:
    settings = FilingStoreSettings(
        source_bucket=os.getenv("SEC_SOURCE_BUCKET", ""),
        vector_bucket=os.getenv("S3_VECTOR_BUCKET", ""),
        index_name=os.getenv("S3_VECTOR_INDEX", ""),
        dimensions=int(os.getenv("SEC_EMBEDDING_DIMENSIONS", "256")),
        max_distance=float(os.getenv("SEC_MAX_DISTANCE", "0.70")),
    )
    session = boto3.Session(region_name=os.getenv("AWS_REGION") or None)
    if not session.region_name:
        raise ValueError("Set AWS_REGION or configure a region in your AWS profile.")
    config = Config(
        connect_timeout=5, read_timeout=30, retries={"mode": "standard", "total_max_attempts": 3}
    )
    runtime = session.client("bedrock-runtime", config=config)
    vectors = session.client("s3vectors", config=config)
    source = session.client("s3", config=config)
    return FilingStore(settings, runtime=runtime, vectors=vectors, source=source)
