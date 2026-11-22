import json
from datetime import UTC, datetime
from io import BytesIO

import boto3
import pytest
from botocore.stub import Stubber

from agentops_lab.filing_store import (
    EMBEDDING_MODEL_ID,
    FilingStore,
    FilingStoreSettings,
    embed_text,
)
from agentops_lab.sec import PILOT_FILING, FilingCorpus, create_chunks

INDEX = {"vectorBucketName": "test-vectors", "indexName": "test-index"}
VECTOR = [0.0625] * 256
OBJECT = {"Bucket": "test-filing-source", "Key": f"sec/NVDA/{PILOT_FILING.accession_number}.json"}


@pytest.fixture
def service_stubs(monkeypatch):
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    session = boto3.Session(
        aws_access_key_id="testing", aws_secret_access_key="testing", region_name="eu-central-1"
    )
    runtime = session.client("bedrock-runtime")
    vectors = session.client("s3vectors")
    source = session.client("s3")
    settings = FilingStoreSettings(
        vector_bucket=INDEX["vectorBucketName"],
        index_name=INDEX["indexName"],
        source_bucket=OBJECT["Bucket"],
    )
    store = FilingStore(settings, runtime=runtime, vectors=vectors, source=source)
    with Stubber(runtime) as titan, Stubber(vectors) as vector_index, Stubber(source) as objects:
        yield store, titan, vector_index, objects
        for stub in (titan, vector_index, objects):
            stub.assert_no_pending_responses()


@pytest.fixture
def filing_corpus():
    sections = {
        "1": ["Main business segment."],
        "1A": ["Supplier risks."],
        "7": ["Revenue growth."],
    }
    chunks = create_chunks(PILOT_FILING, sections)
    return FilingCorpus(html_sha256="a" * 64, chunks=chunks)


def stub_index(stub, dimension=256):
    stub.add_response(
        "get_index",
        {
            "index": {
                **INDEX,
                "indexArn": "test-index-arn",
                "creationTime": datetime(2026, 1, 1, tzinfo=UTC),
                "dataType": "float32",
                "dimension": dimension,
                "distanceMetric": "cosine",
            }
        },
        INDEX,
    )


def stub_embedding(stub, text, vector=VECTOR):
    body = BytesIO(json.dumps({"embedding": vector}).encode())
    stub.add_response(
        "invoke_model",
        {"body": body, "contentType": "application/json"},
        {
            "modelId": EMBEDDING_MODEL_ID,
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({"inputText": text, "dimensions": 256, "normalize": True}),
        },
    )


def test_index_and_embedding_dimensions_are_validated(service_stubs):
    store, titan, vectors, _ = service_stubs
    stub_index(vectors, dimension=512)
    with pytest.raises(ValueError, match="256 dimensions"):
        store.search_filings("suppliers", "NVDA", top_k=3)

    with pytest.raises(ValueError, match="requires 256, 512, or 1024"):
        embed_text(store.runtime, "supplier risk", 200)
    stub_embedding(titan, "supplier risk", vector=[0.2])
    with pytest.raises(ValueError, match="256-dimensional vector"):
        embed_text(store.runtime, "supplier risk", 256)


@pytest.mark.parametrize("cutoff", [None, 0.70])
def test_search_prefers_lower_distances_and_only_filters_with_explicit_cutoff(
    service_stubs,
    filing_corpus,
    cutoff,
):
    store, titan, vectors, source = service_stubs
    store.settings.max_distance = cutoff
    stub_index(vectors)
    stub_embedding(titan, "supply chain")
    business, risk, discussion = filing_corpus.chunks
    metadata = {"ticker": "NVDA", "source_key": OBJECT["Key"]}
    vectors.add_response(
        "query_vectors",
        {
            "distanceMetric": "cosine",
            "vectors": [
                {"key": business.source_id, "distance": 0.41, "metadata": metadata},
                {"key": risk.source_id, "distance": 0.31, "metadata": metadata},
                {"key": discussion.source_id, "distance": 0.7035, "metadata": metadata},
                {
                    "key": "wrong-company",
                    "distance": 0.20,
                    "metadata": {**metadata, "ticker": "OTHER"},
                },
            ],
        },
        {
            **INDEX,
            "topK": 4,
            "queryVector": {"float32": VECTOR},
            "filter": {"ticker": {"$eq": "NVDA"}},
            "returnMetadata": True,
            "returnDistance": True,
        },
    )
    body = BytesIO(filing_corpus.model_dump_json().encode())
    source.add_response("get_object", {"Body": body}, OBJECT)

    hits = store.search_filings("supply chain", " nvda ", top_k=4)

    expected = [risk, business] if cutoff else [risk, business, discussion]
    assert [hit.excerpt for hit in hits] == expected
    assert len(hits) == len(expected)


def test_empty_query_result_does_not_load_source_objects(service_stubs):
    store, titan, vectors, _ = service_stubs
    stub_index(vectors)
    stub_embedding(titan, "supply chain")
    vectors.add_response(
        "query_vectors",
        {"distanceMetric": "cosine", "vectors": []},
        {
            **INDEX,
            "topK": 3,
            "queryVector": {"float32": VECTOR},
            "filter": {"ticker": {"$eq": "NVDA"}},
            "returnMetadata": True,
            "returnDistance": True,
        },
    )
    assert store.search_filings("supply chain", "NVDA", top_k=3) == []


def test_ingestion_reuses_the_same_object_and_vector_keys(service_stubs, filing_corpus):
    store, titan, vectors, source = service_stubs
    encoded = filing_corpus.model_dump_json().encode()
    put_request = {**OBJECT, "Body": encoded, "ContentType": "application/json", "IfNoneMatch": "*"}
    put_vectors = {**INDEX, "vectors": []}
    for excerpt in filing_corpus.chunks:
        put_vectors["vectors"].append(
            {
                "key": excerpt.source_id,
                "data": {"float32": VECTOR},
                "metadata": {
                    "ticker": "NVDA",
                    "source_key": OBJECT["Key"],
                    "accession_number": PILOT_FILING.accession_number,
                    "section": excerpt.filing.section,
                    "filing_date": "2026-02-25",
                },
            }
        )

    for _ in range(2):
        stub_index(vectors)
        for excerpt in filing_corpus.chunks:
            stub_embedding(titan, f"{excerpt.title}\n\n{excerpt.text}")
        vectors.add_response("put_vectors", {}, put_vectors)
    source.add_response("put_object", {}, put_request)
    source.add_client_error("put_object", "PreconditionFailed", expected_params=put_request)
    source.add_response("get_object", {"Body": BytesIO(encoded)}, OBJECT)

    assert store.ingest(filing_corpus) == store.ingest(filing_corpus) == 3
