import boto3
import pytest
from botocore.stub import Stubber


@pytest.fixture
def bedrock_stub(monkeypatch):
    region = "eu-central-1"
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    session = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name=region,
    )
    clients = {service: session.client(service) for service in ("bedrock-runtime", "bedrock")}

    def stubbed_client(service_name, *, config):
        assert config.retries["total_max_attempts"] == 1
        return clients[service_name]

    def stubbed_session(*, region_name):
        assert region_name == region
        return session

    monkeypatch.setattr(session, "client", stubbed_client)
    monkeypatch.setattr(boto3, "Session", stubbed_session)
    monkeypatch.setenv("AWS_REGION", region)
    monkeypatch.setenv("BEDROCK_MODEL_ID", "eu.amazon.nova-micro-v1:0")
    with Stubber(clients["bedrock-runtime"]) as runtime, Stubber(clients["bedrock"]):
        yield runtime
        runtime.assert_no_pending_responses()
