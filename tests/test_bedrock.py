import pytest
from botocore.stub import ANY
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from agentops_lab.bedrock import build_agent_reasoner, build_structured_model, load_bedrock_model


class Summary(BaseModel):
    summary: str


def test_json_mode_uses_caller_schema_and_preserves_first_message(bedrock_stub):
    bedrock_stub.add_response(
        "converse",
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": '{"summary":"Example summary"}'}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
            "metrics": {"latencyMs": 10},
        },
        {
            "modelId": "eu.amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Summarize this passage."}]}],
            "system": [{"text": ANY}],
            "inferenceConfig": {"maxTokens": 64, "temperature": 0},
        },
    )

    model = load_bedrock_model(max_tokens=64)
    summarize = build_structured_model(model, Summary, output_mode="json")
    messages: list[BaseMessage] = [HumanMessage("Summarize this passage.")]
    result = summarize.invoke(messages)

    assert result == Summary(summary="Example summary")


@pytest.mark.parametrize(
    ("content", "output", "validation_codes"),
    [
        ('{"summary":"One public finding"}', Summary(summary="One public finding"), ()),
        (
            '```json\n{"summary":"One public finding"}\n```',
            Summary(summary="One public finding"),
            (),
        ),
        ('{"summary":["private malformed output"]}', None, ("string_type",)),
    ],
)
def test_investment_reasoner_preserves_usage_without_exposing_raw_reply(
    bedrock_stub, content, output, validation_codes
):
    bedrock_stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": content}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
            "metrics": {"latencyMs": 10},
        },
        {
            "modelId": "eu.amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Summarize."}]}],
            "system": [{"text": ANY}],
            "inferenceConfig": {"maxTokens": 64, "temperature": 0},
        },
    )

    reasoner = build_agent_reasoner(load_bedrock_model(max_tokens=64))
    reply = reasoner(Summary, [HumanMessage("Summarize.")])
    assert reply.output == output
    assert (reply.input_tokens, reply.output_tokens) == (12, 5)
    assert reply.validation_codes == validation_codes
    assert "private malformed output" not in repr(reply)
