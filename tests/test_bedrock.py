from botocore.stub import ANY
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from agentops_lab.bedrock import build_structured_model, load_bedrock_model


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
