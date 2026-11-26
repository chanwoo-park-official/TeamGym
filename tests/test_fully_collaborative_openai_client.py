from teamgym.llm.openai_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
    normalize_model_for_provider,
)


def test_bedrock_openai_prefixes_openai_gpt_model_ids() -> None:
    assert normalize_model_for_provider("bedrock_openai", "gpt-5.5") == "openai.gpt-5.5"
    assert normalize_model_for_provider("bedrock_openai", "gpt-oss-120b") == "openai.gpt-oss-120b"


def test_bedrock_openai_keeps_provider_native_model_ids() -> None:
    assert normalize_model_for_provider("bedrock_openai", "qwen.qwen3-32b-v1:0") == "qwen.qwen3-32b-v1:0"
    assert normalize_model_for_provider("bedrock_openai", "openai.gpt-5.5") == "openai.gpt-5.5"


def test_direct_openai_model_ids_are_not_rewritten() -> None:
    assert normalize_model_for_provider("openai", "gpt-5.5") == "gpt-5.5"
    assert normalize_model_for_provider("openai_compatible", "gpt-5.5") == "gpt-5.5"


def test_anthropic_mantle_request_uses_messages_api() -> None:
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_anthropic_mantle",
            model="anthropic.claude-haiku-4-5",
            base_url="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1",
            endpoint="/messages",
            api_key="test-key",
        )
    )
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    client._http_json_request = fake_request  # type: ignore[method-assign]
    payload = client._request("system", "user")

    assert captured["url"].endswith("/anthropic/v1/messages")
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["body"]["model"] == "anthropic.claude-haiku-4-5"
    assert client._extract_output_text(payload) == '{"ok": true}'


def test_anthropic_runtime_request_uses_encoded_inference_profile() -> None:
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_anthropic_runtime",
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            endpoint="/invoke",
            api_key="test-key",
        )
    )
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    client._http_json_request = fake_request  # type: ignore[method-assign]
    client._request("system", "user")

    assert captured["url"].endswith(
        "/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke"
    )
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["body"]["anthropic_version"] == "bedrock-2023-05-31"
    assert "model" not in captured["body"]
