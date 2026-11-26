"""Actor routing and recorded identities agree without credentials or HTTP."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.software.deep_swe import model_actions
from teamgym.data.software.deep_swe.stages.diagnostic_runs import actor_identity


@pytest.mark.parametrize('model,options,provider,endpoint', [
    ('gpt55', {}, 'bedrock_openai', '/responses'),
    ('gpt-5.5', {}, 'bedrock_openai', '/responses'),
    ('custom-model', {}, 'bedrock_openai', '/responses'),
    ('sonnet45', {}, 'bedrock_anthropic_runtime', '/invoke'),
    ('haiku45', {}, 'bedrock_anthropic_runtime', '/invoke'),
    ('qwen3_32b', {}, 'bedrock_openai', '/chat/completions'),
    ('gemma3_4b', {}, 'bedrock_openai', '/chat/completions'),
    ('sonnet45', {'provider': 'openai_compatible'}, 'openai_compatible', '/responses'),
    ('custom-model', {'provider': 'openai'}, 'openai', '/responses'),
    ('custom-model', {'provider': 'bedrock_anthropic_runtime'}, 'bedrock_anthropic_runtime', '/invoke'),
])
def test_actor_client_and_diagnostic_receipt_share_transport_defaults(
        monkeypatch, tmp_path, model, options, provider, endpoint):
    monkeypatch.setattr(model_actions, 'OpenAIJsonClient', lambda cfg: SimpleNamespace(cfg=cfg))
    env_file = tmp_path / 'unused.env'
    actor = model_actions.ModelActions(model, env_file, **options)
    recorded = actor_identity({'model_slug': model, 'env_file': str(env_file), **options})
    assert recorded['env_file'] == str(env_file.resolve())
    for key in ('model', 'provider', 'base_url', 'endpoint', 'api_key_env', 'reasoning_effort', 'temperature'):
        assert getattr(actor.http.cfg, key) == recorded[key]
    assert recorded['provider'] == provider
    assert recorded['endpoint'] == endpoint
    assert not env_file.exists()


def test_explicit_model_configuration_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(model_actions, 'OpenAIJsonClient', lambda cfg: SimpleNamespace(cfg=cfg))
    options = dict(provider='openai_compatible', base_url='https://example.invalid/v1',
                   api_key_env='EXPLICIT_KEY_NAME', reasoning_effort='high')
    actor = model_actions.ModelActions('custom-model', tmp_path / 'unused.env', **options)
    for key, value in options.items():
        assert getattr(actor.http.cfg, key) == value
    official = model_actions.resolve_actor_config('custom-model', provider='openai')
    assert official['base_url'] == 'https://api.openai.com/v1'
    assert official['api_key_env'] == 'OPENAI_API_KEY'


def test_unsupported_actor_provider_fails_before_client_creation(monkeypatch):
    def unexpected_client(cfg):
        raise AssertionError('No client should be created for an unsupported provider')
    monkeypatch.setattr(model_actions, 'OpenAIJsonClient', unexpected_client)
    with pytest.raises(ValueError, match='Unsupported repair-actor provider'):
        model_actions.ModelActions('custom-model', Path('unused.env'), provider='unsupported')


def test_responses_transport_preserves_the_actual_request(monkeypatch):
    calls = []
    response = SimpleNamespace(model_dump=lambda: {'status': 'completed'})
    def create(**body):
        calls.append(body)
        return response
    def client(cfg):
        return SimpleNamespace(cfg=cfg, _client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr(model_actions, 'OpenAIJsonClient', client)
    actor = model_actions.ModelActions('gpt55', Path('unused.env'))
    body = {'model': 'placeholder', 'input': [{'role': 'user', 'content': 'public evidence'}],
            'max_output_tokens': 123, 'reasoning': {'effort': 'medium'}}
    assert actor.create(**body) is response
    assert calls == [{**body, 'model': actor.http.cfg.model}]
    assert 'temperature' not in calls[0]


def test_dispatch_uses_resolved_endpoint_instead_of_model_family(monkeypatch):
    monkeypatch.setattr(model_actions, 'OpenAIJsonClient', lambda cfg: SimpleNamespace(cfg=cfg))
    actor = model_actions.ModelActions('qwen3_32b', Path('unused.env'))
    received = []
    monkeypatch.setattr(actor, '_create_chat', lambda **body: received.append(body) or 'chat')
    assert actor.create(input=[]) == 'chat'
    assert received == [{'input': []}]
