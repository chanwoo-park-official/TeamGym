"""Provider-neutral model actions for the source-restricted patch harness.

Configuration resolution is shared with diagnostic receipts. Responses requests
preserve the caller's actual fields; configured temperature is not implicitly
inserted into that transport.
"""
import json
import re
from types import SimpleNamespace
from urllib.parse import quote
from teamgym.llm.construction_client import OpenAIJsonClient, OpenAIJsonClientConfig

MODELS = {
    'gpt55': 'gpt-5.5',
    'sonnet45': 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
    'haiku45': 'global.anthropic.claude-haiku-4-5-20251001-v1:0',
    'gemma3_4b': 'google.gemma-3-4b-it',
    'gemma3_12b': 'google.gemma-3-12b-it',
    'gemma3_27b': 'google.gemma-3-27b-it',
    'qwen3_32b': 'qwen.qwen3-32b',
    'qwen3_235b_a22b': 'qwen.qwen3-235b-a22b-2507',
}

ANTHROPIC_MODELS = {'sonnet45', 'haiku45'}
CHAT_MODELS = set(MODELS) - ANTHROPIC_MODELS - {'gpt55'}
OPENAI_PROVIDERS = {'openai', 'openai_compatible', 'bedrock_openai'}


def resolve_actor_config(model, *, provider=None, base_url=None, api_key_env=None,
                         reasoning_effort=None):
    """Resolve transport defaults without constructing a client or reading keys."""
    slug = model if model in MODELS else None
    resolved_provider = provider or (
        'bedrock_anthropic_runtime' if slug in ANTHROPIC_MODELS else 'bedrock_openai')
    if resolved_provider not in OPENAI_PROVIDERS | {'bedrock_anthropic_runtime'}:
        raise ValueError(f'Unsupported repair-actor provider: {resolved_provider}')
    if resolved_provider in OPENAI_PROVIDERS:
        endpoint = '/chat/completions' if slug in CHAT_MODELS else '/responses'
        default_url = ('https://api.openai.com/v1' if resolved_provider == 'openai'
            else 'https://bedrock-mantle.us-east-2.api.aws/v1' if endpoint == '/chat/completions'
            else 'https://bedrock-mantle.us-east-2.api.aws/openai/v1')
    else:
        endpoint = '/invoke'
        default_url = 'https://bedrock-runtime.us-east-1.amazonaws.com'
    return {'model_slug': model, 'model': MODELS.get(model, model),
            'provider': resolved_provider, 'base_url': base_url or default_url,
            'endpoint': endpoint, 'api_key_env': api_key_env or (
                'OPENAI_API_KEY' if resolved_provider == 'openai' else 'AWS_BEARER_TOKEN_BEDROCK'),
            'reasoning_effort': reasoning_effort if reasoning_effort is not None
                else ('medium' if slug == 'gpt55' else None),
            'temperature': 0}


def anthropic_body(body):
    system, messages = [], []
    for row in body['input']:
        if row.get('role') == 'developer':
            system.append(row['content'])
        elif row.get('type') == 'function_call':
            messages.append({'role': 'assistant', 'content': [{'type': 'tool_use',
                'id': row['call_id'], 'name': row['name'], 'input': json.loads(row['arguments'])}]})
        elif row.get('type') == 'function_call_output':
            messages.append({'role': 'user', 'content': [{'type': 'tool_result',
                'tool_use_id': row['call_id'], 'content': row['output']}]})
        elif row.get('role') in ('user', 'assistant'):
            messages.append({'role': row['role'], 'content': [{'type': 'text', 'text': row['content']}]})
    merged = []
    for message in messages:
        if merged and merged[-1]['role'] == message['role']:
            merged[-1]['content'].extend(message['content'])
        else:
            merged.append(message)
    tool = body['tools'][0]
    return {'anthropic_version': 'bedrock-2023-05-31', 'system': '\n\n'.join(system),
            'messages': merged, 'max_tokens': body['max_output_tokens'], 'temperature': 0,
            'tools': [{'name': tool['name'], 'description': tool.get('description', ''),
                       'input_schema': tool['parameters']}],
            'tool_choice': {'type': 'tool', 'name': tool['name']}}


class ModelActions:
    def __init__(self, model, env_file, *, provider=None, base_url=None,
                 api_key_env=None, reasoning_effort=None):
        settings = resolve_actor_config(model, provider=provider, base_url=base_url,
            api_key_env=api_key_env, reasoning_effort=reasoning_effort)
        self.slug = model if model in MODELS else None
        self.model = settings['model']
        self.provider = settings['provider']
        self.http = OpenAIJsonClient(OpenAIJsonClientConfig(
            model=self.model, provider=settings['provider'], endpoint=settings['endpoint'],
            base_url=settings['base_url'], api_key_env=settings['api_key_env'],
            env_file=str(env_file), temperature=0,
            reasoning_effort=settings['reasoning_effort'],
            timeout_sec=900, request_retry_attempts=1, max_output_tokens=16000))
        self._client = SimpleNamespace(responses=self)

    def create(self, **body):
        if self.http.cfg.endpoint == '/responses':
            response = self.http._client.responses.create(**{**body, 'model': self.http.cfg.model})
            return response
        if self.http.cfg.endpoint == '/chat/completions':
            return self._create_chat(**body)
        payload = self.http._http_json_request(
            url=f'{self.http.cfg.base_url}/model/{quote(self.model, safe=".")}/invoke',
            headers={'Authorization': f'Bearer {self.http.api_key}', 'Content-Type': 'application/json'},
            body=anthropic_body(body))
        output = [{'type': 'function_call', 'name': c['name'], 'call_id': c['id'],
                   'arguments': json.dumps(c['input'], ensure_ascii=False)}
                  for c in payload.get('content', []) if c.get('type') == 'tool_use']
        usage = payload.get('usage', {})
        # A complete tool-use block is actionable even when the provider marks
        # the response max_tokens: the model may have emitted the tool input
        # before exhausting its remaining generation budget.
        normalized = {'status': 'completed' if len(output) == 1 else 'incomplete',
                      'output': output, 'usage': usage, 'native_response': payload}
        return SimpleNamespace(model_dump=lambda: normalized)

    def _create_chat(self, **body):
        messages = []
        for row in body['input']:
            role = row.get('role')
            if role == 'developer':
                messages.append({'role': 'system', 'content': row['content']})
            elif row.get('type') == 'function_call':
                # Several OpenAI-compatible Bedrock endpoints reject a native
                # tool/tool-result history on the *next* call when it is mixed
                # with the Responses-style adjacent user messages.  Keep the
                # completed action as an ordinary alternating transcript.  The
                # current turn still supplies the real tool schema and forces
                # a native tool call below, so this only normalizes history.
                messages.append({'role': 'assistant', 'content': (
                    f"{row['name']} call arguments:\n{row.get('arguments') or '{}'}")})
            elif row.get('type') == 'function_call_output':
                messages.append({'role': 'user', 'content': (
                    f"Action feedback:\n{row.get('output') or ''}")})
            elif role in {'user', 'assistant'}:
                messages.append({'role': role, 'content': row['content']})
        # Responses-style input permits adjacent user rows (evidence followed
        # by the remaining-budget message). OpenAI-compatible chat endpoints
        # require alternating user/assistant turns, so merge adjacent text rows
        # without changing their order.
        merged = []
        for message in messages:
            if (merged and message['role'] in {'user', 'assistant'}
                    and merged[-1]['role'] == message['role']
                    and isinstance(merged[-1].get('content'), str)
                    and isinstance(message.get('content'), str)):
                merged[-1]['content'] += '\n\n' + message['content']
            else:
                merged.append(message)
        messages = merged
        tool = body['tools'][0]
        schema_instruction = (
            "\n\nYou must answer with exactly one JSON object containing the arguments "
            f"for the `{tool['name']}` action. Do not use markdown. The JSON schema is: "
            + json.dumps(tool['parameters'], ensure_ascii=False)
        )
        for message in messages:
            if message['role'] == 'system':
                message['content'] += schema_instruction
                break
        kwargs = {
            'model': self.model,
            'messages': messages,
            'tools': [{'type': 'function', 'function': {
                'name': tool['name'], 'description': tool.get('description', ''),
                'parameters': tool['parameters']}}],
            'tool_choice': {'type': 'function', 'function': {'name': tool['name']}},
            'temperature': 0,
            'max_completion_tokens': body['max_output_tokens'],
        }
        try:
            response = self.http._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if 'max_completion_tokens' not in str(exc).lower():
                raise
            kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
            response = self.http._client.chat.completions.create(**kwargs)
        raw = response.model_dump()
        choice = (raw.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        output = []
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            output.append({'type': 'function_call', 'name': function.get('name'),
                           'call_id': call.get('id'), 'arguments': function.get('arguments', '{}')})
        if not output and isinstance(message.get('content'), str):
            text = message['content'].strip()
            fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
            if fenced:
                text = fenced.group(1)
            else:
                start, end = text.find('{'), text.rfind('}')
                text = text[start:end + 1] if start >= 0 and end > start else text
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                output.append({'type': 'function_call', 'name': tool['name'],
                               'call_id': 'json_action',
                               'arguments': json.dumps(parsed, ensure_ascii=False)})
        raw_usage = raw.get('usage') or {}
        usage = {'input_tokens': raw_usage.get('prompt_tokens'),
                 'output_tokens': raw_usage.get('completion_tokens'),
                 'total_tokens': raw_usage.get('total_tokens')}
        normalized = {'status': 'completed' if len(output) == 1 else 'incomplete',
                      'output': output, 'usage': usage, 'native_response': raw}
        return SimpleNamespace(model_dump=lambda: normalized)
