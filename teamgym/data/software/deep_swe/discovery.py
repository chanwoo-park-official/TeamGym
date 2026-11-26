#!/usr/bin/env python3
"""Discover task-relevant base-commit source using constructor-only signals.

Reference patches and hidden tests guide discovery but never enter solver
packets. Reads are token-bounded and termination is capped by ``--max-rounds``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import time

ROOT = Path.cwd()
from teamgym.data.software.deep_swe.prompts import load_prompt
from teamgym.data.software.deep_swe.source import load_task
from teamgym.llm.construction_client import OpenAIJsonClient, OpenAIJsonClientConfig
from teamgym.tokenization import count_tokens
from teamgym.data.software.deep_swe.stages.patch_runtime import docker_python, save


SYSTEM = load_prompt("code_discovery_system.txt")

TOOL = {
    'type': 'function', 'name': 'perform_action',
    'description': 'Request source files to read, or declare discovery done.',
    'strict': False,
    'parameters': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['read', 'done']},
        'note': {'type': 'string'},
        'paths': {'type': 'array', 'items': {'type': 'string'}},
        'line_start': {'type': 'integer', 'minimum': 1},
    }, 'required': ['action'], 'additionalProperties': False},
}

LIST_FILES = r'''
import json, subprocess, sys
r = json.load(sys.stdin)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == r['commit']
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
out = subprocess.check_output(['git', 'ls-files'], text=True)
print(json.dumps({'paths': out.splitlines()}))
'''

READ_FILES = r'''
import json, pathlib, subprocess, sys
r = json.load(sys.stdin)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == r['commit']
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True)
tracked = set(subprocess.check_output(['git', 'ls-files'], text=True).splitlines())
files = {}
for path in r['paths']:
    if path not in tracked:
        files[path] = {'kind': 'absent'}
        continue
    raw = subprocess.check_output(['git', 'show', 'HEAD:' + path])
    try:
        files[path] = {'kind': 'file', 'text': raw.decode('utf-8')}
    except UnicodeDecodeError:
        files[path] = {'kind': 'unsupported'}
print(json.dumps({'files': files}))
'''


def valid_repo_path(path) -> bool:
    return (isinstance(path, str) and bool(path) and '\\' not in path
            and not path.startswith('/')
            and all(part not in {'', '.', '..', '.git'} for part in path.split('/'))
            and str(PurePosixPath(path)) == path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-rounds', type=int, default=10)
    ap.add_argument('--round-token-budget', type=int, default=20000,
                    help='Cap on cl100k tokens of file text served in one round.')
    ap.add_argument('--total-output-tokens', type=int, default=60000)
    ap.add_argument('--model', default='gpt-5.4')
    ap.add_argument('--env-file', default=str(ROOT / '.env'))
    ap.add_argument('--runtime-image')
    ap.add_argument('--provider')
    ap.add_argument('--base-url')
    ap.add_argument('--api-key-env')
    ap.add_argument('--reasoning-effort', default='medium')
    ap.add_argument('--resume-from', type=Path)
    ap.add_argument('--gap-feedback', type=Path)
    args = ap.parse_args()

    task = load_task(args.task_dir, runtime_image=args.runtime_image)
    image = task.runtime.docker_image
    commit = task.repository.base_commit_hash
    reference_patch = task.artifacts.solution_patch.read_text(errors='replace')
    hidden_test_patch = task.artifacts.test_patch.read_text(errors='replace')
    f2p_ids = list(task.test_config.get('f2p_node_ids') or [])

    tree = docker_python(image, LIST_FILES, {'commit': commit})['paths']

    request0 = {
        'task_id': task.task_id,
        'instruction': task.instruction,
        'reference_patch': reference_patch,
        'hidden_test_patch': hidden_test_patch,
        'hidden_test_ids': f2p_ids,
        'repository_files': tree,
    }
    if args.gap_feedback:
        request0['implementation_gaps'] = json.loads(args.gap_feedback.read_text())

    cfg = OpenAIJsonClientConfig(
        model=args.model, provider=args.provider or 'bedrock_openai',
        base_url=args.base_url or 'https://bedrock-mantle.us-east-2.api.aws/openai/v1',
        api_key_env=args.api_key_env or 'AWS_BEARER_TOKEN_BEDROCK',
        endpoint='/responses', env_file=args.env_file,
        temperature=None, reasoning_effort=args.reasoning_effort, timeout_sec=900,
        request_retry_attempts=2, max_output_tokens=16000)
    client = OpenAIJsonClient(cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    collected: dict[str, str] = {}
    if args.resume_from and args.resume_from.exists():
        previous = json.loads(args.resume_from.read_text())
        if previous.get('base_commit') != commit or previous.get('task_id') != task.task_id:
            raise ValueError('discovery seed belongs to another base/task')
        collected = {p: row['text'] for p, row in previous.get('files', {}).items()}
    request0['previously_collected_files'] = sorted(collected)
    history, rounds = [], []
    used = 0
    termination = 'max_rounds'
    started = time.monotonic()
    print(json.dumps({'event': 'started', 'task_id': task.task_id,
                      'repository_files': len(tree)}), flush=True)

    for round_no in range(1, args.max_rounds + 1):
        if used >= args.total_output_tokens:
            termination = 'output_token_budget'
            break
        # Keep exact source artifacts on disk; old source responses are archived
        # rather than indefinitely repeated in the API context.
        compact_history = json.loads(json.dumps(history))
        outputs = [item for item in compact_history if item.get('type') == 'function_call_output']
        for item in outputs[:-2]:
            value = json.loads(item['output'])
            if value.get('file_text'):
                value['archived_file_text_paths'] = sorted(value.pop('file_text'))
                item['output'] = json.dumps(value)
        body = {
            'model': cfg.model,
            'input': [{'role': 'developer', 'content': SYSTEM},
                      {'role': 'user', 'content': json.dumps(request0, ensure_ascii=False)},
                      *compact_history],
            'tools': [TOOL], 'tool_choice': {'type': 'function', 'name': 'perform_action'},
            'parallel_tool_calls': False, 'reasoning': {'effort': cfg.reasoning_effort},
            'max_output_tokens': min(16000, args.total_output_tokens - used),
        }
        try:
            response = client._client.responses.create(**body).model_dump()
        except Exception as exc:
            termination = 'api_error'
            save(args.output.parent / 'discover_api_error.json',
                 {'type': type(exc).__name__, 'message': str(exc)})
            break
        usage = response.get('usage') or {}
        if not isinstance(usage.get('output_tokens'), int):
            termination = 'missing_token_usage'
            break
        used += usage['output_tokens']
        if response.get('status') != 'completed':
            termination = 'incomplete_response'
            break
        tool_calls = [x for x in response.get('output', []) if x.get('type') == 'function_call']
        if len(tool_calls) != 1 or tool_calls[0].get('name') != 'perform_action':
            termination = 'tool_protocol_error'
            break
        tool_call = tool_calls[0]
        history.extend(response['output'])
        try:
            action = json.loads(tool_call['arguments'])
        except (ValueError, TypeError):
            action = {'action': 'invalid'}

        if action.get('action') == 'done':
            termination = 'done'
            history.append({'type': 'function_call_output', 'call_id': tool_call['call_id'],
                            'output': json.dumps({'status': 'acknowledged'})})
            rounds.append({'round': round_no, 'action': action})
            break

        requested = [p for p in (action.get('paths') or []) if valid_repo_path(p)]
        start_line = max(1, int(action.get('line_start') or 1))
        new_paths = [p for p in requested if p not in collected or action.get('line_start')]
        served, dropped = {}, []
        excerpts, ranges = {}, {}
        budget = args.round_token_budget
        if new_paths:
            raw = docker_python(image, READ_FILES, {'commit': commit, 'paths': new_paths})['files']
            for path in new_paths:
                entry = raw.get(path, {'kind': 'absent'})
                if entry['kind'] != 'file':
                    served[path] = entry['kind']
                    continue
                text = entry['text']
                lines = text.splitlines(keepends=True)
                if budget < 500:
                    dropped.append(path)
                    continue
                end = start_line
                while end < len(lines) and count_tokens(''.join(lines[start_line - 1:end + 1])) <= budget:
                    end += 1
                excerpt = ''.join(lines[start_line - 1:end])
                if count_tokens(excerpt) > budget:
                    dropped.append(path)
                    continue
                budget -= count_tokens(excerpt)
                collected[path] = text
                served[path] = 'read'
                excerpts[path] = excerpt
                ranges[path] = {'line_start': start_line, 'line_end': min(end, len(lines)),
                                'total_lines': len(lines),
                                'next_line_start': end + 1 if end < len(lines) else None}
        feedback = {'served': served, 'dropped_over_round_budget': dropped,
                    'already_collected': [p for p in requested if p in collected and p not in new_paths],
                    'files_so_far': sorted(collected), 'served_ranges': ranges}
        history.append({'type': 'function_call_output', 'call_id': tool_call['call_id'],
                        'output': json.dumps({'result': feedback,
                                              'file_text': excerpts})})
        rounds.append({'round': round_no, 'action': action, 'feedback': feedback})
        save(args.output.parent / 'discover_rounds.json', rounds)
        print(json.dumps({'event': 'round', 'n': round_no, 'requested': len(requested),
                          'read': sum(1 for v in served.values() if v == 'read'),
                          'dropped': len(dropped), 'output_tokens_used': used}), flush=True)

    artifact = {
        'schema': 'deepswe_discovered_code_v1',
        'task_id': task.task_id,
        'base_commit': commit,
        'discovered_by': cfg.model,
        'termination': termination,
        'rounds_used': len(rounds),
        'output_tokens_used': used,
        'seconds': round(time.monotonic() - started, 3),
        'files': {path: {'text': text} for path, text in collected.items()},
    }
    save(args.output, artifact)
    save(args.output.parent / 'discover_rounds.json', rounds)
    print(json.dumps({'event': 'done', 'termination': termination,
                      'files_collected': len(collected), 'rounds': len(rounds),
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
