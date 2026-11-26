#!/usr/bin/env python3
"""Run one patch diagnostic with a fixed edit budget.

Each submission is one atomic edit transaction.  The solver may select checks
from a frozen, base-validated public catalog to run against the resulting
patch. Official hidden tests run once after the episode. The same bounded
interface measures whether the available evidence supports a verified patch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time

ROOT = Path.cwd()
from teamgym.data.software.deep_swe import load_task
from teamgym.data.software.deep_swe.model_actions import ModelActions, MODELS
from teamgym.tokenization import count_tokens
from teamgym.data.software.deep_swe.candidate import allocated_observation_ids
from teamgym.data.software.deep_swe.stages.patch_runtime import BASE_READ, PUBLIC_CHECK, Editor, digest, docker_python, is_test_path, save, verify


SYSTEM = '''You are the sole software engineer solving the supplied task.
You have only the supplied instruction and source evidence. Do not assume access
to any other repository content. No shell, repository search, read-file tool,
network access, git history, or hidden tests are available. Do not try to reveal
unprovided source through edits or public checks.

Call perform_action once per response. An edit is one atomic transaction over
as many files and operations as needed. The evidence contains a frozen public
check catalog. For each edit, choose zero or more catalog IDs in checks. The
harness runs only those checks after an accepted edit and returns only their
pass/fail/timeout status. You may then repair the patch if
patch turns remain. You cannot provide commands or access checks outside the
catalog. Do not create, modify, or delete test files or test-runner configuration
(including conftest.py). Hidden tests run exactly
once after termination and never provide repair feedback. Rejected edits
consume a patch turn and run no checks.

{"action":"edit","operations":[...],"checks":["public-check-id"]}
  {"op":"replace","path":"relative/path","old":"exact known text","new":"replacement"}
    old must be nonempty, uniquely match, and be wholly contained in supplied
    source or text you authored in this episode.
  {"op":"write","path":"relative/path","text":"complete file content"}
  {"op":"delete","path":"relative/path"}
  Operations are sequential and roll back together if one fails. Paths stay
  inside the repository. Each transaction is limited to 200000 UTF-8 bytes.
{"action":"finish"}
  Submit the latest patch immediately. Finish costs no patch turn.

Use the smallest number of patch turns needed. The user message reports the
remaining patch turns and shared output-token budget. At the limit, the latest
patch is submitted automatically.
'''

EVIDENCE_INSTRUMENTATION = '''

This is an instrumented construction episode. For every edit, include
`evidence_used` in the action. Report at most three code atom IDs and/or
observation IDs in total. Select only supplied evidence that changed the
content of this edit: without it, you would have chosen a different
implementation, replacement anchor, or runtime behavior. Do not cite an ID
merely because it was available, because its file was modified, or because it
is part of the same feature. For every cited ID, give a short counterfactual
role in this form: "without this, I could not determine <specific decision>".
If the old-code anchor is code you authored in an earlier edit, cite no packet
source for that anchor; cite only any supplied evidence that changed the new
edit. This reporting does not replace the edit itself and does not reveal any
hidden-test result.
'''

TOOL = {
    'type': 'function', 'name': 'perform_action',
    'description': 'Submit one patch transaction or finish the episode.',
    'strict': False,
    'parameters': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['edit', 'finish']},
        'note': {'type': 'string'},
        'checks': {'type': 'array', 'items': {'type': 'string'}},
        'operations': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'op': {'type': 'string', 'enum': ['replace', 'write', 'delete']},
            'path': {'type': 'string'}, 'old': {'type': 'string'},
            'new': {'type': 'string'}, 'text': {'type': 'string'},
        }, 'required': ['op', 'path'], 'additionalProperties': False}},
    }, 'required': ['action'], 'additionalProperties': False},
}


def packet_for(candidate: dict, checks: dict, evidence_mode: str, *, include_ids=False) -> dict:
    atoms = candidate['code_atoms'] if evidence_mode == 'all' else []
    allocated = allocated_observation_ids(candidate) if evidence_mode == 'all' else set()
    return {
        'instruction': candidate['instruction'],
        'code_fragments': [{k: a[k] for k in (('id',) if include_ids else ()) + ('path', 'line_start', 'line_end', 'text')} for a in atoms],
        'observations': [{k: o[k] for k in (('id',) if include_ids else ()) + ('argv', 'stdout', 'stderr', 'exit_code')}
                         for o in candidate['observations'] if o['id'] in allocated],
        'public_check_catalog': [
            {'id': check_id, 'argv': argv} for check_id, argv in checks.items()
        ],
    }


def selected_check_ids(action: dict, checks: dict[str, list[str]]) -> list[str]:
    """Validate and de-duplicate solver-selected allowlisted public checks."""

    requested = action.get('checks') or []
    if not isinstance(requested, list) or not all(isinstance(x, str) for x in requested):
        raise ValueError('checks_must_be_string_array')
    selected = list(dict.fromkeys(requested))
    if any(check_id not in checks for check_id in selected):
        raise ValueError('unknown_public_check')
    return selected


def validate_evidence_used(action: dict, candidate: dict) -> str | None:
    """Check construction-only solver citations against the visible packet."""

    evidence = action.get('evidence_used')
    if not isinstance(evidence, dict):
        return 'missing_evidence_used'
    code_atoms = evidence.get('code_atoms')
    observations = evidence.get('observations')
    roles = evidence.get('roles')
    if not isinstance(code_atoms, list) or not all(isinstance(x, str) for x in code_atoms):
        return 'invalid_evidence_code_atoms'
    if not isinstance(observations, list) or not all(isinstance(x, str) for x in observations):
        return 'invalid_evidence_observations'
    if not isinstance(roles, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in roles.items()):
        return 'invalid_evidence_roles'
    cited = set(code_atoms) | set(observations)
    if not cited:
        return 'empty_evidence_used'
    if len(cited) > 3:
        return 'too_many_evidence_citations'
    if set(roles) != cited:
        return 'evidence_roles_do_not_match_citations'
    visible_atoms = {atom['id'] for atom in candidate['code_atoms']}
    visible_observations = allocated_observation_ids(candidate)
    if not set(code_atoms) <= visible_atoms:
        return 'unknown_evidence_code_atom'
    if not set(observations) <= visible_observations:
        return 'unknown_evidence_observation'
    return None


def provider_failure_kind(exc: Exception) -> str:
    """Separate an explicit input-window rejection from a serving failure."""
    message = str(exc).lower()
    if any(marker in message for marker in (
        'context_length_exceeded', 'maximum context length', 'context window exceeded',
        'input is too long', 'input token count exceeds', 'too many input tokens')):
        return 'context_window_failure'
    return 'api_error'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task-dir', type=Path, required=True)
    ap.add_argument('--runtime-image')
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--checks', type=Path, required=True)
    ap.add_argument('--verifier-image', required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--patch-turns', type=int, required=True)
    ap.add_argument('--evidence-mode', choices=('all', 'none'), default='all')
    ap.add_argument('--env-file', default=str(ROOT / '.env'))
    ap.add_argument('--total-output-tokens', type=int, default=32000)
    ap.add_argument('--max-output-tokens-per-call', type=int, default=16000)
    ap.add_argument('--max-checks-per-episode', type=int, default=15)
    ap.add_argument('--model-slug', default='gpt55',
                    help='Actor transport/model from deep_swe.model_actions.MODELS.')
    ap.add_argument('--provider')
    ap.add_argument('--base-url')
    ap.add_argument('--api-key-env')
    ap.add_argument('--reasoning-effort')
    ap.add_argument('--require-evidence-citations', action='store_true',
                    help='construction-only: require solver atom/observation citations on every edit')
    ap.add_argument('--citation-policy', choices=('enforce', 'audit'), default='enforce',
                    help='whether invalid construction citations reject an edit or are only logged')
    args = ap.parse_args()
    if args.patch_turns < 1:
        ap.error('--patch-turns must be positive')
    if min(args.max_checks_per_episode, args.total_output_tokens, args.max_output_tokens_per_call) < 1:
        ap.error('check and token budgets must be positive')

    task = load_task(args.task_dir, runtime_image=args.runtime_image)
    candidate = json.loads(args.candidate.read_text())
    checks = json.loads(args.checks.read_text())
    if not checks or not all(isinstance(k, str) and isinstance(v, list) and v for k, v in checks.items()):
        raise ValueError('checks must be a non-empty object of id -> argv')
    assert candidate['task_id'] == task.task_id and candidate['instruction'] == task.instruction
    image = candidate['image_id']
    verifier_id = json.loads(subprocess.check_output(
        ['docker', 'image', 'inspect', args.verifier_image], text=True))[0]['Id']
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'candidate.frozen.hidden.json', candidate)
    packet = packet_for(candidate, checks, args.evidence_mode,
                        include_ids=args.require_evidence_citations)
    save(out / 'actor_packet.json', packet)
    system_prompt = SYSTEM + (EVIDENCE_INSTRUMENTATION if args.require_evidence_citations else '')
    tool_schema = copy.deepcopy(TOOL)
    if args.require_evidence_citations:
        tool_schema['parameters']['properties']['evidence_used'] = {
            'type': 'object',
            'properties': {
                'code_atoms': {'type': 'array', 'items': {'type': 'string'}},
                'observations': {'type': 'array', 'items': {'type': 'string'}},
                'roles': {'type': 'object', 'additionalProperties': {'type': 'string'}},
            },
            'required': ['code_atoms', 'observations', 'roles'],
            'additionalProperties': False,
        }
    (out / 'system.txt').write_text(system_prompt)

    def read_base(path):
        return docker_python(image, BASE_READ, {'path': path, 'commit': task.repository.base_commit_hash})

    base_cache = {}
    for atom in candidate['code_atoms']:
        raw = base_cache.get(atom['path'])
        if raw is None:
            raw = base_cache[atom['path']] = read_base(atom['path'])
        actual = ''.join(raw['text'].splitlines(keepends=True)[atom['line_start'] - 1:atom['line_end']])
        assert actual == atom['text'] and hashlib.sha256(actual.encode()).hexdigest() == atom['sha256']
    visible_atoms = candidate['code_atoms'] if args.evidence_mode == 'all' else []
    editor = Editor(visible_atoms, read_base)
    action_client = ModelActions(args.model_slug, args.env_file,
        provider=args.provider, base_url=args.base_url, api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort)
    cfg = action_client.http.cfg
    execution = candidate.get('execution_manifest')
    if execution is not None:
        from teamgym.data.software.deep_swe.export import validate_execution_manifest
        from teamgym.llm.openai_client import normalize_model_for_provider
        validate_execution_manifest(execution, task_id=task.task_id,
            candidate_image_id=image, protocol={'patch_turns': args.patch_turns,
                'total_output_tokens': args.total_output_tokens,
                'max_output_tokens_per_call': args.max_output_tokens_per_call,
                'max_checks_per_episode': args.max_checks_per_episode})
        actor = execution['actor']
        expected_model = normalize_model_for_provider(actor['provider'], actor['model'])
        if (execution['source']['base_commit'] != task.repository.base_commit_hash
                or execution['verifier_image']['id'] != verifier_id
                or expected_model != cfg.model or actor['provider'] != cfg.provider
                or actor['base_url'] != cfg.base_url or actor['endpoint'] != cfg.endpoint
                or actor['reasoning_effort'] != cfg.reasoning_effort):
            raise ValueError('episode configuration differs from frozen execution manifest')
    manifest = {
        'task_id': task.task_id, 'model': cfg.model, 'model_slug': args.model_slug,
        'source_base_commit': task.repository.base_commit_hash,
        'runtime_image_id': image,
        'provider': cfg.provider,
        'base_url': cfg.base_url, 'endpoint': cfg.endpoint,
        'reasoning_effort': cfg.reasoning_effort, 'mode': 'single_patch_turns',
        'evidence_mode': args.evidence_mode, 'patch_turn_budget': args.patch_turns,
        'require_evidence_citations': args.require_evidence_citations,
        'citation_policy': args.citation_policy,
        'total_output_token_budget': args.total_output_tokens,
        'max_output_tokens_per_call': args.max_output_tokens_per_call,
        'public_check_mode': 'solver_selected_catalog',
        'public_check_feedback': 'status_only',
        'max_public_checks_per_episode': args.max_checks_per_episode,
        'solver_authored_tests': False,
        'hidden_tests_during_episode': False, 'hidden_tests_at_end': True,
        'candidate_sha256': digest(args.candidate), 'checks_sha256': digest(args.checks),
        'actor_packet_sha256': digest(out / 'actor_packet.json'),
        'harness_sha256': digest(Path(__file__)), 'verifier_image_id': verifier_id,
    }
    save(out / 'manifest.json', manifest)
    (out / 'harness.frozen.py').write_text(Path(__file__).read_text())

    history, trajectory, calls = [], [], []
    used = edits = checks_used = 0
    termination = 'patch_turn_budget'
    started = time.monotonic()
    print(json.dumps({'event': 'started', 'patch_turns': args.patch_turns,
                      'evidence_mode': args.evidence_mode,
                      'packet_cl100k_tokens': count_tokens(json.dumps(packet)),
                      'output_dir': str(out)}), flush=True)
    for call_no in range(1, args.patch_turns + 2):
        if edits >= args.patch_turns or used >= args.total_output_tokens:
            termination = 'patch_turn_budget' if edits >= args.patch_turns else 'output_token_budget'
            break
        remaining = {'patch_turns': args.patch_turns - edits,
                     'output_tokens': args.total_output_tokens - used,
                     'public_checks': args.max_checks_per_episode - checks_used}
        body = {
            'model': cfg.model,
            'input': [{'role': 'developer', 'content': system_prompt},
                      {'role': 'user', 'content': json.dumps({'evidence': packet}, ensure_ascii=False)},
                      *history,
                      {'role': 'user', 'content': json.dumps({'remaining': remaining})}],
            'tools': [tool_schema], 'tool_choice': {'type': 'function', 'name': 'perform_action'},
            'parallel_tool_calls': False, 'reasoning': {'effort': cfg.reasoning_effort},
            'max_output_tokens': min(args.max_output_tokens_per_call,
                                     args.total_output_tokens - used),
        }
        save(out / f'call_{call_no:02d}.request.json', body)
        call_started = time.monotonic()
        try:
            response = action_client._client.responses.create(**body).model_dump()
        except Exception as exc:
            save(out / 'api_error.json', {'type': type(exc).__name__, 'message': str(exc)})
            termination = provider_failure_kind(exc)
            break
        save(out / f'call_{call_no:02d}.response.json', response)
        usage = response.get('usage') or {}
        if not isinstance(usage.get('output_tokens'), int):
            termination = 'missing_token_usage'
            break
        used += usage['output_tokens']
        calls.append({'call': call_no, 'usage': usage,
                      'seconds': round(time.monotonic() - call_started, 3),
                      'status': response.get('status')})
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
        if not isinstance(action, dict):
            action = {'action': 'invalid'}
        if action.get('action') == 'finish':
            trajectory.append({'action': action})
            termination = 'finish'
            break
        edits += 1
        try:
            selected = selected_check_ids(action, checks)
        except ValueError as exc:
            selected = []
            selection_error = str(exc)
        else:
            selection_error = None
        operations = action.get('operations') or []
        evidence_error = (validate_evidence_used(action, candidate)
                          if args.require_evidence_citations else None)
        has_test_edit = any(
            isinstance(operation, dict) and is_test_path(str(operation.get('path') or ''))
            for operation in operations
        )
        if evidence_error is not None and args.citation_policy == 'enforce':
            edit_feedback = {'status': 'rejected'}
            edit_reason = evidence_error
        elif selection_error is not None:
            edit_feedback = {'status': 'rejected'}
            edit_reason = selection_error
        elif checks_used + len(selected) > args.max_checks_per_episode:
            edit_feedback = {'status': 'rejected'}
            edit_reason = 'public_check_budget_exceeded'
        elif has_test_edit:
            edit_feedback = {'status': 'rejected'}
            edit_reason = 'test_file_edits_forbidden'
        elif len(tool_call['arguments'].encode()) > 200000 or action.get('action') != 'edit':
            edit_feedback = {'status': 'rejected'}
            edit_reason = 'action_too_large' if len(tool_call['arguments'].encode()) > 200000 else 'wrong_action'
        else:
            edit_feedback = editor.edit(action.get('operations'))
            edit_reason = editor.last_reason or ('applied' if edit_feedback['status'] == 'applied' else 'transaction_error')
        patch_path = out / f'patch_{edits:02d}.patch'
        patch_path.write_text(editor.patch())
        check_feedback = {}
        for check_id in selected if edit_feedback['status'] == 'applied' else []:
            argv = checks[check_id]
            raw = docker_python(image, PUBLIC_CHECK,
                                {'commit': task.repository.base_commit_hash, 'argv': argv, 'timeout': 120},
                                patch=patch_path, timeout=150)
            save(out / f'patch_{edits:02d}.{check_id}.internal.json', raw)
            check_feedback[check_id] = raw['status']
            checks_used += 1
        feedback = {'edit': edit_feedback['status'], 'public_checks': check_feedback}
        save(out / f'patch_{edits:02d}.edit_diagnostic.json', {
            'status': edit_feedback['status'], 'reason': edit_reason,
            'actor_feedback': edit_feedback['status'],
            'evidence_used': action.get('evidence_used'),
            'evidence_validation_error': evidence_error,
        })
        trajectory.append({'patch_turn': edits, 'action': action, 'feedback': feedback})
        history.append({'type': 'function_call_output', 'call_id': tool_call['call_id'],
                        'output': json.dumps(feedback)})
        save(out / 'trajectory.json', trajectory)
        save(out / 'calls.json', calls)
        print(json.dumps({'event': 'patch_turn', 'n': edits, 'feedback': feedback,
                          'output_tokens_used': used}), flush=True)

    save(out / 'trajectory.json', trajectory)
    save(out / 'calls.json', calls)
    (out / 'model.patch').write_text(editor.patch())
    episode = {'termination': termination, 'patch_turns_used': edits, 'model_calls': len(calls),
               'output_tokens': used, 'input_tokens': sum(c['usage'].get('input_tokens', 0) for c in calls),
               'public_checks_used': checks_used,
               'solver_seconds': round(time.monotonic() - started, 3),
               'patch_sha256': digest(out / 'model.patch')}
    save(out / 'episode.json', episode)
    result = verify(task, verifier_id, out, out / 'model.patch')
    reward = result['reward']
    success = (termination != 'context_window_failure' and result['exit_code'] == 0 and reward.get('reward') == 1
               and reward.get('f2p_total', 0) > 0 and reward.get('p2p_total', 0) > 0
               and reward.get('f2p_passed') == reward.get('f2p_total')
               and reward.get('p2p_passed') == reward.get('p2p_total'))
    result.update(episode=episode, success=success,
                  episode_valid=result.get('verification_valid', True) and termination not in {'api_error', 'missing_token_usage', 'tool_protocol_error'})
    if termination == 'context_window_failure':
        result['failure_kind'] = termination
    save(out / 'result.json', result)
    print(json.dumps({'event': 'verified', 'success': success, 'reward': reward}), flush=True)


if __name__ == '__main__':
    main()
