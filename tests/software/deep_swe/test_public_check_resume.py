"""Offline regressions for compile-check provenance and content-bound resume."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.software.deep_swe import checks, provenance
from teamgym.data.software.deep_swe.integrity import file_sha256


@pytest.fixture
def baseline(monkeypatch, tmp_path):
    patch = tmp_path / 'solution.patch'
    patch.write_text('private implementation reference')
    tests = tmp_path / 'test.patch'
    tests.write_text('private assertions reference')
    task = SimpleNamespace(task_id='example', instruction='Implement stepped slices.',
        source_dir=tmp_path,
        repository=SimpleNamespace(base_commit_hash='a' * 40, language='go'),
        runtime=SimpleNamespace(docker_image='sha256:' + 'c' * 64),
        artifacts=SimpleNamespace(solution_patch=patch, test_patch=tests))
    snapshot = {'head': 'a' * 40, 'sha256': 'b' * 64, 'clean': True, 'status': ''}
    result = {'base_commit': 'a' * 40, 'resolved_base_commit': 'a' * 40,
        'checkout_before': snapshot, 'checkout_after': dict(snapshot),
        'tracked_checkout_unchanged': True, 'exit_code': 0, 'stdout': '? example [no test files]\n',
        'stderr': '', 'timed_out': False}
    catalog = {'compile-main': ['env', 'CONTEXT=abs', 'go', 'test', '.', '-run', '^$']}
    inventory = {'repository_files': ['go.mod', 'main.go', 'main_test.go'],
                 'build_test_manifests': {'go.mod': 'module example\ngo 1.22\n'}}
    seen = []
    class Client:
        def __init__(self, cfg):
            self.cfg = cfg
        def generate_json(self, **kwargs):
            seen.append(kwargs)
            return {'approved': True, 'reason': 'Real compile smoke command from public Go tooling.',
                'public_grounding': 'go.mod and tracked root package; no new behavioral assertions.',
                'nonzero_is_target_behavior': False}
    monkeypatch.setattr(checks, 'OpenAIJsonClient', Client)
    monkeypatch.setattr(checks, 'public_repository_inventory', lambda task: inventory)
    monkeypatch.setattr(checks, 'run_base_check', lambda *a: copy.deepcopy(result))
    source_identity = {'task_id': 'example', 'base_commit': 'a' * 40}
    code = {'files': {'checks.py': 'd' * 64}, 'sha256': 'e' * 64}
    monkeypatch.setattr(checks, 'task_identity', lambda path: copy.deepcopy(source_identity))
    monkeypatch.setattr(checks, 'code_identity', lambda: copy.deepcopy(code))
    identity = {'source': source_identity, 'code': code,
        'runtime_image': {'id': task.runtime.docker_image},
        'environment': {'uv_lock_sha256': file_sha256(Path(checks.__file__).resolve().parents[4] / 'uv.lock')},
        'config': {'constructor_model': 'gpt-5.4'}}
    options = {'model': 'gpt-5.4', 'provider': 'openai_compatible',
        'base_url': 'https://example.invalid/v1', 'reasoning_effort': 'medium', 'trace_dir': tmp_path}
    return SimpleNamespace(task=task, catalog=catalog, seen=seen, Client=Client,
        identity=identity, options=options, result=result, inventory=inventory)


def test_compile_only_check_uses_explicit_baseline_role(baseline):
    result = checks.validate_public_checks(baseline.task, baseline.catalog, **baseline.options)
    request = baseline.seen[0]
    assert json.loads(request['user_prompt'])['artifact_role'] == 'public_baseline_check'
    assert 'It need not exercise, demonstrate, or assert' in request['system_prompt']
    assert 'go test . -run ^$' in request['system_prompt']
    assert 'Reject unconditional-success no-ops' in request['system_prompt']
    audit = result['compile-main']['provenance_audit']
    assert audit['approved'] is True and audit['artifact_role'] == 'public_baseline_check'
    assert len(audit['audit_input_sha256']) == 64
    assert 'env_file' not in audit['auditor'] and 'api_key' not in audit['auditor']


def test_observation_default_keeps_original_criteria_and_request(baseline, tmp_path):
    client = baseline.Client(checks.model_config(**{k: v for k, v in baseline.options.items() if k != 'trace_dir'}, env_file=None))
    provenance.audit_solver_visible(client=client, task=baseline.task,
        artifact={'argv': ['go', 'run', 'public-example.go']}, public_sources=baseline.inventory,
        trace_dir=tmp_path)
    assert baseline.seen[0]['system_prompt'] == provenance.AUDIT_SYSTEM
    assert 'artifact_role' not in json.loads(baseline.seen[0]['user_prompt'])


def test_public_role_does_not_override_provenance_rejection(baseline, monkeypatch):
    monkeypatch.setattr(baseline.Client, 'generate_json', lambda *a, **kw: {
        'approved': False, 'reason': 'Command unconditionally returns success and compiles nothing.',
        'public_grounding': 'No meaningful public build/test operation.'})
    with pytest.raises(checks.PublicCheckError, match='provenance rejected'):
        checks.validate_public_checks(baseline.task, {'fake': ['true']}, **baseline.options)


def test_planner_uses_same_baseline_role(baseline, monkeypatch, tmp_path):
    class PlannerClient(baseline.Client):
        def generate_json(self, **kwargs):
            if 'public_check_planner' in kwargs['prompt_id']:
                return {'checks': [{'id': key, 'argv': argv} for key, argv in baseline.catalog.items()]}
            return super().generate_json(**kwargs)
    monkeypatch.setattr(checks, 'OpenAIJsonClient', PlannerClient)
    options = {k: v for k, v in baseline.options.items() if k != 'trace_dir'}
    planned, rows = checks.plan_public_checks(task=baseline.task, env_file=None,
        trace_dir=tmp_path, **options)
    assert planned == baseline.catalog
    assert rows['compile-main']['provenance_audit']['artifact_role'] == 'public_baseline_check'
    assert json.loads(baseline.seen[0]['user_prompt'])['artifact_role'] == 'public_baseline_check'


def make_cache(baseline, tmp_path):
    rows = checks.validate_public_checks(baseline.task, baseline.catalog, **baseline.options)
    path = tmp_path / 'public_validation.cache.json'
    checks.write_public_check_validation_cache(path, baseline.task, baseline.catalog, rows,
        identity=baseline.identity, **baseline.options)
    return path, file_sha256(path)


def test_exact_checkpointed_cache_skips_model_inventory_and_execution(baseline, monkeypatch, tmp_path):
    path, digest = make_cache(baseline, tmp_path)
    def unexpected(*a, **kw):
        pytest.fail('A verified cache must not launch Docker or create a model client')
    monkeypatch.setattr(checks, 'OpenAIJsonClient', unexpected)
    monkeypatch.setattr(checks, 'public_repository_inventory', unexpected)
    monkeypatch.setattr(checks, 'run_base_check', unexpected)
    rows = checks.validate_public_checks(baseline.task, baseline.catalog,
        validation_cache_path=path, validation_identity=baseline.identity,
        validation_cache_sha256=digest, **baseline.options)
    assert rows['compile-main']['provenance_audit']['approved'] is True


@pytest.mark.parametrize('change', ['untrusted_digest', 'output_bytes', 'argv', 'model',
    'source', 'code', 'runtime', 'uv_lock', 'old_audit_role', 'changed_result', 'dirty_checkout'])
def test_cache_changes_force_real_revalidation(baseline, monkeypatch, tmp_path, change):
    path, digest = make_cache(baseline, tmp_path)
    options = dict(baseline.options)
    catalog = copy.deepcopy(baseline.catalog)
    if change == 'untrusted_digest':
        digest = None
    elif change == 'output_bytes':
        path.write_text(path.read_text() + '\n')
    elif change == 'argv':
        catalog['compile-main'] = ['go', 'test', './parser', '-run', '^$']
    elif change == 'model':
        options['model'] = 'gpt-5.5'
    elif change == 'source':
        monkeypatch.setattr(checks, 'task_identity', lambda path: {'base_commit': 'different'})
    elif change == 'code':
        monkeypatch.setattr(checks, 'code_identity', lambda: {'sha256': 'different'})
    elif change == 'runtime':
        baseline.task.runtime.docker_image = 'sha256:' + 'f' * 64
    elif change == 'uv_lock':
        baseline.identity['environment']['uv_lock_sha256'] = 'f' * 64
    else:
        data = json.loads(path.read_text())
        row = data['results']['compile-main']
        if change == 'old_audit_role':
            row['provenance_audit'].pop('artifact_role')
        elif change == 'changed_result':
            row['stdout'] = 'unreviewed constructor output'
        else:
            row['checkout_after']['clean'] = False
        path.write_text(json.dumps(data))
        digest = file_sha256(path)  # Even a new trusted receipt cannot waive row invariants.
    baseline.seen.clear()
    checks.validate_public_checks(baseline.task, catalog, validation_cache_path=path,
        validation_identity=baseline.identity, validation_cache_sha256=digest, **options)
    assert len(baseline.seen) == 1


def test_writer_refuses_unbound_legacy_approval(baseline, tmp_path):
    rows = {'compile-main': {**baseline.result, 'provenance_audit': {'approved': True}}}
    with pytest.raises(checks.PublicCheckError, match='incomplete or unbound'):
        checks.write_public_check_validation_cache(tmp_path / 'cache.json', baseline.task,
            baseline.catalog, rows, identity=baseline.identity, **baseline.options)
