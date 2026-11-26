from __future__ import annotations

import io
import json
from pathlib import Path

from teamgym.data.software.deep_swe.construction import (
    construct_in_stages, qualification_feedback,
)
from teamgym.data.software.deep_swe.evidence import (
    text_parts, stage_materials, usable_observations,
)
from teamgym.data.software.deep_swe.fragments import extract_complete
from teamgym.tokenization import count_tokens
from tests.software.deep_swe.test_functional_rius import _ConstructorClient, _pool, _proposal


def test_rust_and_unparsed_typescript_source_are_not_lost():
    sources = {
        'lib.rs': '#![allow(dead_code)]\nuse std::cell::Cell;\nimpl Foo {\n    fn f(&self) {}\n}\n',
        'module.ts': 'export async function create() {\n  return 1\n}\nmodule.exports = create\n',
    }
    for path, source in sources.items():
        report = extract_complete(path, source)
        lines = source.splitlines(keepends=True)
        fragments = report.fragments + report.oversized
        covered = set()
        for fragment in fragments:
            assert fragment.text == ''.join(lines[fragment.line_start - 1:fragment.line_end])
            covered.update(range(fragment.line_start, fragment.line_end + 1))
        assert all(n in covered for n, line in enumerate(lines, 1) if line.strip())


def test_large_declaration_becomes_adjacent_readable_base_spans():
    source = 'def huge():\n' + ''.join(f'    value_{n} = {n}\n' for n in range(100))
    report = extract_complete('a.py', source, count_tokens=len, context_tokens=150)
    fragments = sorted(report.fragments + report.oversized, key=lambda f: f.line_start)
    assert ''.join(f.text for f in fragments) == source
    assert all(len(f.text) <= 150 for f in fragments)
    assert all(f.kind == 'source_context' for f in fragments)


def test_material_parts_are_lossless_even_with_single_long_line():
    source = '안녕하세요' * 1_000
    parts = text_parts(source, 100)
    assert ''.join(parts) == source
    assert all(count_tokens(part) <= 100 for part in parts)
    base, materials = stage_materials({'reference_patch': source, 'source_fragments': [],
                                      'runtime_observations': []}, 100)
    assert 'reference_patch' not in base
    assert ''.join(row['text'] for row in materials.values()) == source


def test_big_revision_feedback_is_not_repeated_in_first_request(tmp_path):
    client = _ConstructorClient([
        {'action': 'read', 'fragment_ids': ['frag-0001']}, _proposal(['frag-0001'])])
    huge = 'a long solver patch\n' * 20_000
    _, _, stats = construct_in_stages(
        client=client, request={'source_fragments': _pool()['fragments'],
                               'qualification_feedback': {'summary': {}, 'failing_tests': [],
                                                          'attempts': [{'solver_patch': huge}]}},
        system_prompt='Construct RIUs', pool=_pool(), observations=[], trace_dir=tmp_path,
        max_input_tokens=10_000)
    assert stats['peak_input_tokens'] < 10_000
    assert 'qualification_feedback' not in client.requests[0]
    assert client.requests[0]['constructor_evidence_inventory']


def test_many_collection_failures_remain_readable_without_overflowing_revision_input(tmp_path):
    failures = [{'test': f'tests/test_feature.py::test_case_{i}', 'bucket': 'p2p',
                 'episodes': ['baseline_rep1'], 'detail': 'NameError: missing import; ' * 15}
                for i in range(2_800)]
    request = {'source_fragments': _pool()['fragments'], 'qualification_feedback': {
        'summary': {'passes': 0, 'runs': 3}, 'failing_tests': failures, 'attempts': []}}
    base, materials = stage_materials(request)
    exact = ''.join(row['text'] for row in materials.values()
                    if row['material'] == 'qualification_failures')
    assert json.loads(exact) == failures
    assert base['failure_inventory']['count'] == len(failures)
    assert base['failure_inventory']['by_bucket'] == {'p2p': len(failures)}
    assert 'NameError' not in json.dumps(base)
    client = _ConstructorClient([
        {'action': 'read', 'evidence_ids': ['qualification_failures:part-0001']},
        {'action': 'read', 'fragment_ids': ['frag-0001']}, _proposal(['frag-0001'])])
    _, _, stats = construct_in_stages(
        client=client, request=request, system_prompt='Construct RIUs', pool=_pool(),
        observations=[], trace_dir=tmp_path, max_input_tokens=20_000)
    assert stats['peak_input_tokens'] < 20_000
    assert 'NameError' in client.requests[1]['read_constructor_evidence'][0]['text']


def test_constructor_can_read_observation_material_before_allocating_it(tmp_path):
    obs = {'id': 'observation-01', 'stdout': 'actual base output', 'exit_code': 0}
    client = _ConstructorClient([
        {'action': 'read', 'evidence_ids': ['observation-01:part-0001']},
        {'action': 'propose', 'rius': [{'functional_boundary': 'runtime contract',
                                      'rationale': 'observed base behavior',
                                      'code_fragment_ids': [], 'observation_ids': ['observation-01']}]},
    ])
    plan, _, _ = construct_in_stages(
        client=client, request={'source_fragments': [], 'runtime_observations': [obs]},
        system_prompt='Construct RIUs', pool={'fragments': []}, observations=[obs], trace_dir=tmp_path)
    assert plan['rius'][0]['observation_ids'] == ['observation-01']
    assert 'actual base output' in client.requests[1]['read_constructor_evidence'][0]['text']


def test_archived_invalid_episodes_do_not_contaminate_feedback(tmp_path):
    for name in ['baseline_rep1', 'baseline_rep1.invalid_123']:
        episode = tmp_path / name
        (episode / 'verifier').mkdir(parents=True)
        (episode / 'result.json').write_text(json.dumps({'episode_valid': name == 'baseline_rep1'}))
        (episode / 'verifier/ctrf.json').write_text(json.dumps({'results': {'tests': [
            {'name': f'[f2p] {name}', 'status': 'failed', 'message': 'failure'},
        ]}}))
    feedback = qualification_feedback(tmp_path)
    assert [attempt['episode'] for attempt in feedback['attempts']] == ['baseline_rep1']
    assert len(feedback['failing_tests']) == 1
    assert 'invalid' not in feedback['failing_tests'][0]['test']


def test_probe_failure_classification_keeps_real_error_behavior():
    from tests.software.deep_swe.test_reliability import complete_observation
    rows = [
        complete_observation(id='valid-error'),
        {'id': 'missing-command', 'exit_code': 127, 'failure_kind': 'setup_error'},
        {'id': 'deps', 'exit_code': 1, 'stderr': 'Cannot find package tsx'},
    ]
    usable, rejected = usable_observations(rows)
    assert [row['id'] for row in usable] == ['valid-error']
    assert len(rejected) == 2


def test_probe_runner_records_missing_executable_instead_of_crashing(monkeypatch, capsys):
    import subprocess
    import sys
    from teamgym.data.software.deep_swe.observations import RUNNER
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
        'expected_commit': 'base', 'kind': 'observation', 'argv': ['missing-tsx'], 'timeout_sec': 1})))
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kw: b'base\n' if 'rev-parse' in argv else b'')
    def missing(*args, **kwargs):
        raise FileNotFoundError('missing-tsx')
    monkeypatch.setattr(subprocess, 'run', missing)
    exec(RUNNER, {})
    result = json.loads(capsys.readouterr().out)
    assert result['failure_kind'] == 'setup_error'
    assert result['base_commit'] == 'base'
    assert result['tracked_checkout_unchanged']


def test_feedback_summarizes_nested_actions_without_repeating_source(tmp_path):
    episode = tmp_path / 'baseline_rep1'
    episode.mkdir()
    (episode / 'result.json').write_text(json.dumps({'episode_valid': True}))
    (episode / 'trajectory.json').write_text(json.dumps([{
        'action': {'action': 'edit', 'operations': [{'path': 'a.py', 'old': 'OLD_BODY', 'new': 'NEW_BODY'}]},
        'feedback': {'edit': {'status': 'rejected'}},
    }]))
    summary = qualification_feedback(tmp_path)['attempts'][0]['trajectory_summary']
    assert summary[0]['operation_paths'] == ['a.py']
    assert 'OLD_BODY' not in json.dumps(summary)


def test_public_check_planner_uses_validation_errors_to_repair(monkeypatch, tmp_path):
    from teamgym.data.software.deep_swe import checks
    from types import SimpleNamespace
    requests = []
    class Client:
        def __init__(self, cfg): pass
        def generate_json(self, **kwargs):
            requests.append(json.loads(kwargs['user_prompt']))
            return {'checks': [{'id': 'compile', 'argv': ['missing' if len(requests) == 1 else 'python3']}]}
    monkeypatch.setattr(checks, 'OpenAIJsonClient', Client)
    monkeypatch.setattr(checks, 'public_repository_inventory', lambda task: {})
    from tests.software.deep_swe.test_reliability import complete_observation
    monkeypatch.setattr(checks, 'run_base_check', lambda task, argv: {
        **complete_observation(exit_code=127 if argv == ['missing'] else 0), 'stderr': 'missing executable'})
    monkeypatch.setattr(checks, 'audit_solver_visible', lambda **kw: {'approved': True})
    task = SimpleNamespace(task_id='example', instruction='Implement behavior.',
                           repository=SimpleNamespace(language='python', base_commit_hash='base'))
    accepted, _ = checks.plan_public_checks(task=task, model='gpt-5.4', env_file='', trace_dir=tmp_path)
    assert accepted == {'compile': ['python3']}
    assert requests[1]['repair_feedback']['rejected_checks']['compile']['exit_code'] == 127


def test_verifier_slots_bound_parallel_execution(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from teamgym.data.software.deep_swe.runtime_limits import verifier_slots
    monkeypatch.setenv('DEEPSWE_VERIFIER_CONCURRENCY', '1')
    monkeypatch.setenv('DEEPSWE_RESOURCE_LOCK_DIR', str(tmp_path))
    lock = threading.Lock()
    active = peak = 0
    @verifier_slots
    def verify():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.05)
        with lock:
            active -= 1
        return {'seconds': .05}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: verify(), range(2)))
    assert peak == 1
    assert all(result['resource_slot_wait_seconds'] >= 0 for result in results)
