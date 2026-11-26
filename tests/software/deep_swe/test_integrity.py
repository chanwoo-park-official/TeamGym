"""Offline regressions for stale successful diagnostic and partial-save reuse."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from teamgym.data.software.deep_swe import integrity
from teamgym.data.software.deep_swe.stages import diagnostic_runs as diagnostics


def _success():
    return {'episode_valid': True, 'verification_valid': True, 'success': True,
            'exit_code': 0, 'reward': {'reward': 1, 'f2p_passed': 1, 'f2p_total': 1,
                                      'p2p_passed': 1, 'p2p_total': 1}}


@pytest.fixture
def diagnostic_job(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate.json'
    checks = tmp_path / 'checks.json'
    integrity.atomic_write_json(candidate, {'image_id': 'runtime:tag', 'instruction': 'repair'})
    integrity.atomic_write_json(checks, {'compile': ['python', '-m', 'compileall']})
    state = {'task': 'base-one', 'image': 'sha256:one', 'code': 'harness-one', 'runs': 0}
    monkeypatch.setattr(diagnostics, 'task_identity', lambda _: {
        'runtime_image': 'task:tag', 'tree_sha256': state['task']})
    monkeypatch.setattr(diagnostics, 'image_identity', lambda tag: {
        'requested': tag, 'id': state['image']})
    monkeypatch.setattr(diagnostics, 'code_identity', lambda: {'sha256': state['code']})
    monkeypatch.setattr(diagnostics.time, 'sleep', lambda _: None)
    def fake_episode(cmd, **kwargs):
        state['runs'] += 1
        output = Path(cmd[cmd.index('--output-dir') + 1])
        output.mkdir(parents=True)
        integrity.atomic_write_json(output / 'result.json', _success())
        for name in diagnostics.EPISODE_OUTPUTS:
            if name != 'result.json':
                (output / name).write_text('{}')
        state['last_command'] = cmd
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(diagnostics.subprocess, 'run', fake_episode)
    job = {'label': 'baseline', 'rep': 1, 'task_dir': str(tmp_path / 'task'),
           'candidate': str(candidate), 'checks': str(checks),
           'verifier_image': 'verifier:tag', 'patch_turns': 5, 'model_slug': 'gpt55',
           'require_evidence_citations': False, 'citation_policy': 'enforce',
           'output_dir': tmp_path / 'baseline_rep1'}
    return job, state


def test_matching_diagnostic_reuses_only_receipted_success(diagnostic_job):
    job, state = diagnostic_job
    assert diagnostics.run(job)['cache_status'] == 'fresh'
    row = diagnostics.run(job)
    assert row['success'] and row['cache_status'] == 'cache_hit'
    assert state['runs'] == 1


@pytest.mark.parametrize('change', ['candidate', 'checks', 'task', 'image', 'code',
                                   'model_slug', 'provider', 'base_url', 'api_key_env',
                                   'reasoning_effort', 'runtime_image', 'patch_turns',
                                   'total_output_tokens', 'max_output_tokens_per_call',
                                   'max_checks_per_episode', 'require_evidence_citations'])
def test_changed_inputs_never_adopt_prior_success(diagnostic_job, change):
    job, state = diagnostic_job
    diagnostics.run(job)
    if change in {'candidate', 'checks'}:
        path = Path(job[change])
        value = json.loads(path.read_text())
        value['new_input'] = 'changed'
        integrity.atomic_write_json(path, value)
    elif change in state:
        state[change] += '-changed'
    else:
        job[change] = {'provider': 'openai_compatible',
                       'patch_turns': 4, 'total_output_tokens': 123,
                       'max_output_tokens_per_call': 100, 'max_checks_per_episode': 2,
                       'require_evidence_citations': True}.get(change, 'changed')
    row = diagnostics.run(job)
    assert row['success'] and row['cache_status'] == 'resume'
    assert state['runs'] == 2
    assert list(Path(job['output_dir']).parent.glob('baseline_rep1.invalid_*'))


@pytest.mark.parametrize('damage', ['missing_manifest', 'corrupt_manifest', 'corrupt_result',
                                   'changed_result', 'invalid_result', 'untyped_result', 'missing_packet', 'changed_patch'])
def test_bad_receipt_or_result_never_adopts_prior_success(diagnostic_job, damage):
    job, state = diagnostic_job
    diagnostics.run(job)
    output = Path(job['output_dir'])
    if damage == 'missing_manifest':
        (output / 'cache_manifest.json').unlink()
    elif damage == 'corrupt_manifest':
        (output / 'cache_manifest.json').write_text('{')
    elif damage == 'corrupt_result':
        (output / 'result.json').write_text('{')
    elif damage == 'missing_packet':
        (output / 'actor_packet.json').unlink()
    elif damage == 'changed_patch':
        (output / 'model.patch').write_text('changed patch')
    else:
        value = _success()
        if damage == 'changed_result':
            value['new'] = 'tampered'
        elif damage == 'invalid_result':
            value['episode_valid'] = False
        else:
            value.pop('episode_valid')
        integrity.atomic_write_json(output / 'result.json', value)
        if damage != 'changed_result':
            # Even a matching receipt cannot certify a semantically invalid result.
            inputs = diagnostics.episode_inputs(job)
            integrity.write_manifest(output / 'cache_manifest.json', inputs,
                                     outputs=[output / name for name in diagnostics.EPISODE_OUTPUTS])
    assert diagnostics.run(job)['cache_status'] == 'resume'
    assert state['runs'] == 2


@pytest.mark.parametrize('missing', ['candidate', 'checks'])
def test_missing_current_input_does_not_reuse_success(diagnostic_job, missing):
    job, state = diagnostic_job
    diagnostics.run(job)
    Path(job[missing]).unlink()
    with pytest.raises(FileNotFoundError):
        diagnostics.run(job)
    assert state['runs'] == 1


def test_runtime_configuration_reaches_episode_without_secret_values(diagnostic_job):
    job, state = diagnostic_job
    job.update(runtime_image='pinned-runtime', env_file='/private/config.env', provider='openai',
               base_url='https://example.invalid/v1', api_key_env='PRIVATE_KEY_NAME',
               reasoning_effort='high', total_output_tokens=2000,
               max_output_tokens_per_call=1000, max_checks_per_episode=4)
    diagnostics.run(job)
    command = state['last_command']
    for key in ('runtime_image', 'env_file', 'provider', 'base_url', 'api_key_env',
                'reasoning_effort', 'total_output_tokens', 'max_output_tokens_per_call',
                'max_checks_per_episode'):
        assert command[command.index('--' + key.replace('_', '-')) + 1] == str(job[key])
    receipt = json.loads((Path(job['output_dir']) / 'cache_manifest.json').read_text())
    assert receipt['inputs']['actor']['provider'] == 'openai'
    assert receipt['inputs']['actor']['api_key_env'] == 'PRIVATE_KEY_NAME'
    assert not any('secret' in key for key in receipt['inputs']['actor'])


def test_inputs_changed_during_execution_are_not_certified(diagnostic_job, monkeypatch):
    job, state = diagnostic_job
    original = diagnostics.subprocess.run
    def changing(cmd, **kwargs):
        result = original(cmd, **kwargs)
        state['task'] = 'changed-during-model-call'
        return result
    monkeypatch.setattr(diagnostics.subprocess, 'run', changing)
    with pytest.raises(RuntimeError, match='inputs changed'):
        diagnostics.run(job)
    assert not (Path(job['output_dir']) / 'cache_manifest.json').exists()


def test_atomic_json_interruption_preserves_complete_old_file(tmp_path, monkeypatch):
    target = tmp_path / 'result.json'
    integrity.atomic_write_json(target, {'old': True})
    def interrupted(*_):
        raise OSError('interrupted before atomic replace')
    monkeypatch.setattr(integrity.os, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        integrity.atomic_write_json(target, {'new': True})
    assert integrity.load_json(target) == {'old': True}
    assert list(tmp_path.iterdir()) == [target]


def test_manifest_binds_inputs_and_output_content(tmp_path):
    result = tmp_path / 'result.json'
    manifest = tmp_path / 'manifest.json'
    integrity.atomic_write_json(result, _success())
    inputs = {'model': 'one', 'budget': 5}
    integrity.write_manifest(manifest, inputs, outputs=[result])
    assert integrity.validate_manifest(manifest, inputs, outputs=[result])[0]
    assert not integrity.validate_manifest(manifest, {**inputs, 'model': 'two'}, outputs=[result])[0]
    integrity.atomic_write_json(result, {'success': True})
    assert not integrity.validate_manifest(manifest, inputs, outputs=[result])[0]
    result.unlink()
    assert integrity.validate_manifest(manifest, inputs, outputs=[result]) == (False, 'missing_output')


def test_source_identity_tracks_actual_bytes_not_git_status(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    (task / 'tests').mkdir(parents=True)
    (task / 'task.toml').write_text('[metadata]\nbase_commit_hash="base"\n[environment]\ndocker_image="tag"\n')
    integrity.atomic_write_json(task / 'tests/config.json', {'base_commit': 'base'})
    source = task / 'instruction.md'
    source.write_text('original')
    monkeypatch.setattr(integrity.subprocess, 'check_output', lambda *args, **kwargs: 'unchanged-head\n')
    first = integrity.task_identity(task)
    source.write_text('modified')
    second = integrity.task_identity(task)
    assert first['source_revision'] == second['source_revision']
    assert first['tree']['sha256'] != second['tree']['sha256']


def test_image_identity_resolves_mutable_tag_and_requires_content_id(monkeypatch):
    image = {'Id': 'sha256:first', 'RepoDigests': ['repo@sha256:first'],
             'Architecture': 'arm64', 'Os': 'linux'}
    monkeypatch.setattr(integrity.subprocess, 'check_output', lambda *args, **kwargs: json.dumps([image]))
    first = integrity.image_identity('mutable:tag')
    image['Id'] = 'sha256:second'
    assert first != integrity.image_identity('mutable:tag')
    del image['Id']
    with pytest.raises(ValueError, match='content ID'):
        integrity.image_identity('mutable:tag')


def test_canonical_hash_is_order_independent_and_rejects_partial_values():
    assert integrity.canonical_sha256({'a': 1, 'b': 2}) == integrity.canonical_sha256({'b': 2, 'a': 1})
    with pytest.raises(ValueError):
        integrity.canonical_sha256({'nan': float('nan')})


def test_targeted_cli_runs_three_repetitions_and_keeps_five_turns(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate.json'
    integrity.atomic_write_json(candidate, {})
    monkeypatch.setattr(diagnostics, 'leave_one_out_conditions', lambda *_: [
        {'label': label, 'atom_ids': [], 'observation_ids': []}
        for label in ('baseline', 'loo_A', 'loo_B')])
    monkeypatch.setattr(diagnostics, 'reduce_candidate', lambda *args, **kwargs: {})
    jobs = []
    def episode(job):
        jobs.append(job)
        return {'label': job['label'], 'rep': job['rep'], 'episode_valid': True,
                'success': job['rep'] == 3, 'exit_code': 0, 'f2p': [0, 1], 'p2p': [1, 1]}
    monkeypatch.setattr(diagnostics, 'run', episode)
    monkeypatch.setattr(diagnostics.sys, 'argv', [
        'diagnostics', '--task-dir', str(tmp_path), '--candidate', str(candidate),
        '--checks', str(tmp_path / 'checks.json'), '--verifier-image', 'verifier',
        '--output-dir', str(tmp_path / 'output'), '--patch-turns', '5',
        '--granularity', 'riu', '--only', 'loo_B'])
    diagnostics.main()
    assert [(job['label'], job['rep'], job['patch_turns']) for job in jobs] == [
        ('loo_B', rep, 5) for rep in (1, 2, 3)]
    summary = integrity.load_json(tmp_path / 'output/summary.json')
    assert summary['loo_B']['passes'] == 1 and summary['loo_B']['runs'] == 3


def test_all_qualification_never_skips_third_repetition(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate.json'
    integrity.atomic_write_json(candidate, {})
    monkeypatch.setattr(diagnostics, 'leave_one_out_conditions', lambda *_: [
        {'label': 'baseline', 'atom_ids': [], 'observation_ids': []}])
    monkeypatch.setattr(diagnostics, 'reduce_candidate', lambda *args, **kwargs: {})
    completed = []
    def episode(job):
        completed.append(job['rep'])
        return {'label': 'baseline', 'episode_valid': True, 'success': job['rep'] == 3,
                'exit_code': 0, 'f2p': [1, 1], 'p2p': [1, 1]}
    monkeypatch.setattr(diagnostics, 'run', episode)
    monkeypatch.setattr(diagnostics.sys, 'argv', [
        'diagnostics', '--task-dir', str(tmp_path), '--candidate', str(candidate),
        '--checks', str(tmp_path / 'checks.json'), '--verifier-image', 'verifier',
        '--output-dir', str(tmp_path / 'output'), '--patch-turns', '5',
        '--granularity', 'riu', '--only', 'baseline', '--max-failures', '2'])
    diagnostics.main()
    assert sorted(completed) == [1, 2, 3]
    assert integrity.load_json(tmp_path / 'output/summary.json')['baseline']['passes'] == 1
