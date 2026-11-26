import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.software.deep_swe import pipeline, integrity
from teamgym.data.software.deep_swe.run_state import RunState
from teamgym.data.software.deep_swe.stages.patch_episode import packet_for, provider_failure_kind
from teamgym.data.software.deep_swe.stages.patch_runtime import Editor, save


def test_resume_archives_changed_identity_and_invalidates_tampered_stages(tmp_path):
    output = tmp_path / 'task'
    state = RunState(output, {'source': 'v1'}, resume=False)
    first, later = output / 'first.json', output / 'later.json'
    save(first, {'value': 1})
    save(later, {'value': 2})
    state.checkpoint()
    resumed = RunState(output, {'source': 'v1'}, resume=True)
    assert resumed.reusable(first, requested=True)
    save(first, {'value': 9})
    assert not resumed.reusable(first, requested=True)
    assert not resumed.reusable(later, requested=True)
    RunState(output, {'source': 'v2'}, resume=True)
    assert not first.exists()
    assert next(tmp_path.glob('task.archived_*')).joinpath('first.json').is_file()


def test_normal_actor_packet_does_not_disclose_evidence_ids():
    candidate = {'instruction': 'Repair behavior', 'code_atoms': [{'id': 'private-atom',
        'path': 'a.py', 'line_start': 1, 'line_end': 1, 'text': 'x = 1\n'}],
        'observations': [], 'allocation_candidates': [{'observation_ids': []}]}
    assert 'private-atom' not in json.dumps(packet_for(candidate, {'compile': ['python']}, 'all'))
    assert 'private-atom' in json.dumps(packet_for(candidate, {'compile': ['python']}, 'all', include_ids=True))


def test_explicit_context_window_failure_is_distinguished_from_capacity():
    assert provider_failure_kind(RuntimeError('maximum context length exceeded')) == 'context_window_failure'
    assert provider_failure_kind(RuntimeError('provider capacity exhausted')) == 'api_error'


def test_editor_itself_rejects_test_configuration_edits():
    editor = Editor([], lambda p: {'kind': 'absent'})
    assert editor.edit([{'op': 'write', 'path': 'conftest.py', 'text': '# harmless\n'}])['status'] == 'rejected'
    assert editor.patch() == ''


def test_observation_repair_preserves_good_rows_and_ignores_volatile_duplicate_metadata():
    original = {'id': 'stable', 'task_id': 'task', 'base_commit': 'base', 'argv': ['probe'],
                'stdout': 'actual behavior', 'stderr': '', 'exit_code': 0, 'seconds': 1}
    duplicate = {**original, 'seconds': 2, 'provenance_audit': {'reason': 'Repeated review'}}
    repaired = {**original, 'id': 'replacement', 'argv': ['new-probe']}
    conflicts = {}
    assert pipeline._merge_observations([original], [duplicate, repaired], conflicts=conflicts) == [original, repaired]
    assert not conflicts
    changed_output = {**duplicate, 'stdout': 'different behavior'}
    assert pipeline._merge_observations([original], [changed_output, repaired], conflicts=conflicts) == [repaired]
    assert conflicts['stable']['rejection_reason'] == 'inconsistent_observation_across_collections'
    assert pipeline._merge_observations([repaired], [original], conflicts=conflicts) == [repaired]
    with pytest.raises(pipeline.DeepSWEPipelineError, match='collision'):
        pipeline._merge_observations([], [{**original, 'argv': ['different-probe']}], conflicts=conflicts)


@pytest.mark.parametrize('screening_passes,all_runs,all_passes,minimum,repetitions,expected', [
    (1, 3, 2, 2, 3, 'screening_rejected'), (0, 3, 0, 2, 3, 'qualification_rejected'),
    (0, 3, 1, 2, 3, 'qualification_rejected'), (0, 3, 2, 2, 3, 'accepted'),
    (0, 3, 3, 2, 3, 'accepted'), (0, 2, 2, 2, 3, 'execution_error'),
    (0, 5, 2, 2, 5, 'accepted'), (0, 5, 2, 3, 5, 'qualification_rejected'),
    (0, 5, 3, 3, 5, 'accepted'), (0, 1, 1, 1, 1, 'accepted'),
    (0, 3, 3, 3, 5, 'execution_error')])
def test_representative_command_enforces_current_gates(monkeypatch, tmp_path, screening_passes, all_runs, all_passes, minimum, repetitions, expected):
    from tests.software.deep_swe.test_integration_cli import complete_episode_cache
    from teamgym.llm.openai_client import normalize_model_for_provider
    task = SimpleNamespace(task_id='task', instruction='Repair', source_dir=tmp_path / 'source' / 'task',
        runtime=SimpleNamespace(docker_image='base'), repository=SimpleNamespace(base_commit_hash='base'),
        artifacts=SimpleNamespace(solution_patch=tmp_path / 'solution.patch'))
    monkeypatch.setattr(pipeline, '_RUN_STATE', None)
    monkeypatch.setattr(pipeline, '_OPTIONS', None)
    monkeypatch.setattr(pipeline, 'load_task', lambda *a, **kw: task)
    monkeypatch.setattr(integrity, 'image_identity', lambda p: {'id': 'sha256:base'})
    monkeypatch.setattr(integrity, 'task_identity', lambda p: {'task_id': 'task', 'base_commit': 'base'})
    monkeypatch.setattr(integrity, 'code_identity', lambda: {'files': {
        'teamgym/data/software/deep_swe/stages/patch_episode.py': hashlib.sha256(b'frozen harness\n').hexdigest()}})
    monkeypatch.setattr(pipeline, 'plan_public_checks', lambda **kw: ({'build': ['build']}, {}))
    monkeypatch.setattr(pipeline, 'write_public_check_validation_cache', lambda *a, **kw: None)
    output = tmp_path / 'out' / 'task'
    calls = []

    def diagnostics(argv, passes, runs):
        value = lambda flag: Path(argv[argv.index(flag) + 1])
        root = value('--output-dir')
        candidate = json.loads(value('--candidate').read_text())
        execution = candidate['execution_manifest']; actor = execution['actor']; protocol = execution['protocol']
        packet = packet_for(candidate, json.loads(value('--checks').read_text()), 'all')
        rows = []
        for rep in range(1, runs + 1):
            episode = root / f'baseline_rep{rep}'
            success = rep <= passes
            save(episode / 'actor_packet.json', packet)
            save(episode / 'candidate.frozen.hidden.json', candidate)
            save(episode / 'result.json', {'success': success, 'episode_valid': True, 'exit_code': 0,
                'reward': {'reward': int(success), 'f2p_passed': int(success), 'f2p_total': 1,
                           'p2p_passed': 1, 'p2p_total': 1}})
            save(episode / 'manifest.json', {'task_id': execution['task_id'],
                'model': normalize_model_for_provider(actor['provider'], actor['model']),
                'provider': actor['provider'], 'base_url': actor['base_url'], 'endpoint': actor['endpoint'],
                'reasoning_effort': actor['reasoning_effort'], 'source_base_commit': execution['source'].get('base_commit'),
                'runtime_image_id': execution['runtime_image']['id'], 'verifier_image_id': execution['verifier_image']['id'],
                'patch_turn_budget': protocol['patch_turns'], 'total_output_token_budget': protocol['total_output_tokens'],
                'max_output_tokens_per_call': protocol['max_output_tokens_per_call'],
                'max_public_checks_per_episode': protocol['max_checks_per_episode'],
                'actor_packet_sha256': pipeline.sha256(episode / 'actor_packet.json')})
            complete_episode_cache(episode, execution=execution, rep=rep)
            rows.append({'label': 'baseline', 'rep': rep, 'output_dir': str(episode),
                'success': success, 'episode_valid': True, 'f2p': [int(success), 1], 'p2p': [1, 1]})
        save(root / 'results.json', rows)
        save(root / 'summary.json', {'baseline': {'passes': passes, 'runs': runs,
             'f2p': [r['f2p'] for r in rows], 'p2p': [r['p2p'] for r in rows]}})

    def run(stage, argv, log):
        calls.append(stage)
        def value(flag): return Path(argv[argv.index(flag) + 1])
        if stage == 'instruction_only':
            diagnostics(argv, screening_passes, 3)
        elif stage == '0_discover':
            save(value('--output'), {'termination': 'done', 'files': {'a.py': {'text': 'x=1\n'}}})
        elif stage == '1_2_fragment_pool':
            save(value('--output'), {'fragments': [{'id': 'frag'}]})
        elif stage == '3a_observation_plan':
            save(value('--output'), {})
        elif stage == '3b_observation_collection':
            save(value('--output-dir') / 'observations.json', [])
        elif stage == '4b_functional_riu_round_01':
            save(value('--output'), {'instruction': 'Repair', 'code_atoms': [], 'observations': [],
                'allocation_candidates': [{'riu_id': f'riu-{i}', 'code_atom_ids': [], 'observation_ids': []} for i in (1, 2)]})
        elif stage == '4c_qualification_round_01':
            assert '--max-failures' not in argv
            diagnostics(argv, all_passes, all_runs)
        else:
            raise AssertionError(stage)

    monkeypatch.setattr(pipeline, '_run', run)
    def finish(*args, **kwargs):
        calls.append('distribution')
        args[-1]['decision'] = 'accepted'
    monkeypatch.setattr(pipeline, '_finish_distribution', finish)
    monkeypatch.setattr('sys.argv', ['pipeline', '--task-id', 'task', '--tasks-root', str(tmp_path / 'source'),
        '--output-root', str(tmp_path / 'out'), '--max-riu-rounds', '1',
        '--qualification-min-passes', str(minimum), '--qualification-repetitions', str(repetitions)])
    if expected == 'execution_error':
        with pytest.raises(pipeline.DeepSWEPipelineError, match=f'{repetitions} valid'):
            pipeline.main()
    else:
        pipeline.main()
        report = json.loads((output / 'report.json').read_text())
        assert report['decision'] == expected
        assert report['protocol']['qualification'] == {'minimum_passes': minimum, 'repetitions': repetitions}
        assert not (output / 'export/instance.json').exists()
        assert ('distribution' in calls) is (expected == 'accepted')
    if screening_passes:
        assert calls == ['instruction_only']


@pytest.mark.parametrize('target', ['0', '1'])
def test_cli_cannot_lower_mandatory_qualification_gate(monkeypatch, tmp_path, target):
    monkeypatch.setattr('sys.argv', ['pipeline', '--task-id', 'task',
        '--output-root', str(tmp_path), '--revision-target-passes', target])
    def unexpected_load(*args, **kwargs):
        raise AssertionError('Invalid acceptance target must fail before loading a task')
    monkeypatch.setattr(pipeline, 'load_task', unexpected_load)
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 2
