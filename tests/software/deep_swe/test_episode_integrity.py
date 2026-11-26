"""Offline output-integrity and recovery tests; never run models or Docker."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from teamgym.data.software.deep_swe import pipeline
from teamgym.data.software.deep_swe.integrity import canonical_sha256
from tests.software.deep_swe.test_export import execution_fixture
from tests.software.deep_swe.test_integration_cli import complete_episode_cache


def stage(root):
    root.mkdir(exist_ok=True)
    packet = {'instruction': 'fixed public input', 'public_check_catalog': []}
    execution = execution_fixture()
    execution['code']['files']['teamgym/data/software/deep_swe/stages/patch_episode.py'] = hashlib.sha256(b'frozen harness\n').hexdigest()
    execution['code']['sha256'] = canonical_sha256(execution['code']['files'])
    actor, protocol = execution['actor'], execution['protocol']
    rows = []
    for rep in range(1, 4):
        episode = root / f'baseline_rep{rep}'; episode.mkdir(exist_ok=True)
        (episode / 'actor_packet.json').write_text(json.dumps(packet))
        (episode / 'candidate.frozen.hidden.json').write_text(json.dumps({'execution_manifest': execution}))
        (episode / 'result.json').write_text(json.dumps({'success': False, 'episode_valid': True}))
        manifest = {'task_id': execution['task_id'], 'model': 'openai.gpt-5.5',
            'provider': actor['provider'], 'base_url': actor['base_url'], 'endpoint': actor['endpoint'],
            'reasoning_effort': actor['reasoning_effort'], 'source_base_commit': execution['source']['base_commit'],
            'runtime_image_id': execution['runtime_image']['id'], 'verifier_image_id': execution['verifier_image']['id'],
            'patch_turn_budget': protocol['patch_turns'], 'total_output_token_budget': protocol['total_output_tokens'],
            'max_output_tokens_per_call': protocol['max_output_tokens_per_call'],
            'max_public_checks_per_episode': protocol['max_checks_per_episode'],
            'actor_packet_sha256': pipeline.sha256(episode / 'actor_packet.json')}
        (episode / 'manifest.json').write_text(json.dumps(manifest))
        complete_episode_cache(episode, execution=execution, rep=rep)
        rows.append({'label': 'baseline', 'rep': rep, 'output_dir': str(episode),
            'success': False, 'episode_valid': True, 'f2p': [0, 1], 'p2p': [1, 1]})
    summary = {'passes': 0, 'runs': 3, 'f2p': [[0, 1]] * 3, 'p2p': [[1, 1]] * 3}
    (root / 'results.json').write_text(json.dumps(rows))
    (root / 'summary.json').write_text(json.dumps({'baseline': summary}))
    return packet, execution, summary


def rehash_cache(episode):
    path = episode / 'cache_manifest.json'
    cache = json.loads(path.read_text())
    cache['inputs_sha256'] = canonical_sha256(cache['inputs'])
    cache['outputs'] = {name: pipeline.sha256(Path(name)) for name in cache['outputs']}
    path.write_text(json.dumps(cache))


@pytest.mark.parametrize('change', ['patch', 'missing_ninth', 'wrong_local_path', 'input_hash',
    'schema_bool', 'forged_packet', 'forged_harness', 'forged_patch', 'source', 'actor', 'checks'])
def test_episode_receipt_verifies_all_nine_outputs_and_frozen_inputs(tmp_path, change):
    packet, execution, _ = stage(tmp_path)
    episode = tmp_path / 'baseline_rep1'
    path = episode / 'cache_manifest.json'; cache = json.loads(path.read_text())
    if change == 'patch':
        (episode / 'model.patch').write_text('unreceipted changed repair')
    elif change == 'missing_ninth':
        cache['outputs'].pop(str(episode / 'harness.frozen.py')); path.write_text(json.dumps(cache))
    elif change == 'wrong_local_path':
        digest = cache['outputs'].pop(str(episode / 'model.patch'))
        cache['outputs'][str(tmp_path / 'elsewhere/model.patch')] = digest; path.write_text(json.dumps(cache))
    elif change == 'input_hash':
        cache['inputs_sha256'] = 'f' * 64; path.write_text(json.dumps(cache))
    elif change == 'schema_bool':
        cache['schema_version'] = True; path.write_text(json.dumps(cache))
    elif change == 'forged_packet':
        (episode / 'actor_packet.json').write_text(json.dumps({'instruction': 'other actual input'}))
        manifest = json.loads((episode / 'manifest.json').read_text())
        manifest['actor_packet_sha256'] = pipeline.sha256(episode / 'actor_packet.json')
        (episode / 'manifest.json').write_text(json.dumps(manifest)); rehash_cache(episode)
    elif change == 'forged_harness':
        (episode / 'harness.frozen.py').write_text('changed actor execution code')
        rehash_cache(episode)  # Self-consistent hashes cannot change the frozen harness identity.
    elif change == 'forged_patch':
        (episode / 'model.patch').write_text('changed repair after official verification')
        rehash_cache(episode)
    else:
        if change == 'source': cache['inputs']['task']['base_commit'] = 'a' * 40
        elif change == 'actor': cache['inputs']['actor']['model'] = 'another-model'
        else: cache['inputs']['checks_sha256'] = 'a' * 64
        path.write_text(json.dumps(cache)); rehash_cache(episode)
    with pytest.raises(pipeline.DeepSWEPipelineError):
        pipeline._episode_receipts(tmp_path, expected_packet=packet, expected_execution=execution)


@pytest.mark.parametrize('change', ['outside_group', 'duplicate'])
def test_episode_rows_cannot_import_historical_or_duplicate_output_locations(tmp_path, change):
    run = tmp_path / 'current'; packet, execution, _ = stage(run)
    rows = json.loads((run / 'results.json').read_text())
    if change == 'outside_group':
        historical = tmp_path / 'historical'; stage(historical)
        rows[0]['output_dir'] = str(historical / 'baseline_rep1')
    else:
        rows[1] = deepcopy(rows[0])
    (run / 'results.json').write_text(json.dumps(rows))
    with pytest.raises(pipeline.DeepSWEPipelineError, match='outside its group or duplicated'):
        pipeline._episode_receipts(run, expected_packet=packet, expected_execution=execution)


def test_context_window_failure_remains_valid_failure_even_if_prior_patch_passed_verifier(tmp_path):
    packet, execution, _ = stage(tmp_path)
    episode = tmp_path / 'baseline_rep1'
    episode_data = json.loads((episode / 'episode.json').read_text())
    episode_data['termination'] = 'context_window_failure'
    (episode / 'episode.json').write_text(json.dumps(episode_data))
    result = json.loads((episode / 'result.json').read_text())
    result['episode'] = episode_data
    result['reward'] = {'reward': 1, 'f2p_passed': 1, 'f2p_total': 1, 'p2p_passed': 1, 'p2p_total': 1}
    (episode / 'result.json').write_text(json.dumps(result)); rehash_cache(episode)
    rows = json.loads((tmp_path / 'results.json').read_text()); rows[0]['f2p'] = [1, 1]
    (tmp_path / 'results.json').write_text(json.dumps(rows))
    receipts = pipeline._episode_receipts(tmp_path, expected_packet=packet, expected_execution=execution)
    assert receipts[0]['episode_valid'] is True and receipts[0]['success'] is False


@pytest.mark.parametrize('field', ['reward', 'f2p_passed', 'f2p_total', 'p2p_passed', 'p2p_total'])
def test_rehashed_boolean_reward_or_test_counts_are_not_valid_official_results(tmp_path, field):
    packet, execution, _ = stage(tmp_path)
    episode = tmp_path / 'baseline_rep1'; path = episode / 'result.json'
    result = json.loads(path.read_text()); result['reward'][field] = True
    path.write_text(json.dumps(result)); rehash_cache(episode)
    with pytest.raises(pipeline.DeepSWEPipelineError, match='reward/count schema'):
        pipeline._episode_receipts(tmp_path, expected_packet=packet, expected_execution=execution)


@pytest.mark.parametrize('change', ['patch', 'summary_passes', 'summary_counts', 'result_row_counts', 'summary_checkpoint_miss'])
def test_cached_summary_is_verified_before_paid_stages_and_recovers_as_integrity_error(tmp_path, monkeypatch, change):
    run = tmp_path / 'run'; packet, execution, expected = stage(run)
    if change == 'patch': (run / 'baseline_rep1/model.patch').write_text('changed after summary checkpoint')
    elif change.startswith('summary'):
        value = deepcopy(expected)
        if change in ('summary_passes', 'summary_checkpoint_miss'): value['passes'] = 1
        else: value['f2p'][0] = [1, 1]
        (run / 'summary.json').write_text(json.dumps({'baseline': value}))
    else:
        rows = json.loads((run / 'results.json').read_text()); rows[0]['f2p'] = [1, 1]
        (run / 'results.json').write_text(json.dumps(rows))
    ready = tmp_path / 'export'; ready.mkdir(); (ready / 'instance.json').write_text('old accepted output')
    (tmp_path / 'report.json').write_text(json.dumps({'decision': 'accepted', 'export': {'path': str(ready)}}))
    state = SimpleNamespace(output=tmp_path, dirty=False)
    monkeypatch.setattr(pipeline, '_RUN_STATE', state)
    monkeypatch.setattr(pipeline, '_stage_exists', lambda *a, **kw: change != 'summary_checkpoint_miss')
    invoked = []
    def diagnostic(stage_name, argv, log):
        assert state.dirty is True
        assert not ready.exists()
        assert list((run / 'integrity_rejections').glob('*/summary.json'))
        invoked.append(stage_name)
        # The production diagnostic runner checks each local cache itself;
        # this offline stub returns a repaired valid diagnostic stage.
        stage(run)
    monkeypatch.setattr(pipeline, '_run', diagnostic)
    actual = pipeline._diagnostic_summary('instruction_only', ['diagnostic-placeholder'], tmp_path / 'log',
        run / 'summary.json', skip_existing=True, expected_packet=packet, expected_execution=execution)
    assert invoked == ['instruction_only'] and actual == expected
    recovery = json.loads(next((run / 'integrity_rejections').glob('*/recovery.json')).read_text())
    assert recovery['classification'] == 'cache_integrity_error'
    assert recovery['counted_as_model_failure'] is False
    assert next((tmp_path / 'invalidated_exports').glob('*/instance.json')).read_text() == 'old accepted output'


def test_valid_cached_summary_skips_diagnostic_only_after_all_output_checks(tmp_path, monkeypatch):
    packet, execution, expected = stage(tmp_path)
    monkeypatch.setattr(pipeline, '_stage_exists', lambda *a, **kw: True)
    monkeypatch.setattr(pipeline, '_run', lambda *a, **kw: pytest.fail('Valid cached diagnostics should be reused'))
    assert pipeline._diagnostic_summary('qualification', [], tmp_path / 'log', tmp_path / 'summary.json',
        skip_existing=True, expected_packet=packet, expected_execution=execution) == expected


def test_new_validation_attempt_quarantines_ready_export_even_if_attempt_errors(tmp_path, monkeypatch):
    (tmp_path / 'export').mkdir(); (tmp_path / 'export/instance.json').write_text('accepted original')
    (tmp_path / 'report.json').write_text(json.dumps({'decision': 'accepted', 'export': {'path': 'export/instance.json'}}))
    def failed_attempt():
        pipeline._RUN_STATE = SimpleNamespace(output=tmp_path)
        pipeline._invalidate_ready_export(tmp_path, 'new_validation_attempt')
        raise pipeline.DeepSWEPipelineError('invalid cached output')
    monkeypatch.setattr(pipeline, '_main', failed_attempt)
    with pytest.raises(pipeline.DeepSWEPipelineError): pipeline.main()
    assert not (tmp_path / 'export').exists()
    assert json.loads((tmp_path / 'report.json').read_text())['decision'] == 'execution_error'
    archived = next((tmp_path / 'invalidated_exports').glob('*/instance.json'))
    assert archived.read_text() == 'accepted original'
    assert json.loads((archived.parent / 'previous_report.json').read_text())['decision'] == 'accepted'


def test_preflight_failure_before_run_state_cannot_leave_old_export_ready(tmp_path, monkeypatch):
    output = tmp_path / 'example'; (output / 'export').mkdir(parents=True)
    (output / 'export/instance.json').write_text('accepted prior bytes')
    (output / 'report.json').write_text(json.dumps({'decision': 'accepted'}))
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--task-id', 'example', '--output-root', str(tmp_path)])
    def invalid_source(*a, **kw): raise pipeline.DeepSWEPipelineError('source preflight failed')
    monkeypatch.setattr(pipeline, 'load_task', invalid_source)
    with pytest.raises(pipeline.DeepSWEPipelineError, match='source preflight failed'):
        pipeline.main()
    assert not (output / 'export').exists()
    assert next((output / 'invalidated_exports').glob('*/instance.json')).read_text() == 'accepted prior bytes'
    assert json.loads((output / 'report.json').read_text())['decision'] == 'validation_pending'
