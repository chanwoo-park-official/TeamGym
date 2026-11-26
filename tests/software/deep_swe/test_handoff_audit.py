"""Adversarial offline checks of qualified-producer phase handoff evidence."""

from tests._paths import REPO_ROOT
from pathlib import Path
import importlib.util
import json
import pytest

spec = importlib.util.spec_from_file_location('handoff_audit', REPO_ROOT / 'scripts/data_generation/software/deep_swe/handoff_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


@pytest.fixture
def checks(tmp_path):
    identity = {'source': {'base_commit': 'abc'}}
    argv = ['go', 'test', '.', '-run', '^$']
    clean = {'head': 'abc', 'status': '', 'clean': True, 'sha256': 'a' * 64}
    row = {'base_commit': 'abc', 'resolved_base_commit': 'abc', 'exit_code': 0,
           'timed_out': False, 'tracked_checkout_unchanged': True, 'stdout': 'ok', 'stderr': '',
           'checkout_before': clean, 'checkout_after': dict(clean)}
    artifact = {'id': 'compile', 'argv': argv, 'stdout': 'ok', 'stderr': '', 'exit_code': 0}
    row['provenance_audit'] = {'approved': True, 'status': 'reviewed', 'content_sha256': audit.canonical(artifact)}
    write(tmp_path / 'public_checks.json', {'compile': argv})
    write(tmp_path / 'public_checks.base_validation.json', {'compile': row})
    return tmp_path, identity


def test_public_compile_role_reuses_exact_approved_base_receipt_without_model(checks):
    root, identity = checks
    assert audit.audit_checks(root, identity)['count'] == 1


@pytest.mark.parametrize('mutation', ['argv', 'output', 'approved_int', 'dirty', 'base', 'missing', 'exit_bool'])
def test_public_check_receipt_cannot_be_relabelled(checks, mutation):
    root, identity = checks
    path = root / 'public_checks.base_validation.json'; value = audit.read(path); row = value['compile']
    if mutation == 'argv': write(root / 'public_checks.json', {'compile': ['true']})
    elif mutation == 'output': row['stdout'] = 'invented'
    elif mutation == 'approved_int': row['provenance_audit']['approved'] = 1
    elif mutation == 'dirty': row['checkout_after']['status'] = ' M main.go'
    elif mutation == 'base': row['base_commit'] = 'other'
    elif mutation == 'missing': value = {}
    elif mutation == 'exit_bool': row['exit_code'] = False
    write(path, value)
    with pytest.raises(ValueError): audit.audit_checks(root, identity)


def test_all_background_source_must_be_pinned_even_when_not_canonical(tmp_path):
    identity = {'source': {'task_id': 't', 'base_commit': 'abc'}, 'runtime_image': {'id': 'sha256:image'}}
    write(tmp_path / 'discovered_code.json', {'task_id': 't', 'base_commit': 'abc', 'files': {
        'main.go': {'text': 'public'}, 'unowned.go': {'text': 'invented extra answer'}}})
    write(tmp_path / 'observations.validated.json', [])
    def source_reader(image, commit, paths):
        assert paths == ['main.go', 'unowned.go']
        return {'base_commit': commit, 'clean': True, 'files': {
            'main.go': __import__('hashlib').sha256(b'public').hexdigest(),
            'unowned.go': __import__('hashlib').sha256(b'actual source').hexdigest()}}
    with pytest.raises(ValueError, match='pinned-base'): audit.audit_background(tmp_path, identity, source_reader=source_reader)


def test_background_rejects_unreviewed_distractor_observation(tmp_path):
    identity = {'source': {'task_id': 't', 'base_commit': 'abc'}, 'runtime_image': {'id': 'sha256:image'}}
    write(tmp_path / 'discovered_code.json', {'task_id': 't', 'base_commit': 'abc', 'files': {'main.go': {'text': 'public'}}})
    write(tmp_path / 'observations.validated.json', [{'id': 'looks_valid', 'stdout': 'answer'}])
    def source_reader(image, commit, paths):
        return {'base_commit': commit, 'clean': True, 'files': {'main.go': __import__('hashlib').sha256(b'public').hexdigest()}}
    with pytest.raises(ValueError, match='observation'): audit.audit_background(tmp_path, identity, source_reader=source_reader)


@pytest.fixture
def episode_group(tmp_path, monkeypatch):
    from teamgym.data.software.deep_swe.stages import diagnostic_runs as diag
    from teamgym.data.software.deep_swe.stages.patch_episode import packet_for
    from teamgym.data.software.deep_swe import pipeline
    # External task/image resolution is isolated; exact nine saved outputs,
    # candidate/packet equality and outcome validation remain real.
    monkeypatch.setattr(diag, 'episode_inputs', lambda job: {'condition': {'label': job['label'], 'rep': job['rep']}, 'candidate': audit.digest(job['candidate'])})
    monkeypatch.setattr(pipeline, '_episode_receipts', lambda *a, **k: [])
    candidate = {'instruction': 'repair', 'code_atoms': [], 'observations': [], 'allocation_candidates': [], 'execution_manifest': {}}
    cp = tmp_path / 'candidate.json'; checks = tmp_path / 'checks.json'; group = tmp_path / 'group'
    write(cp, candidate); write(checks, {'compile': ['go', 'test']})
    rows = []
    for rep in range(1, 4):
        out = group / f'baseline_rep{rep}'; out.mkdir(parents=True)
        for name in diag.EPISODE_OUTPUTS: (out / name).write_text('{}')
        (out / 'harness.frozen.py').write_text('frozen harness')
        write(out / 'candidate.frozen.hidden.json', candidate)
        write(out / 'actor_packet.json', packet_for(candidate, audit.read(checks), 'all'))
        write(out / 'manifest.json', {'actor_packet_sha256': audit.digest(out / 'actor_packet.json')})
        write(out / 'result.json', {'episode_valid': True, 'success': False, 'reward': {}, 'verification_valid': True})
        inputs = diag.episode_inputs({'label': 'baseline', 'rep': rep, 'candidate': cp})
        write(out / 'cache_manifest.json', {'schema_version': 1, 'inputs': inputs, 'inputs_sha256': audit.canonical(inputs),
            'outputs': {str(out / name): audit.digest(out / name) for name in diag.EPISODE_OUTPUTS}})
        rows.append({'label': 'baseline', 'rep': rep, 'output_dir': str(out), 'episode_valid': True, 'success': False, 'exit_code': 0})
    write(group / 'results.json', rows)
    identity = {'config': {'actor_model': 'gpt55'}, 'source': {'source_dir': 'task'},
                'code': {'files': {'teamgym/data/software/deep_swe/stages/patch_episode.py': audit.digest(group / 'baseline_rep1/harness.frozen.py')}}}
    return group, cp, checks, identity


def test_complete_failed_episodes_are_valid_evidence(episode_group):
    group, cp, checks, identity = episode_group
    assert audit.audit_episode_group(group, candidate_path=cp, checks_path=checks, identity=identity)['runs'] == 3


@pytest.mark.parametrize('mutation', ['missing_ninth', 'tamper', 'forged_packet_receipt', 'duplicate_rep', 'success_int', 'candidate_changed', 'wrong_harness_receipt'])
def test_episode_handoff_refuses_partial_stale_or_relabelled_artifacts(episode_group, mutation):
    group, cp, checks, identity = episode_group; out = group / 'baseline_rep1'
    if mutation == 'missing_ninth': (out / 'trajectory.json').unlink()
    elif mutation == 'tamper': (out / 'model.patch').write_text('new patch')
    elif mutation == 'forged_packet_receipt':
        write(out / 'actor_packet.json', {'instruction': 'extra hidden answer'})
        receipt = audit.read(out / 'cache_manifest.json'); receipt['outputs'][str(out / 'actor_packet.json')] = audit.digest(out / 'actor_packet.json'); write(out / 'cache_manifest.json', receipt)
    elif mutation == 'duplicate_rep':
        rows = audit.read(group / 'results.json'); rows[1]['rep'] = 1; write(group / 'results.json', rows)
    elif mutation == 'success_int':
        rows = audit.read(group / 'results.json'); rows[0]['success'] = 0; write(group / 'results.json', rows)
    elif mutation == 'candidate_changed':
        value = audit.read(cp); value['instruction'] = 'changed'; write(cp, value)
    elif mutation == 'wrong_harness_receipt':
        (out / 'harness.frozen.py').write_text('different executable')
        receipt = audit.read(out / 'cache_manifest.json'); receipt['outputs'][str(out / 'harness.frozen.py')] = audit.digest(out / 'harness.frozen.py'); write(out / 'cache_manifest.json', receipt)
    with pytest.raises((ValueError, OSError)): audit.audit_episode_group(group, candidate_path=cp, checks_path=checks, identity=identity)


def test_original_producer_change_is_rejected_before_any_postflight_consumption(tmp_path):
    producer = tmp_path / 'producer';producer.mkdir();p = producer / 'evidence.json';write(p, {'old': True})
    before = {'status': 'passed', 'producer': str(producer.resolve()), 'producer_tree': audit.tree_receipt(producer)}
    write(p, {'new': True})
    with pytest.raises(ValueError, match='producer changed'): audit.postflight(producer, {}, p, tmp_path / 'newphase', before)


@pytest.mark.parametrize('minimum,repetitions,passes', [
    (2, 3, 0), (2, 3, 1), (2, 3, 2), (2, 3, 3),
    (3, 5, 2), (3, 5, 3), (1, 1, 1), (2, 5, 2)])
def test_handoff_enforces_recorded_qualification_policy(monkeypatch, tmp_path, minimum, repetitions, passes):
    identity = {'code': {'sha256': 'frozen'}}
    candidate = tmp_path / 'candidate_functional_rius_round_01.json'
    write(tmp_path / 'run_identity.json', identity)
    policy = {'minimum_passes': minimum, 'repetitions': repetitions}
    write(candidate, {'execution_manifest': {'code': identity['code'], 'qualification': policy}})
    monkeypatch.setattr(audit, 'audit_background', lambda *a, **k: {})
    monkeypatch.setattr(audit, 'audit_checks', lambda *a, **k: {})
    def episode_group(directory, **kwargs):
        screening = directory.name == 'stage4a_instruction_only'
        assert kwargs.get('repetitions', 3) == (3 if screening else repetitions)
        return {'runs': 3 if screening else repetitions, 'passes': 0 if screening else passes}
    monkeypatch.setattr(audit, 'audit_episode_group', episode_group)
    if passes < minimum:
        with pytest.raises(ValueError, match='qualification gate failed'):
            audit.preflight(tmp_path, identity, candidate)
    else:
        receipt = audit.preflight(tmp_path, identity, candidate)
        assert receipt['status'] == 'passed'
        assert receipt['qualification_policy'] == policy
