"""Offline regressions for pinned source, probe integrity and provenance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from teamgym.data.software.deep_swe import load_task, DeepSWEValidationError
from teamgym.data.software.deep_swe.source import REQUIRED_FILES
from teamgym.data.software.deep_swe.evidence import usable_observations, setup_failure
from teamgym.data.software.deep_swe.provenance import content_hash, visible_observation


def make_task(tmp_path: Path):
    root = tmp_path / 'example'
    for name in REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('public task\n')
    (root / 'task.toml').write_text('''schema_version = "1.3"
artifacts = []
[task]
name = "datacurve/example"
[metadata]
task_id = "example"
repository_url = "https://github.com/example/repo"
base_commit_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
language = "python"
category = "feature"
display_title = "Example"
display_description = "Example behavior"
[environment]
docker_image = "example:base"
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 1024
gpus = 0
build_timeout_sec = 60
[verifier]
network_mode = "none"
environment_mode = "shared"
timeout_sec = 60
[verifier.environment]
[agent]
timeout_sec = 60
''')
    (root / 'tests/config.json').write_text(json.dumps({
        'base_commit': 'a' * 40, 'f2p_node_ids': ['new behavior'],
        'p2p_node_ids': ['old behavior'], 'grade': {}}))
    return root


def complete_observation(**overrides):
    snapshot = {'head': 'a' * 40, 'sha256': 'b' * 64, 'clean': True, 'status': ''}
    row = {'id': 'obs', 'task_id': 'example', 'base_commit': 'a' * 40,
           'argv': ['python3', '-c', 'raise ValueError("bad input")'],
           'stdout': '', 'stderr': 'ValueError: bad input', 'exit_code': 1,
           'timed_out': False, 'tracked_checkout_unchanged': True,
           'checkout_before': snapshot, 'checkout_after': dict(snapshot),
           'public_sources_verified': True}
    row.update(overrides)
    digest = content_hash(visible_observation(row))
    row['repeat_check'] = {'runs': 2, 'passed': True, 'first_sha256': digest, 'second_sha256': digest}
    row['provenance_audit'] = {'approved': True, 'status': 'reviewed', 'content_sha256': digest,
                               'nonzero_is_target_behavior': row['exit_code'] != 0}
    return row


@pytest.mark.parametrize('change,match', [
    ({'base_commit': 'c' * 40}, 'disagree'),
    ({'f2p_node_ids': [{}]}, 'nonempty strings'),
    ({'f2p_node_ids': []}, 'must not be empty'),
    ({'p2p_node_ids': ['x', 'x']}, 'duplicate test ids'),
    ({'p2p_node_ids': ['new behavior']}, 'overlap'),
])
def test_source_rejects_invalid_verifier_contract(tmp_path, change, match):
    root = make_task(tmp_path)
    path = root / 'tests/config.json'
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(DeepSWEValidationError, match=match):
        load_task(root)


def test_source_checks_schema_and_explicit_runtime_override(tmp_path, monkeypatch):
    root = make_task(tmp_path)
    monkeypatch.setenv('DEEPSWE_RUNTIME_IMAGE', 'unrecorded:override')
    assert load_task(root).runtime.docker_image == 'example:base'
    assert load_task(root, runtime_image='sha256:explicit').runtime.docker_image == 'sha256:explicit'
    path = root / 'task.toml'
    path.write_text(path.read_text().replace('"1.3"', '"999"'))
    with pytest.raises(DeepSWEValidationError, match='unsupported schema'):
        load_task(root)


@pytest.mark.parametrize('field,value', [
    ('argv', {}), ('exit_code', None), ('exit_code', True), ('stdout', None),
    ('task_id', 'other'), ('base_commit', 'c' * 40), ('repeat_check', {}),
    ('provenance_audit', {}), ('public_sources_verified', False), ('id', {}),
])
def test_incomplete_or_foreign_observations_are_not_evidence(field, value):
    row = complete_observation()
    row[field] = value
    good, bad = usable_observations([row], task_id='example', base_commit='a' * 40)
    assert not good and bad[0]['rejection_reason']


def test_repeated_reviewed_nonzero_is_meaningful_and_hashes_bind_output():
    row = complete_observation()
    assert usable_observations([row])[0] == [row]
    row['stdout'] = 'constructor inserted answer'
    assert not usable_observations([row])[0]


def test_duplicate_ids_all_rejected_and_duplicate_content_not_repeated():
    one = complete_observation()
    assert not usable_observations([one, dict(one)])[0]
    good, bad = usable_observations([one, {**one, 'id': 'other-id'}])
    assert len(good) == 1 and bad[0]['rejection_reason'] == 'duplicate_observation_content'


def git_repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'a.py').write_text('VALUE = 1\n')
    subprocess.run(['git', 'add', 'a.py'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                    'commit', '-qm', 'base'], cwd=tmp_path, check=True)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tmp_path, text=True).strip()


def run_script(script, path, request):
    return subprocess.run([sys.executable, '-I', '-c', script], cwd=path,
                          input=json.dumps(request), text=True, capture_output=True)


def test_check_and_probe_detect_real_content_changes_and_dirty_base(tmp_path):
    from teamgym.data.software.deep_swe.checks import _RUN_CHECK
    from teamgym.data.software.deep_swe.observations import RUNNER
    commit = git_repo(tmp_path)
    mutate = [sys.executable, '-c', 'from pathlib import Path; Path("a.py").write_text("VALUE = 2\\n")']
    for script, commit_key in [(_RUN_CHECK, 'commit'), (RUNNER, 'expected_commit')]:
        subprocess.run(['git', 'reset', '--hard', '-q', commit], cwd=tmp_path, check=True)
        request = {commit_key: commit, 'kind': 'observation', 'argv': mutate, 'timeout_sec': 2}
        result = run_script(script, tmp_path, request)
        assert result.returncode == 0, result.stderr
        row = json.loads(result.stdout)
        assert row['exit_code'] == 0 and row['tracked_checkout_unchanged'] is False
        assert row['checkout_before']['sha256'] != row['checkout_after']['sha256']
        # Starting dirty cannot be accepted even if the status would remain M a.py.
        result = run_script(script, tmp_path, request)
        assert result.returncode != 0 and 'not clean' in result.stderr


def test_collector_retains_identical_repeated_outcome_and_rejects_unstable(monkeypatch, tmp_path):
    from teamgym.data.software.deep_swe import observations
    row = complete_observation()
    task = SimpleNamespace(task_id='example', repository=SimpleNamespace(base_commit_hash='a' * 40))
    check = {**row, 'public_source_refs': [], 'timeout_sec': 1}
    monkeypatch.setattr(observations, 'run', lambda *args: dict(row))
    monkeypatch.setattr(observations, 'audit_solver_visible', lambda **kw: row['provenance_audit'])
    collected = observations.collect_observation(task, check, 'image', client=None, trace_dir=tmp_path)
    assert usable_observations([collected])[0] == [collected]
    outcomes = iter([row, {**row, 'stdout': 'different'}])
    monkeypatch.setattr(observations, 'run', lambda *args: next(outcomes))
    unstable = observations.collect_observation(task, check, 'image', client=None, trace_dir=tmp_path)
    assert not unstable['repeat_check']['passed']
    assert not usable_observations([unstable])[0]


def test_public_check_rejects_mutation_despite_exit_zero(monkeypatch, tmp_path):
    from teamgym.data.software.deep_swe import checks
    monkeypatch.setattr(checks, 'public_repository_inventory', lambda task: {})
    monkeypatch.setattr(checks, 'OpenAIJsonClient', lambda cfg: None)
    row = complete_observation(exit_code=0)
    row['tracked_checkout_unchanged'] = False
    monkeypatch.setattr(checks, 'run_base_check', lambda *args: row)
    with pytest.raises(checks.PublicCheckError, match='does not pass on base'):
        checks.validate_public_checks(None, {'check': ['python3']}, trace_dir=tmp_path)


@pytest.mark.parametrize('response', [None, [], {'observations': [None]},
    {'observations': [{'argv': 'python3'}]},
    {'observations': [{'argv': ['python3'], 'timeout_sec': {}}]}])
def test_malformed_proposal_json_is_a_validation_error(response):
    from teamgym.data.software.deep_swe.observation_proposals import normalize_proposals
    with pytest.raises(ValueError):
        normalize_proposals(response, pool={'task_id': 'example', 'base_commit': 'a' * 40, 'fragments': []})


def test_proposal_ids_are_stable_and_distinct_for_replacements():
    from teamgym.data.software.deep_swe.observation_proposals import normalize_proposals
    pool = {'task_id': 'example', 'base_commit': 'a' * 40, 'fragments': []}
    proposal = {'argv': ['python3', '--version'], 'family': 'language_runtime',
                'boundary': 'version', 'why_not_in_instruction': 'runtime version'}
    a = normalize_proposals({'observations': [proposal]}, pool=pool)[0]
    b = normalize_proposals({'observations': [proposal]}, pool=pool)[0]
    c = normalize_proposals({'observations': [{**proposal, 'argv': ['python3', '-V']}]}, pool=pool)[0]
    assert a['id'] == b['id'] != c['id']


def test_single_huge_line_is_an_explicit_budget_error():
    from teamgym.data.software.deep_swe.fragments import extract_complete, FragmentBudgetError
    with pytest.raises(FragmentBudgetError, match='single source line requires 1000 tokens'):
        extract_complete('minified.js', 'x' * 1000, count_tokens=len, context_tokens=100)
