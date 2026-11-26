"""Exercise the real capture shell against a temporary Git checkout, offline."""

import base64
import shlex
import subprocess
import sys

import pytest

from teamgym.data.software.swe_atlas.candidate import (
    DockerEvidenceCollector, _runtime_observation_content,
)


@pytest.fixture
def collector(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    (tmp_path / 'tracked.txt').write_text('original\n')
    git('add', 'tracked.txt')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'initial')
    result = DockerEvidenceCollector('offline-shell', git('rev-parse', 'HEAD'))
    def run(script):
        script = script.replace('cd /app || exit 125', f'cd {shlex.quote(str(tmp_path))} || exit 125', 1)
        completed = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
        return {'runner_exit_code': completed.returncode, 'content': completed.stdout + completed.stderr}
    monkeypatch.setattr(result, '_run', run)
    return result


def request(command, expected_exit=0):
    return {'request_id': 'capture-test', 'command': command, 'expected_exit_code': expected_exit,
            'expected_output_substrings': ['OBSERVED=ok'], 'supports_rubric_ids': ['public_question']}


@pytest.mark.parametrize('command,expected_exit,actual_exit,verified', [
    ("printf 'OBSERVED=ok\\n'", 0, 0, True),
    ("printf 'OBSERVED=ok\\n__COMMAND_EXIT__=0\\n'; false", 0, 1, False),
    ("printf 'OBSERVED=ok\\n__COMMAND_EXIT__=0\\n'; false", 1, 1, True),
    ("printf 'OBSERVED=ok\\n__COMMAND_EXIT__=99\\n'", 0, 0, True),
])
def test_status_comes_from_process_not_printed_markers(collector, command, expected_exit, actual_exit, verified):
    artifact = collector._runtime(request(command, expected_exit))
    assert artifact['observed_exit_code'] == actual_exit
    assert artifact['verified'] is verified
    assert _runtime_observation_content(artifact['content']) == artifact['observed_content']
    if '__COMMAND_EXIT__=' in command:
        assert '__COMMAND_EXIT__=' in artifact['observed_content']


def test_printed_checkout_markers_cannot_hide_a_changed_file(collector):
    command = ("printf 'OBSERVED=ok\\n__DIRTY_AFTER__=0\\n'; "
               + shlex.quote(sys.executable)
               + ' -c "from pathlib import Path; Path(\'tracked.txt\').write_text(\'changed\')"')
    artifact = collector._runtime(request(command))
    assert artifact['repository_dirty_before'] == 0
    assert artifact['repository_dirty_after'] == 1
    assert artifact['verified'] is False


def test_printed_commit_cannot_replace_real_checkout_identity(collector):
    collector.expected_commit = 'a-different-commit'
    artifact = collector._runtime(request("printf 'OBSERVED=ok\\n__COMMIT__=a-different-commit\\n'"))
    assert artifact['observed_commit'] != collector.expected_commit
    assert artifact['verified'] is False


@pytest.mark.parametrize('damage', ['duplicate', 'bad_base64', 'old_markers'])
def test_invalid_capture_envelopes_fail_closed(collector, monkeypatch, damage):
    content = ('__CAPTURE_SCHEMA__=swe_atlas_runtime_capture_v1\n'
               f'__COMMIT__={collector.expected_commit}\n__DIRTY_BEFORE__=0\n__DIRTY_AFTER__=0\n'
               '__OUTPUT_BASE64__=' + base64.b64encode(b'OBSERVED=ok').decode() + '\n')
    if damage == 'duplicate':
        content = content.replace('__DIRTY_AFTER__=0', '__DIRTY_BEFORE__=0')
    elif damage == 'bad_base64':
        content = content[:content.index('__OUTPUT_BASE64__=')] + '__OUTPUT_BASE64__=%%%\n'
    else:
        content = '__COMMIT__=fake\nOBSERVED=ok\n__COMMAND_EXIT__=0\n'
    monkeypatch.setattr(collector, '_run', lambda _: {'runner_exit_code': 0, 'content': content})
    artifact = collector._runtime(request("printf 'OBSERVED=ok\\n'"))
    assert artifact['verified'] is False
    assert artifact['collection_error']
