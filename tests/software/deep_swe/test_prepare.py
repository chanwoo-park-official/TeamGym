"""Source preparation pins commits and refuses to reuse altered input checkouts."""

from pathlib import Path
import subprocess

import pytest

from teamgym.data.software.deep_swe import prepare


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def test_download_pins_independent_clean_checkout(monkeypatch, tmp_path):
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    git(upstream, 'init', '-q')
    source = upstream / 'source.txt'
    commits = []
    for text in ('first', 'second'):
        source.write_text(text)
        git(upstream, 'add', '.')
        git(upstream, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '-qm', text)
        commits.append(git(upstream, 'rev-parse', 'HEAD'))
    monkeypatch.setattr(prepare, 'SOURCE_URL', str(upstream))
    checkout = tmp_path / 'download'
    receipt = prepare.download_source(checkout, commits[0])
    assert receipt['revision'] == commits[0]
    assert (checkout / 'source.txt').read_text() == 'first'
    assert checkout.resolve() != upstream.resolve()
    assert prepare.download_source(checkout, commits[1])['revision'] == commits[1]
    (checkout / 'source.txt').write_text('local work')
    with pytest.raises(ValueError, match='local modifications'):
        prepare.download_source(checkout, commits[0])
    assert (checkout / 'source.txt').read_text() == 'local work'


def test_download_rejects_other_checkout_and_symlink(monkeypatch, tmp_path):
    target = tmp_path / 'other'
    target.mkdir()
    git(target, 'init', '-q')
    git(target, 'remote', 'add', 'origin', 'https://example.invalid/other.git')
    with pytest.raises(ValueError, match='unexpected upstream'):
        prepare.download_source(target)
    link = tmp_path / 'link'
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        prepare.download_source(link)


def test_invalid_revision_fails_before_creating_checkout(tmp_path):
    destination = tmp_path / 'source'
    with pytest.raises(ValueError, match='complete Git commit'):
        prepare.download_source(destination, '--other-option')
    assert not destination.exists()
