"""Content-bound receipts and crash-safe JSON for DeepSWE construction.

A receipt is written only after the referenced outputs are complete. Missing,
malformed, or changed inputs/outputs always cause a cache miss. No credentials
or secret environment values belong in receipt inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

SCHEMA_VERSION = 1


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Serialize fully, fsync, then atomically replace on the same filesystem."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=target.parent, prefix=f'.{target.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_json(path: str | Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return None


def tree_identity(root: str | Path) -> dict:
    root = Path(root).resolve(strict=True)
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in {'.git', '__pycache__', '.pytest_cache'} for part in relative.parts):
            continue
        if path.is_file():
            files[relative.as_posix()] = {
                'sha256': file_sha256(path),
                'executable': bool(path.stat().st_mode & 0o111),
            }
        elif path.is_symlink():
            # A dangling link cannot silently disappear from the source hash.
            files[relative.as_posix()] = {'symlink': os.readlink(path)}
    return {'files': files, 'sha256': canonical_sha256(files)}


def task_identity(task_dir: str | Path) -> dict:
    """Bind all source artifacts, upstream revision, and declared base commits."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    root = Path(task_dir).expanduser().resolve(strict=True)
    config = tomllib.loads((root / 'task.toml').read_text(encoding='utf-8'))
    tests = json.loads((root / 'tests/config.json').read_text(encoding='utf-8'))
    try:
        source_revision = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = None
    return {'task_id': root.name, 'source_dir': str(root),
            'source_revision': source_revision,
            'base_commit': config.get('metadata', {}).get('base_commit_hash'),
            'test_base_commit': tests.get('base_commit'),
            'runtime_image': config.get('environment', {}).get('docker_image'),
            'tree': tree_identity(root)}


def image_identity(image: str) -> dict:
    """Resolve mutable tags to the exact locally executable image and digests.

    Failure to resolve is an execution error, never an identity based on a tag.
    Locally built images may have no registry digest; their content ID remains
    mandatory and identifies the actual executable image.
    """
    rows = json.loads(subprocess.check_output(
        ['docker', 'image', 'inspect', image], text=True))
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f'Cannot resolve Docker image identity: {image}')
    info = rows[0]
    identifier = info.get('Id')
    if not isinstance(identifier, str) or not identifier.startswith('sha256:'):
        raise ValueError(f'Missing Docker content ID: {image}')
    return {'requested': image, 'id': identifier,
            'repo_digests': sorted(info.get('RepoDigests') or []),
            'os': info.get('Os'), 'architecture': info.get('Architecture'),
            'variant': info.get('Variant')}


def code_identity(paths: Iterable[str | Path] | None = None) -> dict:
    """Hash construction/harness/prompts, including local uncommitted changes."""
    package = Path(__file__).resolve().parent
    repository = package.parents[3]
    if paths is None:
        paths = [*package.rglob('*.py'),
                 * (repository / 'teamgym/prompts/software/deep_swe').rglob('*'),
                 repository / 'teamgym/tokenization.py',
                 repository / 'teamgym/llm/construction_client.py',
                 repository / 'teamgym/llm/openai_client.py']
    files = {}
    for raw in sorted(map(Path, paths)):
        if raw.is_dir():
            continue
        if not raw.is_file():
            # Explicit dependencies cannot silently go missing.
            raise FileNotFoundError(raw)
        path = raw.resolve()
        label = str(path.relative_to(repository)) if path.is_relative_to(repository) else str(path)
        files[label] = file_sha256(path)
    return {'files': files, 'sha256': canonical_sha256(files)}


def _output_identity(outputs: Iterable[str | Path]) -> dict:
    return {str(Path(path)): file_sha256(path) for path in outputs}


def write_manifest(path: str | Path, inputs: Any, *,
                   outputs: Iterable[str | Path] | None = None) -> dict:
    receipt = {'schema_version': SCHEMA_VERSION, 'inputs': inputs,
               'inputs_sha256': canonical_sha256(inputs)}
    if outputs is not None:
        receipt['outputs'] = _output_identity(outputs)
    atomic_write_json(path, receipt)
    return receipt


def validate_manifest(path: str | Path, inputs: Any, *,
                      outputs: Iterable[str | Path] | None = None) -> tuple[bool, str]:
    receipt = load_json(path)
    if not isinstance(receipt, dict) or receipt.get('schema_version') != SCHEMA_VERSION:
        return False, 'missing_or_invalid_manifest'
    expected = canonical_sha256(inputs)
    if receipt.get('inputs_sha256') != expected or receipt.get('inputs') != inputs:
        return False, 'inputs_changed'
    if outputs is not None:
        try:
            current = _output_identity(outputs)
        except OSError:
            return False, 'missing_output'
        if receipt.get('outputs') != current:
            return False, 'outputs_changed_or_unreceipted'
    return True, 'manifest_match'
