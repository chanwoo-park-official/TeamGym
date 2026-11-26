"""Bind SWE Atlas construction checkpoints and exports to immutable inputs."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import platform
from importlib import metadata


class ConstructionIdentityError(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), default=str).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def task_identity(task) -> dict:
    payload = asdict(task)
    payload.pop('source_dir')
    payload = json.loads(json.dumps(payload, default=str))
    # The entire loaded task, including question, rubrics and answer, is bound.
    # Raw task files additionally capture loader-visible source revisions.
    files = {str(p.relative_to(task.source_dir)): digest(p)
        for p in sorted(task.source_dir.rglob('*')) if p.is_file() and '.git' not in p.parts}
    return {'task': payload, 'source_files': files, 'sha256': canonical_hash([payload, files])}


def image_identity(image: str) -> dict:
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
    if not str(info.get('Id', '')).startswith('sha256:'):
        raise ConstructionIdentityError('Docker image could not be resolved to a content ID')
    return {'id': info['Id'], 'digests': sorted(info.get('RepoDigests') or []),
            'architecture': info.get('Architecture'), 'os': info.get('Os')}


def run_identity(task, client, config, *, docker_timeout_sec: int) -> dict:
    package = Path(__file__).resolve().parent
    repo = package.parents[3]
    paths = [*package.glob('*.py'), *(repo / 'teamgym/prompts/software/swe_atlas').glob('*.txt'),
             repo / 'teamgym/llm/construction_client.py', repo / 'teamgym/llm/openai_client.py',
             *(repo / 'teamgym/interaction').glob('*.py'), *(repo / 'teamgym/core').glob('*.py'),
             repo / 'teamgym/tokenization.py', repo / 'pyproject.toml', repo / 'uv.lock']
    code = {str(p.relative_to(repo)): digest(p) for p in sorted(paths)}
    cfg = {k: getattr(client.cfg, k, None) for k in ('model', 'provider', 'base_url',
        'endpoint', 'reasoning_effort', 'temperature', 'max_output_tokens', 'api_key_env')}
    return {'schema': 'swe_atlas_construction_identity_v1', 'source': task_identity(task),
            'image': image_identity(task.runtime.docker_image), 'model': cfg,
            'config': config.__dict__, 'docker_timeout_sec': docker_timeout_sec, 'code': code,
            'runtime': runtime_identity()}


def prepare_run(output: Path, identity: dict) -> bool:
    """Reject legacy or changed inputs rather than adopting old accepted flags."""
    path = output / 'input_identity.json'
    if path.is_file():
        try:
            old = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ConstructionIdentityError('Corrupt identity; use a fresh output directory') from exc
        if old != identity:
            raise ConstructionIdentityError('Construction inputs changed; use a fresh output directory')
        validate_checkpoint(output, identity)
        return True
    if output.exists() and any(output.iterdir()):
        raise ConstructionIdentityError('Unbound legacy checkpoint; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(path, identity)
    return False


def _checkpoint_files(output):
    names = {'riu_candidates.json', 'evidence_artifacts.json', 'final_rius.hidden.json',
             'audit_summary.json', 'selected_artifacts.hidden.json', 'display_logs.hidden.json',
             'pipeline_report.json'}
    return {str(p.relative_to(output)): digest(p) for p in output.rglob('*.json')
            if p.name in names and 'llm_traces' not in p.parts}


def write_checkpoint(output: Path, identity: dict) -> None:
    atomic_json(output / 'checkpoint_receipt.json', {
        'identity_sha256': canonical_hash(identity), 'outputs': _checkpoint_files(output)})


def validate_checkpoint(output: Path, identity: dict) -> None:
    try:
        receipt = json.loads((output / 'checkpoint_receipt.json').read_text())
        if receipt['identity_sha256'] != canonical_hash(identity) or receipt['outputs'] != _checkpoint_files(output):
            raise ValueError('checkpoint contents changed')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ConstructionIdentityError('Incomplete or changed checkpoint; use a fresh output directory') from exc


def write_ready_receipt(output: Path, task) -> None:
    files = [output / 'pipeline_report.json', output / '05_final_rius/final_rius.hidden.json',
             output / '05_final_rius/audit_summary.json', output / '05_final_rius/selected_artifacts.hidden.json',
             output / '07_partition/final_partition.hidden.json']
    identity = json.loads((output / 'input_identity.json').read_text())
    atomic_json(output / 'ready_receipt.json', {'source': task_identity(task),
        'input_identity_sha256': canonical_hash(identity),
        'outputs': {str(p.relative_to(output)): digest(p) for p in files}})


def validate_ready_receipt(output: Path, task) -> None:
    try:
        receipt = json.loads((output / 'ready_receipt.json').read_text())
        identity = json.loads((output / 'input_identity.json').read_text())
        if receipt['source'] != task_identity(task) or identity['source'] != receipt['source']:
            raise ValueError('task changed')
        if receipt['input_identity_sha256'] != canonical_hash(identity):
            raise ValueError('input identity changed')
        required = {'pipeline_report.json', '05_final_rius/final_rius.hidden.json',
            '05_final_rius/audit_summary.json', '05_final_rius/selected_artifacts.hidden.json',
            '07_partition/final_partition.hidden.json'}
        if set(receipt['outputs']) != required or any(digest(output / name) != value
                for name, value in receipt['outputs'].items()):
            raise ValueError('frozen outputs changed')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ConstructionIdentityError('Missing/stale ready receipt; rerun construction in a fresh directory') from exc


def runtime_identity() -> dict:
    from teamgym.tokenization import encoding
    packages = {}
    for name in ('openai', 'tiktoken', 'httpx', 'httpcore', 'pydantic', 'pydantic_core',
                 'anyio', 'sniffio', 'certifi', 'distro', 'jiter', 'regex', 'typing_extensions', 'tomli'):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    enc = encoding()
    d = hashlib.sha256()
    for token, rank in sorted(enc._mergeable_ranks.items()):
        d.update(len(token).to_bytes(4, 'big')); d.update(token); d.update(rank.to_bytes(4, 'big'))
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
        'machine': platform.machine(), 'packages': packages, 'tokenizer': {
            'name': enc.name, 'vocabulary_sha256': d.hexdigest(),
            'special_tokens_sha256': canonical_hash(enc._special_tokens),
            'pattern_sha256': hashlib.sha256(enc._pat_str.encode()).hexdigest()}}
