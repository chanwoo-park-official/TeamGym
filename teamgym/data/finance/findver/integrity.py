"""Fresh-run provenance and atomic artifact writes for FinDVer."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import platform
from importlib import metadata
from typing import Any, Iterable


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=target.parent,
                                         prefix=f'.{target.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    _atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def revision(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(Path(path).resolve()), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def client_identity(client: Any) -> dict:
    # Credentials and environment-file contents are never manifest inputs.
    fields = ('model', 'provider', 'base_url', 'endpoint', 'reasoning_effort',
              'temperature', 'max_output_tokens', 'timeout_sec', 'request_retry_attempts',
              'api_key_env', 'env_file', 'credentials_file')
    return {key: getattr(client.cfg, key, None) for key in fields}


def start_run(output: Path, *, qa_path: str | Path, reports_root: str | Path,
              config: dict, clients: dict) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError('FinDVer output directory must be empty; unverified resume/overwrite is forbidden')
    output.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    project = package.parents[3]
    code = code_files(project)
    prompts = project / 'teamgym/prompts/finance/findver'
    manifest = {'schema': 'findver_construction_manifest_v2', 'status': 'running',
                'cache_status': 'fresh', 'resume_policy': 'fresh_output_required',
                'source': {'qa_path': str(Path(qa_path).resolve()), 'qa_sha256': sha256(qa_path),
                           'reports_root': str(Path(reports_root).resolve()),
                           'revision': revision(Path(qa_path).parent)},
                'config': config, 'clients': clients, 'runtime': runtime_identity(),
                'code': {str(p.relative_to(project)): sha256(p) for p in code},
                'prompts': {str(p.relative_to(project)): sha256(p) for p in prompts.glob('*.txt')}}
    write_json(output / 'manifest.json', manifest)
    return manifest


def verify_inputs(manifest: dict) -> None:
    project = Path(__file__).resolve().parents[4]
    if set(manifest['code']) != {str(p.relative_to(project)) for p in code_files(project)}:
        raise RuntimeError('FinDVer code dependency set changed during execution')
    if runtime_identity() != manifest['runtime']:
        raise RuntimeError('FinDVer Python/package/tokenizer runtime changed during execution')
    for group in ('code', 'prompts'):
        for relative, expected in manifest[group].items():
            if sha256(project / relative) != expected:
                raise RuntimeError(f'FinDVer {group} input changed during execution: {relative}')
    if sha256(manifest['source']['qa_path']) != manifest['source']['qa_sha256']:
        raise RuntimeError('FinDVer raw QA input changed during execution')


def code_files(project: Path) -> list[Path]:
    """Conservatively bind the complete shared consumer implementation trees."""
    roots = ('teamgym/data/finance/findver', 'teamgym/core', 'teamgym/interaction',
             'teamgym/llm', 'teamgym/visualization', 'teamgym/scoring')
    paths = {p for root in roots for p in (project / root).rglob('*.py')}
    paths.update(project / relative for relative in ('teamgym/tokenization.py',
        'teamgym/data/cli.py', 'teamgym/data/long_context_reasoning/judge.py',
        'pyproject.toml', 'uv.lock'))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError('Required construction/consumer dependency missing')
    return sorted(paths)


def runtime_identity() -> dict:
    from teamgym.tokenization import encoding
    names = ('openai', 'tiktoken', 'httpx', 'httpcore', 'pydantic', 'pydantic_core',
             'anyio', 'sniffio', 'certifi', 'distro', 'jiter', 'regex', 'typing_extensions', 'tomli')
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    enc = encoding()
    vocab = hashlib.sha256()
    for token, rank in sorted(enc._mergeable_ranks.items()):
        vocab.update(len(token).to_bytes(4, 'big'))
        vocab.update(token)
        vocab.update(rank.to_bytes(4, 'big'))
    specials = json.dumps(enc._special_tokens, sort_keys=True, separators=(',', ':'))
    return {'python': sys.version, 'python_executable': str(Path(sys.executable).resolve()),
            'implementation': platform.python_implementation(), 'machine': platform.machine(),
            'packages': versions, 'tokenizer': {'name': enc.name,
                'mergeable_vocabulary_sha256': vocab.hexdigest(),
                'special_tokens_sha256': hashlib.sha256(specials.encode()).hexdigest(),
                'pattern_sha256': hashlib.sha256(enc._pat_str.encode()).hexdigest()}}
