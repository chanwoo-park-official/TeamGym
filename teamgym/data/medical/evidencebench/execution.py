"""Frozen medical construction dependencies, including actual shared consumers."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from teamgym.tokenization import TOKENIZER, encoding


def dependency_identity():
    package = Path(__file__).resolve().parent
    repository = package.parents[3]
    domain = package.name
    paths = [*package.glob('*.py'),
             *(repository / 'teamgym/prompts/medical' / domain).glob('*.txt'),
             *(repository / 'teamgym/llm').glob('*.py'),
             *(repository / 'teamgym/core').glob('*.py'),
             *(repository / 'teamgym/interaction').glob('*.py'),
             *(repository / 'teamgym/scoring').glob('*.py'),
             repository / 'teamgym/tokenization.py',
             repository / 'teamgym/data/medical/export_schema.py',
             repository / 'teamgym/data/medical/certification.py',
             *(repository / 'teamgym/data/medical').glob('allocation*.py'),
             *(repository / 'teamgym/prompts/medical/allocations').glob('*.txt'),
             repository / 'teamgym/data/cli.py', repository / 'pyproject.toml', repository / 'uv.lock']
    files = {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(set(paths))}
    packages = {}
    for name in ('openai', 'tiktoken', 'httpx', 'pydantic', 'pyarrow', 'pandas'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    vocab = hashlib.sha256()
    for token, rank in sorted(encoding()._mergeable_ranks.items(), key=lambda item: item[1]):
        vocab.update(len(token).to_bytes(4, 'big') + token + rank.to_bytes(8, 'big'))
    identity = {'schema': 'medical_execution_dependencies_v2', 'files': files,
                'packages': packages, 'python': sys.version,
                'tokenizer': {'name': TOKENIZER, 'vocabulary_sha256': vocab.hexdigest()}}
    identity['sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return identity


def assert_frozen_dependencies(identity):
    if dependency_identity() != identity:
        raise RuntimeError('Medical construction shared/domain dependencies changed during execution')


def export_artifacts(root):
    return {name: Path(root) / name for name in (
        'run_manifest.json', '00_source.json', '01_candidates.json', '02_task_rubric.json',
        '03_task_audit.json', '04_boundaries.json', '05_final_rius.hidden.json',
        '06_participant_packets.hidden.json', '07_leakage_audit.json', '08_benchmark_modes.json',
        '09_participant_final_answer_audit.json')}
