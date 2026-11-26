"""Download public DeepSWE sources and build one task's official verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from .integrity import atomic_write_json, image_identity, task_identity
from .source import load_task


SOURCE_URL = 'https://github.com/datacurve-ai/deep-swe.git'


def download_source(destination: Path, revision: str | None = None) -> dict:
    destination = destination.expanduser().absolute()
    if revision is not None and not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('--revision must be a complete Git commit hash')
    if destination.is_symlink():
        raise ValueError('source directory must be independent, not a symlink')
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', SOURCE_URL, str(destination)], check=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(destination), *args], text=True).strip()
    if git('remote', 'get-url', 'origin').removesuffix('.git') != SOURCE_URL.removesuffix('.git'):
        raise ValueError('source directory has an unexpected upstream remote')
    if git('status', '--porcelain'):
        raise ValueError('source directory contains local modifications')
    if revision and git('rev-parse', 'HEAD').lower() != revision.lower():
        subprocess.run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', revision], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', revision], check=True)
    return {'url': SOURCE_URL, 'revision': git('rev-parse', 'HEAD'), 'directory': str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--revision', help='Pin an exact upstream commit; otherwise record the downloaded HEAD')
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--platform', default='linux/amd64')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', args.task_id):
        parser.error('--task-id must be a single directory name')
    source = download_source(args.source_dir, args.revision)
    task_dir = Path(source['directory']) / 'tasks' / args.task_id
    task = load_task(task_dir)
    identity = task_identity(task_dir)
    verifier = f'teamgym-deepswe-verifier:{args.task_id}-{identity["tree"]["sha256"][:12]}'
    subprocess.run(['docker', 'pull', '--platform', args.platform, task.runtime.docker_image], check=True)
    subprocess.run(['docker', 'build', '--platform', args.platform, '-t', verifier,
                    '-f', str(task.artifacts.tests_dockerfile), str(task_dir / 'tests')], check=True)
    receipt = {'schema': 'deepswe_source_preparation_v1', 'source': source, 'task': identity,
               'platform': args.platform, 'runtime_image': image_identity(task.runtime.docker_image),
               'verifier_image': image_identity(verifier), 'verifier_tag': verifier,
               'verifier_recipe': 'upstream tests/Dockerfile without modifications'}
    atomic_write_json(args.receipt, receipt)
    print(json.dumps({'receipt': str(args.receipt), 'source_revision': source['revision'],
                      'tasks_root': str(task_dir.parent), 'verifier_image': verifier}), flush=True)


if __name__ == '__main__':
    main()
