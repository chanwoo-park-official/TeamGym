"""Code reads, atomic edits, and verifier execution for patch diagnostics."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import os
import tempfile
import uuid

from ..runtime_limits import verifier_slots


BASE_READ = r'''
import json, pathlib, subprocess, sys
r=json.load(sys.stdin)
assert subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()==r["commit"]
assert not subprocess.check_output(["git","status","--porcelain"],text=True)
p=pathlib.Path('/app') / r['path']
parts=p.relative_to('/app').parts
cur=pathlib.Path('/app')
for part in parts:
    cur=cur/part
    if cur.is_symlink():
        print(json.dumps({'kind':'unsupported'})); sys.exit()
if p.exists() and not p.is_file():
    print(json.dumps({'kind':'unsupported'})); sys.exit()
tracked=subprocess.run(['git','ls-files','--error-unmatch','--',r['path']],capture_output=True).returncode==0
if p.exists() and not tracked:
    print(json.dumps({'kind':'unsupported'})); sys.exit()
if not tracked:
    print(json.dumps({'kind':'absent'})); sys.exit()
raw=subprocess.check_output(['git','show','HEAD:'+r['path']])
try:
    text=raw.decode('utf-8')
except UnicodeDecodeError:
    print(json.dumps({'kind':'unsupported'})); sys.exit()
print(json.dumps({'kind':'file','text':text}))
'''

PUBLIC_CHECK = r'''
import json, subprocess, sys
r=json.load(sys.stdin)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==r['commit']
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
if open('/candidate.patch','rb').read():
    p=subprocess.run(['git','apply','--binary','/candidate.patch'],capture_output=True)
    if p.returncode:
        print(json.dumps({'harness_error':'patch_did_not_apply'})); sys.exit(2)
try:
    p=subprocess.run(r['argv'],capture_output=True,text=True,errors='replace',timeout=r['timeout'])
    print(json.dumps({'status':'pass' if p.returncode==0 else 'fail',
                      'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
except subprocess.TimeoutExpired:
    print(json.dumps({'status':'timeout'}))
'''


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def is_test_path(path: str) -> bool:
    """Protect tests and test-runner configuration from submitted edits."""
    pure = PurePosixPath(path)
    dirs = {part.lower() for part in pure.parts[:-1]}
    name = pure.name.lower()
    return bool(dirs & {'test', 'tests', '__tests__', 'spec', 'specs', '.github'}) or (
        name in {'conftest.py', 'pytest.ini', '.coveragerc', 'tox.ini',
                 'jest.config.js', 'jest.config.ts', 'jest.config.cjs',
                 'vitest.config.ts', 'vitest.config.js', 'playwright.config.ts'}
        or name.endswith('_test.go') or name.startswith('test_')
        or name.endswith('_test.py') or '.test.' in name or '.spec.' in name)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_path(path) -> bool:
    return (isinstance(path, str) and bool(path) and '\\' not in path
            and not any(ord(c) < 32 or c in '\"' for c in path)
            and not path.startswith('/') and all(p not in {'', '.', '..', '.git'} for p in path.split('/'))
            and str(PurePosixPath(path)) == path)


def make_patch(base: dict, current: dict) -> str:
    result = []
    for path in sorted(current):
        before, after = base[path], current[path]
        if before == after:
            continue
        if before is None and after == '':
            result.extend([f'diff --git a/{path} b/{path}\n', 'new file mode 100644\n', 'index 0000000..e69de29\n'])
            continue
        if before == '' and after is None:
            result.extend([f'diff --git a/{path} b/{path}\n', 'deleted file mode 100644\n', 'index e69de29..0000000\n'])
            continue
        lines = difflib.unified_diff((before or '').splitlines(keepends=True),
                                    (after or '').splitlines(keepends=True),
                                    fromfile='/dev/null' if before is None else 'a/' + path,
                                    tofile='/dev/null' if after is None else 'b/' + path)
        for line in lines:
            result.append(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n')
    return ''.join(result)


def _adjacent_runs(atoms: list, gap_text) -> dict[str, list[str]]:
    """Concatenations of atoms that are contiguous, or blank-separated, in their file.

    Only atoms carrying line numbers participate; whole-file entries added for
    full-context have nothing to join. ``gap_text(path, start, end)`` returns
    the base text strictly between two atoms (1-indexed, inclusive) or None if
    it is not all whitespace; a real gap's exact text is spliced in so the
    joined chunk still reproduces the base verbatim.
    """
    by_path: dict[str, list[dict]] = {}
    for atom in atoms:
        if atom.get('line_start') and atom.get('line_end'):
            by_path.setdefault(atom['path'], []).append(atom)
    runs: dict[str, list[str]] = {}
    for path, rows in by_path.items():
        rows.sort(key=lambda a: a['line_start'])
        run_atoms = [rows[0]]
        run_text = [rows[0]['text']]
        for previous, current in zip(rows, rows[1:]):
            if current['line_start'] == previous['line_end'] + 1:
                run_atoms.append(current)
                run_text.append(current['text'])
                continue
            if current['line_start'] > previous['line_end'] + 1:
                between = gap_text(path, previous['line_end'] + 1, current['line_start'] - 1)
                if between is not None and not between.strip():
                    run_atoms.append(current)
                    run_text.append(between)
                    run_text.append(current['text'])
                    continue
            if len(run_atoms) > 1:
                runs.setdefault(path, []).append(''.join(run_text))
            run_atoms, run_text = [current], [current['text']]
        if len(run_atoms) > 1:
            runs.setdefault(path, []).append(''.join(run_text))
    return runs


class Editor:
    def __init__(self, atoms: list, read_base):
        self.read_base = read_base
        self.base, self.current, self.known = {}, {}, {}
        self.fully_visible_paths = set()
        self.last_reason = None
        for atom in atoms:
            self.known.setdefault(atom['path'], []).append(atom['text'])
        # Atoms that abut in the file, or are separated only by a blank line,
        # are one continuous region of supplied source, so a quote spanning two
        # of them is faithful even though neither chunk contains it whole.
        # Splitting supplied source into atoms must not hide an otherwise
        # visible replacement anchor spanning their original blank boundary.
        def gap_text(path, start, end):
            raw = self.read_base(path)
            if raw.get('kind') != 'file':
                return None
            lines = raw['text'].splitlines(keepends=True)
            if not 1 <= start <= end <= len(lines):
                return None
            return ''.join(lines[start - 1:end])

        for path, spans in _adjacent_runs(atoms, gap_text).items():
            self.known.setdefault(path, []).extend(spans)
        for path, chunks in self.known.items():
            raw = self.read_base(path)
            if raw.get('kind') == 'file' and any(chunk == raw['text'] for chunk in chunks):
                self.fully_visible_paths.add(path)

    def edit(self, operations) -> dict:
        self.last_reason = None
        if not isinstance(operations, list) or not operations:
            self.last_reason = 'empty_operations'
            return {'status': 'rejected'}
        current, known = copy.deepcopy(self.current), copy.deepcopy(self.known)
        try:
            for op in operations:
                path = op.get('path')
                if not valid_path(path):
                    self.last_reason = 'invalid_path'
                    raise ValueError()
                if is_test_path(path):
                    self.last_reason = 'test_file_edits_forbidden'
                    raise ValueError()
                kind = op.get('op')
                if kind not in {'replace', 'write', 'delete'}:
                    self.last_reason = 'invalid_operation'
                    raise ValueError()
                if kind == 'replace':
                    old, new = op.get('old'), op.get('new')
                    if not isinstance(old, str) or not old or not isinstance(new, str):
                        self.last_reason = 'source_not_visible_or_quote_mismatch'
                        raise ValueError()
                    if not any(old in chunk for chunk in known.get(path, [])):
                        # A model copying a source literal containing a bare `"`
                        # (e.g. inside a Go backtick string) routinely escapes it
                        # to `\"` as if re-typing a JSON string, which is not what
                        # is actually in the file. Retry with that one class of
                        # mistake undone before rejecting the whole transaction.
                        unescaped = old.replace('\\"', '"')
                        if unescaped != old and any(unescaped in chunk for chunk in known.get(path, [])):
                            old = unescaped
                        else:
                            self.last_reason = 'source_not_visible_or_quote_mismatch'
                            raise ValueError()
                if kind == 'write' and not isinstance(op.get('text'), str):
                    self.last_reason = 'invalid_write_text'
                    raise ValueError()
                if path not in self.base:
                    raw = self.read_base(path)
                    if raw['kind'] == 'unsupported':
                        raise ValueError()
                    self.base[path] = raw.get('text')
                text = current.get(path, self.base[path])
                if kind == 'write' and self.base[path] is not None \
                        and path not in self.fully_visible_paths:
                    self.last_reason = 'cannot_overwrite_nonvisible_existing_file'
                    raise ValueError()
                if kind == 'replace':
                    if text is None or text.count(old) != 1:
                        self.last_reason = 'quote_not_unique_in_current_file'
                        raise ValueError()
                    current[path] = text.replace(old, new, 1)
                    known[path] = [chunk.replace(old, new, 1) if old in chunk else chunk for chunk in known[path]]
                    known[path].append(new)
                elif kind == 'write':
                    current[path] = op['text']
                    known[path] = [op['text']]
                else:
                    if text is None:
                        self.last_reason = 'delete_missing_file'
                        raise ValueError()
                    if self.base[path] is not None and path not in self.fully_visible_paths:
                        self.last_reason = 'cannot_delete_nonvisible_existing_file'
                        raise ValueError()
                    current[path] = None
                    known[path] = []
        except (ValueError, AttributeError, TypeError):
            if self.last_reason is None:
                self.last_reason = 'transaction_error'
            return {'status': 'rejected'}
        self.current, self.known = current, known
        return {'status': 'applied'}

    def patch(self) -> str:
        return make_patch(self.base, self.current)


def docker_python(image, code, request, *, patch=None, timeout=120) -> dict:
    name = 'deepswe-check-' + uuid.uuid4().hex[:12]
    command = ['docker', 'run', '--rm', '--name', name, '-i', '--platform', 'linux/amd64',
               '--network', 'none', '--cpus', '2', '--memory', '8g', '--pids-limit', '256',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--workdir', '/app']
    if patch:
        command += ['--mount', f'type=bind,src={patch},dst=/candidate.patch,readonly']
    command += ['--entrypoint', 'python3', image, '-I', '-c', code]
    try:
        p = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, timeout=timeout)
        if p.returncode:
            raise RuntimeError('Constructor/check container failed: ' + p.stderr[-1000:] + p.stdout[-1000:])
        return json.loads(p.stdout)
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True)


@verifier_slots
def verify(task, image, out, patch):
    logs = out / 'verifier'
    logs.mkdir()
    name = 'deepswe-verifier-' + uuid.uuid4().hex[:12]
    cmd = ['docker', 'run', '--rm', '--name', name, '--platform', 'linux/amd64', '--network', 'none',
           '--cpus', '2', '--memory', '8g', '--pids-limit', '256', '--cap-drop', 'ALL',
           '--security-opt', 'no-new-privileges',
           '--mount', f'type=bind,src={patch},dst=/logs/artifacts/model.patch,readonly',
           '--mount', f'type=bind,src={logs},dst=/logs/verifier',
           '--entrypoint', '/tests/test.sh', image]
    start = time.monotonic()
    try:
        try:
            with (logs / 'test-stdout.txt').open('w') as stream:
                p = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=task.runtime.verifier_timeout_sec)
        except subprocess.TimeoutExpired:
            # A running verifier exceeding the fixed deadline is a submitted
            # patch failure. Failure to inspect/start Docker remains invalid.
            inspected = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
            state = json.loads(inspected.stdout)[0]['State'] if inspected.returncode == 0 else {}
            valid = bool(state.get('Running')) and not state.get('OOMKilled') and not state.get('Error')
            result = {'exit_code': 124, 'seconds': round(time.monotonic()-start, 3),
                      'reward': {'reward': 0}, 'verification_valid': valid,
                      'failure_kind': 'verifier_timeout' if valid else 'docker_error',
                      'verifier_timeout_sec': task.runtime.verifier_timeout_sec,
                      'docker_state': state}
            (logs / 'timeout.json').write_text(json.dumps(result, indent=2) + '\n')
            return result
        if p.returncode in (125, 126, 127):
            raise RuntimeError(f'Docker verifier failed to launch: exit {p.returncode}')
        if not (logs / 'reward.json').exists():
            return {'exit_code': p.returncode, 'seconds': round(time.monotonic()-start, 3),
                    'reward': {}, 'verification_valid': False, 'failure_kind': 'missing_verifier_reward'}
        reward = json.loads((logs / 'reward.json').read_text())
        return {'exit_code': p.returncode, 'seconds': round(time.monotonic()-start, 3), 'reward': reward}
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
