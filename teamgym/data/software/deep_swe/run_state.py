"""Conservative, content-bound checkpoints for the construction workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

from .stages.patch_runtime import save


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


class RunState:
    """A changed run identity archives the previous run, never relabels it.

    For a compatible resume, any missing/tampered stage output invalidates all
    later stage shortcuts. The per-episode runner independently verifies its
    own manifest before reusing expensive diagnostic results.
    """

    def __init__(self, output: Path, identity: dict, *, resume: bool):
        self.output = output
        self.identity = identity
        self.receipts: dict[str, str] = {}
        self.dirty = not resume
        manifest = output / 'run_identity.json'
        previous = None
        try:
            previous = json.loads(manifest.read_text())
        except (OSError, ValueError):
            pass
        compatible = (resume and previous == identity)
        archive = None
        if output.exists() and any(output.iterdir()) and not compatible:
            archive = output.with_name(output.name + '.archived_' + str(time.time_ns()))
            shutil.move(str(output), archive)
        output.mkdir(parents=True, exist_ok=True)
        if compatible:
            try:
                self.receipts = json.loads((output / 'checkpoints.json').read_text())
                if not isinstance(self.receipts, dict):
                    raise ValueError('invalid checkpoint map')
            except (OSError, ValueError):
                self.receipts = {}
                self.dirty = True
        save(manifest, identity)
        save(output / 'resume.json', {'mode': 'resume' if compatible else 'fresh',
             'archive': str(archive) if archive else None,
             'identity_sha256': object_hash(identity), 'started_at': time.time()})

    def reusable(self, path: Path, *, requested: bool) -> bool:
        key = str(path.relative_to(self.output))
        try:
            valid = self.receipts.get(key) == sha256(path)
            if path.suffix == '.json':
                json.loads(path.read_text())
        except (OSError, ValueError):
            valid = False
        if not requested or self.dirty or not valid:
            self.dirty = True
            return False
        return True

    def checkpoint(self) -> None:
        # Only final stage products are candidates for shortcuts. Capturing a
        # receipt does not bypass diagnostic runner's stronger episode checks.
        self.receipts = {str(p.relative_to(self.output)): sha256(p)
                         for p in self.output.rglob('*.json')
                         if p.name not in {'checkpoints.json', 'resume.json', 'progress.json'}}
        save(self.output / 'checkpoints.json', self.receipts)
