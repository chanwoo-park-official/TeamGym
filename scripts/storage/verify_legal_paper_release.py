#!/usr/bin/env python3
"""Verify the outer legal-paper release and every nested archive digest."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from pathlib import Path
import zipfile

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()


DEFAULT_RELEASE = ARCHIVE_ROOT / 'legal_paper_release_v1/legal_paper_release_v1.zip'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", nargs="?", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()

    release = args.release.expanduser().resolve()
    with zipfile.ZipFile(release) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Outer ZIP CRC failure: {bad}")
        summary = json.loads(archive.read("release_summary.json"))
        expected_members = summary["members"]
        observed = {}
        for member, metadata in expected_members.items():
            digest = hashlib.sha256()
            byte_count = 0
            with archive.open(member) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    byte_count += len(block)
            observed[member] = {"bytes": byte_count, "sha256": digest.hexdigest()}
            if observed[member] != metadata:
                raise RuntimeError(
                    f"Nested member mismatch for {member}: {observed[member]} != {metadata}"
                )

    result = {
        "release": str(release),
        "release_bytes": release.stat().st_size,
        "release_sha256": sha256(release),
        "nested_member_count": len(observed),
        "nested_members": observed,
        "verified": True,
    }
    print(json.dumps(result, indent=2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
