#!/usr/bin/env python3
"""Verify privacy, paths, and byte sizes for the two Hugging Face uploads."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from huggingface_hub import HfApi

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()


UPLOAD_ROOT = ARCHIVE_ROOT / 'huggingface_upload_v1'
REPOSITORIES = {
    "Chanwoo97/collabgym-legal": "collabgym-legal",
    "Chanwoo97/collabgym-legal-results": "collabgym-legal-results",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-root", type=Path, default=UPLOAD_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    api = HfApi()
    report = {"schema": "legal_huggingface_verification_v1", "repositories": {}}
    for repo_id, local_name in REPOSITORIES.items():
        root = args.upload_root / local_name
        expected = {
            path.relative_to(root).as_posix(): path.stat().st_size
            for path in root.rglob("*")
            if path.is_file()
        }
        info = api.dataset_info(repo_id, files_metadata=True)
        observed = {
            sibling.rfilename: sibling.size
            for sibling in info.siblings
            if sibling.size is not None
        }
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected) - {".gitattributes"})
        mismatched = {
            path: {"expected": expected[path], "observed": observed.get(path)}
            for path in expected
            if observed.get(path) != expected[path]
        }
        entry = {
            "private": info.private,
            "expected_file_count": len(expected),
            "remote_file_count": len(observed),
            "missing": missing,
            "extra": extra,
            "size_mismatches": mismatched,
            "verified": bool(info.private and not missing and not extra and not mismatched),
        }
        report["repositories"][repo_id] = entry
        if not entry["verified"]:
            raise RuntimeError(f"Remote verification failed for {repo_id}: {entry}")

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
