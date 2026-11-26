"""Assemble the files required for a verified FinLongDocQA evaluator export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.data.finance.finlongdocqa.evaluation import _producer_evidence, capture_model_observations  # noqa: E402
from teamgym.data.finance.finlongdocqa.packet_budget import PAPER_CONTEXT_BUDGET  # noqa: E402
from teamgym.data.finance.finlongdocqa.provenance import REPO_ROOT, file_digest  # noqa: E402
from teamgym.data.finance.finlongdocqa.runtime import write_json  # noqa: E402


def assemble_producer_bundle(
    *,
    run_dir: str | Path,
    qa_path: str | Path,
    reports_zip: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    generated, qa, reports, output = (
        Path(path).expanduser().resolve()
        for path in (run_dir, qa_path, reports_zip, output_dir)
    )
    if output.exists():
        raise FileExistsError(f"Producer bundle output already exists: {output}")
    if output.is_relative_to(generated):
        raise ValueError("Producer bundle output must be outside the generation directory")
    manifest_path = generated / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["selected_report_ids"]) != 1:
        raise ValueError("Use a completed producer run for one report selected with --report-ids")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".finlongdocqa-", dir=output.parent) as scratch:
        bundle = Path(scratch) / "bundle"
        sources = bundle / "source"
        sources.mkdir(parents=True)
        shutil.copy2(qa, sources / "dataset_qa.jsonl")
        shutil.copy2(reports, sources / "reports.zip")
        shutil.copytree(generated, bundle / "generated")
        for name, expected_hash in manifest["code_and_prompts"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid producer dependency path: {name}")
            dependency = REPO_ROOT / relative
            if file_digest(dependency) != expected_hash:
                raise ValueError(f"Use the producer code revision. Dependency changed: {name}")
            stored = bundle / "code_snapshot" / relative
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dependency, stored)

        raw, _, evidence = _producer_evidence(bundle)
        write_json(bundle / "consumer_inputs.json", {
            "capture": "offline_replay_of_verified_accepted_producer",
            "producer_fingerprint": evidence["producer_fingerprint"],
            "accepted_commit_payload_sha256": evidence["accepted_commit_payload_sha256"],
            "materialized_context_token_budget": PAPER_CONTEXT_BUDGET,
            "inputs": capture_model_observations(raw, budget=PAPER_CONTEXT_BUDGET),
        })
        if output.exists():
            raise FileExistsError(f"Producer bundle output already exists: {output}")
        bundle.rename(output)
    return {
        "producer_run": str(output),
        "report_id": raw["source"]["report_id"],
        "producer_fingerprint": evidence["producer_fingerprint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qa-path", type=Path, required=True)
    parser.add_argument("--reports-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assemble_producer_bundle(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
