"""Recertify changed whole-context packets from an immutable, live-audited PIU set.

This is a post-PIU stage. It never regenerates or edits the source question,
canonical evidence, or ALL/NONE/LOO outcomes. Packet changes receive fresh real
mechanical, complete-packet leakage, and salience checks before export readiness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from teamgym.llm.construction_client import OpenAIJsonClient, OpenAIJsonClientConfig
from .integrity import (atomic_json, canonical_hash, digest, run_identity,
                        validate_ready_receipt, write_ready_receipt, write_checkpoint)
from .pipeline import (PipelineConfig, PipelineError, fit_actual_reader_budget,
    materialize_modes, run_leakage_repair, run_salience_repair, validate_partition_mechanically)
from .source import load_qa_task


def release(*, parent: Path, producer_snapshot: Path, task, output: Path, client,
            config: PipelineConfig) -> dict:
    parent, output = parent.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise PipelineError('Post-PIU release requires a fresh empty output directory')
    validate_ready_receipt(parent, task)
    old_identity = json.loads((parent / 'input_identity.json').read_text())
    for relative, expected in old_identity['code'].items():
        if digest(producer_snapshot / relative) != expected:
            raise PipelineError(f'Original producer snapshot is missing or changed: {relative}')
    cfg = config.checked()
    output.mkdir(parents=True, exist_ok=True)
    identity = run_identity(task, client, cfg, docker_timeout_sec=old_identity['docker_timeout_sec'])
    atomic_json(output / 'input_identity.json', identity)
    immutable = ['05_final_rius/final_rius.hidden.json', '05_final_rius/audit_summary.json',
                 '05_final_rius/selected_artifacts.hidden.json']
    for name in immutable:
        target = output / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, target)
    immutable_hashes = {name: digest(output / name) for name in immutable}
    rius = json.loads((output / immutable[0]).read_text())['rius']
    parent_report = json.loads((parent / 'pipeline_report.json').read_text())
    if not parent_report.get('all_none_loo_accept') or not json.loads((output / immutable[1]).read_text()).get('accept'):
        raise PipelineError('Frozen PIU set lacks a valid ALL/NONE/LOO acceptance')
    partition = json.loads((parent / '07_partition/final_partition.hidden.json').read_text())
    partition, capacity = fit_actual_reader_budget(partition, cfg)
    atomic_json(output / 'reader_capacity_repair.json', capacity)
    mechanical = validate_partition_mechanically(partition, cfg)
    if not mechanical['valid']:
        raise PipelineError('Capacity release failed mechanical validation')
    partition, leakage = run_leakage_repair(client=client, task=task, rius=rius, partition=partition,
        output_dir=output / '08_leakage', config=cfg)
    partition, salience = run_salience_repair(client=client, task=task, partition=partition,
        output_dir=output / '08_leakage/salience', config=cfg)
    final_partition, final_capacity = fit_actual_reader_budget(partition, cfg)
    if final_capacity['removed_neutral_contexts']:
        raise PipelineError('Audited packet still requires a capacity change; do not certify without re-audit')
    partition = final_partition
    mechanical = validate_partition_mechanically(partition, cfg)
    if not mechanical['valid']:
        raise PipelineError('Final release failed mechanical validation')
    for name, expected in immutable_hashes.items():
        if digest(output / name) != expected:
            raise PipelineError('Frozen canonical PIU/audit inputs changed during post-PIU release')
    if run_identity(task, client, cfg, docker_timeout_sec=old_identity['docker_timeout_sec']) != identity:
        raise PipelineError('Release inputs changed during packet recertification')
    atomic_json(output / '07_partition/final_partition.hidden.json', partition)
    paths = materialize_modes(task=task, rius=rius, partition=partition,
        output_dir=output / '09_benchmark', config=cfg)
    report = {**parent_report, 'schema': 'swe_atlas_post_piu_release_v1',
        'status': 'benchmark_ready', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_source': 'immutable_live_audited_PIUs_from_parent_run',
        'parent_build': str(parent), 'post_piu_release': True,
        'assigned_context_count': len(partition['contexts']),
        'assigned_code_tokens': sum(row['estimated_tokens'] for row in partition['contexts'] if row['context_type'] == 'code'),
        'parent_packet_leakage_removed_context_count': parent_report['leakage_removed_context_count'],
        'reader_capacity_removed_context_count': len(capacity['removed_neutral_contexts']),
        'actual_reader_capacity_verified': True,
        'leakage_repair_rounds': leakage['rounds_run'],
        'leakage_removed_context_count': len(leakage['removed_context_ids']),
        'semantic_leakage_accept': leakage['accept'], 'salience_accept': salience['accept'],
        'mechanical_partition_valid': mechanical['valid'], 'benchmark_paths': paths}
    atomic_json(output / 'pipeline_report.json', report)
    atomic_json(output / 'post_piu_release_receipt.json', {
        'stage': 'post_PIU_packet_release', 'parent_ready_receipt_sha256': digest(parent / 'ready_receipt.json'),
        'parent_input_identity_sha256': digest(parent / 'input_identity.json'),
        'immutable_canonical_and_ALL_NONE_LOO_hashes': immutable_hashes,
        'source_question_and_PIUs_unchanged': True, 'new_real_packet_audits': True,
        'reader_capacity_repair': capacity, 'reader_final_validation': final_capacity,
        'final_partition_sha256': digest(output / '07_partition/final_partition.hidden.json')})
    write_ready_receipt(output, task); write_checkpoint(output, identity)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent-build', type=Path, required=True)
    ap.add_argument('--producer-snapshot', type=Path, required=True)
    ap.add_argument('--task-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--env-file', required=True)
    ap.add_argument('--model', default='gpt-5.5')
    ap.add_argument('--reasoning-effort', default='medium')
    args = ap.parse_args()
    old = json.loads((args.parent_build / 'input_identity.json').read_text())
    cfg = PipelineConfig(**old['config'])
    client = OpenAIJsonClient(OpenAIJsonClientConfig(provider='bedrock_openai', model=args.model,
        base_url='https://bedrock-mantle.us-east-2.api.aws/openai/v1', env_file=args.env_file,
        reasoning_effort=args.reasoning_effort, max_output_tokens=12000, timeout_sec=600,
        request_retry_attempts=3, request_log_path=str(args.output_dir / 'llm_requests.jsonl')))
    print(json.dumps(release(parent=args.parent_build, producer_snapshot=args.producer_snapshot,
        task=load_qa_task(args.task_dir), output=args.output_dir, client=client, config=cfg), indent=2))


if __name__ == '__main__': main()
