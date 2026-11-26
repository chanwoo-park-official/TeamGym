#!/usr/bin/env python3
"""Run repeated patch diagnostics, optionally with evidence omitted."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil

from teamgym.data.software.deep_swe.candidate import leave_one_out_conditions, reduce_candidate
from teamgym.data.software.deep_swe.integrity import (
    atomic_write_json, code_identity, file_sha256, image_identity, load_json,
    task_identity, validate_manifest, write_manifest,
)


def actor_identity(job: dict) -> dict:
    from teamgym.data.software.deep_swe.model_actions import resolve_actor_config
    settings = resolve_actor_config(job.get('model_slug', 'gpt55'),
        **{key: job.get(key) for key in ('provider', 'base_url', 'api_key_env', 'reasoning_effort')})
    return {**settings,
            'env_file': str(Path(job.get('env_file') or Path.cwd() / '.env').expanduser().resolve())}


def episode_inputs(job: dict) -> dict:
    """Recompute identity before reuse; no identity can be supplied by a caller."""
    candidate = json.loads(Path(job['candidate']).read_text(encoding='utf-8'))
    checks = json.loads(Path(job['checks']).read_text(encoding='utf-8'))
    if not isinstance(candidate, dict) or not isinstance(checks, dict) or not checks:
        raise ValueError('candidate and nonempty checks must be JSON objects')
    source = task_identity(job['task_dir'])
    runtime = job.get('runtime_image') or source['runtime_image']
    return {'task': source,
            'candidate_sha256': file_sha256(job['candidate']),
            'checks_sha256': file_sha256(job['checks']),
            'candidate_runtime': image_identity(candidate['image_id']),
            'task_runtime': image_identity(runtime),
            'verifier': image_identity(job['verifier_image']),
            'actor': actor_identity(job), 'code': code_identity(),
            'protocol': {'patch_turns': job['patch_turns'], 'evidence_mode': 'all',
                         'total_output_tokens': job.get('total_output_tokens', 32000),
                         'max_output_tokens_per_call': job.get('max_output_tokens_per_call', 16000),
                         'max_checks_per_episode': job.get('max_checks_per_episode', 15),
                         'require_evidence_citations': job.get('require_evidence_citations', False),
                         'citation_policy': job.get('citation_policy', 'enforce')},
            'condition': {'label': job['label'], 'rep': job['rep']}}


def valid_episode_result(result: object) -> bool:
    if not (isinstance(result, dict) and result.get('episode_valid') is True
            and isinstance(result.get('success'), bool)
            and isinstance(result.get('reward'), dict)):
        return False
    if result.get('verification_valid') is False:
        return False
    if result['success']:
        reward = result['reward']
        return (result.get('exit_code') == 0 and reward.get('reward') == 1
                and all(isinstance(reward.get(f'{kind}_total'), int)
                        and reward[f'{kind}_total'] > 0
                        and reward.get(f'{kind}_passed') == reward[f'{kind}_total']
                        for kind in ('f2p', 'p2p')))
    return True


EPISODE_OUTPUTS = ('result.json', 'manifest.json', 'actor_packet.json',
                   'candidate.frozen.hidden.json', 'model.patch', 'episode.json',
                   'calls.json', 'trajectory.json', 'harness.frozen.py')


def run(job: dict) -> dict:
    out = Path(job['output_dir'])
    cmd = [sys.executable, '-m', f'{__package__}.patch_episode',
           '--task-dir', str(job['task_dir']), '--candidate', str(job['candidate']), '--checks', str(job['checks']),
           '--verifier-image', job['verifier_image'], '--output-dir', str(out),
           '--patch-turns', str(job['patch_turns']), '--evidence-mode', 'all',
           '--model-slug', job.get('model_slug', 'gpt55')]
    for key in ('runtime_image', 'env_file', 'provider', 'base_url', 'api_key_env', 'reasoning_effort',
                'total_output_tokens', 'max_output_tokens_per_call', 'max_checks_per_episode'):
        if job.get(key) is not None:
            cmd.extend(['--' + key.replace('_', '-'), str(job[key])])
    if job.get('require_evidence_citations'):
        cmd.extend(['--require-evidence-citations', '--citation-policy', job.get('citation_policy', 'enforce')])
    result = {}
    proc = None
    cache_status, cache_reason = 'fresh', 'no_prior_episode'
    for attempt in range(3):
        inputs = episode_inputs(job)
        result_path = out / 'result.json'
        receipt_path = out / 'cache_manifest.json'
        completed_outputs = [out / name for name in EPISODE_OUTPUTS]
        result = load_json(result_path)
        matches, reason = validate_manifest(receipt_path, inputs, outputs=completed_outputs)
        if matches and valid_episode_result(result):
            cache_status, cache_reason = 'cache_hit', 'manifest_and_valid_result_match'
            proc = subprocess.CompletedProcess(cmd, 0, '', '')
            atomic_write_json(out / 'cache.json', {'status': cache_status, 'reason': cache_reason})
            break
        if out.exists():
            cache_status = 'resume' if attempt == 0 else 'retry'
            cache_reason = reason if not matches else 'invalid_episode_result'
            shutil.move(str(out), str(out) + f'.invalid_{time.time_ns()}')
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result = load_json(result_path)
        # A subprocess success is insufficient: inputs must also remain frozen.
        unchanged = episode_inputs(job) == inputs
        if proc.returncode == 0 and valid_episode_result(result) and unchanged:
            write_manifest(receipt_path, inputs, outputs=completed_outputs)
            atomic_write_json(out / 'cache.json', {'status': cache_status, 'reason': cache_reason})
            break
        if not unchanged:
            raise RuntimeError('DeepSWE inputs changed during an episode; refusing its result')
        if attempt < 2:
            time.sleep(2 ** attempt)
    assert proc is not None
    valid = proc.returncode == 0 and valid_episode_result(result)
    result = result if isinstance(result, dict) else {}
    reward = result.get('reward') or {}
    row = {'label': job['label'], 'rep': job['rep'], 'exit_code': proc.returncode,
           'success': valid and result.get('success') is True,
           'episode_valid': valid,
           'f2p': [reward.get('f2p_passed'), reward.get('f2p_total')],
           'p2p': [reward.get('p2p_passed'), reward.get('p2p_total')],
           'output_dir': str(out), 'failure_kind': result.get('failure_kind'),
           'cache_status': cache_status, 'cache_reason': cache_reason}
    if not valid:
        row['error'] = (proc.stderr or proc.stdout or 'missing or invalid episode result')[-800:]
    print(json.dumps({'event': 'episode', **row}), flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task-dir', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--checks', type=Path, required=True)
    ap.add_argument('--verifier-image', required=True)
    ap.add_argument('--runtime-image')
    ap.add_argument('--env-file', default=str(Path.cwd() / '.env'))
    ap.add_argument('--provider')
    ap.add_argument('--base-url')
    ap.add_argument('--api-key-env')
    ap.add_argument('--reasoning-effort')
    ap.add_argument('--total-output-tokens', type=int, default=32000)
    ap.add_argument('--max-output-tokens-per-call', type=int, default=16000)
    ap.add_argument('--max-checks-per-episode', type=int, default=15)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--patch-turns', type=int, required=True,
                    help='Patch-turn budget applied to every condition.')
    ap.add_argument('--granularity', choices=('riu', 'unit'), required=True)
    ap.add_argument('--repetitions', type=int, default=3)
    ap.add_argument('--concurrency', type=int, default=2)
    ap.add_argument('--model-slug', default='gpt55')
    ap.add_argument('--max-failures', type=int,
                    help='Stop a targeted LOO condition after this many failed repetitions. '
                         'ALL always runs every repetition. Jobs launch in bounded batches so a later repetition '
                         'is not started before the stop decision can be made.')
    ap.add_argument('--require-evidence-citations', action='store_true',
                    help='Record construction-only evidence citations in every edit.')
    ap.add_argument('--citation-policy', choices=('enforce', 'audit'), default='enforce',
                    help='Whether invalid construction citations reject an edit or are only logged.')
    ap.add_argument('--only', nargs='*', help='Optional labels; baseline is included only if named.')
    args = ap.parse_args()
    if min(args.patch_turns, args.repetitions, args.concurrency, args.total_output_tokens,
           args.max_output_tokens_per_call, args.max_checks_per_episode) < 1:
        ap.error('numeric budgets must be positive')
    if args.max_failures is not None and args.max_failures < 1:
        ap.error('--max-failures must be positive')

    candidate = json.loads(args.candidate.read_text())
    conditions = leave_one_out_conditions(candidate, args.granularity)
    if args.only is not None:
        wanted = set(args.only)
        unknown = wanted - {c['label'] for c in conditions}
        if unknown:
            ap.error(f'Unknown condition labels: {sorted(unknown)}')
        conditions = [c for c in conditions if c['label'] in wanted]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    packets = root / 'packets'
    packets.mkdir(exist_ok=True)
    jobs = []
    for condition in conditions:
        reduced = reduce_candidate(candidate, atom_ids=condition['atom_ids'],
                                   observation_ids=condition['observation_ids'])
        packet = packets / f"{condition['label']}.json"
        atomic_write_json(packet, reduced)
        for rep in range(1, args.repetitions + 1):
            jobs.append({'label': condition['label'], 'rep': rep, 'task_dir': str(args.task_dir),
                         'candidate': str(packet), 'checks': str(args.checks),
                         'verifier_image': args.verifier_image, 'patch_turns': args.patch_turns,
                         'model_slug': args.model_slug,
                         **{key: getattr(args, key) for key in (
                             'runtime_image', 'env_file', 'provider', 'base_url', 'api_key_env',
                             'reasoning_effort', 'total_output_tokens',
                             'max_output_tokens_per_call', 'max_checks_per_episode')},
                         'require_evidence_citations': args.require_evidence_citations,
                         'citation_policy': args.citation_policy,
                         'output_dir': root / f"{condition['label']}_rep{rep}"})
    atomic_write_json(root / 'design.json', {
        'granularity': args.granularity, 'patch_turns': args.patch_turns,
        'repetitions': args.repetitions, 'conditions': [c['label'] for c in conditions],
        'actor': actor_identity(vars(args)),
        'model_slug': args.model_slug,
        'baseline_meaning': 'all PIUs present', 'hidden_feedback_reused': False,
        'max_failures': args.max_failures,
        'require_evidence_citations': args.require_evidence_citations,
        'citation_policy': args.citation_policy,
    })
    results = []
    early_stops = {}
    if args.max_failures is None:
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for row in pool.map(run, jobs):
                results.append(row)
                atomic_write_json(root / 'results.json', results)
    else:
        # Bounded batches allow explicit targeted-LOO stopping. ALL must always
        # complete every repetition, including a possible success on rep three.
        for condition in conditions:
            condition_jobs = [job for job in jobs if job['label'] == condition['label']]
            failures = 0
            completed = 0
            for offset in range(0, len(condition_jobs), args.concurrency):
                batch = condition_jobs[offset:offset + args.concurrency]
                with futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    batch_rows = list(pool.map(run, batch))
                results.extend(batch_rows)
                completed += len(batch_rows)
                failures += sum(row['episode_valid'] and not row['success'] for row in batch_rows)
                atomic_write_json(root / 'results.json', results)
                if any(not row['episode_valid'] for row in batch_rows):
                    raise RuntimeError('Infrastructure/protocol error: stop without counting it as a solver failure')
                if condition['label'] != 'baseline' and failures >= args.max_failures:
                    early_stops[condition['label']] = {
                        'reason': 'failure_threshold',
                        'failures': failures,
                        'completed_repetitions': completed,
                        'planned_repetitions': args.repetitions,
                    }
                    print(json.dumps({'event': 'early_stop',
                                      'label': condition['label'],
                                      **early_stops[condition['label']]}), flush=True)
                    break
    summary = {}
    for row in results:
        cell = summary.setdefault(row['label'], {'passes': 0, 'runs': 0, 'f2p': [], 'p2p': []})
        if not row['episode_valid']:
            cell['invalid_runs'] = cell.get('invalid_runs', 0) + 1
            continue
        cell['runs'] += 1
        cell['passes'] += int(row['success'])
        cell['f2p'].append(row['f2p'])
        cell['p2p'].append(row['p2p'])
    atomic_write_json(root / 'summary.json', summary)
    atomic_write_json(root / 'early_stops.json', early_stops)
    if any(row['exit_code'] != 0 or not row['episode_valid'] for row in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
