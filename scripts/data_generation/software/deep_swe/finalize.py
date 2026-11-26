"""Finish a fully qualified DeepSWE producer run using its exact frozen code.

This is an explicit phase handoff, not cache migration. An interrupted optional
improvement attempt is retained. All required screening and qualification episodes
must already be complete and content-bound. Packet review, targeted LOO and
participant diagnostics are new runs under the unchanged producer epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def validate_episode_files(episode, code):
    """Validate immutable outputs before importing or executing frozen code."""
    receipt = read(episode / 'cache_manifest.json')
    encoded = json.dumps(receipt['inputs'], ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    if receipt.get('schema_version') != 1 or hashlib.sha256(encoded).hexdigest() != receipt['inputs_sha256']:
        raise ValueError('invalid producer episode receipt')
    if receipt['inputs']['code'] != code:
        raise ValueError('producer episode used another code epoch')
    required = {'result.json', 'manifest.json', 'actor_packet.json',
                'candidate.frozen.hidden.json', 'model.patch', 'episode.json',
                'calls.json', 'trajectory.json', 'harness.frozen.py'}
    outputs = receipt.get('outputs', {})
    if {Path(p).name for p in outputs} != required or len(outputs) != len(required):
        raise ValueError('incomplete producer episode outputs')
    if any(Path(p).resolve().parent != episode.resolve() or digest(p) != h
           for p, h in outputs.items()):
        raise ValueError('changed producer episode output')
    return receipt


def validate_producer(producer, snapshot, round_number):
    from teamgym.data.software.deep_swe.qualification import QualificationPolicy
    identity = read(producer / 'run_identity.json')
    code = identity['code']
    for label, expected in code['files'].items():
        if digest(snapshot / label) != expected:
            raise ValueError(f'changed frozen producer code: {label}')
    if digest(snapshot / 'uv.lock') != identity['environment']['uv_lock_sha256']:
        raise ValueError('changed frozen lock file')
    expected_map = read(producer / 'checkpoints.json')
    candidate = producer / f'candidate_functional_rius_round_{round_number:02d}.json'
    qualification = producer / f'stage4c_qualification_round_{round_number:02d}'
    required = [candidate, producer / 'candidate_instruction_only.json',
                producer / 'public_checks.json', producer / 'public_checks.base_validation.json',
                producer / 'discovered_code.json', producer / 'observations.validated.json']
    for path in required:
        if expected_map.get(str(path.relative_to(producer))) != digest(path):
            raise ValueError(f'producer input lacks matching checkpoint: {path.name}')
    checks, validation = read(required[2]), read(required[3])
    if set(checks) != set(validation) or not checks:
        raise ValueError('public check validation incomplete')
    for result in validation.values():
        before = result.get('checkout_before', {})
        if (result.get('exit_code') != 0 or result.get('timed_out') is not False
                or not before.get('clean') or before != result.get('checkout_after')
                or before.get('head') != identity['source']['base_commit']
                or result.get('provenance_audit', {}).get('approved') is not True):
            raise ValueError('public check was not clean, base-passing and approved')
    policy = QualificationPolicy.from_manifest(read(candidate)['execution_manifest'])
    groups = []
    for directory, screening in ((producer / 'stage4a_instruction_only', True),
                                 (qualification, False)):
        rows = read(directory / 'results.json')
        repetitions = 3 if screening else policy.repetitions
        if (len(rows) != repetitions or {r['rep'] for r in rows} != set(range(1, repetitions + 1))
                or any(type(r.get('rep')) is not int or r['label'] != 'baseline' for r in rows)):
            raise ValueError(f'{repetitions} complete producer repetitions required')
        for row in rows:
            episode = Path(row['output_dir'])
            receipt = validate_episode_files(episode, code)
            result = read(episode / 'result.json')
            if (row.get('episode_valid') is not True or result.get('episode_valid') is not True
                    or type(row.get('success')) is not bool or result.get('success') is not row['success']
                    or receipt['inputs']['checks_sha256'] != digest(required[2])
                    or receipt['inputs']['task'] != identity['source']):
                raise ValueError('invalid producer episode outcome or input')
            if not screening and receipt['inputs']['candidate_sha256'] != digest(candidate):
                raise ValueError('qualification did not use selected frozen candidate')
        passes = sum(r['success'] for r in rows)
        if (screening and passes != 0) or (not screening and passes < policy.minimum_passes):
            raise ValueError('producer fails screening or recorded qualification gate')
        groups.append({'directory': str(directory), 'passes': passes, 'runs': repetitions})
    return identity, candidate, qualification, groups


def execute(args):
    # This branch runs with cwd/PYTHONPATH set to the verified isolated epoch.
    from teamgym.data.software.deep_swe import pipeline
    from teamgym.data.software.deep_swe.integrity import code_identity, image_identity, task_identity
    from teamgym.data.software.deep_swe.run_state import RunState
    from teamgym.data.software.deep_swe.source import load_task
    from teamgym.data.software.deep_swe.qualification import QualificationPolicy
    producer, snapshot, output = args.producer, args.snapshot, args.output
    identity, candidate_path, qualification, groups = validate_producer(producer, snapshot, args.round)
    if code_identity() != identity['code']:
        raise ValueError('isolated executable epoch differs from frozen producer')
    epoch = Path.cwd().resolve()
    origins = {name: str(Path(module.__file__).resolve())
               for name, module in tuple(sys.modules.items())
               if name.startswith('teamgym') and getattr(module, '__file__', None)}
    if any(not Path(path).is_relative_to(epoch) for path in origins.values()):
        raise ValueError('a package import escaped the isolated producer epoch')
    from handoff_audit import preflight, postflight, tree_receipt
    preflight_receipt = preflight(producer, identity, candidate_path)
    source_record = read(output.parent / (output.name + '.handoff.json')).get('audit_source')
    if args.packet_audit_source is not None:
        if (not source_record or source_record['path'] != str(args.packet_audit_source)
                or source_record['files'] != tree_receipt(args.packet_audit_source)):
            raise ValueError('inherited audit source changed before finalization')
    config = identity['config'].copy()
    if args.diagnostic_concurrency is not None:
        config['concurrency'] = args.diagnostic_concurrency
    for name in ('tasks_root', 'env_file', 'public_checks'):
        if config.get(name) is not None:
            config[name] = Path(config[name])
    options = argparse.Namespace(**config)
    task = load_task(identity['source']['source_dir'], runtime_image=options.runtime_image)
    if task_identity(task.source_dir) != identity['source']:
        raise ValueError('upstream raw source changed')
    for name in ('runtime_image', 'verifier_image'):
        if image_identity(getattr(options, name))['id'] != identity[name]['id']:
            raise ValueError('pinned image changed')
    output.mkdir(parents=True, exist_ok=True)
    pipeline._OPTIONS = options
    pipeline._RUN_STATE = RunState(output, identity, resume=True)
    shutil.copy2(output.parent / (output.name + '.handoff.json'), output / 'handoff.json')
    write(output / 'preflight.json', preflight_receipt)
    versions = {}
    for name in ('openai', 'httpx', 'tiktoken', 'python-dotenv', 'pydantic', 'requests'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write(output / 'finalizer_runtime.json', {'python': sys.version, 'package_versions': versions,
          'module_origins': origins, 'cwd': str(epoch),
          'diagnostic_concurrency': options.concurrency,
          'verifier_concurrency': os.environ.get('DEEPSWE_VERIFIER_CONCURRENCY'),
          'epoch_files': {str(p.relative_to(epoch)): digest(p) for p in sorted(epoch.rglob('*'))
                          if p.is_file()}, 'driver_sha256': digest(__file__)})
    # These paths are read-only inputs to _finish_distribution. No original
    # producer file or episode cache is rewritten by the handoff.
    for name in ('candidate_instruction_only.json', 'stage4a_instruction_only'):
        link = output / name
        if not link.exists():
            link.symlink_to(producer / name, target_is_directory=(producer / name).is_dir())
    report = {'schema': 'deepswe_qualified_phase_handoff_v1', 'producer': str(producer),
              'producer_code': identity['code']['sha256'], 'selected_round': args.round,
              'producer_diagnostics': groups, 'stages': {},
              'scheduling': {'producer_concurrency': identity['config']['concurrency'],
                             'finalizer_concurrency': options.concurrency,
                             'verifier_concurrency': os.environ.get('DEEPSWE_VERIFIER_CONCURRENCY'),
                             'actor_protocol_unchanged': True},
              'optional_improvement': 'not used by this finalization phase; qualification uses the selected complete round',
              'qualification_filter': QualificationPolicy.from_manifest(
                  read(candidate_path)['execution_manifest']).to_dict(),
              'limitation': 'separate post-producer phase; not a new cold run of later forward fixes'}
    finish_distribution = pipeline._finish_distribution
    stage_options = {}
    if args.distribution_module is not None:
        spec = importlib.util.spec_from_file_location(
            'teamgym.data.software.deep_swe.handoff_distribution', args.distribution_module)
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        finish_distribution = stage.finish_distribution
        stage_options['audit_system'] = args.packet_review_system_file.read_text()
        report['postproducer_stage'] = {'path': str(args.distribution_module),
            'sha256': digest(args.distribution_module),
            'packet_review_system_sha256': digest(args.packet_review_system_file),
            'boundary': 'new distribution/audit coordinator; frozen producer actor, harness, source and budgets'}
    if args.packet_audit_source is not None:
        stage_options['packet_audit_source'] = args.packet_audit_source
        report['audit_source'] = {'path': str(args.packet_audit_source),
            'policy': 'inherit only actual clear component reviews of identical semantic material; declare opaque global bundle hash change'}
    if args.packing_module is not None:
        spec = importlib.util.spec_from_file_location(
            'teamgym.data.software.deep_swe.handoff_packing', args.packing_module)
        packing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packing)
        stage_options['pack_function'] = packing.pack
        if args.background_exclusions is not None:
            stage_options['background_exclusions'] = read(args.background_exclusions)
        if args.background_policy is not None:
            stage_options['background_policy'] = args.background_policy
        if args.observation_background_exclusions is not None:
            stage_options['observation_background_exclusions'] = read(args.observation_background_exclusions)
        report['packet_revision'] = {'packing_module_sha256': digest(args.packing_module),
            'exclusions_sha256': digest(args.background_exclusions) if args.background_exclusions else None,
            'background_policy': args.background_policy,
            'observation_exclusions_sha256': (digest(args.observation_background_exclusions)
                if args.observation_background_exclusions else None),
            'policy': 'separately versioned public background; canonical candidate and qualification unchanged'}
    try:
        finish_distribution(options, task, read(candidate_path), candidate_path,
            read(producer / 'public_checks.json'), producer / 'public_checks.json',
            producer / 'discovered_code.json', read(producer / 'observations.validated.json'),
            options.verifier_image, output, qualification, report, **stage_options)
        if args.packet_audit_source is not None and tree_receipt(args.packet_audit_source) != source_record['files']:
            raise ValueError('inherited audit source changed during finalization')
        if report['decision'] == 'accepted':
            audit = postflight(producer, identity, candidate_path, output, preflight_receipt)
        else:
            if tree_receipt(producer) != preflight_receipt['producer_tree']:
                raise ValueError('producer changed after rejected finalization')
            audit = {'status': 'not_accepted', 'producer_tree_unchanged': True,
                     'decision': report['decision']}
        write(output / 'postflight.json', audit)
        pipeline._write_json(output / 'report.json', report)
        pipeline._RUN_STATE.checkpoint()
    except BaseException as exc:
        report['decision'] = 'execution_error'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        pipeline._write_json(output / 'report.json', report)
        raise
    print(json.dumps({'decision': report['decision'], 'report': str(output / 'report.json')}), flush=True)
    if report['decision'] != 'accepted':
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--snapshot', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--output', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--distribution-module', type=lambda p: Path(p).resolve())
    parser.add_argument('--packet-review-system-file', type=lambda p: Path(p).resolve())
    parser.add_argument('--packing-module', type=lambda p: Path(p).resolve())
    parser.add_argument('--background-exclusions', type=lambda p: Path(p).resolve())
    parser.add_argument('--background-policy', choices=('discovered', 'neutral_repository'))
    parser.add_argument('--observation-background-exclusions', type=lambda p: Path(p).resolve())
    parser.add_argument('--packet-audit-source', type=lambda p: Path(p).resolve())
    parser.add_argument('--diagnostic-concurrency', type=int,
                        help='Post-producer scheduling only; preserves every per-episode budget.')
    parser.add_argument('--execute-frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.diagnostic_concurrency is not None and args.diagnostic_concurrency < 1:
        parser.error('diagnostic concurrency must be positive')
    if (args.distribution_module is None) != (args.packet_review_system_file is None):
        parser.error('distribution module and explicit packet audit prompt must be supplied together')
    if args.packet_audit_source is not None and args.distribution_module is None:
        parser.error('audit inheritance requires an explicit distribution stage')
    if ((args.packing_module is not None and args.distribution_module is None)
            or ((args.background_exclusions is not None or args.background_policy is not None
                 or args.observation_background_exclusions is not None)
                and args.packing_module is None)):
        parser.error('packing revision requires distribution and packing modules')
    if args.execute_frozen:
        execute(args)
        return
    identity, _, _, groups = validate_producer(args.producer, args.snapshot, args.round)
    root = Path(__file__).resolve().parents[4]
    epoch = args.output.parent / (args.output.name + '-frozen-epoch')
    if not epoch.exists():
        # Preserve non-causal package infrastructure too; credentials and other
        # datasets are never copied. Every DeepSWE source/prompt is then replaced
        # by the producer snapshot and checked in the isolated child process.
        shutil.copytree(root / 'teamgym', epoch / 'teamgym',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        deep = epoch / 'teamgym/data/software/deep_swe'
        shutil.rmtree(deep)
        for label in identity['code']['files']:
            target = epoch / label
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.snapshot / label, target)
        shutil.copy2(args.snapshot / 'uv.lock', epoch / 'uv.lock')
    args.output.mkdir(parents=True, exist_ok=True)
    stage_inputs = {}
    if args.distribution_module is not None:
        stage_root = args.output.parent / (args.output.name + '-stage')
        stage_root.mkdir(parents=True, exist_ok=True)
        for key in ('distribution_module', 'packet_review_system_file', 'packing_module',
                    'background_exclusions', 'observation_background_exclusions'):
            source = getattr(args, key)
            if source is None:
                continue
            target = stage_root / source.name
            if target.exists() and digest(target) != digest(source):
                raise ValueError('post-producer stage changed; use a new output directory')
            shutil.copy2(source, target)
            setattr(args, key, target)
            stage_inputs[key] = {'path': str(target), 'sha256': digest(target)}
    audit_source = None
    if args.packet_audit_source is not None:
        from handoff_audit import tree_receipt
        audit_source = {'path': str(args.packet_audit_source), 'files': tree_receipt(args.packet_audit_source)}
    write(args.output.parent / (args.output.name + '.handoff.json'), {'schema': 'deepswe_frozen_phase_handoff_v1',
          'producer': str(args.producer), 'snapshot': str(args.snapshot), 'epoch': str(epoch),
          'producer_code_sha256': identity['code']['sha256'], 'diagnostics': groups,
          'driver': str(Path(__file__).resolve()), 'driver_sha256': digest(__file__), 'stage_inputs': stage_inputs,
          'diagnostic_concurrency': args.diagnostic_concurrency,
          'background_policy': args.background_policy,
          'audit_source': audit_source,
          'verifier_concurrency': os.environ.get('DEEPSWE_VERIFIER_CONCURRENCY'),
          'policy': 'do not rewrite producer manifests or relabel interrupted improvement outcomes'})
    env = dict(os.environ, PYTHONPATH=str(epoch), PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, str(Path(__file__).resolve()), '--producer', str(args.producer),
               '--snapshot', str(args.snapshot), '--output', str(args.output),
               '--round', str(args.round), '--execute-frozen']
    if stage_inputs:
        command += ['--distribution-module', str(args.distribution_module),
                    '--packet-review-system-file', str(args.packet_review_system_file)]
    if args.packing_module is not None:
        command += ['--packing-module', str(args.packing_module)]
    if args.background_exclusions is not None:
        command += ['--background-exclusions', str(args.background_exclusions)]
    if args.background_policy is not None:
        command += ['--background-policy', args.background_policy]
    if args.observation_background_exclusions is not None:
        command += ['--observation-background-exclusions', str(args.observation_background_exclusions)]
    if args.packet_audit_source is not None:
        command += ['--packet-audit-source', str(args.packet_audit_source)]
    if args.diagnostic_concurrency is not None:
        command += ['--diagnostic-concurrency', str(args.diagnostic_concurrency)]
    raise SystemExit(subprocess.run(command, cwd=epoch, env=env).returncode)


if __name__ == '__main__':
    main()
