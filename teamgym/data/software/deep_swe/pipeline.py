"""Construct DeepSWE evidence bundles and record patch diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .candidate import assemble_instruction_only_candidate
from .checks import (plan_public_checks, validate_public_checks,
                     write_public_check_validation_cache)
from .construction import qualification_feedback
from .qualification import DEFAULT_QUALIFICATION_POLICY, QualificationPolicy
from .source import load_task
from .run_state import RunState, sha256
from .stages.patch_runtime import save


WORKSPACE = Path.cwd()
_RUN_STATE: RunState | None = None
_OPTIONS = None


class DeepSWEPipelineError(RuntimeError):
    """Raised when a required construction or evaluation stage fails."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    save(path, value)


def _run(stage: str, argv: list[str], log: Path) -> None:
    argv = list(argv)
    if _OPTIONS is not None:
        module = argv[0].rsplit('.', 1)[-1]
        if module in {'discovery', 'construction', 'observation_proposals', 'observations'}:
            argv += ['--env-file', str(_OPTIONS.env_file), '--provider', _OPTIONS.provider,
                     '--base-url', _OPTIONS.base_url, '--api-key-env', _OPTIONS.api_key_env,
                     '--reasoning-effort', _OPTIONS.reasoning_effort]
        if module == 'observations':
            argv += ['--audit-model', _OPTIONS.constructor_model]
        if '--task-dir' in argv and '--runtime-image' not in argv:
            argv += ['--runtime-image', _OPTIONS.runtime_image]
        if module == 'diagnostic_runs':
            argv += ['--env-file', str(_OPTIONS.env_file), '--model-slug', _OPTIONS.actor_model,
                     '--provider', _OPTIONS.provider, '--base-url', _OPTIONS.base_url,
                     '--api-key-env', _OPTIONS.api_key_env, '--reasoning-effort', _OPTIONS.reasoning_effort,
                     '--total-output-tokens', str(_OPTIONS.total_output_tokens),
                     '--max-output-tokens-per-call', str(_OPTIONS.max_output_tokens_per_call),
                     '--max-checks-per-episode', str(_OPTIONS.max_checks_per_episode)]
    started = time.monotonic()
    _write_json(log.parent.parent / "progress.json", {
        "stage": stage, "state": "running", "started_at": time.time(), "log": str(log)
    })
    completed = subprocess.run(
        [sys.executable, "-m", argv[0], *argv[1:]],
        capture_output=True, text=True, cwd=WORKSPACE,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        completed.stdout
        + ("\n--- stderr ---\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": stage,
        "exit_code": completed.returncode,
        "seconds": round(time.monotonic() - started, 2),
    }), flush=True)
    _write_json(log.parent.parent / "progress.json", {
        "stage": stage, "state": "completed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode, "seconds": round(time.monotonic() - started, 2),
        "log": str(log), "updated_at": time.time(),
    })
    if completed.returncode:
        raise DeepSWEPipelineError(f"{stage} failed; see {log}")
    if _RUN_STATE is not None:
        _RUN_STATE.checkpoint()


def _summary(path: Path) -> dict:
    payload = _read_json(path)
    row = payload.get("baseline") or {}
    return {
        "passes": int(row.get("passes", 0)),
        "runs": int(row.get("runs", 0)),
        "f2p": row.get("f2p", []),
        "p2p": row.get("p2p", []),
    }


def _stage_exists(path: Path, *, skip_existing: bool) -> bool:
    if _RUN_STATE is not None:
        return _RUN_STATE.reusable(path, requested=skip_existing)
    return False


def _episode_receipts(run_dir: Path, *, label='baseline', expected_packet=None,
                      expected_execution=None) -> list[dict]:
    """Read real episode artifacts, checking their provenance before export."""
    rows = _read_json(run_dir / 'results.json')
    receipts = []
    locations = set()
    for row in rows:
        if row['label'] != label:
            continue
        episode = Path(row['output_dir'])
        if (type(row.get('rep')) is not int or episode.name != f"{label}_rep{row['rep']}"
                or episode.resolve().parent != run_dir.resolve() or episode.resolve() in locations):
            raise DeepSWEPipelineError('episode output directory is outside its group or duplicated')
        locations.add(episode.resolve())
        result = _read_json(episode / 'result.json')
        manifest = _read_json(episode / 'manifest.json')
        actor_path = episode / 'actor_packet.json'
        if manifest['actor_packet_sha256'] != sha256(actor_path):
            raise DeepSWEPipelineError('actor packet changed after episode')
        if expected_packet is not None and _read_json(actor_path) != expected_packet:
            raise DeepSWEPipelineError('episode did not use the frozen evaluation packet')
        if expected_execution is not None:
            from teamgym.llm.openai_client import normalize_model_for_provider
            frozen = _read_json(episode / 'candidate.frozen.hidden.json')
            if frozen.get('execution_manifest') != expected_execution:
                raise DeepSWEPipelineError('episode execution manifest mismatch')
            actor, protocol = expected_execution['actor'], expected_execution['protocol']
            expected = {'task_id': expected_execution['task_id'],
                'model': normalize_model_for_provider(actor['provider'], actor['model']),
                'provider': actor['provider'], 'base_url': actor['base_url'], 'endpoint': actor['endpoint'],
                'reasoning_effort': actor['reasoning_effort'], 'patch_turn_budget': protocol['patch_turns'],
                'total_output_token_budget': protocol['total_output_tokens'],
                'max_output_tokens_per_call': protocol['max_output_tokens_per_call'],
                'max_public_checks_per_episode': protocol['max_checks_per_episode'],
                'runtime_image_id': expected_execution['runtime_image']['id'],
                'verifier_image_id': expected_execution['verifier_image']['id'],
                'source_base_commit': expected_execution['source']['base_commit'],
                'harness_sha256': expected_execution['code']['files'][
                    'teamgym/data/software/deep_swe/stages/patch_episode.py']}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise DeepSWEPipelineError('episode did not use the declared execution protocol')
        _verify_episode_cache(episode, row, manifest, expected_execution)
        if result.get('success') is not row['success'] or result.get('episode_valid') is not row['episode_valid']:
            raise DeepSWEPipelineError('episode result and summary disagree')
        receipt = {'episode_valid': row['episode_valid'], 'success': row['success'],
            'output_dir': str(episode), 'manifest_sha256': sha256(episode / 'manifest.json'),
            'result_sha256': sha256(episode / 'result.json'),
            'cache_manifest_sha256': sha256(episode / 'cache_manifest.json')}
        if expected_execution is not None:
            from .export import sha256 as json_sha256
            receipt['execution_manifest_sha256'] = json_sha256(expected_execution)
        receipts.append(receipt)
    return receipts


def _verify_episode_cache(episode, row, manifest, expected_execution):
    from .integrity import SCHEMA_VERSION, canonical_sha256
    from .stages.diagnostic_runs import EPISODE_OUTPUTS, valid_episode_result
    try:
        cache = _read_json(episode / 'cache_manifest.json')
        if (not isinstance(cache, dict) or set(cache) != {'schema_version', 'inputs', 'inputs_sha256', 'outputs'}
                or type(cache['schema_version']) is not int or cache['schema_version'] != SCHEMA_VERSION
                or not isinstance(cache['inputs'], dict)
                or cache['inputs_sha256'] != canonical_sha256(cache['inputs'])):
            raise ValueError('invalid episode cache schema or input hash')
        expected_outputs = {str(episode / name): sha256(episode / name) for name in EPISODE_OUTPUTS}
        if cache['outputs'] != expected_outputs or any((episode / name).is_symlink() for name in EPISODE_OUTPUTS):
            raise ValueError('episode cache outputs changed or are not exactly nine local files')
        inputs = cache['inputs']
        candidate_sha = sha256(episode / 'candidate.frozen.hidden.json')
        if (manifest.get('candidate_sha256') != candidate_sha or inputs.get('candidate_sha256') != candidate_sha
                or manifest.get('checks_sha256') != inputs.get('checks_sha256')
                or not isinstance(inputs.get('checks_sha256'), str) or len(inputs['checks_sha256']) != 64
                or set(inputs['checks_sha256']) - set('0123456789abcdef')
                or manifest.get('harness_sha256') != sha256(episode / 'harness.frozen.py')
                or inputs.get('condition', {}).get('label') != row['label']
                or ('rep' in row and inputs.get('condition', {}).get('rep') != row['rep'])):
            raise ValueError('episode cache candidate/checks/harness/condition mismatch')
        result = _read_json(episode / 'result.json')
        if not valid_episode_result(result):
            raise ValueError('episode result is invalid')
        episode_data = _read_json(episode / 'episode.json')
        if (canonical_sha256(result.get('episode')) != canonical_sha256(episode_data)
                or episode_data.get('patch_sha256') != sha256(episode / 'model.patch')):
            raise ValueError('episode patch/result binding mismatch')
        reward = result['reward']
        if (type(reward.get('reward')) not in (int, float) or reward['reward'] not in (0, 1)
                or any(type(reward.get(f'{kind}_{field}')) is not int
                       for kind in ('f2p', 'p2p') for field in ('passed', 'total'))
                or any(not 0 <= reward[f'{kind}_passed'] <= reward[f'{kind}_total']
                       or reward[f'{kind}_total'] < 1 for kind in ('f2p', 'p2p'))):
            raise ValueError('episode reward/count schema is invalid')
        actual_success = (episode_data.get('termination') != 'context_window_failure'
            and result.get('exit_code') == 0 and reward['reward'] == 1
            and all(reward[f'{kind}_passed'] == reward[f'{kind}_total'] for kind in ('f2p', 'p2p')))
        if result['success'] is not actual_success:
            raise ValueError('episode success contradicts verifier result')
        for kind in ('f2p', 'p2p'):
            counts = [result['reward'].get(f'{kind}_passed'), result['reward'].get(f'{kind}_total')]
            if kind in row and canonical_sha256(row[kind]) != canonical_sha256(counts):
                raise ValueError('episode result and summary counts disagree')
        if expected_execution is not None:
            actor = {k: v for k, v in inputs.get('actor', {}).items() if k != 'env_file'}
            expected_source = expected_execution['source']
            if (any(canonical_sha256(inputs.get('task', {}).get(k)) != canonical_sha256(expected_source.get(k))
                    for k in ('task_id', 'base_commit', 'test_base_commit', 'source_revision', 'tree'))
                    or canonical_sha256(inputs.get('code')) != canonical_sha256(expected_execution['code'])
                    or canonical_sha256(actor) != canonical_sha256(expected_execution['actor'])
                    or inputs.get('candidate_runtime', {}).get('id') != expected_execution['runtime_image']['id']
                    or inputs.get('task_runtime', {}).get('id') != expected_execution['runtime_image']['id']
                    or inputs.get('verifier', {}).get('id') != expected_execution['verifier_image']['id']
                    or canonical_sha256({k: inputs.get('protocol', {}).get(k) for k in expected_execution['protocol']})
                       != canonical_sha256(expected_execution['protocol'])
                    or inputs.get('protocol', {}).get('evidence_mode') != 'all'):
                raise ValueError('episode cache inputs do not match frozen execution')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DeepSWEPipelineError(f'episode cache integrity failed: {exc}') from exc


def _invalidate_ready_export(output, reason):
    """Preserve prior accepted output, but withdraw its current ready path."""
    ready = output / 'export'
    if not ready.exists():
        return
    archive = output / 'invalidated_exports' / str(time.time_ns())
    archive.parent.mkdir(parents=True, exist_ok=True)
    ready.rename(archive)
    report_path = output / 'report.json'
    if report_path.is_file():
        old = _read_json(report_path)
        _write_json(archive / 'previous_report.json', old)
        _write_json(report_path, {**old, 'decision': 'validation_pending',
            'export': None, 'previous_export_archive': str(archive)})
    _write_json(output / 'export_invalidation.json', {'reason': reason, 'archive': str(archive)})


def _validated_diagnostic_summary(path, *, expected_packet, expected_execution, expected_repetitions=3):
    from .integrity import canonical_sha256
    _episode_receipts(path.parent, expected_packet=expected_packet, expected_execution=expected_execution)
    rows = [r for r in _read_json(path.parent / 'results.json') if r['label'] == 'baseline']
    if (len(rows) != expected_repetitions or any(type(r.get('rep')) is not int for r in rows)
            or {r['rep'] for r in rows} != set(range(1, expected_repetitions + 1))):
        raise DeepSWEPipelineError(f'diagnostic summary requires {expected_repetitions} valid outcomes with unique repetitions')
    actual = {'passes': sum(r['success'] is True for r in rows), 'runs': len(rows),
              'f2p': [r['f2p'] for r in rows], 'p2p': [r['p2p'] for r in rows]}
    if canonical_sha256(_read_json(path)['baseline']) != canonical_sha256(actual):
        raise DeepSWEPipelineError('diagnostic summary disagrees with actual episode rows')
    return actual


def _diagnostic_summary(stage, argv, log, path, *, skip_existing, expected_packet, expected_execution,
                        expected_repetitions=3):
    reuse = _stage_exists(path, skip_existing=skip_existing)
    if reuse or (skip_existing and path.exists()):
        try:
            validated = _validated_diagnostic_summary(path, expected_packet=expected_packet,
                                                       expected_execution=expected_execution,
                                                       expected_repetitions=expected_repetitions)
            if reuse:
                return validated
        except (DeepSWEPipelineError, OSError, ValueError, KeyError, TypeError) as exc:
            if _RUN_STATE is not None:
                _RUN_STATE.dirty = True
                _invalidate_ready_export(_RUN_STATE.output, 'diagnostic_cache_integrity_failure')
            rejected = path.parent / 'integrity_rejections' / str(time.time_ns())
            rejected.mkdir(parents=True)
            for name in ('summary.json', 'results.json'):
                previous = path.parent / name
                if previous.exists():
                    previous.rename(rejected / name)
            _write_json(rejected / 'recovery.json', {'stage': stage, 'error': str(exc),
                'classification': 'cache_integrity_error', 'action': 'rerun_diagnostic',
                'counted_as_model_failure': False})
    _run(stage, argv, log)
    return _validated_diagnostic_summary(path, expected_packet=expected_packet,
                                         expected_execution=expected_execution,
                                         expected_repetitions=expected_repetitions)


def _verify_frozen_inputs(task, args):
    from .integrity import code_identity, image_identity, task_identity
    from .distribution import packet_audit_source_sha256
    identity = _RUN_STATE.identity
    if (identity['source'] != task_identity(task.source_dir) or identity['code'] != code_identity()
            or identity['runtime_image']['id'] != image_identity(args.runtime_image)['id']
            or identity['verifier_image']['id'] != image_identity(args.verifier_image)['id']
            or identity['environment']['uv_lock_sha256'] != sha256(WORKSPACE / 'uv.lock')
            or (args.public_checks and identity['public_checks_sha256'] != sha256(args.public_checks))
            or (getattr(args, 'background_exclusions', None)
                and identity['background_exclusions_sha256'] != sha256(args.background_exclusions))
            or (getattr(args, 'observation_background_exclusions', None)
                and identity['observation_background_exclusions_sha256'] != sha256(args.observation_background_exclusions))
            or (getattr(args, 'packet_audit_source', None)
                and identity['packet_audit_source_sha256'] != packet_audit_source_sha256(args.packet_audit_source))):
        raise DeepSWEPipelineError('Construction inputs changed during execution')


def _merge_observations(original, repaired, *, conflicts):
    """Merge stable observations and exclude commands with conflicting output."""
    by_id = {row['id']: row for row in original}
    identity_fields = ('task_id', 'base_commit', 'argv')
    outcome_fields = ('stdout', 'stderr', 'exit_code')
    for row in repaired:
        identifier = row['id']
        conflict = conflicts.get(identifier)
        prior = conflict or by_id.get(identifier)
        if prior is None:
            by_id[identifier] = row
            continue
        if any(prior.get(key) != row.get(key) for key in identity_fields):
            raise DeepSWEPipelineError('observation ID collision during repair')
        if conflict is None and any(prior.get(key) != row.get(key) for key in outcome_fields):
            del by_id[identifier]
            conflicts[identifier] = {
                'id': identifier,
                **{key: row[key] for key in identity_fields},
                'rejection_reason': 'inconsistent_observation_across_collections',
            }
    return list(by_id.values())


def _execution_manifest(task, args, identity):
    from .stages.diagnostic_runs import actor_identity
    actor = actor_identity({'model_slug': args.actor_model, 'provider': args.provider,
        'base_url': args.base_url, 'api_key_env': args.api_key_env,
        'reasoning_effort': args.reasoning_effort})
    actor.pop('env_file', None)
    return {'schema': 'deepswe_execution_manifest_v1', 'task_id': task.task_id,
        'qualification': QualificationPolicy(args.qualification_min_passes, args.repetitions).to_dict(),
        'source': identity['source'], 'runtime_image': identity['runtime_image'],
        'verifier_image': identity['verifier_image'], 'actor': actor,
        'protocol': {'patch_turns': 5, 'total_output_tokens': args.total_output_tokens,
                     'max_output_tokens_per_call': args.max_output_tokens_per_call,
                     'max_checks_per_episode': args.max_checks_per_episode},
        'code': identity['code'], 'environment': identity['environment']}


def _finish_distribution(args, task, candidate, candidate_path, checks, checks_path,
                         discovered, observations, verifier, output, qualification_dir, report,
                         *, audit_system=None, pack_function=None, background_exclusions=None,
                         background_policy=None, background_language=None,
                         observation_background_exclusions=None, packet_audit_source=None):
    from .distribution import finish_distribution
    return finish_distribution(args, task, candidate, candidate_path, checks, checks_path,
        discovered, observations, verifier, output, qualification_dir, report,
        audit_system=audit_system, pack_function=pack_function, background_exclusions=background_exclusions,
        background_policy=background_policy, background_language=background_language,
        observation_background_exclusions=observation_background_exclusions, packet_audit_source=packet_audit_source)


def _main() -> None:
    global _RUN_STATE, _OPTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--tasks-root", type=Path, default=WORKSPACE / ".external_datasets/deep-swe/tasks"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verifier-image")
    parser.add_argument("--runtime-image")
    parser.add_argument("--public-checks", type=Path)
    parser.add_argument("--probes", type=int, default=10)
    parser.add_argument("--patch-turns", type=int, default=5)
    parser.add_argument("--qualification-repetitions", "--repetitions", dest="repetitions", type=int,
                        default=DEFAULT_QUALIFICATION_POLICY.repetitions,
                        help="Complete-evidence repair attempts per round (default: 3)")
    parser.add_argument("--qualification-min-passes", type=int,
                        default=DEFAULT_QUALIFICATION_POLICY.minimum_passes,
                        help="Required official verifier successes (default: 2)")
    parser.add_argument("--revision-target-passes", type=int,
                        help="Optional improvement target; defaults to --qualification-min-passes")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--constructor-model", default="gpt-5.4")
    parser.add_argument("--actor-model", default="gpt55")
    parser.add_argument("--env-file", type=Path, default=WORKSPACE / '.env')
    parser.add_argument("--provider", default="bedrock_openai")
    parser.add_argument("--base-url", default="https://bedrock-mantle.us-east-2.api.aws/openai/v1")
    parser.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--total-output-tokens", type=int, default=32000)
    parser.add_argument("--max-output-tokens-per-call", type=int, default=16000)
    parser.add_argument("--max-checks-per-episode", type=int, default=15)
    parser.add_argument("--packet-tokens", type=int, default=40000)
    parser.add_argument("--background-exclusions", type=Path,
                        help="JSON list of participant-specific background ranges to omit during packing")
    parser.add_argument("--observation-background-exclusions", type=Path,
                        help="JSON list of participant-specific foreign observation distractors to omit")
    parser.add_argument("--packet-audit-source", type=Path,
                        help="Prior actual audit directory; reuse only exact clear component reviews")
    parser.add_argument("--background-policy", choices=('discovered', 'neutral_repository'),
                        default='neutral_repository',
                        help="Source of filler code; neutral_repository reads other pinned public base files")
    parser.add_argument("--loo-repetitions", type=int, default=1)
    parser.add_argument("--max-riu-rounds", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    try:
        qualification_policy = QualificationPolicy(args.qualification_min_passes, args.repetitions)
    except ValueError as exc:
        parser.error(str(exc))
    if args.revision_target_passes is None:
        args.revision_target_passes = qualification_policy.minimum_passes
    if not qualification_policy.minimum_passes <= args.revision_target_passes <= args.repetitions:
        parser.error("--revision-target-passes must be between --qualification-min-passes and --qualification-repetitions")
    if args.patch_turns != 5:
        parser.error("patch diagnostics use five edit turns")
    if args.max_riu_rounds < 1:
        parser.error("--max-riu-rounds must be positive")
    if min(args.total_output_tokens, args.max_output_tokens_per_call, args.max_checks_per_episode,
           args.packet_tokens, args.loo_repetitions, args.concurrency, args.probes) < 1:
        parser.error("all execution budgets must be positive")

    output = (args.output_root / args.task_id).resolve()
    _invalidate_ready_export(output, 'new_validation_attempt')
    task_dir = (args.tasks_root / args.task_id).expanduser().resolve()
    task = load_task(task_dir, runtime_image=args.runtime_image)
    verifier = args.verifier_image or f"codex-deepswe-verifier:{task.task_id}"
    from .integrity import image_identity, task_identity, code_identity
    runtime_identity = image_identity(task.runtime.docker_image)
    verifier_identity = image_identity(verifier)
    args.runtime_image = runtime_identity['id']
    verifier = verifier_identity['id']
    args.verifier_image = verifier
    task = load_task(task_dir, runtime_image=args.runtime_image)
    _OPTIONS = args
    from .distribution import packet_audit_source_sha256
    identity = {'schema': 'deepswe_construction_run_v2', 'source': task_identity(task_dir),
        'code': code_identity(), 'runtime_image': runtime_identity, 'verifier_image': verifier_identity,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'uv_lock_sha256': sha256(WORKSPACE / 'uv.lock')},
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                   if k not in {'skip_existing', 'output_root'}},
        'public_checks_sha256': sha256(args.public_checks) if args.public_checks else None,
        'background_exclusions_sha256': sha256(args.background_exclusions) if args.background_exclusions else None,
        'observation_background_exclusions_sha256': sha256(args.observation_background_exclusions)
            if args.observation_background_exclusions else None,
        'packet_audit_source_sha256': packet_audit_source_sha256(args.packet_audit_source) if args.packet_audit_source else None}
    background_exclusions = _read_json(args.background_exclusions) if args.background_exclusions else None
    observation_background_exclusions = (_read_json(args.observation_background_exclusions)
                                        if args.observation_background_exclusions else None)
    _RUN_STATE = RunState(output, identity, resume=args.skip_existing)
    logs = output / "logs"
    report_path = output / "report.json"
    report: dict[str, Any] = {
        "schema": "deepswe_functional_riu_pipeline_v1",
        "task_id": task.task_id,
        "protocol": {
            "patch_turns": 5,
            "repetitions": args.repetitions,
            "instruction_only_role": "screening_0_of_3",
            "revision_target_passes": args.revision_target_passes,
            "minimum_qualification_passes": qualification_policy.minimum_passes,
            "qualification": qualification_policy.to_dict(),
            "max_riu_rounds": args.max_riu_rounds,
            "public_check_feedback": "status_only",
            "max_public_checks_per_episode": args.max_checks_per_episode,
            "solver_authored_tests": False,
        },
        "stages": {},
    }

    checks_path = output / "public_checks.json"
    check_validation_path = output / "public_checks.base_validation.json"
    check_cache_path = output / 'public_checks.validation_cache.json'
    check_options = dict(model=args.constructor_model, env_file=str(args.env_file),
        trace_dir=output / 'prompts' / 'public_check_validation', provider=args.provider,
        base_url=args.base_url, api_key_env=args.api_key_env, reasoning_effort=args.reasoning_effort)
    cache_options = dict(validation_cache_path=check_cache_path, validation_identity=identity,
        validation_cache_sha256=(_RUN_STATE.receipts.get(str(check_cache_path.relative_to(output)))
            if args.skip_existing and not _RUN_STATE.dirty else None))
    if args.public_checks:
        checks = _read_json(args.public_checks)
        check_validation = validate_public_checks(task, checks, **check_options, **cache_options)
    elif _stage_exists(checks_path, skip_existing=args.skip_existing):
        checks = _read_json(checks_path)
        check_validation = validate_public_checks(task, checks, **check_options, **cache_options)
    else:
        checks, check_validation = plan_public_checks(
            task=task,
            model=args.constructor_model,
            env_file=str(args.env_file),
            trace_dir=output / "prompts" / "public_check_planner",
            provider=args.provider, base_url=args.base_url, api_key_env=args.api_key_env,
            reasoning_effort=args.reasoning_effort,
        )
    _write_json(checks_path, checks)
    _write_json(check_validation_path, check_validation)
    write_public_check_validation_cache(check_cache_path, task, checks, check_validation,
                                        identity=identity, **check_options)
    report["stages"]["public_checks"] = {
        "count": len(checks), "base_passed": len(check_validation)
    }
    _RUN_STATE.checkpoint()

    instruction_only_candidate = output / "candidate_instruction_only.json"
    execution_manifest = _execution_manifest(task, args, identity)
    instruction_candidate = assemble_instruction_only_candidate(task)
    instruction_candidate['execution_manifest'] = execution_manifest
    _write_json(instruction_only_candidate, instruction_candidate)
    baseline_dir = output / "stage4a_instruction_only"
    baseline_summary_path = baseline_dir / "summary.json"
    from .stages.patch_episode import packet_for
    baseline = _diagnostic_summary("instruction_only", [
            f"{__package__}.stages.diagnostic_runs", "--task-dir", str(task_dir),
            "--candidate", str(instruction_only_candidate), "--checks", str(checks_path),
            "--verifier-image", verifier, "--output-dir", str(baseline_dir),
            "--patch-turns", "5", "--granularity", "riu", "--repetitions", "3",
            "--concurrency", str(args.concurrency), "--only", "baseline",
        ], logs / "instruction_only.log", baseline_summary_path,
        skip_existing=args.skip_existing, expected_packet=packet_for(instruction_candidate, checks, 'all'),
        expected_execution=execution_manifest)
    report["stages"]["instruction_only"] = baseline
    _write_json(report_path, report)
    if baseline["runs"] != 3:
        raise DeepSWEPipelineError(f"instruction-only produced {baseline['runs']} rows, expected 3")
    if baseline['passes'] != 0:
        report['decision'] = 'screening_rejected'
        _write_json(report_path, report)
        print(json.dumps({'event': 'done', 'decision': report['decision'],
                          'instruction_only': baseline}), flush=True)
        return
    discovered = output / "discovered_code.json"
    if not _stage_exists(discovered, skip_existing=args.skip_existing):
        _run("0_discover", [
            f"{__package__}.discovery", "--task-dir", str(task_dir),
            "--output", str(discovered), "--model", args.constructor_model,
        ], logs / "stage0.log")
    discovery = _read_json(discovered)
    if discovery.get("termination") != "done":
        _run("0_discover_continue", [
            f"{__package__}.discovery", "--task-dir", str(task_dir),
            "--output", str(discovered), "--model", args.constructor_model,
            "--resume-from", str(discovered), "--max-rounds", "10",
        ], logs / "stage0_continue.log")
        discovery = _read_json(discovered)
    if not discovery.get("files") or discovery.get("termination") not in {
        "done", "max_rounds", "output_token_budget"
    }:
        raise DeepSWEPipelineError(
            f"code discovery did not reach fixpoint: {discovery.get('termination')}"
        )
    report["stages"]["discovery"] = {
        "rounds": discovery.get("rounds_used"), "files": len(discovery.get("files") or {}),
        "termination": discovery.get("termination"),
        "coverage_complete": discovery.get("termination") == "done",
    }

    pool_path = output / "fragment_pool.json"
    if not _stage_exists(pool_path, skip_existing=args.skip_existing):
        _run("1_2_fragment_pool", [
            f"{__package__}.stages.build_deepswe_fragment_pool", "--task-dir", str(task_dir),
            "--discovered", str(discovered), "--output", str(pool_path),
        ], logs / "stage12.log")
    pool = _read_json(pool_path)
    if not (pool.get("fragments") or pool.get("oversized")):
        raise DeepSWEPipelineError("the evidence pool needs at least one source fragment")
    report["stages"]["fragment_pool"] = pool.get("summary")

    plan_path = output / "observation_plan.json"
    if not _stage_exists(plan_path, skip_existing=args.skip_existing):
        _run("3a_observation_plan", [
            f"{__package__}.observation_proposals", "--task-dir", str(task_dir), "--pool", str(pool_path),
            "--discovered", str(discovered), "--reference-patch", str(task.artifacts.solution_patch),
            "--count", str(args.probes), "--model", args.constructor_model,
            "--output", str(plan_path),
        ], logs / "stage3a.log")

    observations_dir = output / "observations"
    observations_path = observations_dir / "observations.json"
    if not _stage_exists(observations_path, skip_existing=args.skip_existing):
        _run("3b_observation_collection", [
            f"{__package__}.observations", "--task-dir", str(task_dir),
            "--plan", str(plan_path), "--output-dir", str(observations_dir),
        ], logs / "stage3b.log")
    observations = _read_json(observations_path)
    # Broken probe setup is repairable and must not be sent as behavioral evidence.
    from .evidence import usable_observations
    observations, rejected = usable_observations(observations, task_id=task.task_id,
        base_commit=task.repository.base_commit_hash)
    observation_conflicts = {}
    feedback_file = observations_dir / "collection_feedback.json"
    for repair_no in range(1, 3):
        feedback = _read_json(feedback_file) if feedback_file.exists() else {"rejected": rejected}
        if not feedback.get("rejected"):
            break
        if not feedback_file.exists():
            _write_json(feedback_file, feedback)
        repair_plan = output / f"observation_plan_repair_{repair_no:02d}.json"
        repair_dir = output / f"observations_repair_{repair_no:02d}"
        if not _stage_exists(repair_plan, skip_existing=args.skip_existing):
            _run(f"3a_probe_repair_{repair_no}", [
                f"{__package__}.observation_proposals", "--task-dir", str(task_dir), "--pool", str(pool_path),
                "--discovered", str(discovered), "--reference-patch", str(task.artifacts.solution_patch),
                "--count", str(args.probes), "--model", args.constructor_model,
                "--collection-feedback", str(feedback_file), "--prior-plan", str(plan_path),
                "--output", str(repair_plan),
            ], logs / f"stage3a_repair_{repair_no}.log")
        if not _stage_exists(repair_dir / "observations.json", skip_existing=args.skip_existing):
            _run(f"3b_probe_repair_{repair_no}", [
                f"{__package__}.observations", "--task-dir", str(task_dir),
                "--plan", str(repair_plan), "--output-dir", str(repair_dir),
            ], logs / f"stage3b_repair_{repair_no}.log")
        plan_path, observations_dir = repair_plan, repair_dir
        observations_path = observations_dir / "observations.json"
        repaired, rejected = usable_observations(_read_json(observations_path),
            task_id=task.task_id, base_commit=task.repository.base_commit_hash)
        observations = _merge_observations(observations, repaired, conflicts=observation_conflicts)
        feedback_file = observations_dir / "collection_feedback.json"
        if observation_conflicts:
            feedback = _read_json(feedback_file) if feedback_file.exists() else {"rejected": rejected}
            feedback_file = output / f"observation_repair_feedback_{repair_no:02d}.json"
            _write_json(feedback_file, {**feedback, "usable": len(observations),
                "rejected": [*feedback["rejected"], *observation_conflicts.values()]})
    observations_path = output / 'observations.validated.json'
    _write_json(observations_path, observations)
    report["stages"]["observations"] = len(observations)
    if observation_conflicts:
        report["stages"]["observation_conflicts"] = list(observation_conflicts.values())
    report["stages"]["probe_feedback"] = str(feedback_file)
    _write_json(report_path, report)

    prior_candidate: Path | None = None
    prior_qualification: Path | None = None
    report["stages"]["riu_rounds"] = []
    qualification = {"passes": 0, "runs": 0, "f2p": [], "p2p": []}
    for round_no in range(1, args.max_riu_rounds + 1):
        round_label = f"round_{round_no:02d}"
        if prior_candidate and prior_qualification:
            prior = _read_json(prior_candidate).get("constructor_metadata") or {}
            gaps = {"uncovered": prior.get("uncovered", []), "diagnosis": prior.get("diagnosis"),
                    "qualification": _summary(prior_qualification / "summary.json"),
                    "failing_tests": qualification_feedback(prior_qualification).get("failing_tests", [])}
            gap_path = output / f"source_gaps_{round_label}.json"
            expansion_marker = output / f"source_expansion_{round_label}.json"
            if (gaps["uncovered"] or gaps["diagnosis"] or gaps['failing_tests']) and not _stage_exists(expansion_marker, skip_existing=args.skip_existing):
                _write_json(gap_path, gaps)
                _run(f"0_expand_{round_label}", [
                    f"{__package__}.discovery", "--task-dir", str(task_dir),
                    "--output", str(discovered), "--model", args.constructor_model,
                    "--resume-from", str(discovered), "--gap-feedback", str(gap_path),
                    "--max-rounds", "5",
                ], logs / f"stage0_expand_{round_label}.log")
                expanded = _read_json(discovered)
                if expanded.get("termination") not in {"done", "max_rounds", "output_token_budget"}:
                    raise DeepSWEPipelineError(f"source expansion failed: {expanded.get('termination')}")
                _run(f"1_2_expand_{round_label}", [
                    f"{__package__}.stages.build_deepswe_fragment_pool", "--task-dir", str(task_dir),
                    "--discovered", str(discovered), "--output", str(pool_path),
                    "--prior-pool", str(pool_path),
                ], logs / f"stage12_expand_{round_label}.log")
                _write_json(expansion_marker, {"gap_feedback": str(gap_path),
                            "files": len(expanded.get("files", {})), "termination": expanded.get("termination")})
        candidate_path = output / f"candidate_functional_rius_{round_label}.json"
        functional_plan_path = output / f"functional_riu_plan_{round_label}.json"
        constructor_argv = [
            f"{__package__}.construction", "--task-dir", str(task_dir),
            "--pool", str(pool_path), "--observations", str(observations_path),
            "--baseline-run", str(baseline_dir), "--public-checks", str(checks_path),
            "--model", args.constructor_model, "--plan-output", str(functional_plan_path),
            "--output", str(candidate_path),
        ]
        if prior_candidate and prior_qualification:
            constructor_argv.extend([
                "--prior-candidate", str(prior_candidate),
                "--qualification-run", str(prior_qualification),
            ])
        if not _stage_exists(candidate_path, skip_existing=args.skip_existing):
            _run(
                f"4b_functional_riu_{round_label}", constructor_argv,
                logs / f"stage4b_{round_label}.log",
            )
        candidate = _read_json(candidate_path)
        # Freeze all solver-visible evidence before its qualification. Audit
        # outcomes live in external receipts so candidate hashes stay stable.
        if candidate.get('status') != 'candidate_frozen':
            candidate['status'] = 'candidate_frozen'
            candidate['execution_manifest'] = execution_manifest
            candidate['audits'] = {'status': 'external_receipts', 'report': 'report.json'}
            _write_json(candidate_path, candidate)
            _RUN_STATE.checkpoint()

        qualification_dir = output / f"stage4c_qualification_{round_label}"
        qualification_summary_path = qualification_dir / "summary.json"
        qualification = _diagnostic_summary(f"4c_qualification_{round_label}", [
                f"{__package__}.stages.diagnostic_runs", "--task-dir", str(task_dir),
                "--candidate", str(candidate_path), "--checks", str(checks_path),
                "--verifier-image", verifier, "--output-dir", str(qualification_dir),
                "--patch-turns", "5", "--granularity", "riu", "--repetitions", str(args.repetitions),
                "--concurrency", str(args.concurrency),
                "--only", "baseline",
            ], logs / f"stage4c_{round_label}.log", qualification_summary_path,
            skip_existing=args.skip_existing, expected_packet=packet_for(candidate, checks, 'all'),
            expected_execution=execution_manifest, expected_repetitions=args.repetitions)
        feedback_path = output / f"qualification_feedback_{round_label}.json"
        feedback = qualification_feedback(qualification_dir)
        _write_json(feedback_path, feedback)
        round_report = {
            "round": round_no,
            "revision": round_no > 1,
            "candidate": str(candidate_path),
            "rius": len(candidate.get("allocation_candidates") or []),
            "atoms": len(candidate.get("code_atoms") or []),
            "observations": len(candidate.get("observations") or []),
            "diagnosis": (candidate.get("constructor_metadata") or {}).get("diagnosis"),
            "qualification": qualification,
            "qualification_dir": str(qualification_dir),
            "qualification_feedback": str(feedback_path),
            "failing_tests": len(feedback.get("failing_tests") or []),
        }
        report["stages"]["riu_rounds"].append(round_report)
        _write_json(report_path, report)
        if (
            qualification["runs"] == args.repetitions
            and qualification["passes"] >= args.revision_target_passes
        ):
            break
        prior_candidate = candidate_path
        prior_qualification = qualification_dir

    # Keep the best completed bundle meeting the recorded qualification policy.
    usable = [r for r in report['stages']['riu_rounds']
              if r['qualification']['runs'] == qualification_policy.repetitions
              and r['qualification']['passes'] >= qualification_policy.minimum_passes]
    if usable:
        selected = max(usable, key=lambda r: (r['qualification']['passes'], -r['round']))
        candidate_path = Path(selected['candidate'])
        candidate = _read_json(candidate_path)
        qualification = selected['qualification']
        qualification_dir = Path(selected['qualification_dir'])
        report['selected_round'] = selected['round']
    report["decision"] = "construction_complete"
    report["diagnostic_target_met"] = (
        qualification["runs"] == args.repetitions
        and qualification["passes"] >= args.revision_target_passes
    )
    if qualification['runs'] != qualification_policy.repetitions:
        raise DeepSWEPipelineError(f'qualification requires {qualification_policy.repetitions} valid outcomes')
    if qualification['passes'] < qualification_policy.minimum_passes:
        report['decision'] = 'qualification_rejected'
    else:
        _finish_distribution(args, task, candidate, candidate_path, checks, checks_path,
                             discovered, observations, verifier, output, qualification_dir, report,
                             background_exclusions=background_exclusions,
                             background_policy=args.background_policy,
                             observation_background_exclusions=observation_background_exclusions,
                             packet_audit_source=args.packet_audit_source)
    _write_json(report_path, report)
    _RUN_STATE.checkpoint()
    print(json.dumps({
        "event": "done", "task_id": task.task_id,
        "instruction_only": f"{baseline['passes']}/{baseline['runs']}",
        "riu_full": f"{qualification['passes']}/{qualification['runs']}",
        "decision": report["decision"],
    }), flush=True)


def main() -> None:
    global _RUN_STATE, _OPTIONS
    _RUN_STATE = _OPTIONS = None
    try:
        _main()
    except Exception as exc:
        if _RUN_STATE is not None:
            path = _RUN_STATE.output / 'report.json'
            report = _read_json(path) if path.is_file() else {}
            report.update(decision='execution_error', error_type=type(exc).__name__, error=str(exc))
            _write_json(path, report)
        raise


if __name__ == '__main__':
    main()
