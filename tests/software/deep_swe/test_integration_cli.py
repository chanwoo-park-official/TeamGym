"""Real parser/API integration checks without API requests or Docker execution."""
from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from teamgym.data.software.deep_swe import pipeline
from teamgym.data.software.deep_swe.stages.patch_episode import packet_for
from tests.software.deep_swe.test_export import fixture, clean_review


PACKAGE = 'teamgym.data.software.deep_swe'


class Parsed(Exception):
    def __init__(self, namespace): self.namespace = namespace


def parse_module_only(monkeypatch, module_name, argv):
    """Stop directly after the module's *actual* argparse parser accepts argv."""
    module = importlib.import_module(module_name)
    original = argparse.ArgumentParser.parse_args
    def parse(parser, args=None, namespace=None):
        value = original(parser, args=args, namespace=namespace)
        raise Parsed(value)
    with monkeypatch.context() as context:
        context.setattr(argparse.ArgumentParser, 'parse_args', parse)
        context.setattr(sys, 'argv', [module_name, *argv])
        with pytest.raises(Parsed) as result:
            module.main()
    return result.value.namespace


@pytest.fixture
def options():
    return SimpleNamespace(env_file=Path('/credential-placeholder.env'), provider='openai_compatible',
        base_url='https://example.invalid/v1', api_key_env='PLACEHOLDER_API_KEY',
        reasoning_effort='high', runtime_image='sha256:' + 'a' * 64,
        constructor_model='constructor-placeholder', actor_model='actor-placeholder',
        total_output_tokens=32000, max_output_tokens_per_call=16000, max_checks_per_episode=15)


@pytest.mark.parametrize('suffix,required,expected', [
    ('discovery', ['--task-dir', '/task', '--output', '/output', '--model', 'constructor-placeholder'], 'constructor-placeholder'),
    ('construction', ['--task-dir', '/task', '--pool', '/pool', '--observations', '/obs',
        '--baseline-run', '/baseline', '--output', '/output', '--model', 'constructor-placeholder'], 'constructor-placeholder'),
    ('observation_proposals', ['--task-dir', '/task', '--pool', '/pool', '--output', '/output',
        '--model', 'constructor-placeholder'], 'constructor-placeholder'),
    ('observations', ['--task-dir', '/task', '--plan', '/plan', '--output-dir', '/output'], 'constructor-placeholder'),
    ('stages.build_deepswe_fragment_pool', ['--task-dir', '/task', '--discovered', '/discovered', '--output', '/output'], None),
    ('stages.diagnostic_runs', ['--task-dir', '/task', '--candidate', '/candidate', '--checks', '/checks',
        '--verifier-image', 'verifier', '--output-dir', '/output', '--patch-turns', '5',
        '--granularity', 'riu', '--only', 'baseline'], 'actor-placeholder'),
])
def test_pipeline_propagated_flags_are_accepted_by_real_stage_parsers(monkeypatch, tmp_path, options, suffix, required, expected):
    module_name = f'{PACKAGE}.{suffix}'
    captured = []
    def run(command, **kwargs):
        assert command[:2] == [sys.executable, '-m']
        namespace = parse_module_only(monkeypatch, command[2], command[3:])
        captured.append(namespace)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(pipeline, '_OPTIONS', options)
    monkeypatch.setattr(pipeline, '_RUN_STATE', None)
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    pipeline._run('parser-contract', [module_name, *required], tmp_path / 'task/logs/test.log')
    namespace = captured[0]
    assert namespace.runtime_image == options.runtime_image
    if suffix == 'stages.build_deepswe_fragment_pool': return
    assert str(namespace.env_file) == str(options.env_file)
    assert namespace.provider == options.provider and namespace.base_url == options.base_url
    assert namespace.api_key_env == options.api_key_env and namespace.reasoning_effort == options.reasoning_effort
    if suffix == 'observations': assert namespace.audit_model == expected
    elif suffix == 'stages.diagnostic_runs':
        assert namespace.model_slug == expected
        assert namespace.total_output_tokens == options.total_output_tokens
        assert namespace.max_output_tokens_per_call == options.max_output_tokens_per_call
        assert namespace.max_checks_per_episode == options.max_checks_per_episode
    else: assert namespace.model == expected


@pytest.mark.parametrize('suffix', ['discovery', 'construction', 'observation_proposals', 'observations',
                                  'stages.build_deepswe_fragment_pool', 'stages.diagnostic_runs', 'stages.patch_episode'])
def test_real_stage_help_runs_without_external_inputs(suffix):
    result = subprocess.run([sys.executable, '-m', f'{PACKAGE}.{suffix}', '--help'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--runtime-image' in result.stdout


def test_diagnostic_forwarded_actor_runtime_options_match_real_episode_parser(monkeypatch, options):
    argv = ['--task-dir', '/task', '--candidate', '/candidate', '--checks', '/checks',
            '--verifier-image', 'verifier', '--output-dir', '/out', '--patch-turns', '5',
            '--evidence-mode', 'all', '--model-slug', options.actor_model]
    for key in ['runtime_image', 'env_file', 'provider', 'base_url', 'api_key_env', 'reasoning_effort',
                'total_output_tokens', 'max_output_tokens_per_call', 'max_checks_per_episode']:
        argv.extend(['--' + key.replace('_', '-'), str(getattr(options, key))])
    namespace = parse_module_only(monkeypatch, f'{PACKAGE}.stages.patch_episode', argv)
    assert namespace.runtime_image == options.runtime_image and namespace.model_slug == options.actor_model


@pytest.mark.parametrize('entrypoint', ['wrapper_default', 'wrapper_override', 'external_override'])
@pytest.mark.parametrize('packing_case', ['default', 'explicit_default', 'explicit_exclusions',
                                        'explicit_policy', 'explicit_policy_exclusions',
                                        'observation_exclusions', 'all_exclusions'])
def test_finish_distribution_uses_actual_client_signature_and_packet_audit_schema(monkeypatch, tmp_path, options, entrypoint, packing_case):
    from teamgym.llm.construction_client import OpenAIJsonClient
    from teamgym.data.software.deep_swe import distribution, packet_audit
    explicit_prompt = 'Versioned packet audit rules supplied explicitly. Return JSON.'
    expected_prompt = packet_audit.AUDIT_SYSTEM if entrypoint == 'wrapper_default' else explicit_prompt
    actual_signature = inspect.signature(OpenAIJsonClient.generate_json)
    reviewed = []
    class Client:
        def __init__(self, cfg):
            self.cfg = cfg
            # Persist the complete hidden distribution/corpus before even
            # constructing the client that can issue the first packet call.
            assert json.loads((tmp_path / 'distribution.hidden.json').read_text()) == report['stages']['distribution']
        def generate_json(self, **kwargs):
            actual_signature.bind(self, **kwargs)
            assert kwargs['trace_dir'] is not None
            assert kwargs['system_prompt'] == expected_prompt
            request = json.loads(kwargs['user_prompt'])
            assert len(request['views']) == 1
            reviewed.append(request['views'][0])
            return clean_review(request)
    monkeypatch.setattr('teamgym.llm.construction_client.OpenAIJsonClient', Client)
    candidate, _ = fixture()
    files = {atom['path']: '# background ' + atom['path'].split('.')[0] + '\n' + atom['text']
             for atom in candidate['code_atoms']}
    discovered = tmp_path / 'discovered.json'
    discovered.write_text(json.dumps({'files': {p: {'text': text} for p, text in files.items()}}))
    options.packet_tokens = 1000
    options.loo_repetitions = 1
    options.concurrency = 1
    class ReachedDiagnostics(Exception): pass
    def run(stage, argv, log):
        assert stage == 'targeted_loo'
        targets = argv[argv.index('--only') + 1:]
        assert targets == [r['riu_id'] for r in candidate['allocation_candidates']]
        raise ReachedDiagnostics()
    monkeypatch.setattr(pipeline, '_run', run)
    report = {'stages': {}}
    finish = pipeline._finish_distribution
    prompt_options = {} if entrypoint == 'wrapper_default' else {'audit_system': explicit_prompt}
    packed = []
    if packing_case != 'default':
        from teamgym.data.software.deep_swe.full_context import pack
        exclusions = [{'participant_id': 'participant_01', 'path': candidate['code_atoms'][0]['path'],
            'line_start': 1, 'line_end': 1, 'reason': 'background formatting fixture'}]
        observation_exclusions = [{'participant_id': 'participant_01', 'observation_id': 'observation-01',
                                   'reason': 'foreign distractor fixture'}]
        def explicit_pack(*args, **kwargs):
            packed.append((args, kwargs))
            # Domain pack's removal behavior is tested separately. This adapter
            # verifies first-class dependency forwarding without changing gates.
            return pack(*args)
        prompt_options['pack_function'] = explicit_pack
        if packing_case in ('explicit_exclusions', 'explicit_policy_exclusions', 'all_exclusions'):
            prompt_options['background_exclusions'] = exclusions
        if packing_case in ('explicit_policy', 'explicit_policy_exclusions'):
            prompt_options['background_policy'] = 'neutral_repository'
        if packing_case in ('observation_exclusions', 'all_exclusions'):
            prompt_options['observation_background_exclusions'] = observation_exclusions
    if entrypoint == 'external_override':
        # The finalizer may load this stage by path under a frozen package
        # namespace, while its relative imports continue to resolve there.
        spec = importlib.util.spec_from_file_location(f'{PACKAGE}.external_distribution_phase', distribution.__file__)
        external = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(external)
        assert external.__package__ == PACKAGE
        finish = external.finish_distribution
    with pytest.raises(ReachedDiagnostics):
        finish(options, SimpleNamespace(source_dir=tmp_path, repository=SimpleNamespace(language='go')), candidate,
            tmp_path / 'candidate.json', {'syntax': ['python', '-m', 'compileall']}, tmp_path / 'checks.json',
            discovered, candidate['observations'], 'verifier', tmp_path, tmp_path / 'qualification', report,
            **prompt_options)
    assert report['stages']['packet_audit']['passed']
    assert len(reviewed) == len(candidate['allocation_candidates']) * 2
    if packing_case != 'default':
        assert len(packed) == 1 and len(packed[0][0]) == 4
        assert packed[0][0][0] is candidate
        expected_options = {}
        if packing_case in ('explicit_exclusions', 'explicit_policy_exclusions', 'all_exclusions'):
            expected_options['background_exclusions'] = exclusions
        if packing_case in ('explicit_policy', 'explicit_policy_exclusions'):
            expected_options['background_policy'] = 'neutral_repository'
            expected_options['background_language'] = 'go'
        if packing_case in ('observation_exclusions', 'all_exclusions'):
            expected_options['observation_background_exclusions'] = observation_exclusions
        assert packed[0][1] == expected_options


@pytest.mark.parametrize('language,override,policy,expected', [
    ('go', None, 'neutral_repository', 'go'),
    ('typescript', None, 'neutral_repository', 'typescript'),
    ('python', None, 'neutral_repository', 'python'),
    ('rust', None, 'neutral_repository', 'rust'),
    ('javascript', None, 'neutral_repository', 'javascript'),
    ('go', 'python', 'neutral_repository', 'python'),
    ('go', 'python', 'discovered', None),
    ('go', 'python', None, None),
])
def test_distribution_forwards_declared_or_overridden_language_only_for_neutral_background(
        tmp_path, options, language, override, policy, expected):
    candidate, _ = fixture()
    discovered = tmp_path / 'discovered.json'
    discovered.write_text(json.dumps({'files': {'a.py': {'text': 'public base source\n'}}}))
    options.packet_tokens = 40000
    captured = []
    class ReachedPacking(Exception): pass
    def pack(*args, **kwargs):
        captured.append((args, kwargs))
        raise ReachedPacking()
    with pytest.raises(ReachedPacking):
        pipeline._finish_distribution(options,
            SimpleNamespace(source_dir=tmp_path, repository=SimpleNamespace(language=language)),
            candidate, tmp_path / 'candidate.json', {}, tmp_path / 'checks.json',
            discovered, [], 'verifier', tmp_path, tmp_path / 'qualification', {'stages': {}},
            pack_function=pack, background_policy=policy, background_language=override)
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[0] is candidate and args[1] == {'a.py': 'public base source\n'}
    assert kwargs.get('background_language') == expected
    assert ('background_language' in kwargs) is (expected is not None)
    assert ('background_policy' in kwargs) is (policy is not None)


def test_neutral_distribution_rejects_unsupported_language_before_model_or_docker(monkeypatch, tmp_path, options):
    from teamgym.data.software.deep_swe import full_context
    candidate, _ = fixture()
    discovered = tmp_path / 'discovered.json'
    discovered.write_text(json.dumps({'files': {a['path']: {'text': '\n' + a['text']}
                                              for a in candidate['code_atoms']}}))
    options.packet_tokens = 40000
    def unexpected(*a, **kw):
        pytest.fail('Unsupported language must fail before model or container execution')
    monkeypatch.setattr('teamgym.llm.construction_client.OpenAIJsonClient', unexpected)
    monkeypatch.setattr(full_context.subprocess, 'run', unexpected)
    with pytest.raises(ValueError, match='neutral_repository_unsupported_language'):
        pipeline._finish_distribution(options,
            SimpleNamespace(source_dir=tmp_path, repository=SimpleNamespace(language='unknown-language')),
            candidate, tmp_path / 'candidate.json', {}, tmp_path / 'checks.json',
            discovered, candidate['observations'], 'verifier', tmp_path, tmp_path / 'qualification',
            {'stages': {}}, background_policy='neutral_repository')


@pytest.mark.parametrize('policy', [None, 'discovered', 'neutral_repository'])
def test_native_pipeline_parser_accepts_background_controls_without_external_inputs(monkeypatch, policy):
    argv = ['--task-id', 'example', '--output-root', '/output', '--background-exclusions', '/exclusions.json',
            '--observation-background-exclusions', '/observation-exclusions.json',
            '--packet-audit-source', '/prior-audit']
    if policy is not None:
        argv += ['--background-policy', policy]
    args = parse_module_only(monkeypatch, f'{PACKAGE}.pipeline', argv)
    assert args.background_exclusions == Path('/exclusions.json')
    assert args.observation_background_exclusions == Path('/observation-exclusions.json')
    assert args.packet_audit_source == Path('/prior-audit')
    assert args.background_policy == (policy or 'neutral_repository')


def test_all_and_participant_packets_match_actual_episode_renderer():
    from teamgym.data.software.deep_swe.export import MODES, participant_view_candidates
    candidate, bundle = fixture()
    checks = {r['id']: r['argv'] for r in bundle['modes'][MODES[0]][0]['packet']['public_check_catalog']}
    assert packet_for(candidate, checks, 'all') == bundle['modes'][MODES[0]][0]['packet']
    for mode, pid, reduced in participant_view_candidates(candidate, bundle):
        expected = next(v['packet'] for v in bundle['modes'][mode] if v['participant_id'] == pid)
        assert packet_for(reduced, checks, 'all') == expected


def complete_episode_cache(episode, *, execution=None, label='baseline', rep=1):
    from teamgym.data.software.deep_swe.integrity import write_manifest
    from teamgym.data.software.deep_swe.stages.diagnostic_runs import EPISODE_OUTPUTS
    for name in EPISODE_OUTPUTS:
        path = episode / name
        if not path.exists():
            path.write_text('{}' if name.endswith('.json') else 'frozen harness\n' if name.endswith('.py') else '')
    result = json.loads((episode / 'result.json').read_text())
    result.setdefault('reward', {'reward': 0, 'f2p_passed': 0, 'f2p_total': 1, 'p2p_passed': 1, 'p2p_total': 1})
    result.setdefault('exit_code', 0)
    episode_data = json.loads((episode / 'episode.json').read_text())
    episode_data.setdefault('termination', 'patch_turn_budget')
    episode_data['patch_sha256'] = pipeline.sha256(episode / 'model.patch')
    (episode / 'episode.json').write_text(json.dumps(episode_data))
    result['episode'] = episode_data
    (episode / 'result.json').write_text(json.dumps(result))
    manifest = json.loads((episode / 'manifest.json').read_text())
    manifest.setdefault('candidate_sha256', pipeline.sha256(episode / 'candidate.frozen.hidden.json'))
    manifest.setdefault('checks_sha256', 'c' * 64)
    manifest.setdefault('harness_sha256', pipeline.sha256(episode / 'harness.frozen.py'))
    (episode / 'manifest.json').write_text(json.dumps(manifest))
    inputs = {'candidate_sha256': manifest['candidate_sha256'], 'checks_sha256': manifest['checks_sha256'],
              'condition': {'label': label, 'rep': rep}}
    if execution is not None:
        inputs.update(task=deepcopy(execution['source']), code=deepcopy(execution['code']),
            actor=deepcopy(execution['actor']), candidate_runtime=deepcopy(execution['runtime_image']),
            task_runtime=deepcopy(execution['runtime_image']), verifier=deepcopy(execution['verifier_image']),
            protocol={**execution['protocol'], 'evidence_mode': 'all'})
    write_manifest(episode / 'cache_manifest.json', inputs, outputs=[episode / name for name in EPISODE_OUTPUTS])


def test_episode_receipts_reject_actor_packet_and_summary_drift(tmp_path):
    from teamgym.data.software.deep_swe.run_state import sha256
    episode = tmp_path / 'baseline_rep1'; episode.mkdir()
    packet = {'instruction': 'same frozen input'}
    (episode / 'actor_packet.json').write_text(json.dumps(packet))
    (episode / 'result.json').write_text(json.dumps({'success': False, 'episode_valid': True}))
    (episode / 'manifest.json').write_text(json.dumps({'actor_packet_sha256': sha256(episode / 'actor_packet.json')}))
    complete_episode_cache(episode)
    rows = [{'label': 'baseline', 'rep': 1, 'output_dir': str(episode), 'success': False, 'episode_valid': True}]
    (tmp_path / 'results.json').write_text(json.dumps(rows))
    assert len(pipeline._episode_receipts(tmp_path, expected_packet=packet)) == 1
    with pytest.raises(pipeline.DeepSWEPipelineError, match='frozen evaluation packet'):
        pipeline._episode_receipts(tmp_path, expected_packet={'instruction': 'different'})
    rows[0]['success'] = True
    (tmp_path / 'results.json').write_text(json.dumps(rows))
    with pytest.raises(pipeline.DeepSWEPipelineError, match='summary disagree'):
        pipeline._episode_receipts(tmp_path, expected_packet=packet)
    (episode / 'actor_packet.json').write_text(json.dumps({'instruction': 'tampered'}))
    with pytest.raises(pipeline.DeepSWEPipelineError, match='packet changed'):
        pipeline._episode_receipts(tmp_path)


@pytest.mark.parametrize('drift', ['source', 'code', 'runtime', 'verifier', 'checks', 'lock',
                                   'background_exclusions', 'observation_background_exclusions', 'packet_audit_source'])
def test_final_input_drift_check_covers_every_live_execution_dependency(monkeypatch, tmp_path, drift):
    from teamgym.data.software.deep_swe import integrity
    from teamgym.data.software.deep_swe.run_state import sha256
    (tmp_path / 'uv.lock').write_text('frozen dependency lock')
    checks = tmp_path / 'checks.json'; checks.write_text('{}')
    exclusions = tmp_path / 'exclusions.json'; exclusions.write_text('[]')
    observation_exclusions = tmp_path / 'observation-exclusions.json'; observation_exclusions.write_text('[]')
    audit_source = tmp_path / 'audit-source'; audit_source.mkdir()
    for name in ('run_identity.json', 'evaluation_bundle.hidden.json', 'packet_audit.json', 'checkpoints.json'):
        (audit_source / name).write_text('{}')
    (audit_source / 'packet_audit').mkdir()
    (audit_source / 'packet_audit/01.request.json').write_text('{}')
    from teamgym.data.software.deep_swe.distribution import packet_audit_source_sha256
    actual = {'source': {'tree': 'source'}, 'code': {'sha256': 'code'},
              'runtime': 'runtime-id', 'verifier': 'verifier-id'}
    identity = {'source': deepcopy(actual['source']), 'code': deepcopy(actual['code']),
                'runtime_image': {'id': actual['runtime']}, 'verifier_image': {'id': actual['verifier']},
                'environment': {'uv_lock_sha256': sha256(tmp_path / 'uv.lock')},
                'public_checks_sha256': sha256(checks), 'background_exclusions_sha256': sha256(exclusions),
                'observation_background_exclusions_sha256': sha256(observation_exclusions),
                'packet_audit_source_sha256': packet_audit_source_sha256(audit_source)}
    monkeypatch.setattr(pipeline, 'WORKSPACE', tmp_path)
    monkeypatch.setattr(pipeline, '_RUN_STATE', SimpleNamespace(identity=identity))
    monkeypatch.setattr(integrity, 'task_identity', lambda _: actual['source'])
    monkeypatch.setattr(integrity, 'code_identity', lambda: actual['code'])
    monkeypatch.setattr(integrity, 'image_identity', lambda image: {'id': actual[image]})
    args = SimpleNamespace(runtime_image='runtime', verifier_image='verifier', public_checks=checks,
                           background_exclusions=exclusions, observation_background_exclusions=observation_exclusions,
                           packet_audit_source=audit_source)
    task = SimpleNamespace(source_dir=tmp_path / 'source')
    pipeline._verify_frozen_inputs(task, args)
    if drift in ['source', 'code']: actual[drift] = {'changed': True}
    elif drift in ['runtime', 'verifier']: actual[drift] += '-changed'
    elif drift == 'checks': checks.write_text('{"changed": true}')
    elif drift == 'background_exclusions': exclusions.write_text('[{"changed": true}]')
    elif drift == 'observation_background_exclusions': observation_exclusions.write_text('[{"changed": true}]')
    elif drift == 'packet_audit_source': (audit_source / 'packet_audit/01.request.json').write_text('{"changed": true}')
    else: (tmp_path / 'uv.lock').write_text('changed lock')
    with pytest.raises(pipeline.DeepSWEPipelineError, match='inputs changed'):
        pipeline._verify_frozen_inputs(task, args)


@pytest.mark.parametrize('field', ['model', 'provider', 'endpoint', 'reasoning_effort', 'source_base_commit',
                                   'runtime_image_id', 'verifier_image_id', 'max_output_tokens_per_call',
                                   'total_output_token_budget', 'max_public_checks_per_episode', 'harness_sha256'])
def test_episode_receipts_verify_actual_frozen_execution_fields(tmp_path, field):
    from teamgym.data.software.deep_swe.run_state import sha256 as file_sha256
    from teamgym.data.software.deep_swe.export import sha256 as object_sha256
    from teamgym.llm.openai_client import normalize_model_for_provider
    from tests.software.deep_swe.test_export import execution_fixture
    execution = execution_fixture()
    execution['code']['files']['teamgym/data/software/deep_swe/stages/patch_episode.py'] = hashlib.sha256(b'frozen harness\n').hexdigest()
    execution['code']['sha256'] = object_sha256(execution['code']['files'])
    actor, protocol = execution['actor'], execution['protocol']
    episode = tmp_path / 'baseline_rep1'; episode.mkdir()
    packet = {'instruction': 'same frozen packet'}
    (episode / 'actor_packet.json').write_text(json.dumps(packet))
    (episode / 'candidate.frozen.hidden.json').write_text(json.dumps({'execution_manifest': execution}))
    (episode / 'result.json').write_text(json.dumps({'success': False, 'episode_valid': True}))
    manifest = {'actor_packet_sha256': file_sha256(episode / 'actor_packet.json'), 'task_id': 'fixture',
        'model': normalize_model_for_provider(actor['provider'], actor['model']), 'provider': actor['provider'],
        'base_url': actor['base_url'], 'endpoint': actor['endpoint'], 'reasoning_effort': actor['reasoning_effort'],
        'patch_turn_budget': protocol['patch_turns'], 'total_output_token_budget': protocol['total_output_tokens'],
        'max_output_tokens_per_call': protocol['max_output_tokens_per_call'],
        'max_public_checks_per_episode': protocol['max_checks_per_episode'],
        'runtime_image_id': execution['runtime_image']['id'], 'verifier_image_id': execution['verifier_image']['id'],
        'source_base_commit': execution['source']['base_commit'],
        'harness_sha256': execution['code']['files']['teamgym/data/software/deep_swe/stages/patch_episode.py']}
    (episode / 'manifest.json').write_text(json.dumps(manifest))
    complete_episode_cache(episode, execution=execution)
    (tmp_path / 'results.json').write_text(json.dumps([
        {'label': 'baseline', 'rep': 1, 'output_dir': str(episode), 'success': False, 'episode_valid': True}]))
    receipts = pipeline._episode_receipts(tmp_path, expected_packet=packet, expected_execution=execution)
    assert receipts[0]['execution_manifest_sha256'] == object_sha256(execution)
    manifest[field] = 'changed'
    (episode / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(pipeline.DeepSWEPipelineError, match='declared execution protocol'):
        pipeline._episode_receipts(tmp_path, expected_packet=packet, expected_execution=execution)
