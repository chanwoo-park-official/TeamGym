"""Distribution and export gates, separable from the construction driver.

Dependencies are resolved lazily from this module's package, so a separately
versioned finalizer can execute the stage against its frozen producer package.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet_audit_source_sha256(source):
    """Bind only the complete audit evidence used by content-scoped reuse."""
    source = Path(source)
    files = [source / name for name in ('run_identity.json', 'evaluation_bundle.hidden.json',
                                       'packet_audit.json', 'checkpoints.json')]
    if (source / 'finalizer_runtime.json').is_file():
        files.append(source / 'finalizer_runtime.json')
    files += sorted(p for p in (source / 'packet_audit').rglob('*') if p.is_file())
    return _json_hash({str(p.relative_to(source)): _file_hash(p) for p in files})


def _audit_config(cfg):
    from teamgym.llm.openai_client import normalize_model_for_provider
    result = {key: getattr(cfg, key) for key in ('model', 'provider', 'base_url', 'endpoint',
        'reasoning_effort', 'temperature', 'max_output_tokens', 'api_key_env', 'env_file')}
    result['model'] = normalize_model_for_provider(result['provider'], result['model'])
    return result


def _inherit_packet_review(source, candidate, unit, system, cfg):
    """Reuse a clear component fact, never the rejected bundle's decision.

    The sole ignored field is the opaque global bundle digest. Every actual
    semantic input, inference setting, and retained output must agree.
    """
    from .packet_audit import audit_request, parse_audit_response
    from .model_config import model_config
    source = Path(source)
    lineage = {'method': 'fresh_review', 'source': str(source),
               'ignored_fields': ['bundle_sha256']}
    used = []
    def read(path):
        used.append(path)
        return json.loads(path.read_text())
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)
    try:
        import importlib
        import importlib.metadata
        identity = read(source / 'run_identity.json')
        original = identity['config']
        original_cfg = model_config(model=original['constructor_model'], env_file=original.get('env_file'),
            provider=original['provider'], base_url=original['base_url'],
            api_key_env=original['api_key_env'], endpoint=original.get('endpoint', '/responses'),
            reasoning_effort=original['reasoning_effort'])
        require(identity['code']['files']['teamgym/data/software/deep_swe/model_config.py'] ==
                _file_hash(Path(model_config.__code__.co_filename)), 'source_model_config_code_changed')
        for name in ('construction_client', 'openai_client'):
            module = importlib.import_module(f'teamgym.llm.{name}')
            require(identity['code']['files'][f'teamgym/llm/{name}.py'] == _file_hash(module.__file__),
                    'source_constructor_transport_changed')
        require(_audit_config(original_cfg) == _audit_config(cfg), 'constructor_config_changed')
        runtime = source / 'finalizer_runtime.json'
        if runtime.is_file():
            packages = read(runtime)['package_versions']
            current_packages = {}
            for name in packages:
                try:
                    current_packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    current_packages[name] = None
            require(current_packages == packages, 'constructor_runtime_packages_changed')
            lineage['source_runtime_packages'] = packages
            lineage['current_runtime_packages'] = current_packages
        bundle = read(source / 'evaluation_bundle.hidden.json')
        receipt = read(source / 'packet_audit.json')
        checkpoints = read(source / 'checkpoints.json')
        checkpoint_names = ['packet_audit.json', 'evaluation_bundle.hidden.json', 'run_identity.json']
        if runtime.is_file() or 'finalizer_runtime.json' in checkpoints:
            checkpoint_names.append('finalizer_runtime.json')
        for name in checkpoint_names:
            require(checkpoints.get(name) == _file_hash(source / name), 'source_checkpoint_mismatch')
        expected = audit_request(candidate, bundle)
        require(receipt['request'] == expected and receipt['request_sha256'] == _json_hash(expected)
                and receipt['candidate_sha256'] == expected['candidate_sha256']
                and receipt['bundle_sha256'] == _json_hash(bundle), 'source_global_request_mismatch')
        keys = {(v['mode'], v['participant_id']) for v in expected['views']}
        parsed_global = parse_audit_response(receipt['response'], keys)
        require(receipt['response_sha256'] == _json_hash(parsed_global), 'source_global_response_hash')
        clear = not any(a['issue'] for v in parsed_global['views'] for a in v['assessments'].values())
        require(receipt['passed'] is clear, 'source_global_decision_mismatch')
        key = (unit['views'][0]['mode'], unit['views'][0]['participant_id'])
        matches = [(i, v) for i, v in enumerate(expected['views'], 1)
                   if (v['mode'], v['participant_id']) == key]
        require(len(matches) == 1, 'source_view_not_unique')
        index, view = matches[0]
        old_unit = {**expected, 'views': [view], 'exact_findings': [r for r in expected['exact_findings']
                    if (r['mode'], r['participant_id']) == key]}
        require(read(source / f'packet_audit/{index:02d}.request.json') == old_unit, 'source_unit_changed')
        normalized_old = {k: v for k, v in old_unit.items() if k != 'bundle_sha256'}
        normalized_current = {k: v for k, v in unit.items() if k != 'bundle_sha256'}
        require(json.dumps(normalized_old, ensure_ascii=False) ==
                json.dumps(normalized_current, ensure_ascii=False), 'semantic_request_changed')
        response = read(source / f'packet_audit/{index:02d}.response.json')
        response = parse_audit_response(response, {key})
        require(response['views'] == [v for v in parsed_global['views']
            if (v['mode'], v['participant_id']) == key], 'source_unit_response_changed')
        require(not any(a['issue'] for a in response['views'][0]['assessments'].values()), 'source_view_rejected')
        users = list((source / 'packet_audit/llm_traces').glob(f'deep_swe.packet_audit.view_{index}.*.user.txt'))
        require(len(users) == 1, 'source_trace_not_unique')
        user = users[0]
        prefix = str(user)[:-len('.user.txt')]
        system_path = Path(prefix + '.system.txt')
        used.extend([user, system_path])
        require(user.read_bytes() == json.dumps(old_unit, ensure_ascii=False).encode('utf-8'), 'actual_user_trace_changed')
        require(system_path.read_bytes() == system.encode('utf-8'), 'actual_system_prompt_changed')
        require(read(Path(prefix + '.parsed.json')) == response, 'actual_parsed_trace_changed')
        outputs = list(user.parent.glob(Path(prefix).name + '.attempt_*.output.txt'))
        require(len(outputs) == 1, 'source_output_attempt_not_unique')
        output = outputs[0]
        used.append(output)
        require(json.loads(output.read_text()) == response, 'actual_output_trace_changed')
        raw = read(Path(str(output)[:-len('.output.txt')] + '.raw_response.json'))
        require(raw.get('status') == 'completed' and raw.get('error') is None
            and raw.get('model') == _audit_config(cfg)['model']
            and raw.get('reasoning', {}).get('effort') == cfg.reasoning_effort
            and raw.get('max_output_tokens') == cfg.max_output_tokens
            and raw.get('temperature') == (1.0 if cfg.temperature is None else cfg.temperature),
            'actual_response_inference_changed')
        raw_text = raw.get('output_text')
        if not isinstance(raw_text, str):
            raw_text = '\n'.join(c['text'] for item in raw.get('output', []) for c in item.get('content', [])
                                 if isinstance(c.get('text'), str))
        require(raw_text.encode('utf-8') == output.read_bytes(), 'actual_response_output_changed')
        lineage.update(method='inherited_clear_component', fresh_api_calls=0,
            source_view_index=index, source_bundle_sha256=expected['bundle_sha256'],
            current_bundle_sha256=unit['bundle_sha256'],
            source_normalized_request_sha256=_json_hash(normalized_old),
            current_normalized_request_sha256=_json_hash(normalized_current),
            actual_system_sha256=_file_hash(system_path), constructor_config=_audit_config(cfg),
            source_response_reported_temperature=raw['temperature'],
            source_files_sha256={str(p.relative_to(source)): _file_hash(p) for p in used})
        return response, lineage
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        lineage['fallback_reason'] = str(exc)
        return None, lineage


def finish_distribution(args, task, candidate, candidate_path, checks, checks_path,
                         discovered, observations, verifier, output, qualification_dir, report,
                        *, audit_system=None, pack_function=None, background_exclusions=None,
                        background_policy=None, background_language=None,
                        observation_background_exclusions=None, packet_audit_source=None):
    from .pipeline import (_read_json, _write_json, _run, _episode_receipts,
                           _verify_frozen_inputs, DeepSWEPipelineError, sha256)
    from .full_context import pack
    from .export import (MODES, build_evaluation_bundle, freeze_export, load_export,
                         make_validation_receipt, participant_view_candidates)
    from .packet_audit import AUDIT_SYSTEM, audit_evaluation_bundle
    from .model_config import model_config
    from teamgym.llm.construction_client import OpenAIJsonClient
    from .stages.patch_episode import packet_for

    if len(candidate['allocation_candidates']) < 2:
        report['decision'] = 'collaboration_rejected'
        report['reason'] = 'at least two functional PIUs are needed for a collaborative export'
        return
    files = {path: row['text'] for path, row in _read_json(discovered)['files'].items()}
    packing_options = {}
    if background_exclusions is not None:
        packing_options['background_exclusions'] = background_exclusions
    if observation_background_exclusions is not None:
        packing_options['observation_background_exclusions'] = observation_background_exclusions
    if background_policy is not None:
        packing_options['background_policy'] = background_policy
    if background_policy == 'neutral_repository':
        packing_options['background_language'] = (
            task.repository.language if background_language is None else background_language)
    participants, distribution = (pack_function or pack)(candidate, files, observations, args.packet_tokens,
        **packing_options)
    bundle = build_evaluation_bundle(candidate, checks, participants,
        patch_turns=5, total_output_tokens=args.total_output_tokens,
        max_output_tokens_per_call=args.max_output_tokens_per_call,
        max_checks_per_episode=args.max_checks_per_episode)
    _write_json(output / 'evaluation_bundle.hidden.json', bundle)
    report['stages']['distribution'] = distribution
    _write_json(output / 'distribution.hidden.json', distribution)
    cfg = model_config(model=args.constructor_model, env_file=str(args.env_file),
        provider=args.provider, base_url=args.base_url, api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort)
    client = None
    source_sha256 = packet_audit_source_sha256(packet_audit_source) if packet_audit_source is not None else None
    component_lineage = []

    def semantic_auditor(request):
        nonlocal client
        # Each call retains one complete view. Chunking removes other views,
        # never evidence from the packet being certified.
        responses = []
        trace = output / 'packet_audit'
        for index, view in enumerate(request['views'], 1):
            unit = {**request, 'views': [view], 'exact_findings': [r for r in request['exact_findings']
                if (r['mode'], r['participant_id']) == (view['mode'], view['participant_id'])]}
            _write_json(trace / f'{index:02d}.request.json', unit)
            system = AUDIT_SYSTEM if audit_system is None else audit_system
            response, lineage = (None, {'method': 'fresh_review'})
            if packet_audit_source is not None:
                response, lineage = _inherit_packet_review(packet_audit_source, candidate, unit, system, cfg)
            if response is None:
                if client is None:
                    client = OpenAIJsonClient(cfg)
                response = client.generate_json(prompt_id=f'deep_swe.packet_audit.view_{index}',
                    system_prompt=system, user_prompt=json.dumps(unit, ensure_ascii=False),
                    trace_dir=trace / 'llm_traces')
                lineage['fresh_api_calls'] = 1
            _write_json(trace / f'{index:02d}.lineage.json', lineage)
            component_lineage.append({'mode': view['mode'], 'participant_id': view['participant_id'], **lineage})
            _write_json(trace / f'{index:02d}.response.json', response)
            responses.extend(response.get('views', []))
        return {'views': responses}

    packet_audit = audit_evaluation_bundle(candidate, bundle, semantic_auditor=semantic_auditor)
    if packet_audit_source is not None and packet_audit_source_sha256(packet_audit_source) != source_sha256:
        raise DeepSWEPipelineError('packet audit source changed during review')
    _write_json(output / 'packet_audit_lineage.json', {'source_sha256': source_sha256,
        'components': component_lineage,
        'inherited_components': sum(r['method'] == 'inherited_clear_component' for r in component_lineage),
        'fresh_api_calls': sum(r['fresh_api_calls'] for r in component_lineage)})
    _write_json(output / 'packet_audit.json', packet_audit)
    report['stages']['packet_audit'] = {'passed': packet_audit['passed'],
                                      'receipt': str(output / 'packet_audit.json')}
    if not packet_audit['passed']:
        report['decision'] = 'packet_audit_rejected'
        return

    loo_dir = output / 'targeted_loo'
    targets = [r['riu_id'] for r in candidate['allocation_candidates']]
    _run('targeted_loo', [f'{__package__}.stages.diagnostic_runs', '--task-dir', str(task.source_dir),
        '--candidate', str(candidate_path), '--checks', str(checks_path), '--verifier-image', verifier,
        '--output-dir', str(loo_dir), '--patch-turns', '5', '--granularity', 'riu',
        '--repetitions', str(args.loo_repetitions), '--concurrency', str(args.concurrency),
        '--only', *targets], output / 'logs' / 'targeted_loo.log')
    loo = {'targets': [{'riu_id': target,
        'rationale': 'Inspect functional contribution under the fixed protocol; no mandatory failure threshold.',
        'episodes': _episode_receipts(loo_dir, label=target,
            expected_execution=candidate['execution_manifest'])} for target in targets]}
    report['stages']['targeted_loo'] = loo
    participant_results = []
    for mode, pid, view_candidate in participant_view_candidates(candidate, bundle):
        view_path = output / 'participant_candidates' / mode / f'{pid}.json'
        _write_json(view_path, view_candidate)
        view_dir = output / 'participant_diagnostics' / mode / pid
        _run(f'{mode}_{pid}', [f'{__package__}.stages.diagnostic_runs',
            '--task-dir', str(task.source_dir), '--candidate', str(view_path), '--checks', str(checks_path),
            '--verifier-image', verifier, '--output-dir', str(view_dir), '--patch-turns', '5',
            '--granularity', 'riu', '--repetitions', '3', '--concurrency', str(args.concurrency),
            '--only', 'baseline'], output / 'logs' / f'{mode}_{pid}.log')
        expected = next(r['packet'] for r in bundle['modes'][mode] if r['participant_id'] == pid)
        episodes = _episode_receipts(view_dir, expected_packet=expected,
            expected_execution=candidate['execution_manifest'])
        participant_results.append({'mode': mode, 'participant_id': pid, 'episodes': episodes})
    report['stages']['participant_only'] = participant_results
    if any(r['success'] for v in participant_results for r in v['episodes']):
        report['decision'] = 'participant_only_rejected'
        return
    screening_candidate = _read_json(output / 'candidate_instruction_only.json')
    screening = _episode_receipts(output / 'stage4a_instruction_only',
        expected_packet=packet_for(screening_candidate, checks, 'all'),
        expected_execution=candidate['execution_manifest'])
    qualification = _episode_receipts(qualification_dir,
        expected_packet=bundle['modes'][MODES[0]][0]['packet'],
        expected_execution=candidate['execution_manifest'])
    receipt = make_validation_receipt(candidate, bundle, screening=screening,
        qualification=qualification, loo=loo, participant_results=participant_results,
        packet_audit=packet_audit)
    _verify_frozen_inputs(task, args)
    export_path = output / 'export' / 'instance.json'
    freeze_export(export_path, candidate, bundle, receipt)
    loaded = load_export(export_path)
    for mode in MODES:
        for view in bundle['modes'][mode]:
            actual = loaded.render_model_input(mode, view['participant_id'])
            _write_json(output / 'export' / 'model_inputs' / mode / f"{view['participant_id']}.json", actual)
            if json.loads(actual['input'][1]['content'])['evidence'] != view['packet']:
                raise DeepSWEPipelineError('export consumer changed frozen evidence')
    report['decision'] = 'accepted'
    report['export'] = {'path': str(export_path), 'sha256': sha256(export_path),
                        'consumer': 'teamgym.data.software.deep_swe.export.load_export'}
