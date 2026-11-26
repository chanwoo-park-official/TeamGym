"""Synthetic offline traces test strict component reuse; no generation claims."""
from copy import deepcopy
import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.software.deep_swe import distribution, packet_audit
from teamgym.data.software.deep_swe.model_config import model_config
from tests.software.deep_swe.test_export import fixture, clean_review


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    traces = root / 'packet_audit/llm_traces'
    traces.mkdir(parents=True)
    candidate, bundle = fixture()
    request = packet_audit.audit_request(candidate, bundle)
    response = clean_review(request)
    # A rejected bundle may contain independently clear component facts.
    response['views'][-1]['assessments']['complete_repair_possible']['issue'] = True
    cfg = model_config(model='gpt-5.4', env_file=str(tmp_path / 'credential-placeholder.env'), provider='openai_compatible',
        base_url='https://example.invalid/v1', reasoning_effort='medium')
    system = 'Exact versioned audit system.\nReturn JSON.'
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False))
    identity = {'config': {'constructor_model': cfg.model, 'provider': cfg.provider,
        'base_url': cfg.base_url, 'api_key_env': cfg.api_key_env,
        'endpoint': cfg.endpoint, 'reasoning_effort': cfg.reasoning_effort, 'env_file': cfg.env_file},
        'code': {'files': {'teamgym/data/software/deep_swe/model_config.py':
            distribution._file_hash(Path(model_config.__code__.co_filename))}}}
    for name in ('construction_client', 'openai_client'):
        identity['code']['files'][f'teamgym/llm/{name}.py'] = distribution._file_hash(
            importlib.import_module(f'teamgym.llm.{name}').__file__)
    receipt = {'request': request, 'response': response,
        'request_sha256': distribution._json_hash(request),
        'response_sha256': distribution._json_hash(response),
        'candidate_sha256': request['candidate_sha256'], 'bundle_sha256': distribution._json_hash(bundle),
        'passed': False}
    for name, value in [('run_identity.json', identity), ('evaluation_bundle.hidden.json', bundle),
                        ('packet_audit.json', receipt)]:
        write(root / name, value)
    write(root / 'checkpoints.json', {name: distribution._file_hash(root / name)
        for name in ('run_identity.json', 'evaluation_bundle.hidden.json', 'packet_audit.json')})
    units = []
    for i, view in enumerate(request['views'], 1):
        key = (view['mode'], view['participant_id'])
        unit = {**request, 'views': [view], 'exact_findings': [r for r in request['exact_findings']
            if (r['mode'], r['participant_id']) == key]}
        units.append(unit)
        result = {'views': [response['views'][i - 1]]}
        write(root / f'packet_audit/{i:02d}.request.json', unit)
        write(root / f'packet_audit/{i:02d}.response.json', result)
        prefix = traces / f'deep_swe.packet_audit.view_{i}.synthetic'
        Path(str(prefix) + '.user.txt').write_text(json.dumps(unit, ensure_ascii=False))
        Path(str(prefix) + '.system.txt').write_text(system)
        write(Path(str(prefix) + '.parsed.json'), result)
        text = json.dumps(result, ensure_ascii=False)
        Path(str(prefix) + '.attempt_01.output.txt').write_text(text)
        write(Path(str(prefix) + '.attempt_01.raw_response.json'), {
            'status': 'completed', 'error': None, 'model': cfg.model,
            'reasoning': {'effort': cfg.reasoning_effort}, 'temperature': 1.0,
            'max_output_tokens': cfg.max_output_tokens,
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]})
    return SimpleNamespace(root=root, candidate=candidate, cfg=cfg, units=units, system=system)


def inherit(source, unit=None, system=None, cfg=None):
    return distribution._inherit_packet_review(source.root, source.candidate,
        deepcopy(source.units[0]) if unit is None else unit,
        source.system if system is None else system, source.cfg if cfg is None else cfg)


def test_rejected_bundle_can_supply_clear_component_with_only_opaque_hash_changed(source):
    unit = deepcopy(source.units[0]); unit['bundle_sha256'] = 'f' * 64
    result, lineage = inherit(source, unit)
    assert result and not any(a['issue'] for a in result['views'][0]['assessments'].values())
    assert lineage['method'] == 'inherited_clear_component' and lineage['fresh_api_calls'] == 0
    assert lineage['source_normalized_request_sha256'] == lineage['current_normalized_request_sha256']
    assert lineage['ignored_fields'] == ['bundle_sha256']
    assert lineage['constructor_config']['temperature'] is None
    assert lineage['source_response_reported_temperature'] == 1.0
    assert lineage['source_files_sha256']


@pytest.mark.parametrize('field', ['packet', 'owned_evidence', 'foreign_functional_code',
    'foreign_functional_observations', 'protocol', 'constructor_functional_boundaries',
    'candidate_sha256', 'exact_findings', 'limitations', 'new_semantic_field'])
def test_only_global_hash_is_ignored_no_visible_or_semantic_field_is_waived(source, field):
    unit = deepcopy(source.units[0])
    if field in unit['views'][0]:
        unit['views'][0][field] = 'changed semantic evidence'
    else:
        unit[field] = 'changed semantic input'
    result, lineage = inherit(source, unit)
    assert result is None and lineage['fallback_reason'] == 'semantic_request_changed'


@pytest.mark.parametrize('change', ['field_order', 'json_numeric_type'])
def test_normalized_request_serialization_preserves_order_and_json_types(source, change):
    unit = deepcopy(source.units[0])
    if change == 'field_order':
        unit = dict(reversed(list(unit.items())))
    else:
        unit['protocol']['patch_turns'] = float(unit['protocol']['patch_turns'])
    assert inherit(source, unit=unit)[1]['fallback_reason'] == 'semantic_request_changed'


@pytest.mark.parametrize('setting,value', [('model', 'gpt-5.5'), ('reasoning_effort', 'high'),
    ('temperature', 0), ('provider', 'openai'), ('base_url', 'https://other.invalid/v1'),
    ('endpoint', '/chat/completions'), ('max_output_tokens', 8000),
    ('api_key_env', 'OTHER_CREDENTIAL_NAMESPACE'), ('env_file', '/other-placeholder.env')])
def test_changed_constructor_inference_configuration_cannot_inherit(source, setting, value):
    cfg = deepcopy(source.cfg); setattr(cfg, setting, value)
    assert inherit(source, cfg=cfg)[0] is None


def test_changed_system_or_failed_view_cannot_inherit(source):
    assert inherit(source, system=source.system + ' changed')[0] is None
    assert inherit(source, unit=source.units[-1])[0] is None


def test_actual_system_bytes_do_not_normalize_different_line_endings(source):
    path = next((source.root / 'packet_audit/llm_traces').glob('*view_1*.system.txt'))
    path.write_bytes(source.system.replace('\n', '\r\n').encode('utf-8'))
    assert inherit(source)[1]['fallback_reason'] == 'actual_system_prompt_changed'


@pytest.mark.parametrize('trace', ['user.txt', 'system.txt', 'parsed.json', 'attempt_01.output.txt',
    'attempt_01.raw_response.json'])
def test_missing_or_changed_actual_trace_prevents_inheritance(source, trace):
    path = source.root / 'packet_audit/llm_traces' / ('deep_swe.packet_audit.view_1.synthetic.' + trace)
    path.write_text('{}')
    assert inherit(source)[0] is None


@pytest.mark.parametrize('field,value', [('model', 'wrong-model'), ('temperature', 0),
    ('max_output_tokens', 8000), ('reasoning', {'effort': 'high'}), ('status', 'incomplete'),
    ('output', [{'content': [{'text': '{}'}]}])])
def test_raw_actual_inference_and_output_must_match_recorded_control(source, field, value):
    path = next((source.root / 'packet_audit/llm_traces').glob('*view_1*.raw_response.json'))
    data = json.loads(path.read_text()); data[field] = value; path.write_text(json.dumps(data))
    assert inherit(source)[0] is None


def test_source_checkpoint_and_global_receipt_are_not_self_attested_shortcuts(source):
    path = source.root / 'packet_audit.json'
    data = json.loads(path.read_text()); data['passed'] = True; path.write_text(json.dumps(data))
    assert inherit(source)[0] is None
    checkpoints = json.loads((source.root / 'checkpoints.json').read_text())
    checkpoints['packet_audit.json'] = distribution._file_hash(path)
    (source.root / 'checkpoints.json').write_text(json.dumps(checkpoints))
    assert inherit(source)[1]['fallback_reason'] == 'source_global_decision_mismatch'


def test_source_fingerprint_includes_actual_trace_bytes(source):
    before = distribution.packet_audit_source_sha256(source.root)
    path = next((source.root / 'packet_audit/llm_traces').glob('*view_1*.user.txt'))
    path.write_text(path.read_text() + '\n')
    assert distribution.packet_audit_source_sha256(source.root) != before


@pytest.mark.parametrize('name', ['construction_client', 'openai_client'])
def test_constructor_transport_change_prevents_reuse_even_with_same_prompt_and_settings(source, name):
    path = source.root / 'run_identity.json'
    data = json.loads(path.read_text())
    data['code']['files'][f'teamgym/llm/{name}.py'] = 'f' * 64
    path.write_text(json.dumps(data))
    assert inherit(source)[1]['fallback_reason'] == 'source_constructor_transport_changed'


@pytest.mark.parametrize('change', ['tamper', 'missing'])
def test_optional_runtime_evidence_is_fingerprinted_and_checkpoint_bound(source, change):
    runtime = source.root / 'finalizer_runtime.json'
    runtime.write_text(json.dumps({'package_versions': {}, 'python': 'offline test'}))
    checkpoints = json.loads((source.root / 'checkpoints.json').read_text())
    checkpoints['finalizer_runtime.json'] = distribution._file_hash(runtime)
    (source.root / 'checkpoints.json').write_text(json.dumps(checkpoints))
    assert inherit(source)[0] is not None
    fingerprint = distribution.packet_audit_source_sha256(source.root)
    if change == 'missing':
        runtime.unlink()
    else:
        runtime.write_text(json.dumps({'package_versions': {}, 'python': 'changed runtime evidence'}))
    assert distribution.packet_audit_source_sha256(source.root) != fingerprint
    assert inherit(source)[0] is None


@pytest.mark.parametrize('mutate_source', [False, True])
def test_distribution_only_calls_model_for_failed_component_and_checks_source_drift(source, monkeypatch, tmp_path, mutate_source):
    from teamgym.data.software.deep_swe import pipeline
    output = tmp_path / 'output'; output.mkdir()
    discovered = output / 'discovered.json'
    discovered.write_text(json.dumps({'files': {a['path']: {'text': '# background ' + a['path'][0] + '\n' + a['text']}
                                               for a in source.candidate['code_atoms']}}))
    cfg = source.cfg
    args = SimpleNamespace(constructor_model=cfg.model, env_file=Path(cfg.env_file), provider=cfg.provider,
        base_url=cfg.base_url, api_key_env=cfg.api_key_env, reasoning_effort=cfg.reasoning_effort,
        packet_tokens=1000, total_output_tokens=32000, max_output_tokens_per_call=16000,
        max_checks_per_episode=15, loo_repetitions=1, concurrency=1)
    called = []
    class Client:
        def __init__(self, config): pass
        def generate_json(self, **kwargs):
            request = json.loads(kwargs['user_prompt'])
            called.append(request['views'][0]['participant_id'])
            if mutate_source:
                path = next((source.root / 'packet_audit/llm_traces').glob('*view_1*.user.txt'))
                path.write_text(path.read_text() + '\n')
            return clean_review(request)
    monkeypatch.setattr('teamgym.llm.construction_client.OpenAIJsonClient', Client)
    class ReachedDiagnostics(Exception): pass
    def run(*a, **kw): raise ReachedDiagnostics()
    monkeypatch.setattr(pipeline, '_run', run)
    expected_error = pipeline.DeepSWEPipelineError if mutate_source else ReachedDiagnostics
    with pytest.raises(expected_error):
        distribution.finish_distribution(args, SimpleNamespace(source_dir=tmp_path), source.candidate,
            output / 'candidate.json', {'syntax': ['python', '-m', 'compileall']}, output / 'checks.json',
            discovered, source.candidate['observations'], 'verifier', output, output / 'qualification',
            {'stages': {}}, audit_system=source.system, packet_audit_source=source.root)
    assert called == [source.units[-1]['views'][0]['participant_id']]
    if not mutate_source:
        lineage = json.loads((output / 'packet_audit_lineage.json').read_text())
        assert lineage['inherited_components'] == len(source.units) - 1
        assert lineage['fresh_api_calls'] == 1
