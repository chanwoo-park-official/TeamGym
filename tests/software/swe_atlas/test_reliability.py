"""Offline adversarial tests for SWE Atlas provenance and criterion boundaries.

Model/Docker responses are mocked explicitly; none of these are live acceptance.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.software.swe_atlas import candidate, export_common, integrity, pipeline
from teamgym.data.software.swe_atlas.schema import Rubric
from tests.software.swe_atlas.test_pipeline import _task, _bundle


def _write(path, value):
    integrity.atomic_json(Path(path), value)


def test_full_release_reselects_when_audited_rius_share_allocation_artifacts(tmp_path, monkeypatch):
    task = replace(_task(), source_dir=tmp_path / "source")
    task.source_dir.mkdir()
    client = SimpleNamespace(cfg=SimpleNamespace(model="offline"))
    monkeypatch.setattr(integrity, "image_identity", lambda _: {"id": "sha256:fixture"})

    selections = []

    def generate(**kwargs):
        candidates, artifacts = _bundle()
        if not selections:
            artifacts[1]["path"] = artifacts[0]["path"]
            candidates[0]["evidence_mode"] = "hybrid"
            candidates[0]["evidence"].append(deepcopy(candidates[1]["evidence"][1]))
        selections.append(kwargs)
        _write(kwargs["output_dir"] / "evidence_artifacts.json", {"artifacts": artifacts})
        return {"task_id": task.task_id, "candidates": candidates}

    def consolidate(**kwargs):
        return {"groups": [
            {"group_id": row["candidate_id"], "candidate_ids": [row["candidate_id"]],
             "hidden_claim": row["claim"], "selected_evidence": row["evidence"]}
            for row in kwargs["candidate_result"]["candidates"]
        ]}

    audits = []

    def audit(**kwargs):
        audits.append(kwargs["rius"])
        return {"accept": True, "errors": [], "all": {"judge_result": {}},
                "none": {"judge_result": {}}, "leave_one_out": []}

    class ReachedPartitionPreparation(Exception):
        pass

    def stop_at_log_preparation(**kwargs):
        assert len(selections) == 2 and len(audits) == 2
        assert kwargs["rius"] == audits[1]
        assert "collapse all collaborative RIUs" in selections[1]["construction_feedback"]["reason"]
        raise ReachedPartitionPreparation

    monkeypatch.setattr(pipeline, "generate_riu_candidates", generate)
    monkeypatch.setattr(pipeline, "_consolidate_candidates", consolidate)
    monkeypatch.setattr(pipeline, "run_riu_audit", audit)
    monkeypatch.setattr(pipeline.DockerEvidenceCollector, "inventory", lambda _: {})
    monkeypatch.setattr(pipeline.DockerEvidenceCollector, "tool_paths", lambda _: {})
    monkeypatch.setattr(pipeline, "build_experiment_logs", stop_at_log_preparation)
    output = tmp_path / "run"
    with pytest.raises(ReachedPartitionPreparation):
        pipeline.build_task(task=task, output_dir=output, client=client,
            config=pipeline.PipelineConfig(audit_repair_attempts=0, riu_reselection_attempts=1))

    first = output / "04_riu_audit/selection_00/attempt_00"
    assert json.loads((first / "summary.json").read_text())["accept"] is True
    assert json.loads((first / "partitionability_failure.json").read_text())["overlap_groups"] == [["riu_01", "riu_02"]]


@pytest.fixture
def bound_build(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'raw_task.json').write_text('{"original":"source"}')
    task = replace(_task(), source_dir=source)
    state = {'image_id': 'sha256:original'}
    monkeypatch.setattr(integrity, 'image_identity', lambda _: {'id': state['image_id']})
    client = SimpleNamespace(cfg=SimpleNamespace(model='original-model', provider='offline'))
    config = pipeline.PipelineConfig(stop_after_riu_audit=True, seed=7)
    identity = integrity.run_identity(task, client, config.checked(), docker_timeout_sec=900)
    output = tmp_path / 'build'
    integrity.prepare_run(output, identity)
    _write(output / '05_final_rius/audit_summary.json', {'accept': True, 'errors': []})
    _write(output / '05_final_rius/final_rius.hidden.json', {'rius': [{'riu_id': 'old-success'}]})
    integrity.write_checkpoint(output, identity)
    # Any stale-checkpoint adoption would reach candidate/model work, which is forbidden.
    monkeypatch.setattr(pipeline, 'generate_riu_candidates', lambda **kwargs:
        (_ for _ in ()).throw(AssertionError('stale checkpoint reached model work')))
    return task, client, config, output, state


@pytest.mark.parametrize('change', ['raw_source', 'question', 'base_commit', 'reference_answer',
                                   'rubric', 'model', 'image', 'seed', 'context_budget',
                                   'provider', 'reasoning', 'harness', 'prompt'])
def test_current_build_rejects_old_acceptance_when_any_bound_input_changes(bound_build, change, monkeypatch):
    task, client, config, output, state = bound_build
    if change == 'raw_source':
        (task.source_dir / 'raw_task.json').write_text('{"modified":"same Git status"}')
    elif change == 'question':
        task = replace(task, public=replace(task.public, question='New question with same task ID'))
    elif change == 'base_commit':
        task = replace(task, repository=replace(task.repository, base_commit='different-base'))
    elif change == 'reference_answer':
        task = replace(task, evaluation=replace(task.evaluation, reference_answer='New reference'))
    elif change == 'rubric':
        task = replace(task, evaluation=replace(task.evaluation, rubrics=(
            Rubric('r1', 'different criterion', 'positive', 'must have', {}),)))
    elif change == 'model':
        client.cfg.model = 'new-model'
    elif change == 'image':
        state['image_id'] = 'sha256:changed-tag-target'
    elif change == 'seed':
        config = replace(config, seed=8)
    elif change == 'provider':
        client.cfg.provider = 'different-provider'
    elif change == 'reasoning':
        client.cfg.reasoning_effort = 'high'
    elif change in {'harness', 'prompt'}:
        original_digest = integrity.digest
        target = 'candidate.py' if change == 'harness' else 'candidate_plan_system.txt'
        monkeypatch.setattr(integrity, 'digest', lambda path:
            'changed-content-sha256' if Path(path).name == target else original_digest(path))
    else:
        config = replace(config, participant_context_token_cap=10)
    with pytest.raises(integrity.ConstructionIdentityError, match='inputs changed'):
        pipeline.build_task(task=task, output_dir=output, client=client, config=config)


@pytest.mark.parametrize('damage', ['missing_identity', 'corrupt_identity', 'missing_receipt',
                                   'corrupt_receipt', 'changed_audit', 'changed_rius'])
def test_legacy_or_tampered_accepted_checkpoint_is_rejected_before_model_work(bound_build, damage):
    task, client, config, output, _ = bound_build
    if damage == 'missing_identity':
        (output / 'input_identity.json').unlink()
    elif damage == 'corrupt_identity':
        (output / 'input_identity.json').write_text('{')
    elif damage == 'missing_receipt':
        (output / 'checkpoint_receipt.json').unlink()
    elif damage == 'corrupt_receipt':
        (output / 'checkpoint_receipt.json').write_text('{')
    elif damage == 'changed_audit':
        _write(output / '05_final_rius/audit_summary.json', {'accept': True, 'errors': ['edited after audit']})
    else:
        _write(output / '05_final_rius/final_rius.hidden.json', {'rius': [{'riu_id': 'different-success'}]})
    with pytest.raises(integrity.ConstructionIdentityError):
        pipeline.build_task(task=task, output_dir=output, client=client, config=config)


@pytest.fixture
def ready_build(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'instruction.md').write_text('original source instruction')
    task = replace(_task(), source_dir=source)
    output = tmp_path / 'ready'
    _write(output / 'input_identity.json', {'source': integrity.task_identity(task), 'model': 'original'})
    _write(output / 'pipeline_report.json', {'task_id': task.task_id, 'status': 'benchmark_ready',
        'all_none_loo_accept': True, 'semantic_leakage_accept': True, 'salience_accept': True})
    for path in ['05_final_rius/final_rius.hidden.json', '05_final_rius/audit_summary.json',
                 '05_final_rius/selected_artifacts.hidden.json', '07_partition/final_partition.hidden.json']:
        _write(output / path, {'frozen': True})
    integrity.write_ready_receipt(output, task)
    integrity.validate_ready_receipt(output, task)
    monkeypatch.setattr(export_common, 'load_qa_task', lambda _: task)
    return task, output


@pytest.mark.parametrize('change', ['source', 'question', 'commit', 'missing_receipt',
                                   'identity', 'partition', 'selected_artifacts', 'audit'])
def test_export_refuses_ready_flags_with_stale_or_tampered_provenance(ready_build, monkeypatch, change):
    task, output = ready_build
    if change == 'source':
        (task.source_dir / 'instruction.md').write_text('changed source bytes')
    elif change in {'question', 'commit'}:
        task = (replace(task, public=replace(task.public, question='new question')) if change == 'question'
                else replace(task, repository=replace(task.repository, base_commit='new-commit')))
        monkeypatch.setattr(export_common, 'load_qa_task', lambda _: task)
    elif change == 'missing_receipt':
        (output / 'ready_receipt.json').unlink()
    elif change == 'identity':
        payload = json.loads((output / 'input_identity.json').read_text())
        payload['model'] = 'new model after acceptance'
        _write(output / 'input_identity.json', payload)
    else:
        path = {'partition': '07_partition/final_partition.hidden.json',
                'selected_artifacts': '05_final_rius/selected_artifacts.hidden.json',
                'audit': '05_final_rius/audit_summary.json'}[change]
        _write(output / path, {'frozen': False})
    with pytest.raises(export_common.ExportError, match='ready receipt'):
        export_common.export_build(build_dir=output, task_dir=task.source_dir)


@pytest.mark.parametrize("failed_recollections", [0, 1, 2])
def test_same_local_artifact_ids_can_reselect_new_source_without_failed_history_shadowing(
    tmp_path, monkeypatch, failed_recollections
):
    task = replace(_task(), source_dir=tmp_path / 'source')
    task.source_dir.mkdir()
    monkeypatch.setattr(integrity, 'image_identity', lambda _: {'id': 'sha256:offline'})
    clients = SimpleNamespace(cfg=SimpleNamespace(model='offline'))
    generated = []
    candidates, artifacts = _bundle()
    def generate(**kwargs):
        turn = len(generated)
        generated.append(kwargs)
        if 0 < turn <= failed_recollections:
            raise candidate.CandidateCollectionExhausted("Unverified evidence artifacts: ['probe']")
        selected, collected = deepcopy(candidates), deepcopy(artifacts)
        if turn:
            # A legitimate new selection reuses s1 but changes the exact span.
            selected[0]['evidence'][0]['quote'] = 'code one NEW'
            collected[0]['content'] = 'code one NEW'
            collected[0]['line_range'] = '1-20'
        root = kwargs['output_dir']
        _write(root / 'evidence_artifacts.json', {'artifacts': collected})
        return {'task_id': task.task_id, 'candidates': selected}
    monkeypatch.setattr(pipeline, 'generate_riu_candidates', generate)
    def consolidate(**kwargs):
        return {'groups': [{'group_id': f'g{n}', 'hidden_claim': c['claim'],
            'candidate_ids': [c['candidate_id']], 'supports_rubric_ids': c['supports_rubric_ids'],
            'refutes_rubric_ids': [], 'selected_evidence': c['evidence']}
            for n, c in enumerate(kwargs['candidate_result']['candidates'], 1)]}
    monkeypatch.setattr(pipeline, '_consolidate_candidates', consolidate)
    count = 0
    def audit(**kwargs):
        nonlocal count
        count += 1
        return {'accept': count == 2, 'errors': [] if count == 2 else ['first selection insufficient'],
                'all': {'judge_result': {}}, 'none': {'judge_result': {}}, 'leave_one_out': []}
    monkeypatch.setattr(pipeline, 'run_riu_audit', audit)
    output = tmp_path / 'run'
    config = pipeline.PipelineConfig(stop_after_riu_audit=True, audit_repair_attempts=0,
                                     riu_reselection_attempts=2)
    if failed_recollections == config.riu_reselection_attempts:
        with pytest.raises(pipeline.PipelineError, match="candidate collection failed"):
            pipeline.build_task(task=task, output_dir=output, client=clients, config=config)
        exhausted = json.loads((output / '04_riu_audit/reselection_exhausted.json').read_text())
        assert exhausted['total_selection_attempts'] == len(generated) == 3
        assert [entry['riu_count'] for entry in exhausted['failure_history']] == [2, 0, 0]
        assert count == 1
        assert not (output / 'pipeline_report.json').exists()
        return
    report = pipeline.build_task(task=task, output_dir=output, client=clients, config=config)
    assert report['all_none_loo_accept'] is True and len(generated) == 2 + failed_recollections
    if failed_recollections:
        failures = generated[-1]['construction_feedback']['failure_history']
        assert [entry['riu_count'] for entry in failures] == [2, 0]
        assert "Unverified evidence artifacts" in failures[-1]['reason']
    selected = json.loads((output / '05_final_rius/selected_artifacts.hidden.json').read_text())['artifacts']
    assert next(a['content'] for a in selected if a['local_artifact_id'] == 's1') == 'code one NEW'
    history = json.loads((output / '03_candidates/selection_00/artifact_snapshot.json').read_text())['artifacts']
    assert next(a['content'] for a in history if a['local_artifact_id'] == 's1') == 'code one'
    failed_id = next(a['artifact_id'] for a in history if a['local_artifact_id'] == 's1')
    assert failed_id not in {a['artifact_id'] for a in selected}
    final = json.loads((output / '05_final_rius/final_rius.hidden.json').read_text())['rius']
    pipeline._validate_final_riu_artifacts(final, selected)


@pytest.mark.parametrize('provenance', ['path', 'line_range', 'command'])
def test_equal_text_different_provenance_never_aliases_one_artifact(provenance):
    candidates, artifacts = _bundle()
    result = {'task_id': 'x', 'candidates': candidates}
    first, one = pipeline._scope_candidate_bundle(result, artifacts)
    changed = deepcopy(artifacts)
    changed[0][provenance] = {'path': 'different.go', 'line_range': '5-10', 'command': 'different command'}[provenance]
    second, two = pipeline._scope_candidate_bundle(result, changed)
    assert one[0]['artifact_id'] != two[0]['artifact_id']
    assert first['candidates'][0]['evidence'][0]['artifact_id'] != second['candidates'][0]['evidence'][0]['artifact_id']
    again, unchanged = pipeline._scope_candidate_bundle(first, one)
    assert unchanged == one and again == first


def test_planner_and_repair_never_receive_hidden_criteria_or_reference(tmp_path, monkeypatch):
    hidden_title = 'HIDDEN_CRITERION_SENTINEL'
    task = replace(_task(), evaluation=replace(_task().evaluation,
        rubrics=(Rubric('private_rubric', hidden_title, 'positive', 'PRIVATE_IMPORTANCE', {}),),
        reference_answer='HIDDEN_REFERENCE_SENTINEL'))
    calls = []
    class Collector:
        def __init__(self, *args, **kwargs): pass
        def preflight(self):
            return {'commit': 'abc', 'paths': ['a.go'], 'tool_paths': {}, 'config_files': [],
                    'runtime_uid': 0, 'runtime_user': 'root'}
        def collect(self, requests):
            return [{'artifact_id': 'source-one', 'kind': 'source', 'path': 'a.go',
                     'content': 'func One() {}', 'verified': True}]
    monkeypatch.setattr(candidate, 'DockerEvidenceCollector', Collector)
    monkeypatch.setattr(integrity, 'image_identity', lambda _: {'id': 'sha256:offline'})
    class Client:
        cfg = SimpleNamespace(model='offline')
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            if 'candidate_public_coverage' in kwargs['prompt_id']:
                return {'adequate_for_public_question': True, 'missing_public_aspects': [], 'replacement_plan': None}
            if 'candidate_synthesis' in kwargs['prompt_id']:
                return {'candidates': [{'candidate_id': 'c', 'claim': 'private grouped claim',
                    'evidence_mode': 'source', 'supports_rubric_ids': ['private_rubric'],
                    'refutes_rubric_ids': [], 'evidence': [{'artifact_id': 'source-one',
                                                          'kind': 'source', 'quote': 'func One() {}'}]}]}
            if len(calls) == 1:
                return {'rubric_requirements': [], 'evidence_requests': []}  # triggers a real repair prompt
            return {'rubric_requirements': [{'rubric_id': 'public_question', 'evidence_mode': 'source'}],
                    'evidence_requests': [{'request_id': 'source-one', 'kind': 'source', 'path': 'a.go',
                                           'supports_rubric_ids': ['public_question']}]}
    result = candidate.generate_riu_candidates(task=task, output_dir=tmp_path / 'candidate',
        client=Client(), repair_attempts=1, construction_feedback={
            'hidden_previous_failure': hidden_title, 'private_audit_feedback': 'PRIVATE_FEEDBACK_SENTINEL'})
    planner_calls = [call for call in calls if 'candidate_synthesis' not in call['prompt_id']]
    assert len(planner_calls) == 3
    for call in planner_calls:
        payload = call['user_prompt']
        assert hidden_title not in payload
        assert 'PRIVATE_IMPORTANCE' not in payload
        assert 'private_rubric' not in payload
        assert 'HIDDEN_REFERENCE_SENTINEL' not in payload
    synthesis = next(call for call in calls if 'candidate_synthesis' in call['prompt_id'])
    assert hidden_title in synthesis['user_prompt']
    assert 'PRIVATE_FEEDBACK_SENTINEL' in synthesis['user_prompt']
    assert 'HIDDEN_REFERENCE_SENTINEL' not in synthesis['user_prompt']
    assert result['repository_search_policy'].startswith('public_question_only')


@pytest.fixture
def standalone_candidates(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'instruction.md').write_text('Original raw instruction')
    task = replace(_task(), source_dir=source, evaluation=replace(_task().evaluation,
        rubrics=(Rubric('r1', 'static fact', 'positive', 'must have', {}),)))
    class Collector:
        def __init__(self, *args, **kwargs): pass
        def preflight(self):
            return {'commit': 'abc', 'paths': ['a.go'], 'tool_paths': {}, 'config_files': [],
                    'runtime_uid': 0, 'runtime_user': 'root'}
        def collect(self, requests):
            return [{'artifact_id': 'source-one', 'kind': 'source', 'path': 'a.go',
                     'content': 'func One() {}', 'verified': True}]
    monkeypatch.setattr(candidate, 'DockerEvidenceCollector', Collector)
    monkeypatch.setattr(integrity, 'image_identity', lambda _: {'id': 'sha256:offline'})
    class Client:
        cfg = SimpleNamespace(model='offline')
        def __init__(self): self.calls = []
        def generate_json(self, **kwargs):
            self.calls.append(kwargs['prompt_id'])
            if 'candidate_public_coverage' in kwargs['prompt_id']:
                return {'adequate_for_public_question': True, 'missing_public_aspects': [], 'replacement_plan': None}
            if 'candidate_synthesis' in kwargs['prompt_id']:
                return {'candidates': [{'candidate_id': 'c', 'claim': 'static fact',
                    'evidence_mode': 'source', 'supports_rubric_ids': ['r1'], 'refutes_rubric_ids': [],
                    'evidence': [{'artifact_id': 'source-one', 'kind': 'source', 'quote': 'func One() {}'}]}]}
            return {'rubric_requirements': [{'rubric_id': 'public_question', 'evidence_mode': 'source'}],
                'evidence_requests': [{'request_id': 'source-one', 'kind': 'source', 'path': 'a.go',
                                       'supports_rubric_ids': ['public_question']}]}
    client = Client()
    output = tmp_path / 'candidate'
    candidate.generate_riu_candidates(task=task, output_dir=output, client=client)
    client.calls.clear()
    return task, client, output


@pytest.mark.parametrize('change', ['missing_receipt', 'corrupt_receipt', 'plan', 'source', 'model', 'path_budget'])
def test_standalone_plan_cache_never_reuses_changed_or_unbound_inputs(standalone_candidates, change):
    task, client, output = standalone_candidates
    kwargs = {}
    if change == 'missing_receipt':
        (output / 'plan_cache_receipt.json').unlink()
    elif change == 'corrupt_receipt':
        (output / 'plan_cache_receipt.json').write_text('{')
    elif change == 'plan':
        plan = json.loads((output / 'evidence_plan.json').read_text())
        plan['evidence_requests'][0]['purpose'] = 'Changed after original validation'
        _write(output / 'evidence_plan.json', plan)
    elif change == 'source':
        (task.source_dir / 'instruction.md').write_text('Different raw task bytes')
    elif change == 'model':
        client.cfg.model = 'new-model'
    else:
        kwargs['max_repository_paths'] = 1
    with pytest.raises(candidate.CandidateValidationError, match='plan receipt'):
        candidate.generate_riu_candidates(task=task, output_dir=output, client=client,
                                         reuse_plan=True, **kwargs)
    assert not client.calls


def test_standalone_plan_reuse_is_allowed_only_for_matching_receipt(standalone_candidates):
    task, client, output = standalone_candidates
    candidate.generate_riu_candidates(task=task, output_dir=output, client=client, reuse_plan=True)
    assert len(client.calls) == 2
    assert 'candidate_public_coverage' in client.calls[0] and 'candidate_synthesis' in client.calls[1]


@pytest.mark.parametrize('stop_after_audit', [True, False])
def test_mid_run_prompt_drift_cannot_be_certified_at_either_success_return(tmp_path, monkeypatch, stop_after_audit):
    task = replace(_task(), source_dir=tmp_path / 'source')
    task.source_dir.mkdir()
    client = SimpleNamespace(cfg=SimpleNamespace(model='offline'))
    monkeypatch.setattr(integrity, 'image_identity', lambda _: {'id': 'sha256:offline'})
    state = {'prompt_changed': False}
    original_digest = integrity.digest
    monkeypatch.setattr(integrity, 'digest', lambda path:
        'different-prompt-content' if state['prompt_changed'] and Path(path).name == 'candidate_plan_system.txt'
        else original_digest(path))
    candidates, artifacts = _bundle()
    def generate(**kwargs):
        _write(kwargs['output_dir'] / 'evidence_artifacts.json', {'artifacts': artifacts})
        return {'task_id': task.task_id, 'candidates': deepcopy(candidates)}
    monkeypatch.setattr(pipeline, 'generate_riu_candidates', generate)
    monkeypatch.setattr(pipeline, '_consolidate_candidates', lambda **kwargs: {'groups': [
        {'group_id': f'g{n}', 'hidden_claim': row['claim'], 'candidate_ids': [row['candidate_id']],
         'supports_rubric_ids': row['supports_rubric_ids'], 'refutes_rubric_ids': [],
         'selected_evidence': row['evidence']}
        for n, row in enumerate(kwargs['candidate_result']['candidates'], 1)]})
    def audit(**kwargs):
        state['prompt_changed'] = stop_after_audit
        return {'accept': True, 'errors': [], 'all': {}, 'none': {}, 'leave_one_out': []}
    monkeypatch.setattr(pipeline, 'run_riu_audit', audit)
    class Collector:
        def __init__(self, *args, **kwargs): pass
        def inventory(self): return {'paths': []}
        def tool_paths(self): return {}
    monkeypatch.setattr(pipeline, 'DockerEvidenceCollector', Collector)
    monkeypatch.setattr(pipeline, 'build_experiment_logs', lambda **kwargs:
        {'canonical_logs': [], 'distractor_logs': []})
    monkeypatch.setattr(pipeline, 'collect_code_corpus', lambda **kwargs: [])
    partition = {'participants': [{}, {}], 'contexts': [], 'excluded_code': []}
    monkeypatch.setattr(pipeline, 'build_partition', lambda **kwargs: deepcopy(partition))
    monkeypatch.setattr(pipeline, 'validate_partition_mechanically', lambda *args: {'valid': True, 'errors': []})
    monkeypatch.setattr(pipeline, 'run_leakage_repair', lambda **kwargs:
        (kwargs['partition'], {'accept': True, 'rounds_run': 1, 'removed_context_ids': []}))
    def salience(**kwargs):
        state['prompt_changed'] = True
        return kwargs['partition'], {'accept': True}
    monkeypatch.setattr(pipeline, 'run_salience_repair', salience)
    monkeypatch.setattr(pipeline, 'materialize_modes', lambda **kwargs: {})
    output = tmp_path / 'run'
    with pytest.raises(pipeline.PipelineError, match='inputs changed during execution'):
        pipeline.build_task(task=task, output_dir=output, client=client,
            config=pipeline.PipelineConfig(stop_after_riu_audit=stop_after_audit))
    assert not (output / 'pipeline_report.json').exists()
    assert not (output / 'ready_receipt.json').exists()


@pytest.mark.parametrize('field,value', [('has_material_error', 1), ('overall_complete_and_correct', 0),
                                       ('content_correct', 1), ('grounded_in_available_material', 0),
                                       ('supporting_evidence_numbers', [True])])
def test_judge_numbers_never_substitute_for_booleans_or_evidence_indices(field, value):
    rubric = Rubric('r', 'criterion', 'positive', 'must have', {})
    judgment = {'has_material_error': False, 'overall_complete_and_correct': True,
        'rubric_results': [{'rubric_id': 'r', 'status': 'satisfied', 'content_correct': True,
                           'grounded_in_available_material': True, 'supporting_evidence_numbers': [1]}]}
    target = judgment if field in {'has_material_error', 'overall_complete_and_correct'} else judgment['rubric_results'][0]
    target[field] = value
    assert pipeline.validate_audit_judge(judgment, (rubric,), 1)


def _two_groups(candidates):
    return {'groups': [{'group_id': f'g{n}', 'hidden_claim': row['claim'],
        'candidate_ids': [row['candidate_id']], 'supports_rubric_ids': row['supports_rubric_ids'],
        'refutes_rubric_ids': [], 'selected_evidence': deepcopy(row['evidence'])}
        for n, row in enumerate(candidates, 1)]}


def test_consolidation_derives_rubric_links_from_validated_candidates():
    task = _task()
    candidates, artifacts = _bundle()
    options = {"task": task, "candidates": candidates, "artifacts": artifacts}
    response = _two_groups(candidates)
    expected = pipeline.validate_consolidation(response, **options)
    for group in response["groups"]:
        del group["supports_rubric_ids"]
        del group["refutes_rubric_ids"]
    assert pipeline.validate_consolidation(response, **options) == expected

    response["groups"][0]["supports_rubric_ids"] = ["r2"]
    with pytest.raises(pipeline.PipelineError, match="must equal member-candidate union"):
        pipeline.validate_consolidation(response, **options)


def test_encoded_docker_output_is_used_through_packet_rendering():
    task = _task()
    candidates, artifacts = _bundle()
    result = {"task_id": task.task_id, "candidates": candidates}
    runtime = artifacts[2]
    output = "PASS\nObserved result: 42"
    runtime["content"] = (
        "__CAPTURE_SCHEMA__=swe_atlas_runtime_capture_v1\n"
        "__COMMIT__=abc\n__DIRTY_BEFORE__=0\n__DIRTY_AFTER__=0\n"
        f"__OUTPUT_BASE64__={base64.b64encode(output.encode()).decode()}\n"
    )
    runtime["observed_content"] = output
    requirements = [{"rubric_id": "r1", "evidence_mode": "source"},
                    {"rubric_id": "r2", "evidence_mode": "hybrid"}]

    candidate.validate_candidates(result, task.evaluation.rubrics, requirements, artifacts)
    pipeline._validate_candidate_bundle(task, result, artifacts)
    rius = pipeline.validate_consolidation(_two_groups(candidates), task=task,
                                          candidates=candidates, artifacts=artifacts)
    pipeline._validate_final_riu_artifacts(rius, artifacts)
    assert pipeline._display_log(runtime, canonical=True)["text"].endswith(f"output:\n{output}")

    candidates[1]["evidence"][1]["quote"] = "__DIRTY_AFTER__=0"
    with pytest.raises(candidate.CandidateValidationError, match="quote is not exact"):
        candidate.validate_candidates(result, task.evaluation.rubrics, requirements, artifacts)


def test_private_revision_can_reselect_new_exact_span_from_frozen_artifact():
    candidates, artifacts = _bundle()
    artifacts[0]['content'] += '\nNEW previously omitted startup order source span'
    response = _two_groups(candidates)
    response['groups'][0]['selected_evidence'][0]['quote'] = 'NEW previously omitted startup order source span'
    rius = pipeline.validate_consolidation(response, task=_task(), candidates=candidates, artifacts=artifacts)
    assert rius[0]['evidence_atoms'][0]['quote'].startswith('NEW previously omitted')
    assert rius[0]['evidence_atoms'][0]['artifact_id'] == 's1'


@pytest.mark.parametrize('invalid', ['fabricated', 'kind_spoof', 'runtime_metadata'])
def test_private_revision_never_accepts_fabricated_spans_or_collector_metadata(invalid):
    candidates, artifacts = _bundle()
    response = _two_groups(candidates)
    if invalid == 'fabricated':
        response['groups'][0]['selected_evidence'][0]['quote'] = 'UNOBSERVED SOURCE'
    elif invalid == 'kind_spoof':
        response['groups'][0]['selected_evidence'][0]['kind'] = 'runtime'
    else:
        artifacts[2]['content'] = '__COMMAND__=go test\nPASS'
        artifacts[2]['observed_content'] = 'PASS'
        response['groups'][1]['selected_evidence'][1]['quote'] = '__COMMAND__=go test'
    with pytest.raises(pipeline.PipelineError):
        pipeline.validate_consolidation(response, task=_task(), candidates=candidates, artifacts=artifacts)


def test_private_consolidator_receives_frozen_full_artifacts_without_search(tmp_path, monkeypatch):
    candidates, artifacts = _bundle()
    artifacts[0]['content'] += '\nomitted source span available for grounded revision'
    seen = {}
    def call(*args, **kwargs):
        seen.update(args[3])
        return _two_groups(candidates)
    monkeypatch.setattr(pipeline, '_call_prompt', call)
    pipeline._consolidate_candidates(client=None, task=_task(),
        candidate_result={'candidates': candidates}, artifacts=artifacts,
        audit_feedback={'errors': ['missing startup ordering']}, trace_dir=tmp_path)
    assert 'omitted source span' in seen['VERIFIED_ARTIFACTS_JSON'][0]['content']
    assert 'command' not in seen['VERIFIED_ARTIFACTS_JSON'][2]


def test_public_coverage_revision_uses_raw_source_then_freezes_before_private_feedback(tmp_path, monkeypatch):
    task = replace(_task(), source_dir=tmp_path / 'source', evaluation=replace(_task().evaluation,
        rubrics=(Rubric('r1', 'PRIVATE_CRITERION', 'positive', 'must have', {}),)))
    task.source_dir.mkdir()
    monkeypatch.setattr(integrity, 'image_identity', lambda _: {'id': 'sha256:offline'})
    collected = []
    class Collector:
        def __init__(self, *args, **kwargs): pass
        def preflight(self):
            return {'commit': 'abc', 'paths': ['a.go'], 'tool_paths': {}, 'config_files': [],
                    'runtime_uid': 0, 'runtime_user': 'root'}
        def collect(self, requests):
            collected.append(deepcopy(requests))
            rows = [{'artifact_id': 's', 'kind': 'source', 'path': 'a.go', 'verified': True,
                     'content': 'PUBLIC RAW SOURCE: startup calls the listener',
                     'supports_rubric_ids': ['DO_NOT_FORWARD_MAPPING']}]
            if any(q['kind'] == 'runtime' for q in requests):
                rows.append({'artifact_id': 'run', 'kind': 'runtime', 'verified': True,
                             'content': 'ACTUAL LISTENER OUTPUT'})
            return rows
    monkeypatch.setattr(candidate, 'DockerEvidenceCollector', Collector)
    plan = {'rubric_requirements': [{'rubric_id': 'public_question', 'evidence_mode': 'source'}],
            'evidence_requests': [{'request_id': 's', 'kind': 'source', 'path': 'a.go',
                                   'supports_rubric_ids': ['public_question']}]}
    replacement = deepcopy(plan)
    replacement['evidence_requests'].append({'request_id': 'run', 'kind': 'runtime',
        'command': 'printf observation', 'expected_output_substrings': ['observation'], 'supports_rubric_ids': ['public_question']})
    calls = []
    class Client:
        cfg = SimpleNamespace(model='offline')
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            prompt_id = kwargs['prompt_id']
            if 'candidate_public_coverage_0' in prompt_id:
                return {'adequate_for_public_question': False, 'missing_public_aspects': ['startup observation'],
                        'reason': 'raw source describes a listener but no runtime observation exists',
                        'replacement_plan': replacement}
            if 'candidate_public_coverage_1' in prompt_id:
                return {'adequate_for_public_question': True, 'missing_public_aspects': [], 'replacement_plan': None}
            if 'candidate_synthesis' in prompt_id:
                return {'candidates': [{'candidate_id': 'c', 'claim': 'grounded fact', 'evidence_mode': 'source',
                    'supports_rubric_ids': ['r1'], 'refutes_rubric_ids': [], 'evidence': [
                        {'artifact_id': 's', 'kind': 'source', 'quote': 'PUBLIC RAW SOURCE: startup calls the listener'}]}]}
            return plan
    result = candidate.generate_riu_candidates(task=task, output_dir=tmp_path / 'run', client=Client(),
        runtime_repair_attempts=1, construction_feedback={'failed_positive_rubrics': ['PRIVATE_FEEDBACK']})
    assert len(collected) == 2 and result['collection_attempt_limit'] == 2
    coverage_calls = [call for call in calls if 'candidate_public_coverage' in call['prompt_id']]
    assert 'PUBLIC RAW SOURCE' in coverage_calls[0]['user_prompt']
    assert 'ACTUAL LISTENER OUTPUT' in coverage_calls[1]['user_prompt']
    for call in calls:
        if 'candidate_synthesis' not in call['prompt_id']:
            assert 'PRIVATE_FEEDBACK' not in call['user_prompt']
            assert 'PRIVATE_CRITERION' not in call['user_prompt']
            assert 'DO_NOT_FORWARD_MAPPING' not in call['user_prompt']
    assert 'PRIVATE_FEEDBACK' in calls[-1]['user_prompt']


def test_actual_reader_fit_drops_only_minimal_whole_neutral_units():
    from teamgym.interaction.actor_utils import count_tokens
    text = 'func A() { return 1 }'
    neutral = text + '\n// neutral'
    cap = count_tokens(text + '\n\n' + neutral)
    assert count_tokens(text) + count_tokens(neutral) + count_tokens('\n\n') > cap
    partition = {'participants': [{'participant_id': 'p', 'private_context_unit_ids': ['canonical', 'neutral'],
                                  'private_context_tokens': cap}],
        'contexts': [
            {'context_unit_id': 'canonical', 'text': text, 'assigned_to': ['p'],
             'estimated_tokens': count_tokens(text), 'hidden_contains_riu_ids': ['riu']},
            {'context_unit_id': 'neutral', 'text': neutral, 'assigned_to': ['p'],
             'estimated_tokens': count_tokens(neutral), 'hidden_contains_riu_ids': []}],
        'riu_ownership': {'riu': 'p'}, 'excluded_code': []}
    original = deepcopy(partition)
    fitted, report = pipeline.fit_actual_reader_budget(partition,
        pipeline.PipelineConfig(participant_context_token_cap=cap))
    assert partition == original
    assert [row['context_unit_id'] for row in fitted['contexts']] == ['canonical']
    assert report['canonical_preserved'] and report['participant_results'][0]['byte_exact_untruncated']
    assert [row['context_unit_id'] for row in report['removed_neutral_contexts']] == ['neutral']
    # Canonical-only overflow is a rejection, never permission to change or trim PIUs.
    partition['contexts'][1]['hidden_contains_riu_ids'] = ['riu2']
    with pytest.raises(pipeline.PipelineError, match='Canonical material exceeds'):
        pipeline.fit_actual_reader_budget(partition, pipeline.PipelineConfig(participant_context_token_cap=cap))


def test_actual_reader_fit_preserves_participant_packet_order():
    from teamgym.interaction.llm_actors import _private_context_payload
    from teamgym.interaction.types import BudgetSpec

    partition = {
        "participants": [
            {"participant_id": "A", "private_context_unit_ids": ["second", "first"]},
            {"participant_id": "B", "private_context_unit_ids": ["third"]},
        ],
        "contexts": [
            {"context_unit_id": "first", "text": "first excerpt", "assigned_to": ["A"]},
            {"context_unit_id": "third", "text": "third excerpt", "assigned_to": ["B"]},
            {"context_unit_id": "second", "text": "second excerpt", "assigned_to": ["A"]},
        ],
    }
    original = deepcopy(partition)
    fitted, _ = pipeline.fit_actual_reader_budget(partition, pipeline.PipelineConfig())
    assert partition == original
    assert fitted["participants"] == partition["participants"]
    datapoint = {"private_context_units": fitted["contexts"]}
    assert _private_context_payload(datapoint, fitted["participants"][0], BudgetSpec()) == "second excerpt\n\nfirst excerpt"
    assert _private_context_payload(datapoint, fitted["participants"][1], BudgetSpec()) == "third excerpt"


def test_default_partition_uses_actual_reader_fit_before_audits(tmp_path, monkeypatch):
    called = []
    def fitter(partition, config):
        called.append(deepcopy(partition))
        return partition, {'byte_exact_before_audits': True}
    monkeypatch.setattr(pipeline, 'fit_actual_reader_budget', fitter)
    candidates, artifacts = _bundle()
    rius = pipeline.validate_consolidation(_two_groups(candidates), task=_task(), candidates=candidates, artifacts=artifacts)
    code = [{'path': path, 'module_id': path, 'text': text, 'estimated_tokens': 10}
            for path, text in [('a.go', 'code one'), ('b.go', 'code two')]]
    result = pipeline.build_partition(task=_task(), rius=rius, artifacts=artifacts, code_corpus=code,
        experiment_bundle={'canonical_logs': [], 'distractor_logs': []}, config=pipeline.PipelineConfig())
    assert called and result['reader_budget_fit']['byte_exact_before_audits'] is True


def test_failed_optional_control_artifact_is_persisted_even_when_verified_minimum_met(tmp_path, monkeypatch):
    plan = {'experiments': [
        {'experiment_id': 'good', 'family': 'environment_probe', 'command': 'printf good',
         'purpose': 'neutral', 'why_not_canonical': 'not canonical'},
        {'experiment_id': 'bad', 'family': 'expected_failure', 'command': 'printf failure',
         'purpose': 'negative control', 'why_not_canonical': 'not canonical'}]}
    monkeypatch.setattr(pipeline, '_call_prompt', lambda *args, **kwargs: plan)
    rows = [{'artifact_id': 'good', 'kind': 'runtime', 'command': 'printf good', 'verified': True,
             'content': 'good', 'observed_exit_code': 0},
            {'artifact_id': 'bad', 'kind': 'runtime', 'command': 'printf failure', 'verified': False,
             'content': 'ACTUAL_FAILED_STDOUT', 'observed_exit_code': 1}]
    result = pipeline.build_experiment_logs(client=None, task=_task(), rius=[], artifacts=[],
        collector=SimpleNamespace(collect=lambda requests: deepcopy(rows)), tool_paths={}, output_dir=tmp_path,
        config=pipeline.PipelineConfig(min_distractor_logs=1))
    assert len(result['distractor_logs']) == 1
    raw = json.loads((tmp_path / 'all_collected_artifacts.attempt_00.hidden.json').read_text())
    assert raw['planned_count'] == 2 and len(raw['artifacts']) == 2
    failed = next(row for row in raw['artifacts'] if row['artifact_id'] == 'bad')
    assert failed['verified'] is False and failed['content'] == 'ACTUAL_FAILED_STDOUT'
    assert failed['observed_exit_code'] == 1 and failed['family'] == 'expected_failure'


@pytest.mark.parametrize('invalid', [
    {'cmd': 'grep -R implementation scapy'},
    {'adequate_for_public_question': 'true'},
    {'adequate_for_public_question': True, 'missing_public_aspects': ['still missing'], 'replacement_plan': None},
    ValueError('invalid JSON'),
])
def test_public_coverage_recovers_malformed_model_output_without_executing_commands(tmp_path, monkeypatch, invalid):
    responses = [invalid, {'adequate_for_public_question': True,
                          'missing_public_aspects': [], 'replacement_plan': None}]
    calls = []
    class Client:
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            value = responses.pop(0)
            if isinstance(value, Exception): raise value
            return value
    def never_execute(*args, **kwargs):
        raise AssertionError('Coverage review must not execute model commands')
    monkeypatch.setattr(candidate.subprocess, 'run', never_execute)
    result = candidate._review_public_coverage(
        client=Client(), task_id='public-task', planner_input={'question': 'Public question'},
        plan={}, artifacts=[{'artifact_id': 's', 'kind': 'source', 'content': 'PUBLIC SOURCE',
                             'supports_rubric_ids': ['PRIVATE_MAPPING']}],
        remaining_revisions=1, trace_dir=tmp_path, round_number=0, validation_retries=1)
    assert result['adequate_for_public_question'] is True
    assert len(calls) == 2
    assert 'validation_error' in calls[1]['user_prompt']
    assert all('PRIVATE_MAPPING' not in call['user_prompt'] for call in calls)


def test_public_coverage_format_retries_are_bounded(tmp_path):
    calls = []
    class Client:
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            return {'cmd': 'not a coverage review'}
    with pytest.raises(candidate.CandidateValidationError, match='format retries exhausted'):
        candidate._review_public_coverage(
            client=Client(), task_id='task', planner_input={}, plan={}, artifacts=[],
            remaining_revisions=0, trace_dir=tmp_path, round_number=0, validation_retries=2)
    assert len(calls) == 3


def test_hidden_regrouping_feedback_retains_specific_all_failures_without_repeating_material():
    failed = {'rubric_id': 'r1', 'status': 'missing', 'content_correct': False,
              'grounded_in_available_material': False, 'reason': 'Chosen TTL was omitted'}
    passed = {'rubric_id': 'r2', 'status': 'satisfied', 'content_correct': True,
              'grounded_in_available_material': True, 'reason': 'Supported'}
    audit = {'all': {'judge_result': {'overall_complete_and_correct': False,
                                    'overall_reason': 'Incomplete', 'rubric_results': [failed, passed]},
                     'material': 'DO_NOT_REPEAT_FULL_MATERIAL'},
             'none': {'judge_result': {'overall_complete_and_correct': False}}, 'leave_one_out': []}
    feedback = pipeline._compact_audit_feedback(audit)
    assert feedback['all_failed_rubrics'] == [failed]
    assert feedback['all_overall_reason'] == 'Incomplete'
    assert 'DO_NOT_REPEAT_FULL_MATERIAL' not in json.dumps(feedback)
