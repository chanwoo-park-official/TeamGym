from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.finance.findver import free_response_funnel as funnel
from teamgym.data.finance.findver import integrity
from teamgym.data.finance.findver.source import findver_report_candidates, load_financial_report
from teamgym.data.finance.findver.free_response_validation import validate_packet_riu_audit
from tests.finance.findver.test_free_response_funnel import _candidate, _selection, _plan


def _client():
    return SimpleNamespace(cfg=SimpleNamespace(model='offline-fixture'))


def _judgment():
    return {'final_answer_correct': True, 'source_grounded': True, 'claim_results': [
        {'claim_id': 'claim_01', 'content_correct': True,
         'grounded_in_available_evidence': True, 'supporting_riu_ids': [], 'reason': 'grounded'}],
        'disqualifying_errors': []}


def test_native_knowledge_never_becomes_common_information():
    candidate = _candidate()
    candidate['source_qas'] = [{'qa_id': 'q1', 'subset': 'knowledge',
                               'public_knowledge': ['PRIVATE_NATIVE_SENTINEL']}]
    selection = _selection(candidate)
    selection['source_qa_ids_used'] = ['q1']
    assert 'PRIVATE_NATIVE_SENTINEL' not in json.dumps(funnel._public_information(candidate, selection))
    candidate['source_knowledge'] = ['PRIVATE_NATIVE_SENTINEL']
    scenario = funnel.build_full_context_partition(candidate=candidate, selection=selection,
        plan=_plan(selection), audits={}, seed=1, allocation_mode='one_each', min_participants=2)
    assert 'PRIVATE_NATIVE_SENTINEL' not in json.dumps(scenario['common_datapoint']['initial_common_information'])


def test_solver_and_judge_share_public_information_and_solver_has_no_piu_labels(tmp_path, monkeypatch):
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        return ({'written_answer': '24%', 'confidence': 1} if kwargs['stage'] == 'free_solver_user'
                else _judgment())
    monkeypatch.setattr(funnel, '_call', call)
    oracle = _plan(_selection(_candidate()))['oracle']
    panel, errors = funnel._run_answerability_panel(cases=[{
        'case_id': 'full', 'is_full_set': True, 'available_riu_ids': ['secret_piu'],
        'evidence': [{'riu_id': 'secret_piu', 'information_feature': 'hidden feature', 'text': 'source text'}]}],
        public_task={'question': 'review'}, public_information={'shared_fact': 'allowed common fact'},
        oracle=oracle, blind_clients=[('solver', _client())], judge_client=_client(),
        trace_root=tmp_path, instance_id='case', plan_attempt=1, audit_attempts=1,
        stages=[], require_full_passes=1, reject_any_non_full_pass=True, panel_name='boundary')
    assert not errors and panel['passed']
    assert calls[0]['variables']['PUBLIC_INFORMATION_JSON'] == calls[1]['variables']['PUBLIC_INFORMATION_JSON']
    assert calls[0]['variables']['RIU_EVIDENCE_JSON'] == [{'text': 'source text'}]


@pytest.mark.parametrize('field,value', [('passes', False), ('required_step_coverage', 0)])
def test_judge_cannot_override_contradictory_summary(tmp_path, field, value):
    oracle = _plan(_selection(_candidate()))['oracle']
    judgment = _judgment()
    judgment[field] = value
    errors = funnel._validate_normalized_judgment(judgment, oracle=oracle, available_riu_ids=[])
    assert any('contradict' in error for error in errors)


def test_packet_audit_accepts_collective_reconstruction_and_validates_empty_packet():
    audit = {'participant_id': 'a', 'unassigned_riu_results': [{
        'riu_id': 'current', 'visible': True, 'visibility_type': 'arithmetic',
        'evidence_context_ids': [11, 12], 'reason': '6.0 + 6.4 = 12.4'}]}
    assert not validate_packet_riu_audit(audit, all_rui_ids=['current', 'prior'],
        assigned_or_shared_rui_ids=['prior'], packet_context_ids=[11, 12], expected_participant_id='a')
    errors = validate_packet_riu_audit(audit, all_rui_ids=['current', 'prior'],
        assigned_or_shared_rui_ids=['prior'], packet_context_ids=[], expected_participant_id='a')
    assert any('not in this packet' in error for error in errors)


def test_full_local_view_detects_collective_solution_beyond_owned_piu(tmp_path, monkeypatch):
    candidate = _candidate()
    selection = _selection(candidate)
    scenario = funnel.build_full_context_partition(candidate=candidate, selection=selection,
        plan=_plan(selection), audits={}, seed=1, allocation_mode='one_each', min_participants=2)
    common = scenario['common_datapoint']
    participant = common['participants'][0]
    hidden_id = participant['private_context_unit_ids'][0]
    unit = next(row for row in common['private_context_units'] if row['context_unit_id'] == hidden_id)
    unit['text'] += '\nOther period can be reconstructed: 6.0 + 6.4 = 12.4.'
    cases = funnel._participant_answerability_cases(scenario, token_budget=20000)
    assert any('6.0 + 6.4' in case['evidence'][0]['text'] for case in cases)
    monkeypatch.setattr(funnel, '_call', lambda **kwargs:
        {'written_answer': '24%', 'confidence': 1} if kwargs['stage'] == 'free_solver_user' else _judgment())
    _, errors = funnel._run_answerability_panel(cases=cases, public_task=_plan(selection)['public_task'],
        public_information={}, oracle=_plan(selection)['oracle'], blind_clients=[('solver', _client())],
        judge_client=_client(), trace_root=tmp_path, instance_id='case', plan_attempt=1,
        audit_attempts=1, stages=[], require_full_passes=0, reject_any_non_full_pass=True,
        panel_name='participant_full_view')
    assert any('answer recovered' in error for error in errors)


def test_post_removal_reaudit_and_participant_gate_are_required_before_acceptance(tmp_path, monkeypatch):
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(selection)
    generated = {**plan, 'selection': selection}
    monkeypatch.setattr(funnel, '_call', lambda **kwargs: deepcopy(generated))
    monkeypatch.setattr(funnel, '_run_structured_audit', lambda **kwargs: ({}, []))
    panels = []
    def panel(**kwargs):
        panels.append(kwargs['panel_name'])
        return ({'passed': True, 'case_results': []},
                ['participant can solve'] if kwargs['panel_name'] == 'participant_full_view' else [])
    monkeypatch.setattr(funnel, '_run_answerability_panel', panel)
    audits = []
    def packet(**kwargs):
        audits.append(deepcopy(kwargs['scenario']))
        return [], [], {15} if len(audits) == 1 else set()
    monkeypatch.setattr(funnel, '_run_packet_audits', packet)
    result, scenario, byproduct = funnel._try_candidate(candidate=candidate, candidate_index=1,
        output_root=tmp_path, trace_root=tmp_path, generator_client=_client(), audit_client=_client(),
        blind_clients=[('solver', _client())], allocation_mode='one_each', min_participants=2,
        min_units=2, max_units=2, plan_attempts=1, audit_attempts=1, full_boundary_min_passes=1,
        seed=1, question_generate_stage=funnel.QUESTION_GENERATE_STAGE)
    assert scenario is None and result['status'] == 'audit_failed'
    assert len(audits) == 2
    assert any(row['source_ref']['context_id'] == 15 for row in audits[0]['common_datapoint']['private_context_units'])
    assert all(row['source_ref']['context_id'] != 15 for row in audits[1]['common_datapoint']['private_context_units'])
    assert panels[-1] == 'participant_full_view'
    assert 'participant can solve' in result['errors']


def test_existing_output_is_preserved_instead_of_unverified_overwrite(tmp_path):
    qa = tmp_path / 'qa.json'
    integrity.write_json(qa, [])
    output = tmp_path / 'output'
    output.mkdir()
    old = output / 'scenarios.jsonl'
    old.write_text('preserve old accepted record\n')
    with pytest.raises(ValueError, match='empty'):
        funnel.build_dataset(qa_path=qa, reports_root=tmp_path, output_root=output,
            generator_client=_client(), audit_client=_client(), blind_clients=[('solver', _client())], subset='all')
    assert old.read_text() == 'preserve old accepted record\n'


def test_atomic_jsonl_interruption_preserves_prior_complete_file(tmp_path, monkeypatch):
    target = tmp_path / 'scenarios.jsonl'
    integrity.write_jsonl(target, [{'accepted': True}])
    monkeypatch.setattr(integrity.os, 'replace', lambda *_: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(OSError):
        integrity.write_jsonl(target, [{'accepted': False}])
    assert json.loads(target.read_text())['accepted'] is True
    assert len(list(tmp_path.iterdir())) == 1


def test_loader_requires_explicit_mixed_split_and_rejects_duplicate_identity(tmp_path):
    qa = tmp_path / 'qa.json'
    base = {'example_id': 'q', 'subset': 'ie', 'report': 'r.json', 'relevant_context': [1]}
    integrity.write_json(tmp_path / 'r.json', {'context': [{'id': 1, 'context': 'source', 'type': 'paragraph'}]})
    integrity.write_json(qa, [{**base, 'split': 'train'}, {**base, 'split': 'testmini'}])
    with pytest.raises(ValueError, match='Mixed source splits'):
        list(findver_report_candidates(qa_path=qa, reports_root=tmp_path, subset='all'))
    only_testmini = list(findver_report_candidates(qa_path=qa, reports_root=tmp_path, subset='all', source_split='testmini'))
    assert len(only_testmini) == 1 and only_testmini[0]['split'] == 'testmini'
    integrity.write_json(qa, [{**base, 'split': 'testmini'}, {**base, 'split': 'testmini'}])
    with pytest.raises(ValueError, match='Duplicate'):
        list(findver_report_candidates(qa_path=qa, reports_root=tmp_path, subset='all', source_split='testmini'))


def test_source_identity_hashes_exact_report_and_native_knowledge_is_internal(tmp_path):
    qa = tmp_path / 'testmini.json'
    integrity.write_json(qa, [{'example_id': 'q', 'subset': 'knowledge', 'report': 'r.json',
        'relevant_context': [1], 'knowledge': ['native answer annotation'], 'split': 'testmini'}])
    integrity.write_json(tmp_path / 'r.json', {'context': [{'id': 1, 'context': 'source', 'type': 'paragraph'}]})
    candidate = next(findver_report_candidates(qa_path=qa, reports_root=tmp_path, subset='all', source_split='testmini'))
    assert candidate['source_provenance']['report_sha256'] == integrity.sha256(tmp_path / 'r.json')
    assert candidate['source_qas'][0]['construction_knowledge'] == ['native answer annotation']
    assert 'public_knowledge' not in candidate['source_qas'][0]


def test_report_path_cannot_escape_raw_source_root(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        load_financial_report(tmp_path, '../outside.json')


def test_numeric_grounding_rejects_duplicate_step_and_nonfinite_oracle():
    from teamgym.data.finance.findver.free_response_validation import (
        validate_free_grounding_audit, validate_free_response_plan)
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _plan(selection)
    plan['oracle']['answer_spec']['value'] = float('nan')
    assert any('numeric value' in error for error in validate_free_response_plan(plan, selection, candidate))
    step = {'step': 1, 'claim_id': 'claim_01', 'grounded': True, 'uses_match': True}
    audit = {key: True for key in ('valid', 'question_answerable_from_full_riu_set', 'short_answer_correct',
                                  'canonical_response_grounded', 'all_selected_rius_necessary', 'answer_spec_correct')}
    audit.update(step_results=[step, deepcopy(step)], unsupported_claims=[])
    assert any('every reasoning step' in error for error in validate_free_grounding_audit(audit, plan))


def test_manifest_binds_config_and_input_changes(tmp_path):
    qa = tmp_path / 'qa.json'
    integrity.write_json(qa, [])
    manifest = integrity.start_run(tmp_path / 'output', qa_path=qa, reports_root=tmp_path,
        config={'source_split': 'testmini', 'seed': 12}, clients={'solver': integrity.client_identity(_client())})
    integrity.verify_inputs(manifest)
    integrity.write_json(qa, [{'changed': True}])
    with pytest.raises(RuntimeError, match='QA input changed'):
        integrity.verify_inputs(manifest)


def _count_plan(question):
    selection = _selection(_candidate())
    plan = _plan(selection)
    plan['public_task']['question'] = question
    plan['oracle']['answer_spec'] = {'type': 'composite', 'items': [
        {'item_id': 'count', 'claim_ids': ['claim_01'], 'type': 'numeric', 'value': 4,
         'unit': 'quarterly payment periods', 'absolute_tolerance': 0},
        {'item_id': 'change', 'claim_ids': ['claim_01'], 'type': 'numeric', 'value': 24,
         'unit': 'percent', 'absolute_tolerance': 0.1}]}
    return plan


def test_objective_gate_rejects_word_cardinality_even_if_auditor_overlooks_it():
    from teamgym.data.finance.findver.free_response_validation import validate_objective_disclosure_audit
    plan = _count_plan('For 2023, total the deficiencies for the four quarterly payment periods and report their number.')
    errors = validate_objective_disclosure_audit(
        {'valid': True, 'discloses_answer_information': False, 'findings': []}, plan)
    assert any('unit/cardinality' in error for error in errors)


def test_objective_gate_allows_year_scope_and_item_enumeration():
    from teamgym.data.finance.findver.free_response_validation import validate_objective_disclosure_audit
    plan = _count_plan('For reporting year 2023: (3) total the deficiencies; (4) report the number of quarterly payment periods dated in 2023.')
    assert not validate_objective_disclosure_audit(
        {'valid': True, 'discloses_answer_information': False, 'findings': []}, plan)


def test_independent_objective_gate_rejects_semantic_answer_disclosure():
    from teamgym.data.finance.findver.free_response_validation import validate_objective_disclosure_audit
    plan = _count_plan('Explain why distributions exceeded the deficiencies and calculate the change.')
    audit = {'valid': False, 'discloses_answer_information': True, 'findings': [{
        'public_quote': 'distributions exceeded the deficiencies', 'answer_item_ids': ['classification'],
        'reason': 'Question states the requested comparison conclusion.'}]}
    assert any('discloses or may disclose' in error for error in validate_objective_disclosure_audit(audit, plan))


def test_objective_disclosure_rejected_before_any_grounding_or_solver_call(tmp_path, monkeypatch):
    candidate = _candidate()
    selection = _selection(candidate)
    plan = _count_plan('For 2023, report how many of the four quarterly payment periods had deficiencies.')
    generated = {**plan, 'selection': selection}
    monkeypatch.setattr(funnel, '_call', lambda **kwargs: deepcopy(generated))
    def forbidden(**kwargs):
        raise AssertionError('A disclosed answer must not reach grounding or solver work')
    monkeypatch.setattr(funnel, '_run_structured_audit', forbidden)
    monkeypatch.setattr(funnel, '_run_answerability_panel', forbidden)
    result, scenario, _ = funnel._try_candidate(candidate=candidate, candidate_index=1,
        output_root=tmp_path, trace_root=tmp_path, generator_client=_client(), audit_client=_client(),
        blind_clients=[('solver', _client())], allocation_mode='one_each', min_participants=2,
        min_units=2, max_units=2, plan_attempts=1, audit_attempts=1, full_boundary_min_passes=1,
        seed=1, question_generate_stage=funnel.QUESTION_GENERATE_STAGE)
    assert scenario is None and any('unit/cardinality' in error for error in result['errors'])


def test_actual_single_and_multi_agent_entries_receive_identical_common_information():
    from teamgym.interaction.llm_actors import LLMParticipantActor, PromptTraceRecorder
    from teamgym.interaction.run_single_agent_rui_evaluation import run_single_agent_rui_episode
    from teamgym.interaction.rui_context import rui_only_datapoint
    from teamgym.interaction.types import BudgetSpec
    candidate = _candidate()
    selection = _selection(candidate)
    common = funnel.build_full_context_partition(candidate=candidate, selection=selection,
        plan=_plan(selection), audits={}, seed=1, allocation_mode='one_each', min_participants=2)['common_datapoint']
    assert 'dataset' not in common['initial_common_information']
    class Client:
        last_request_meta = {}
        def generate_json(self, **kwargs):
            return {'utterance': 'offline', 'written_answer': 'offline', 'answer': 'free_response', 'confidence': 0}
    commons = []
    for row in (common, rui_only_datapoint(common)):
        recorder = PromptTraceRecorder()
        LLMParticipantActor(Client(), recorder).respond(datapoint=row,
            participant_id=row['participants'][0]['participant_id'], state={'turn': 1, 'messages': []},
            request=None, budgets=BudgetSpec(participant_context_token_budget=40000))
        commons.append(json.loads(recorder.prompt_records[0]['user_prompt'])['task']['initial_common_information'])
    recorder = PromptTraceRecorder()
    run_single_agent_rui_episode(common, client=Client(), recorder=recorder,
        args=SimpleNamespace(context_token_budget=40000, final_rationale_token_budget=120,
                             final_reasoning_token_budget=256))
    commons.append(json.loads(recorder.prompt_records[0]['user_prompt'])['task']['initial_common_information'])
    assert commons[0] == commons[1] == commons[2]
    assert 'dataset' not in json.dumps(commons)


def test_manifest_binds_actual_consumer_dependencies_and_tokenizer_runtime(tmp_path, monkeypatch):
    qa = tmp_path / 'qa.json'
    integrity.write_json(qa, [])
    manifest = integrity.start_run(tmp_path / 'output', qa_path=qa, reports_root=tmp_path,
        config={}, clients={})
    required = {'teamgym/interaction/actor_utils.py', 'teamgym/interaction/public_reading.py',
        'teamgym/interaction/run_single_agent_rui_evaluation.py', 'teamgym/core/dataset.py',
        'teamgym/core/io.py', 'teamgym/llm/construction_client.py', 'teamgym/llm/openai_client.py',
        'teamgym/tokenization.py', 'teamgym/data/cli.py', 'pyproject.toml', 'uv.lock'}
    assert required <= set(manifest['code'])
    assert manifest['runtime']['packages']['tiktoken']
    assert manifest['runtime']['tokenizer']['mergeable_vocabulary_sha256']
    original = integrity.sha256
    monkeypatch.setattr(integrity, 'sha256', lambda path:
        'different-renderer' if Path(path).name == 'actor_utils.py' else original(path))
    with pytest.raises(RuntimeError, match='actor_utils'):
        integrity.verify_inputs(manifest)


def test_manifest_rejects_runtime_or_tokenizer_drift(tmp_path, monkeypatch):
    qa = tmp_path / 'qa.json'
    integrity.write_json(qa, [])
    manifest = integrity.start_run(tmp_path / 'output', qa_path=qa, reports_root=tmp_path,
        config={}, clients={})
    changed = deepcopy(manifest['runtime'])
    changed['tokenizer']['mergeable_vocabulary_sha256'] = 'different-tokenizer'
    monkeypatch.setattr(integrity, 'runtime_identity', lambda: changed)
    with pytest.raises(RuntimeError, match='runtime changed'):
        integrity.verify_inputs(manifest)


def test_actual_generic_grader_receives_complete_strict_finance_contract():
    from teamgym.scoring.short_answers import judge_one_answer, requires_all_steps_for_pass
    candidate = _candidate()
    selection = _selection(candidate)
    common = funnel.build_full_context_partition(candidate=candidate, selection=selection,
        plan=_plan(selection), audits={}, seed=1, allocation_mode='one_each', min_participants=2)['common_datapoint']
    assert requires_all_steps_for_pass(common)
    captured = {}
    class Captured(Exception): pass
    class Client:
        def generate_json(self, **kwargs):
            captured.update(json.loads(kwargs['user_prompt']))
            raise Captured()
    with pytest.raises(Captured):
        judge_one_answer(client=Client(), datapoint=common,
            answer_row={'answer_text': common['oracle']['canonical_response']}, trace_dir=None, index=1)
    assert captured['judge_rubric'] == common['oracle']['judge_rubric']
    assert captured['required_decision_steps'] == common['oracle']['required_decision_steps']
    assert captured['canonical_response'] == common['oracle']['canonical_response']
    assert captured['judge_rubric']['require_all_steps_for_pass'] is True
    assert captured['judge_rubric']['answer_spec'] == common['oracle']['answer_spec']
    assert captured['pass_policy']['require_all_steps_for_pass'] is True
