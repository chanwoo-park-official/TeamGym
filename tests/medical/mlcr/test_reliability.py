from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.data.medical.mlcr.boundary import validate_judgment, run_boundaries
from teamgym.data.medical.mlcr.construction import validate_grounding_audit
from teamgym.data.medical.mlcr.partition import validate_leakage_audit
from teamgym.data.medical.mlcr.llm import ResponseCache


def judgment():
    return {'fully_correct': True, 'content_complete': True,
            'claim_results': [{'claim_id': 'c', 'content_correct': True,
                              'grounded_in_available_material': True, 'satisfied': True,
                              'material_numbers': ['material_01']}],
            'incomplete_claim_ids': [], 'prohibited_errors_present': []}


def test_claim_correctness_grounding_and_error_flags_are_consistent():
    good = judgment()
    assert validate_judgment(good, claim_ids={'c'}, available_numbers={'material_01'}) == []
    assert validate_judgment({**good, 'prohibited_errors_present': ['clinical error']}, claim_ids={'c'})
    assert validate_judgment({**good, 'content_complete': False}, claim_ids={'c'})
    assert validate_judgment(good, claim_ids={'c'}, available_numbers=set())
    wrong = deepcopy(good)
    wrong['claim_results'][0]['grounded_in_available_material'] = False
    assert validate_judgment(wrong, claim_ids={'c'})


def test_duplicate_grounding_rows_cannot_hide_a_failed_row():
    task = {'rubric': {'required_claims': [{'claim_id': 'c'}]}}
    good = {'claim_audits': [{'claim_id': 'c', 'grounded': True, 'scope_preserved': True}],
            'reference_answer_fully_grounded': True, 'all_units_materially_necessary': True, 'passed': True}
    assert not validate_grounding_audit(good, task=task)
    good['claim_audits'].insert(0, {'claim_id': 'c', 'grounded': False, 'scope_preserved': False})
    assert validate_grounding_audit(good, task=task)


def test_leakage_global_failure_and_duplicates_cannot_be_overwritten():
    good = {'passed': True, 'issues': [], 'foreign_riu_audits': [
        {'foreign_unit_id': 'foreign', 'recoverable': False, 'canonical_overlap': False,
         'removable_context_ids': []}]}
    args = dict(foreign_unit_ids={'foreign'}, packet_context_ids={'x'}, canonical_context_ids={'x'})
    assert not validate_leakage_audit(good, **args)
    assert validate_leakage_audit({**good, 'passed': False, 'issues': ['unresolved']}, **args)
    assert validate_leakage_audit({**good, 'foreign_riu_audits': good['foreign_riu_audits'] * 2}, **args)


def test_boundary_judge_receives_each_actual_available_packet(monkeypatch, tmp_path):
    from teamgym.data.medical.mlcr import boundary
    judges = []
    def call(**kw):
        if kw['stage'].startswith('boundary_solver'):
            return {'answer': 'same answer', 'insufficient': False, 'missing_answer_elements': []}, []
        judges.append(deepcopy(kw['variables']))
        return {'fully_correct': False, 'content_complete': False,
                'claim_results': [{'claim_id': 'c', 'content_correct': False,
                                  'grounded_in_available_material': False, 'satisfied': False,
                                  'material_numbers': []}],
                'incomplete_claim_ids': ['c'], 'prohibited_errors_present': []}, []
    monkeypatch.setattr(boundary, 'structured_call', call)
    task = {'public_task': {'question': 'Q'}, 'rubric': {'required_claims': [
        {'claim_id': 'c', 'supporting_candidate_ids': ['a', 'b']}]}}
    units = [{'unit_id': x, 'candidate_id': x, 'text': f'fact {x}'} for x in ['a', 'b']]
    run_boundaries(case_id='case', task=task, evidence_units=units, client=None, cache=None,
                   trace_root=tmp_path, attempts=1, seed=1)
    assert [len(r['DISPLAY_MATERIALS_JSON']) for r in judges] == [2, 0, 1, 1]
    assert all('CONDITION_JSON' in r and 'MATERIAL_MAPPING_JSON' in r for r in judges)
    assert len({json.dumps(r, sort_keys=True) for r in judges}) == 4


def test_mapping_cache_binds_qa_source_and_model(monkeypatch, tmp_path):
    from teamgym.data.medical.mlcr import pipeline
    calls = []
    def call(**kw):
        calls.append(kw['stage'])
        return {'prompt_id': 'q', 'answer_reconstructable': False, 'candidate_spans': []}, []
    monkeypatch.setattr(pipeline, 'structured_call', call)
    case = {'case_uuid': 'case', 'source_questions': [{'prompt_id': 'q', 'question': 'Q', 'answer': 'old'}]}
    shared = {}
    client = SimpleNamespace(cfg=SimpleNamespace(model='model1'))
    def run():
        return pipeline.run_case(case_index=case, output_root=tmp_path,
             generation_client=client, verification_client=client, qa_mapping_cache=shared, min_source_questions=1)
    run(); run()
    assert len(calls) == 1
    case['source_questions'][0]['answer'] = 'new answer'
    run()
    assert len(calls) == 2
    client.cfg.model = 'model2'
    run()
    assert len(calls) == 3


def test_mapping_retrieval_has_an_explicit_exact_atom_budget():
    from teamgym.data.medical.mlcr.pipeline import mapping_source_supply
    atoms = [{'atom_id': str(i), 'source_document_id': 'd', 'text': f'Date {i}. Exact recorded sentence.'}
             for i in range(30)]
    supplied, report = mapping_source_supply(atoms, {'question': 'Date 20?'}, token_budget=150)
    assert report['tokens'] <= 150 and len(supplied) < len(atoms)
    assert all(r in atoms for r in supplied)
    assert report['policy'] == 'question_lexical_atoms_with_neighbors'


def test_response_cache_binds_provider_and_recovers_partial_file(tmp_path):
    calls = []
    class Client:
        def __init__(self, provider): self.cfg = SimpleNamespace(model='same', provider=provider)
        def generate_json(self, **kw): calls.append(1); return {'ok': True}
    path = tmp_path / 'cache.jsonl'; path.write_text('{partial')
    cache = ResponseCache(path)
    args = dict(stage='s', case_id='case', system_prompt='system', user_prompt='user', trace_dir=tmp_path)
    cache.call(client=Client('a'), **args)
    cache.call(client=Client('a'), **args)
    cache.call(client=Client('b'), **args)
    assert len(calls) == 2


def test_main_cli_exposes_merge_policy_explicitly(tmp_path, monkeypatch):
    from teamgym.data.medical.mlcr import cli
    monkeypatch.setattr(cli, 'load_inspectable', lambda _: ([], []))
    monkeypatch.setattr(cli, 'build_case_indexes', lambda *_: [{'case_uuid': 'case'}])
    monkeypatch.setattr(cli, 'OpenAIJsonClient', lambda _: None)
    options = {}
    def run(**kw): options.update(kw); return {'status': 'failed'}
    monkeypatch.setattr(cli, 'run_case', run)
    monkeypatch.setattr('sys.argv', ['mlcr', '--source-dir', str(tmp_path), '--case-id', 'case',
                                   '--output-root', str(tmp_path / 'run'), '--allow-ownership-merges'])
    cli.main()
    assert options['strict_one_each'] is False


def test_export_identity_survives_directory_move_and_tracks_task_change(tmp_path):
    from teamgym.data.medical.mlcr.evaluation import build_evaluation_datapoint
    units = [{'unit_id': f'u{i}', 'candidate_id': f'c{i}', 'text': f'Exact fact {i}.',
              'supports_claim_ids': [f'claim{i}'], 'source_document_id': f'doc{i}',
              'raw_start': 0, 'raw_end': 13, 'sha256': str(i) * 64} for i in [1, 2]]
    report = {'status': 'accepted', 'case_id': 'case', 'grounding_audit': {'passed': True},
              'boundary_audits': {'passed': True}, 'packet_leakage_audit': {'passed': True},
              'task': {'public_task': {'question': 'State both facts.'}, 'evidence_units': units,
                       'oracle': {'reference_answer': 'Both facts.'},
                       'rubric': {'required_claims': [
                           {'claim_id': f'claim{i}', 'description': f'Fact {i}',
                            'expected_answer_element': f'Exact fact {i}', 'grading_rule': 'Exact fact'}
                           for i in [1, 2]]}},
              'partition': {'participants': [
                  {'participant_id': f'p{i}', 'hidden_owned_unit_ids': [f'u{i}'],
                   'context_ids': [f'x{i}'], 'context_tokens': 5} for i in [1, 2]],
                  'contexts': [{'context_id': f'x{i}', 'text': f'Exact fact {i}.',
                                'source_document_id': f'doc{i}', 'source_file': f'{i}.txt',
                                'document_order': i, 'source_order': 1} for i in [1, 2]],
                  'excluded_context_ids': []}}
    paths = [tmp_path / name / 'run_report.json' for name in ['one', 'moved']]
    for p in paths:
        p.parent.mkdir(); p.write_text(json.dumps(report))
        from tests.medical.certified_fixtures import complete_fixture
        complete_fixture(p.parent, 'mlcr')
    a, b = [build_evaluation_datapoint(p) for p in paths]
    assert a['instance_id'] == b['instance_id']
    assert a['format_validation']['valid']
    report['task']['public_task']['question'] = 'State both recorded findings precisely.'
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='certification'):
        build_evaluation_datapoint(paths[1])
    complete_fixture(paths[1].parent, 'mlcr')
    assert build_evaluation_datapoint(paths[1])['instance_id'] != a['instance_id']


def test_unreconstructable_raw_qa_does_not_discard_other_eligible_qas(monkeypatch, tmp_path):
    from teamgym.data.medical.mlcr import pipeline
    class ReachedGeneration(Exception): pass
    def call(**kw):
        if kw['stage'].startswith('generate_composite_task'):
            raise ReachedGeneration()
        pid = kw['stage'].split('.')[-1]
        return {'prompt_id': pid, 'answer_reconstructable': pid != 'q3',
                'candidate_spans': [{'candidate_id': 'one'}] if pid != 'q3' else []}, []
    monkeypatch.setattr(pipeline, 'structured_call', call)
    monkeypatch.setattr(pipeline, 'hydrate_candidates', lambda mapping, **_: mapping['candidate_spans'])
    case = {'case_uuid': 'case', 'source_questions': [
        {'prompt_id': q, 'question': q, 'answer': q} for q in ['q1', 'q2', 'q3']]}
    with pytest.raises(ReachedGeneration):
        pipeline.run_case(case_index=case, output_root=tmp_path, generation_client=None,
                          verification_client=None, min_source_questions=2)


def test_full_context_file_structure_matches_actual_actor_at_declared40k_budget():
    from teamgym.data.medical.mlcr.evaluation import private_context_units, consumer_full_context_materials
    from teamgym.interaction.llm_actors import LLMParticipantActor, PromptTraceRecorder
    from teamgym.interaction.types import BudgetSpec
    contexts = [{'context_id': f'c{i}', 'text': f'Exact fact {i}.', 'source_document_id': f'd{i}',
                 'source_file': f'{i:03d}.txt', 'document_order': i, 'source_order': 1}
                for i in [1, 2, 3]]
    partition = {'contexts': contexts, 'participants': [
        {'participant_id': 'p1', 'context_ids': ['c1', 'c3']},
        {'participant_id': 'p2', 'context_ids': ['c2']}]}
    expected = 'Source file: 001.txt\n\nExact fact 1.\n\nSource file: 003.txt\n\nExact fact 3.'
    materials = consumer_full_context_materials(partition, token_budget=40000)
    assert materials['p1'] == expected
    calls = []
    class Client:
        last_request_meta = {}
        def generate_json(self, **kw): calls.append(json.loads(kw['user_prompt'])); return {'utterance': 'capture'}
    data = {'instance_id': 'fixture', 'dataset': 'mlcr_aa', 'task_description': {'question': 'Q'},
            'participants': [{'participant_id': p['participant_id'], 'private_context_unit_ids': p['context_ids']}
                             for p in partition['participants']],
            'private_context_units': private_context_units(partition)}
    actor = LLMParticipantActor(Client(), PromptTraceRecorder())
    actor.respond(datapoint=data, participant_id='p1', state={'turn': 0, 'messages': []}, request=None,
                  budgets=BudgetSpec(participant_context_token_budget=40000))
    assert calls[0]['private_source_material'] == expected
    with pytest.raises(ValueError, match='differs'):
        consumer_full_context_materials(partition, token_budget=1)


def test_dependency_manifest_includes_actual_shared_clients_tokenizer_schema_and_consumers():
    from teamgym.data.medical.mlcr.execution import dependency_identity, assert_frozen_dependencies
    identity = dependency_identity()
    for path in ['teamgym/llm/openai_client.py', 'teamgym/llm/construction_client.py',
                 'teamgym/tokenization.py', 'teamgym/core/schema.py', 'teamgym/data/medical/export_schema.py',
                 'teamgym/interaction/llm_actors.py', 'teamgym/interaction/actor_utils.py',
                 'teamgym/interaction/run_single_agent_rui_evaluation.py', 'uv.lock']:
        assert path in identity['files']
    assert identity['tokenizer']['name'] == 'cl100k_base'
    assert len(identity['tokenizer']['vocabulary_sha256']) == 64
    assert_frozen_dependencies(identity)
    identity['files']['teamgym/interaction/llm_actors.py'] = 'changed'
    with pytest.raises(RuntimeError, match='changed during execution'):
        assert_frozen_dependencies(identity)
