"""Exercise the actual generic grader against domain-owned export adapters."""
import json
from pathlib import Path
import pytest

from teamgym.scoring.short_answers import judge_one_answer


def export_fixture(tmp_path, domain):
    root = tmp_path / domain; root.mkdir()
    units = [{'unit_id': f'u{i}', 'riu_id': f'u{i}', 'candidate_id': f'cand{i}',
              'text': text, 'hidden_claim': text, 'supports_claim_ids': [f'claim{i}'],
              'required_claim_ids': [f'claim{i}'], 'source_document_id': f'doc{i}',
              'raw_start': 0, 'raw_end': len(text), 'sha256': str(i) * 64,
              'evidence_atoms': [{'text': text, 'sentence_indices': [i]}]}
             for i, text in enumerate(['Alpha.', 'Beta.'], 1)]
    contexts = [{'context_id': f'x{i}', 'text': u['text'], 'source_document_id': f'doc{i}',
                 'source_file': f'{i:03d}.txt', 'document_order': i, 'source_order': 1,
                 'sentence_index': i} for i, u in enumerate(units, 1)]
    participants = [{'participant_id': f'p{i}', 'hidden_owned_unit_ids': [f'u{i}'],
                     'hidden_owned_riu_ids': [f'u{i}'], 'context_ids': [f'x{i}'], 'context_tokens': 20}
                    for i in [1, 2]]
    claims = [{'claim_id': f'claim{i}', 'description': text, 'expected_answer_element': text,
               'criterion': text, 'grading_rule': 'State the exact recorded fact.',
               'support_logic': 'ALL_REQUIRED', 'supporting_candidate_ids': [f'cand{i}']}
              for i, text in enumerate(['Alpha.', 'Beta.'], 1)]
    rubric = {'required_claims': claims, 'scope_constraints': ['Preserve clinical scope.'],
              'prohibited_errors': ['Do not invent a clinical diagnosis.'],
              'disqualifying_errors': ['Do not invent a measurement.']}
    task = {'public_task': {'question': 'State both recorded findings.'}, 'evidence_units': units,
            'rubric': rubric, 'oracle': {'reference_answer': 'Alpha. Beta.'}, 'canonical_answer': 'Alpha. Beta.'}
    if domain == 'mlcr':
        from teamgym.data.medical.mlcr.evaluation import build_evaluation_datapoint
        report = {'status': 'accepted', 'case_id': 'case', 'task': task,
                  'grounding_audit': {'passed': True}, 'boundary_audits': {'passed': True},
                  'packet_leakage_audit': {'passed': True},
                  'partition': {'participants': participants, 'contexts': contexts, 'excluded_context_ids': []}}
        path = root / 'run_report.json'; path.write_text(json.dumps(report))
        from tests.medical.certified_fixtures import complete_fixture
        complete_fixture(root, domain)
        return build_evaluation_datapoint(path)
    from teamgym.data.medical.evidencebench.evaluation import build_evaluation_datapoint
    files = {'summary.json': {'status': 'accepted', 'instance_id': 'fixture'},
             '02_task_rubric.json': task, '05_final_rius.hidden.json': units,
             '06_participant_packets.hidden.json': {'allocation_policy': 'one_each_disjoint_full_paper',
                                                    'participants': participants, 'contexts': contexts}}
    for name, value in files.items(): (root / name).write_text(json.dumps(value))
    from tests.medical.certified_fixtures import complete_fixture
    complete_fixture(root, domain)
    return build_evaluation_datapoint(root)


@pytest.mark.parametrize('domain', ['evidencebench', 'mlcr'])
@pytest.mark.parametrize('partial', [False, True])
def test_real_generic_grader_receives_all_claims_scope_errors_and_strict_policy(tmp_path, domain, partial):
    datapoint = export_fixture(tmp_path, domain)
    calls = []
    class Client:
        last_request_meta = {}
        def generate_json(self, **kwargs):
            payload = json.loads(kwargs['user_prompt']); calls.append(payload)
            assert len(payload['required_decision_steps']) == 2
            assert [r['step'] for r in payload['required_decision_steps']] == [1, 2]
            assert len(payload['judge_rubric']['required_claims']) == 2
            assert payload['judge_rubric']['scope_constraints'] == ['Preserve clinical scope.']
            assert payload['disqualifying_errors']
            assert payload['pass_policy']['require_all_steps_for_pass'] is True
            covered = [True, not partial]
            return {'passed': not partial, 'score': .5 if partial else 1,
                    'conclusion_correct': not partial, 'covered_step_count': sum(covered),
                    'total_step_count': 2, 'step_scores': [
                        {'step': i, 'covered': value, 'score': int(value), 'reason': 'Fixture claim evidence.'}
                        for i, value in enumerate(covered, 1)],
                    'missing_steps': ['Beta.'] if partial else [], 'incorrect_claims': [],
                    'disqualifying_errors_found': [], 'reason': 'Complete' if not partial else 'Second claim omitted'}
    result = judge_one_answer(client=Client(), datapoint=datapoint,
        answer_row={'answer_text': 'Alpha.' if partial else 'Alpha. Beta.'}, trace_dir=None, index=1)
    assert len(calls) == 1
    assert result['judgment']['passed'] is (not partial)
    assert result['judgment']['pass_policy'] == 'all_required_decision_steps'
