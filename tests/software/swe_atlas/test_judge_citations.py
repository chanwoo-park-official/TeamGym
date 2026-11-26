"""Positive grounded judgments need citations, including after JSON repair."""

from types import SimpleNamespace

import pytest

from teamgym.data.software.swe_atlas.pipeline import run_riu_audit, validate_audit_judge
from teamgym.data.software.swe_atlas.schema import Rubric


def judgment(polarity='positive', satisfied=True, citations=None):
    rubric = Rubric('r', 'A repository-specific fact', polarity, 'required', {})
    row = {'rubric_id': 'r', 'status': 'satisfied' if satisfied else 'missing',
           'content_correct': satisfied, 'grounded_in_available_material': satisfied,
           'supporting_evidence_numbers': citations}
    return rubric, {'rubric_results': [row], 'has_material_error': False,
                    'overall_complete_and_correct': satisfied}


@pytest.mark.parametrize('citations', [[], None, '', 0, False, {}, [0], [2], [True]])
def test_positive_grounded_judgment_requires_nonempty_valid_citations(citations):
    rubric, result = judgment(citations=citations)
    assert validate_audit_judge(result, (rubric,), 1)


def test_positive_grounded_judgment_cannot_omit_citation_field():
    rubric, result = judgment(citations=[])
    del result['rubric_results'][0]['supporting_evidence_numbers']
    assert validate_audit_judge(result, (rubric,), 1)


@pytest.mark.parametrize('polarity,satisfied,evidence_count,citations', [
    ('positive', True, 1, [1]),
    ('positive', False, 0, []),
    ('negative', True, 0, []),
])
def test_valid_support_and_empty_support_for_nonpositive_outcomes_remain_allowed(
        polarity, satisfied, evidence_count, citations):
    rubric, result = judgment(polarity, satisfied, citations)
    assert validate_audit_judge(result, (rubric,), evidence_count) == []


def test_citation_repair_receives_actual_material_and_solver_answer(tmp_path):
    rubrics = tuple(Rubric(f'r{i}', f'Fact {i}', 'positive', 'required', {}) for i in (1, 2))
    task = SimpleNamespace(task_id='citation-fixture',
        public=SimpleNamespace(question='PUBLIC_QUESTION'),
        evaluation=SimpleNamespace(rubrics=rubrics, reference_answer='REFERENCE_FIXTURE'))
    rius = [{'riu_id': f'unit_{i}', 'supports_rubric_ids': [f'r{i}'],
             'evidence_atoms': [{'kind': 'source', 'quote': f'FACT_{i}', 'source_path': f'{i}.py'}]}
            for i in (1, 2)]
    repairs = []
    class Client:
        def generate_json(self, **kwargs):
            pid = kwargs['prompt_id']
            if pid.endswith('.solver'):
                return {'answer': 'ANSWER_FIXTURE'}
            repaired = pid.endswith('.judge_repair')
            if repaired:
                repairs.append(kwargs)
                for text in ('PUBLIC_QUESTION', 'REFERENCE_FIXTURE', 'FACT_1', 'FACT_2', 'ANSWER_FIXTURE'):
                    assert text in kwargs['user_prompt']
                assert '{{' not in kwargs['user_prompt']
            rows = []
            available = [] if '.none.' in pid else ([2] if '.without_unit_1.' in pid
                        else [1] if '.without_unit_2.' in pid else [1, 2])
            for i in (1, 2):
                supported = i in available
                rows.append({'rubric_id': f'r{i}', 'status': 'satisfied' if supported else 'missing',
                    'content_correct': supported, 'grounded_in_available_material': supported,
                    'supporting_evidence_numbers': [available.index(i) + 1] if supported and
                        (repaired or len(available) == 1) else []})
            return {'rubric_results': rows, 'has_material_error': False,
                    'overall_complete_and_correct': len(available) == 2}
    audit = run_riu_audit(client=Client(), task=task, rius=rius, trace_dir=tmp_path, max_workers=1)
    assert len(repairs) == 1
    assert audit['accept'] is True
    assert audit['all']['judge_validation_errors'] == []
