import pytest

from teamgym.scoring.short_answers import normalize_judgment


def full_response():
    return {'passed': True, 'conclusion_correct': True, 'score': 1,
        'covered_step_count': 2, 'total_step_count': 2,
        'step_scores': [{'step': 1, 'covered': True, 'score': 1},
                        {'step': 2, 'covered': True, 'score': 1}],
        'missing_steps': [], 'incorrect_claims': [], 'disqualifying_errors_found': []}


STEPS = [{'step': 1, 'claim': 'First independent claim'},
         {'step': 2, 'claim': 'Second independent claim'}]


def normalized(response):
    return normalize_judgment(response, required_steps=STEPS, require_all_steps_for_pass=True)


def test_consistent_complete_answer_passes_strict_policy():
    result = normalized(full_response())
    assert result['passed'] is True
    assert result['covered_step_count'] == result['total_step_count'] == 2
    assert result['judgment_consistency_errors'] == []


def test_aggregate_cannot_claim_uncovered_step_was_covered():
    response = full_response()
    response['step_scores'][1]['covered'] = False
    result = normalized(response)
    assert result['passed'] is False
    assert result['covered_step_count'] == 1
    assert 'aggregate_coverage_disagrees_with_steps' in result['judgment_consistency_errors']


@pytest.mark.parametrize('bad_id', [1, 'unknown'])
def test_duplicate_or_unknown_rows_cannot_replace_an_expected_step(bad_id):
    response = full_response()
    response['step_scores'][1]['step'] = bad_id
    result = normalized(response)
    assert result['passed'] is False
    assert result['covered_step_count'] == 1


@pytest.mark.parametrize('bad_covered', ['false', 'true', 1, 0, None])
def test_only_literal_boolean_true_counts_as_step_coverage(bad_covered):
    response = full_response()
    response['step_scores'][1]['covered'] = bad_covered
    assert normalized(response)['passed'] is False


@pytest.mark.parametrize('field,value', [('covered_step_count', True), ('covered_step_count', 9),
    ('covered_step_count', '2'), ('total_step_count', 1), ('conclusion_correct', 'false'),
    ('passed', 'true')])
def test_invalid_or_contradictory_summaries_do_not_pass(field, value):
    response = full_response()
    response[field] = value
    assert normalized(response)['passed'] is False


@pytest.mark.parametrize('field', ['incorrect_claims', 'missing_steps', 'disqualifying_errors_found'])
def test_material_error_lists_veto_a_claimed_complete_answer(field):
    response = full_response()
    response[field] = ['A required clinical scope is missing']
    assert normalized(response)['passed'] is False


def test_partial_answer_is_valid_but_not_strictly_complete():
    response = full_response()
    response['step_scores'][1]['covered'] = False
    response['covered_step_count'] = 1
    response['missing_steps'] = ['Second independent claim']
    result = normalized(response)
    assert result['passed'] is False
    assert result['judgment_consistency_errors'] == []


def test_holistic_policy_remains_separate_from_all_step_policy():
    response = full_response()
    response['step_scores'][1]['covered'] = False
    response['covered_step_count'] = 1
    response['missing_steps'] = ['Second claim']
    assert normalize_judgment(response, required_steps=STEPS)['passed'] is True
    assert normalized(response)['passed'] is False


def test_explicit_negative_overall_judgment_vetoes_strict_success():
    response = full_response()
    response['passed'] = False
    assert normalized(response)['passed'] is False


def test_model_cannot_invent_required_steps_when_strict_config_is_empty():
    result = normalize_judgment(full_response(), required_steps=[], require_all_steps_for_pass=True)
    assert result['passed'] is False
    assert 'strict_policy_missing_required_steps' in result['judgment_consistency_errors']


@pytest.mark.parametrize('identifier_field', ['step_id', 'claim_id', 'id'])
def test_unique_displayed_id_aliases_count_as_the_same_required_steps(identifier_field):
    required = [{'step': i, identifier_field: f'claim_{i:02d}'} for i in (1, 2)]
    response = full_response()
    for i, row in enumerate(response['step_scores'], 1):
        row['step'] = f'claim_{i:02d}'
    result = normalize_judgment(response, required_steps=required, require_all_steps_for_pass=True)
    assert result['passed'] is True
    assert [row['step'] for row in result['step_scores']] == [1, 2]
    assert result['judgment_consistency_errors'] == []


def test_duplicate_coverage_via_two_aliases_is_still_rejected():
    response = full_response()
    response['step_scores'][1]['step'] = 'claim_one'
    result = normalize_judgment(response, required_steps=[
        {'step': 1, 'claim_id': 'claim_one'}, {'step': 2, 'claim_id': 'claim_two'}],
        require_all_steps_for_pass=True)
    assert result['passed'] is False
    assert 'duplicate_judged_step_id' in result['judgment_consistency_errors']


def test_an_alias_shared_by_different_steps_does_not_credit_either():
    response = full_response()
    response['step_scores'][0]['step'] = 'shared'
    result = normalize_judgment(response, required_steps=[
        {'step': 1, 'claim_id': 'shared'}, {'step': 2, 'claim_id': 'shared'}],
        require_all_steps_for_pass=True)
    assert result['passed'] is False
    assert 'ambiguous_judged_step_id' in result['judgment_consistency_errors']


def test_declared_string_ids_accept_unambiguous_positional_numbers():
    result = normalize_judgment(full_response(), required_steps=[
        {'step': 'claim_a'}, {'step': 'claim_b'}], require_all_steps_for_pass=True)
    assert result['passed'] is True
    assert [row['step'] for row in result['step_scores']] == ['claim_a', 'claim_b']


def test_actual_grader_reports_inconsistent_json_as_judge_error_not_wrong_answer():
    from teamgym.scoring.short_answers import judge_one_answer

    class Client:
        last_request_meta = {}
        def generate_json(self, **kw):
            response = full_response()
            response['step_scores'][1]['covered'] = False
            return response

    with pytest.raises(ValueError, match='Inconsistent strict judgment'):
        judge_one_answer(client=Client(), datapoint={'instance_id': 'case',
            'oracle': {'required_decision_steps': STEPS,
                       'judge_rubric': {'require_all_steps_for_pass': True}}},
            answer_row={'answer_text': 'answer'}, trace_dir=None, index=1)
