from copy import deepcopy

import pytest

from teamgym.interaction.rui_context import all_rui_context_text, model_common_information, rui_context_units, rui_only_datapoint


@pytest.mark.parametrize('dataset', ['findver', 'finlongdocqa', 'evidencebench', 'mlcr_aa', 'swe_atlas'])
def test_all_shared_piu_renderers_exclude_constructor_annotations(dataset):
    row = {'dataset': dataset, 'reference_information_units': [{
        'rui_id': 'PRIVATE_ID_SENTINEL', 'information_feature': 'PRIVATE_FEATURE_SENTINEL',
        'summary': 'PRIVATE_SUMMARY_SENTINEL', 'text': 'Exact public source.\n'}],
        'participants': [{'participant_id': 'p1', 'owned_rui_ids': ['PRIVATE_ID_SENTINEL']}]}
    original = deepcopy(row)
    transformed = rui_only_datapoint(row)
    assert transformed['private_context_units'][0]['text'] == 'Exact public source.\n'
    assert all_rui_context_text(row) == 'Exact public source.\n'
    assert rui_context_units(row)[0]['text'] == 'Exact public source.\n'
    assert row == original


def test_missing_source_cannot_fall_back_to_hidden_summary():
    with pytest.raises(ValueError, match='source text'):
        all_rui_context_text({'dataset': 'evidencebench', 'reference_information_units': [
            {'rui_id': 'private', 'summary': 'gold answer'}]})


def test_legal_renderer_keeps_its_existing_contract():
    text = all_rui_context_text({'dataset': 'maud', 'reference_information_units': [
        {'rui_id': 'r1', 'text': 'legal source', 'summary': 'existing annotation'}]})
    assert 'RIU r1' in text and 'Summary: existing annotation' in text


def test_actual_actor_task_receives_public_raw_facts_without_common_piu_annotations():
    from teamgym.interaction.llm_actors import _public_task_payload
    row = {'dataset': 'findver', 'initial_common_information': {
        'public_facts': 'Public company name', 'common_reference_units': [{
            'rui_id': 'PRIVATE_ID', 'summary': 'PRIVATE_SUMMARY',
            'information_feature': 'PRIVATE_FEATURE', 'text': 'Exact public source'}]}}
    expected = {'public_facts': 'Public company name',
                'common_reference_units': [{'text': 'Exact public source'}]}
    assert model_common_information(row) == expected
    assert _public_task_payload(row)['initial_common_information'] == expected
    assert row['initial_common_information']['common_reference_units'][0]['summary'] == 'PRIVATE_SUMMARY'
