from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from teamgym.data.medical.evidencebench.validation import (
    validate_candidates, validate_boundary_judgment, validate_leakage_audit,
)
from teamgym.data.medical.evidencebench.llm import ResponseCache
from tests.medical.evidencebench.test_independent_pipeline import _record, _candidates, _rubric


def judgment():
    return {'content_complete': True, 'grounded_complete': True,
            'claim_results': [{'claim_id': 'claim_01', 'content_correct': True,
                              'grounded_in_available_material': True, 'satisfied': True,
                              'evidence_numbers': ['e1', 'e2']}],
            'unsupported_claims': [], 'critical_errors': []}


def validate(row):
    return validate_boundary_judgment(row, rubric=_rubric(), available_numbers={'e1', 'e2'},
                                     candidate_by_number={'e1': 'cand_02', 'e2': 'cand_03'})


def test_judge_global_flags_and_duplicate_claims_cannot_override_rows():
    good = judgment()
    assert not validate(good)
    assert validate({**good, 'content_complete': False})
    assert validate({**good, 'grounded_complete': False})
    assert validate({**good, 'claim_results': [*good['claim_results'], *good['claim_results']]})


def test_candidate_complete_span_is_contiguous_and_own_anchor_local():
    record = _record()
    record['paper_as_candidate_pool'] += ['irrelevant'] * 10
    candidates = _candidates()
    anchors = [{'anchor_id': f'anchor_{i:02d}', 'sentence_index': n,
                'adjacent_context': [{'sentence_index': j} for j in (n, 12)]}
               for i, n in enumerate((1, 2, 3), 1)]
    candidates[0]['evidence_atoms'].append({'atom_id': 'remote', 'sentence_indices': [12], 'text': 'irrelevant'})
    assert 'complete candidate span must be contiguous' in ' '.join(validate_candidates(
        {'valid': True, 'candidate_rius': candidates}, record=record, anchors=anchors))
    anchors[0]['adjacent_context'] = [{'sentence_index': 1}]
    assert 'own anchor neighborhood' in ' '.join(validate_candidates(
        {'valid': True, 'candidate_rius': candidates}, record=record, anchors=anchors))


def test_leakage_canonical_reconstruction_and_id_contradictions():
    row = {'participant_id': 'p', 'foreign_riu_results': [{
        'riu_id': 'foreign', 'reconstructable_from_packet': False, 'canonical_overlap': True,
        'leakage_support_context_ids': ['c'], 'removable_noncanonical_context_ids': []}], 'errors': []}
    args = dict(participant_id='p', foreign_riu_ids={'foreign'}, packet_ids={'c'}, canonical_ids={'c'})
    assert validate_leakage_audit(row, **args)
    row['foreign_riu_results'][0].update(reconstructable_from_packet=True, canonical_overlap=False)
    assert not validate_leakage_audit(row, **args)
    row['foreign_riu_results'] *= 2
    assert validate_leakage_audit(row, **args)


def test_cache_binds_provider_inference_and_recovers_truncated_jsonl(tmp_path):
    calls = []
    class Client:
        def __init__(self, provider): self.cfg = SimpleNamespace(model='same', provider=provider)
        def generate_json(self, **kwargs): calls.append(1); return {'valid': True}
    path = tmp_path / 'cache.jsonl'
    path.write_text('{truncated')
    cache = ResponseCache(path)
    args = dict(stage='test', instance_id='id', system_prompt='system', user_prompt='user', trace_dir=tmp_path)
    cache.call(client=Client('one'), **args)
    cache.call(client=Client('one'), **args)
    cache.call(client=Client('two'), **args)
    assert len(calls) == 2


def test_third_packet_repair_gets_a_final_recheck(monkeypatch, tmp_path):
    from teamgym.data.medical.evidencebench import partition
    packet = {'participants': [{'participant_id': 'p', 'hidden_owned_riu_ids': ['own'],
               'hidden_canonical_context_ids': ['own'], 'context_ids': ['own', 'x', 'y', 'z']}],
              'contexts': [{'context_id': c, 'text': c, 'sentence_index': i}
                           for i, c in enumerate(['own', 'x', 'y', 'z'])]}
    calls = []
    def structured(**kwargs):
        calls.append(kwargs['stage'])
        ids = [r['context_id'] for r in kwargs['variables']['PACKET_JSON'] if r['context_id'] != 'own']
        return {'foreign_riu_results': [{'riu_id': 'foreign', 'reconstructable_from_packet': bool(ids),
                 'canonical_overlap': False, 'leakage_support_context_ids': ids[:1],
                 'removable_noncanonical_context_ids': ids[:1]}]}, []
    monkeypatch.setattr(partition, 'structured_call', structured)
    rius = [{'riu_id': c, 'hidden_claim': c, 'required_claim_ids': [], 'evidence_atoms': []}
            for c in ['own', 'foreign']]
    cleaned, audit = partition.audit_and_sanitize_packets(instance_id='id', public_task={}, final_rius=rius,
        partition=packet, client=None, cache=None, trace_root=tmp_path, attempts=1, max_rounds=3)
    assert audit['passed'] and len(calls) == 4
    assert cleaned['participants'][0]['context_ids'] == ['own']
