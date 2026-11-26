"""Frozen Finance redistribution must run the same final-view gates as generation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES
from teamgym.core.schema import normalize_release_datapoint


@pytest.mark.parametrize('mode', PAPER_ALLOCATION_MODES)
@pytest.mark.parametrize('failure', [None, 'post_packet', 'full_context', 'piu_only'])
def test_findver_rechecks_and_both_consumer_conditions_gate_export(tmp_path, monkeypatch, mode, failure):
    from teamgym.data.finance.findver import rematerialize_allocations as module
    from tests.finance.findver.test_free_response_funnel import _three_riu_fixture
    candidate, selection, plan = _three_riu_fixture()
    monkeypatch.setattr(module, '_fixed_task_inputs', lambda *_a, **_kw: (candidate, selection, plan))
    packet_calls, panel_calls = [], []
    def packets(**kw):
        packet_calls.append(1)
        return [], ['failed recheck'] if failure == 'post_packet' and len(packet_calls)>1 else [], set()
    def panel(**kw):
        name = kw['panel_name']; panel_calls.append(name)
        assert kw['cases']
        return {'cases': kw['cases']}, ['participant solved alone'] if failure and name.endswith(failure) else []
    monkeypatch.setattr(module, '_run_packet_audits', packets)
    monkeypatch.setattr(module, '_run_answerability_panel', panel)
    result, scenario = module._rematerialize_one(wrapper={'instance_id':'fixture'}, reports_root=tmp_path,
        allocation_mode=mode, audit_client=None, trace_root=tmp_path, audit_attempts=1, seed=7)
    assert len(packet_calls) >= 2
    if failure:
        assert scenario is None and result['status'] != 'accepted'
    else:
        assert result['status'] == 'accepted'
        assert panel_calls == ['allocation_full_context','allocation_piu_only']
        assert scenario['common_datapoint']['instance_id'] != 'fixture'
        assert scenario['common_datapoint']['format_validation']['valid']


@pytest.mark.parametrize('mode', PAPER_ALLOCATION_MODES)
@pytest.mark.parametrize('failure', [None, 'salience', 'initial_view'])
def test_finlongdocqa_frozen_repartition_runs_final_gates_and_grader_export(tmp_path, monkeypatch, mode, failure):
    from teamgym.data.finance.finlongdocqa import generate_frozen_allocation_modes as module
    from teamgym.data.finance.finlongdocqa.partition import build_partitioned_datapoint
    from tests.finance.finlongdocqa.test_pipeline import _report_index, _three_riu_task, _mappings
    base = build_partitioned_datapoint(report_id='TEST_2024',company='TEST',year='2024',report_source={},
        report_index=_report_index(), task=_three_riu_task(),qa_mappings=_mappings(),solver_audit={},allocation_mode='one_each')
    original = deepcopy(base); calls=[]
    monkeypatch.setattr(module,'build_client',lambda *_:None)
    monkeypatch.setattr(module,'run_packet_leakage_repair',lambda **kw:(kw['datapoint'],[],set(),[]))
    def salience(**kw):
        calls.append('salience'); return [], ['bad salience'] if failure=='salience' else []
    def local(**kw):
        calls.append('initial_view'); assert kw['task']['rubric']==original['dataset_specific']['free_response_rubric']
        return [], ['participant answered'] if failure=='initial_view' else []
    monkeypatch.setattr(module,'run_packet_salience_audit',salience)
    monkeypatch.setattr(module,'run_initial_view_audit',local)
    summary=module._run_mode(mode=mode,rows=[base],input_path=tmp_path/'input.jsonl',output_root=tmp_path,
        allocation_seed=7,audit_workers=1,packet_repair_rounds=1,packet_audit_max_chars=100000,max_items=1)
    assert calls[0]=='salience'
    path=tmp_path/mode/'common_datapoints.jsonl'
    if failure:
        assert summary['accepted_count']==0 and not path.exists()
    else:
        assert calls==['salience','initial_view']
        assert summary['accepted_count']==1
        row=json.loads(path.read_text().splitlines()[0])
        assert row['format_validation']['valid']
        assert row['oracle']['required_decision_steps']
        assert row['task_description']==original['task_description']
        assert row['schema_version'] == 'teamgym_release_datapoint_v1'
        assert normalize_release_datapoint(row)['reference_information_units'] == original['reference_information_units']
    assert base==original


def test_finlongdocqa_audit_client_options_are_not_hardcoded(tmp_path):
    from teamgym.data.finance.finlongdocqa.generate_frozen_allocation_modes import _client_args
    args=_client_args(tmp_path,SimpleNamespace(model='custom-auditor',env_file='/custom.env',max_output_tokens=321))
    assert args.model=='custom-auditor' and args.env_file=='/custom.env' and args.max_output_tokens==321
