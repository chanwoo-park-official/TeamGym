"""Offline negative controls for the explicit foreign-observation deletion delta."""

from tests._paths import REPO_ROOT
from pathlib import Path
import importlib.util
import json
import pytest

spec=importlib.util.spec_from_file_location('obs_delta',REPO_ROOT/'scripts/data_generation/software/deep_swe/observation_delta_audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


@pytest.fixture
def case():
    observations=[{'id':oid,'argv':['echo',oid],'stdout':oid,'stderr':'','exit_code':0} for oid in ('a','b','c','unallocated')]
    candidate={'observations':observations,'allocation_candidates':[{'observation_ids':[oid]} for oid in ('a','b','c')]}
    packet={'instruction':'public','code_fragments':[{'path':'source.go','line_start':1,'line_end':1,'text':'source\n'}],
            'observations':[a.visible(o) for o in observations[:3]]}
    before={'identity':'same','modes':{'single_agent_all_piu':[{'participant_id':'single','packet':json.loads(json.dumps(packet))}],
        'multi_agent_piu_only':[{'participant_id':f'participant_{i}','packet':json.loads(json.dumps(packet))} for i in (1,2,3)],
        a.FC:[{'participant_id':f'participant_{i}','packet':json.loads(json.dumps(packet))} for i in (1,2,3)]}}
    after=json.loads(json.dumps(before));after['modes'][a.FC][2]['packet']['observations'].pop(1)
    plan=[{'participant_id':'participant_3','observation_id':'b','reason':'specific retained packet finding'}]
    return before,after,candidate,plan


def test_only_foreign_fact_is_removed_and_other_six_views_source_remain_exact(case):
    result=a.verify(*case);assert result['all_other_views_exact']==6
    assert result['changed_FC_participants']==['participant_3']
    assert result['all_FC_source_code_exact'] and result['foreign_context_still_present']


@pytest.mark.parametrize('change',['owned','unallocated','not_visible','duplicate_spec','missing_reason','source_redistributed','common_changed','additional_removed','replacement_added','all_foreign_removed'])
def test_narrow_deletion_cannot_cover_other_packet_changes(case,change):
    before,after,candidate,plan=case;target=after['modes'][a.FC][2]['packet']
    if change=='owned':plan[0]['observation_id']='c'
    elif change=='unallocated':plan[0]['observation_id']='unallocated'
    elif change=='not_visible':
        before['modes'][a.FC][2]['packet']['observations'].pop(1)
    elif change=='duplicate_spec':plan.append(dict(plan[0]))
    elif change=='missing_reason':plan[0]['reason']=''
    elif change=='source_redistributed':target['code_fragments'][0]['text']='different source\n'
    elif change=='common_changed':target['instruction']='new requirements'
    elif change=='additional_removed':target['observations'].pop(0)
    elif change=='replacement_added':target['observations'].append(a.visible(candidate['observations'][3]))
    elif change=='all_foreign_removed':
        plan.append({'participant_id':'participant_3','observation_id':'a','reason':'second exclusion'})
        target['observations'].pop(0)
    with pytest.raises(AssertionError):a.verify(before,after,candidate,plan)
