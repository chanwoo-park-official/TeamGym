"""Exact independent audit of a declared post-pack foreign-observation deletion.

No model/container calls. The only allowed visible difference is removal of the
specified already-present, allocated non-owned observation(s). Source and all
other packet fields remain byte material identical; no refill or redistribution.
"""
from pathlib import Path
import argparse
import hashlib
import json

FC='multi_agent_full_context'
VISIBLE=('argv','stdout','stderr','exit_code')


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def visible(observation):return {key:observation[key] for key in VISIBLE}


def verify(before,after,candidate,specs):
    assert isinstance(specs,list) and specs,'Explicit exclusions required'
    assert set(before)==set(after)
    for key in set(before)-{'modes'}:assert before[key]==after[key],('bundle_metadata_changed',key)
    assert set(before['modes'])==set(after['modes'])
    observations={o['id']:o for o in candidate['observations']};owners={}
    for number,allocation in enumerate(candidate['allocation_candidates'],1):
        for oid in allocation['observation_ids']:
            assert oid in observations and oid not in owners
            owners[oid]=f'participant_{number}'
    oldviews={view['participant_id']:view for view in before['modes'][FC]}
    expected=json.loads(json.dumps(before));targetviews={view['participant_id']:view for view in expected['modes'][FC]}
    seen=set();removed=[]
    for spec in specs:
        assert isinstance(spec,dict) and set(spec)=={'participant_id','observation_id','reason'}
        pid,oid=spec['participant_id'],spec['observation_id'];assert isinstance(spec['reason'],str) and spec['reason'].strip()
        assert pid in targetviews and oid in owners and owners[oid]!=pid,'Exclusion must be allocated foreign evidence'
        assert (pid,oid) not in seen;seen.add((pid,oid))
        packet=targetviews[pid]['packet'];needle=visible(observations[oid]);matches=[n for n,row in enumerate(packet['observations']) if row==needle]
        assert len(matches)==1,'Exclusion must be exactly one currently visible observation'
        packet['observations'].pop(matches[0]);removed.append({'participant_id':pid,'observation_id':oid,'foreign_owner':owners[oid],'visible_observation_sha256':canonical(needle)})
    assert expected==after,'Visible delta exceeds declared observation deletion (no source redistribution/refill permitted)'
    unaffected=0;changed=[]
    for mode,views in before['modes'].items():
        new={v['participant_id']:v for v in after['modes'][mode]};assert len(new)==len(views)
        for view in views:
            pid=view['participant_id'];value=new[pid]
            if view==value:unaffected+=1
            else:
                assert mode==FC;changed.append(pid)
                assert view['packet']['code_fragments']==value['packet']['code_fragments'],'Full-context code changed'
                owned=[visible(observations[oid]) for oid,owner in owners.items() if owner==pid]
                assert all(value['packet']['observations'].count(o)==1 for o in owned),'Owned observations changed'
                foreign=[visible(observations[oid]) for oid,owner in owners.items() if owner!=pid]
                assert any(o in value['packet']['observations'] for o in foreign),'All foreign runtime context removed'
    return {'schema':'deepswe_independent_observation_exclusion_delta_v1','status':'passed',
        'only_declared_visible_observations_removed':True,'removed':removed,'changed_FC_participants':changed,
        'all_other_views_exact':unaffected,'all_FC_source_code_exact':True,'SA_and_PIU_only_exact':True,
        'owned_observations_exact':True,'foreign_context_still_present':True,'no_refill_rebalance_replacement':True,
        'before_bundle_sha256':canonical(before),'after_bundle_sha256':canonical(after),'model_calls':0,'container_runs':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','candidate','spec','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=verify(read(args.before),read(args.after),read(args.candidate),read(args.spec))
    result['input_file_sha256']={name:digest(getattr(args,name)) for name in ('before','after','candidate','spec')}
    assert args.output.resolve() not in {getattr(args,k).resolve() for k in ('before','after','candidate','spec')}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
