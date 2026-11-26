import hashlib
import json
import pytest
from copy import deepcopy
from teamgym.data.software.deep_swe.full_context import pack, audit_packet, exclude_background, tokens
from teamgym.data.software.deep_swe.model_actions import anthropic_body


def sources():
    files={'a.py':'# context A\ndef a():\n    return "A"\n# extra A\n',
           'b.py':'# context B\ndef b():\n    return "B"\n# extra B\n',
           'other.py':'# Background\n'+'print("unrelated")\n'*60}
    atoms=[]
    for i,path in enumerate(('a.py','b.py'),1):
        text=''.join(files[path].splitlines(keepends=True)[1:3])
        atoms.append({'id':f'atom-{i:02d}','path':path,'line_start':2,'line_end':3,
                      'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()})
    return {'code_atoms':atoms,'observations':[], 'allocation_candidates':[
        {'riu_id':f'riu-{i:02d}','code_atom_ids':[a['id']],'observation_ids':[]} for i,a in enumerate(atoms,1)]},files


def test_packing_keeps_evidence_exclusive_and_grounded():
    c,files=sources()
    p,audit=pack(c,files,[{'id':'obs-a','argv':['test','a'],'stdout':'pass','stderr':'','exit_code':0},
                        {'id':'obs-b','argv':['test','b'],'stdout':'pass','stderr':'','exit_code':0}],1000)
    for participant in p:
        audit_packet(c,participant,files,1000)
        body=json.dumps(participant['private_source_material'])
        assert 'riu-' not in body and 'atom-' not in body
        assert participant['private_source_material']['observations']==[]  # Unallocated pool is not eligible distractor evidence.
    assert 'return \\"B\\"' not in json.dumps(p[0]['private_source_material'])
    with pytest.raises(ValueError,match='foreign_riu'):
        p[0]['private_source_material']['code_fragments'].append(
            {k:c['code_atoms'][1][k] for k in ('path','line_start','line_end','text')})
        audit_packet(c,p[0],files,2000)


def test_native_tool_history_retains_assistant_and_tool_result():
    b={'max_output_tokens':64,'tools':[{'name':'edit','parameters':{'type':'object'}}],
       'input':[{'role':'developer','content':'rules'}, {'role':'user','content':'task'},
          {'type':'function_call','call_id':'c1','name':'edit','arguments':'{"action":"edit"}'},
          {'type':'function_call_output','call_id':'c1','output':'{"edit":"applied"}'},
          {'role':'user','content':'remaining=4'}]}
    n=anthropic_body(b)
    assert [x['role'] for x in n['messages']]==['user','assistant','user']
    assert n['messages'][1]['content'][0]['input']=={'action':'edit'}
    assert n['messages'][2]['content'][0]['tool_use_id']=='c1'
    assert len(n['messages'][2]['content'])==2
    assert n['tool_choice']=={'type':'tool','name':'edit'}


def exclusion(**changes):
    return {'participant_id': 'participant_1', 'path': 'other.py',
            'line_start': 20, 'line_end': 30,
            'reason': 'Independent audit found reconstruction through this background.', **changes}


def test_post_pack_exclusion_splits_exact_source_without_refill_or_other_view_changes():
    c, files = sources()
    pool = [{'id': 'extra', 'argv': ['probe'], 'stdout': 'existing', 'stderr': '', 'exit_code': 0}]
    before, old_manifest = pack(c, files, pool, 1000)
    assert pack(c, files, pool, 1000, background_exclusions=None) == (before, old_manifest)
    assert pack(c, files, pool, 1000, background_exclusions=[]) == (before, old_manifest)
    after, manifest = pack(c, files, pool, 1000, background_exclusions=[exclusion()])
    assert after[1] == before[1]
    assert c['code_atoms'] == sources()[0]['code_atoms']
    assert after[0]['private_source_material']['observations'] == before[0]['private_source_material']['observations']
    original_blocks = [s for s in before[0]['private_source_material']['code_fragments'] if s['path'] != 'other.py']
    assert original_blocks == [s for s in after[0]['private_source_material']['code_fragments'] if s['path'] != 'other.py']
    pieces = [s for s in after[0]['private_source_material']['code_fragments'] if s['path'] == 'other.py']
    assert [(s['line_start'], s['line_end']) for s in pieces] == [(1, 19), (31, 60)]
    for piece in pieces:
        assert piece['text'] == ''.join(files['other.py'].splitlines(keepends=True)[piece['line_start']-1:piece['line_end']])
    ledger = manifest['background_exclusions_hidden']
    removed = ledger['entries'][0]['exclusions'][0]['removed_intervals'][0]
    assert removed['text_sha256'] == hashlib.sha256(''.join(files['other.py'].splitlines(keepends=True)[19:30]).encode()).hexdigest()
    assert ledger['phase'] == 'post_pack_no_refill_or_redistribution'
    assert 'reason' not in json.dumps(after[0]['private_source_material'])
    assert 'exclusion' not in json.dumps(after[0]['private_source_material'])
    for participant in after:
        audit_packet(c, participant, files, 1000)


def test_requested_source_function_may_extend_beyond_visible_background_but_ledger_is_exact():
    c, files = sources()
    before, _ = pack(c, files, [], 1000)
    after, manifest = pack(c, files, [], 1000,
        background_exclusions=[exclusion(line_start=50, line_end=61)])
    assert after[1] == before[1]
    intervals = manifest['background_exclusions_hidden']['entries'][0]['exclusions'][0]['removed_intervals']
    assert [(r['line_start'], r['line_end']) for r in intervals] == [(50, 60)]
    assert any(r['path']=='other.py' and r['line_end']==49 for r in after[0]['private_source_material']['code_fragments'])


@pytest.mark.parametrize('spec,error', [
    (exclusion(participant_id='unknown'), 'unknown_participant'),
    (exclusion(path='../other.py'), 'invalid_path'),
    (exclusion(path='/other.py'), 'invalid_path'),
    (exclusion(path='a/../other.py'), 'invalid_path'),
    (exclusion(path='missing.py'), 'invalid_path'),
    (exclusion(line_start=True), 'invalid_bounds'),
    (exclusion(line_start=0), 'invalid_bounds'),
    (exclusion(line_end=62), 'invalid_bounds'),
    (exclusion(line_start=31), 'invalid_bounds'),
    (exclusion(reason=' '), 'reason_required'),
    (exclusion(path='a.py', line_start=1, line_end=2), 'canonical_overlap'),
    (exclusion(path='b.py', line_start=2, line_end=3), 'canonical_overlap'),
    (exclusion(line_start=61, line_end=61), 'not_visible'),
])
def test_exclusions_reject_invalid_plan_or_any_owned_or_foreign_canonical_overlap(spec,error):
    c, files = sources()
    with pytest.raises(ValueError, match=error):
        pack(c, files, [], 1000, background_exclusions=[spec])


def test_exclusion_plan_rejects_overlaps_and_validates_every_spec_before_changes():
    c, files = sources()
    participants, _ = pack(c, files, [], 1000)
    before = deepcopy(participants)
    with pytest.raises(ValueError, match='overlapping_specs'):
        exclude_background(c, participants, files, 1000, [exclusion(), exclusion(line_start=30, line_end=35)])
    assert participants == before


def test_interval_split_is_reaudited_for_metadata_token_overhead():
    c, files = sources()
    participants, _ = pack(c, files, [], 1000)
    before_budget = tokens(participants[0]['private_source_material'])
    # Removing one short line creates a second normal source block; the extra
    # path/range envelope can outweigh the deleted source and must be counted.
    with pytest.raises(ValueError, match='private_packet_budget'):
        exclude_background(c, participants, files, before_budget,
                           [exclusion(line_start=20, line_end=20)])


@pytest.mark.parametrize('owner_count', [3,5])
def test_observation_distractors_cap_each_foreign_piu_preserve_owned_and_exclude_unknown_pool(owner_count):
    c,files=sources();pool=[]
    for owner in range(owner_count):
        ids=[]
        for offset in range(2):
            oid=f'owned-{owner}-{offset}';ids.append(oid)
            pool.append({'id':oid,'argv':['probe',str(owner),str(offset)],'stdout':str(offset),'stderr':'','exit_code':0})
        if owner>=len(c['allocation_candidates']):
            c['allocation_candidates'].append({'riu_id':f'riu-{owner+1:02d}','code_atom_ids':[],'observation_ids':[]})
        c['allocation_candidates'][owner]['observation_ids']=ids
    c['observations']=deepcopy(pool)
    pool.insert(1,{'id':'unallocated','argv':['unknown-family'],'stdout':'not eligible','stderr':'','exit_code':0})
    participants,manifest=pack(c,files,pool,2000)
    assert manifest['unallocated_observation_distractors_excluded_hidden']==['unallocated']
    for index,p in enumerate(participants):
        ledger=manifest['observation_allocation_hidden'][p['participant_id']]
        assert len(ledger['distractor'])==min(3,owner_count-1)
        assert len(set(ledger['distractor_foreign_piu_ids']))==len(ledger['distractor'])
        assert c['allocation_candidates'][index]['riu_id'] not in ledger['distractor_foreign_piu_ids']
        visible=p['private_source_material']['observations']
        expected_ids=set(ledger['owned']+ledger['distractor'])
        expected=[{k:row[k] for k in ('argv','stdout','stderr','exit_code')} for row in pool if row['id'] in expected_ids]
        assert visible==expected  # Existing collection order, including both owned observations.
        assert len(ledger['owned'])==2
        assert 'unknown-family' not in json.dumps(visible)
        assert 'riu-' not in json.dumps(visible) and 'owned-' not in json.dumps(visible)


def observation_sources():
    c,files=sources();pool=[]
    c['allocation_candidates'].append({'riu_id':'riu-03','code_atom_ids':[],'observation_ids':[]})
    for i,allocation in enumerate(c['allocation_candidates']):
        for j in range(2):
            oid=f'obs-{i}-{j}';allocation['observation_ids'].append(oid)
            pool.append({'id':oid,'argv':['probe',str(i),str(j)],'stdout':str(j),'stderr':'','exit_code':0})
    c['observations']=deepcopy(pool)
    pool.append({'id':'unallocated','argv':['unallocated'],'stdout':'unused','stderr':'','exit_code':0})
    return c,files,pool


def test_observation_exclusion_is_post_pack_without_code_rebalance_refill_or_other_changes():
    c,files,pool=observation_sources();frozen=deepcopy((c,files,pool))
    before,manifest=pack(c,files,pool,1000)
    assert pack(c,files,pool,1000,observation_background_exclusions=None)==(before,manifest)
    assert pack(c,files,pool,1000,observation_background_exclusions=[])==(before,manifest)
    target='participant_3';oid=manifest['observation_allocation_hidden'][target]['distractor'][0]
    spec={'participant_id':target,'observation_id':oid,'reason':'Actual audit identified a substituted foreign functional boundary.'}
    after,receipt=pack(c,files,pool,1000,observation_background_exclusions=[spec])
    assert before[:2]==after[:2]
    for old,new in zip(before,after):
        assert old['private_source_material']['code_fragments']==new['private_source_material']['code_fragments']
        assert old['allocation_hidden']==new['allocation_hidden']
    visible={k:next(o for o in pool if o['id']==oid)[k] for k in ('argv','stdout','stderr','exit_code')}
    expected=deepcopy(before[2]);expected['private_source_material']['observations'].remove(visible)
    assert after[2]==expected
    assert receipt['existing_observation_distractors_per_participant'][target]==1
    assert (c,files,pool)==frozen
    ledger=receipt['observation_background_exclusions_hidden']
    assert ledger['phase']=='post_pack_no_refill_replacement_or_redistribution'
    entry=ledger['entries'][0]
    assert entry['visible_sha256']==hashlib.sha256(json.dumps(visible,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert entry['before_tokens']==tokens(before[2]['private_source_material'])
    assert entry['after_tokens']==tokens(after[2]['private_source_material'])
    assert 'reason' not in json.dumps(after[2]['private_source_material'])
    canonical=__import__('teamgym.data.software.deep_swe.full_context',fromlist=['build_participants']).build_participants(c)
    for owned in canonical[2]['private_source_material']['observations']:
        assert owned in after[2]['private_source_material']['observations']


@pytest.mark.parametrize('mutation,error',[
    (lambda s:s.update(participant_id='unknown'),'unknown_participant'),
    (lambda s:s.update(observation_id='absent'),'unknown_observation'),
    (lambda s:s.update(observation_id='unallocated'),'unallocated'),
    (lambda s:s.update(observation_id='obs-2-0'),'owned'),
    (lambda s:s.update(reason=' '),'reason_required'),
    (lambda s:s.update(extra='field'),'schema'),
])
def test_observation_exclusions_reject_invalid_or_canonical_removal(mutation,error):
    c,files,pool=observation_sources();_,manifest=pack(c,files,pool,1000)
    spec={'participant_id':'participant_3','observation_id':manifest['observation_allocation_hidden']['participant_3']['distractor'][0],'reason':'Actual audit'}
    mutation(spec)
    with pytest.raises(ValueError,match=error):
        pack(c,files,pool,1000,observation_background_exclusions=[spec])


def test_observation_exclusions_reject_duplicate_or_nonvisible_allocated_observation():
    c,files,pool=observation_sources();_,manifest=pack(c,files,pool,1000)
    selected=manifest['observation_allocation_hidden']['participant_3']['distractor']
    spec={'participant_id':'participant_3','observation_id':selected[0],'reason':'Actual audit'}
    with pytest.raises(ValueError,match='duplicate'):
        pack(c,files,pool,1000,observation_background_exclusions=[spec,spec])
    omitted=next(o['id'] for o in c['observations'] if o['id'] not in selected and o['id'] not in c['allocation_candidates'][2]['observation_ids'])
    with pytest.raises(ValueError,match='not_selected_visible'):
        pack(c,files,pool,1000,observation_background_exclusions=[{**spec,'observation_id':omitted}])
