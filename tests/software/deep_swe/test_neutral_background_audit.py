"""Offline adversarial checks for the independent neutral-packet audit."""

from tests._paths import REPO_ROOT
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('neutral_audit',REPO_ROOT/'scripts/data_generation/software/deep_swe/neutral_background_audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def blob(text):
    raw=text.encode();return {'text':text,'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'bytes':len(raw)}


@pytest.fixture
def case():
    original={'first.go':{'text':'own first\nprivate other\n'},'second.go':{'text':'own second\nprivate third\n'}}
    atoms=[{'id':'a','path':'first.go','line_start':1,'line_end':1,'text':'own first\n'},
           {'id':'b','path':'second.go','line_start':1,'line_end':1,'text':'own second\n'}]
    execution={'source':{'base_commit':'abc'},'runtime_image':{'id':'sha256:runtime'}}
    candidate={'task_id':'task','execution_manifest':execution,'code_atoms':atoms,'allocation_candidates':[
        {'riu_id':'riu-01','code_atom_ids':['a']},{'riu_id':'riu-02','code_atom_ids':['b']} ]}
    neutral={'neutral.go':blob('neutral first\nneutral second\n')}
    catalog=[{'path':p,'git_blob_sha1':row['git_blob_sha1'],'bytes':row['bytes']} for p,row in neutral.items()]
    corpus={'schema':'deepswe_neutral_repository_source_v1','base_commit':'abc','runtime_image_id':'sha256:runtime',
        'clean_before':True,'clean_after':True,'excluded_discovered_paths':sorted(original),'files':neutral,
        'limits':{'max_files':600,'max_total_bytes':12000000},'selected_bytes':sum(x['bytes'] for x in neutral.values()),
        'eligible_files':catalog,'excluded_by_budget':[]}
    full=[]
    for n,atom in enumerate(atoms,1):
        full.append({'participant_id':f'participant_{n}','packet':{'instruction':'public requirements','public_check_catalog':[],
            'observations':[{'stdout':'existing approved base fact'}],'code_fragments':[
                {k:atom[k] for k in ('path','line_start','line_end','text')},
                {'path':'neutral.go','line_start':n,'line_end':n,'text':neutral['neutral.go']['text'].splitlines(keepends=True)[n-1]}]}})
    after={'execution_manifest':execution,'modes':{a.SA:[{'packet':'same all'}],a.PIU:[{'packet':'same sparse'}],a.FC:full}}
    before=json.loads(json.dumps(after))
    return before,after,candidate,{'files':original,'base_commit':'abc','task_id':'task'},{'neutral_repository_source_hidden':corpus}


def reader_for(case,extra=False):
    before,after,candidate,discovered,distribution=case
    def read_source(image,commit,paths,excluded,policy=None):
        files={**discovered['files'],**distribution['neutral_repository_source_hidden']['files']};rows={}
        for path in paths:
            row=blob(files[path]['text']);rows[path]={key:row[key] for key in ('sha256','git_blob_sha1','bytes')};rows[path]['mode']='100644'
        catalog=list(distribution['neutral_repository_source_hidden']['eligible_files'])
        if extra:catalog.append({'path':'omitted.go','bytes':10,'git_blob_sha1':'a'*40})
        return {'base_commit':commit,'clean_before':True,'clean_after':True,'files':rows,'eligible_files':catalog},{'mock_only':True}
    return read_source


def test_independent_source_and_packet_audit_accepts_complete_fixture(case):
    d=a.verify(*case,read_source=reader_for(case));assert d['status']=='passed';assert d['neutral_corpus_files']==1


@pytest.mark.parametrize('change',['missing_canonical','original_hotspot','duplicate_source','observation','SA','actor_identity','test_source','fresh_catalog_omission','fresh_blob_tamper'])
def test_neutral_label_cannot_hide_changed_or_missing_material(case,change):
    before,after,candidate,discovered,distribution=case;packet=after['modes'][a.FC][0]['packet'];reader=reader_for(case)
    if change=='missing_canonical':packet['code_fragments'].pop(0)
    elif change=='original_hotspot':packet['code_fragments'].append({'path':'first.go','line_start':2,'line_end':2,'text':'private other\n'})
    elif change=='duplicate_source':after['modes'][a.FC][1]['packet']['code_fragments'][1]=dict(packet['code_fragments'][1])
    elif change=='observation':packet['observations'][0]['stdout']='new answer'
    elif change=='SA':after['modes'][a.SA][0]['packet']='changed all'
    elif change=='actor_identity':after['execution_manifest']={'actor':'changed'}
    elif change=='test_source':
        corpus=distribution['neutral_repository_source_hidden'];corpus['files']['neutral_test.go']=corpus['files'].pop('neutral.go')
        corpus['eligible_files'][0]['path']='neutral_test.go'
    elif change=='fresh_catalog_omission':reader=reader_for(case,extra=True)
    elif change=='fresh_blob_tamper':
        base_reader=reader
        def reader(*args):
            value,trace=base_reader(*args);value['files']['neutral.go']['sha256']='0'*64;return value,trace
    with pytest.raises(AssertionError):a.verify(*case,read_source=reader)


@pytest.fixture
def case_v2(case):
    before,after,candidate,discovered,distribution=case
    observations=[]
    for pid in (1,2):
        ids=[]
        for n in (1,2):
            oid=f'owner{pid}_fact{n}';ids.append(oid)
            observations.append({'id':oid,'argv':['echo',oid],'stdout':oid,'stderr':'','exit_code':0})
        candidate['allocation_candidates'][pid-1]['observation_ids']=ids
    candidate['observations']=observations
    visible=lambda o:{k:o[k] for k in ('argv','stdout','stderr','exit_code')}
    for pid in (1,2):
        own=observations[(pid-1)*2:pid*2];foreign=observations[(2-pid)*2:(3-pid)*2]
        before['modes'][a.FC][pid-1]['packet']['observations']=[visible(o) for o in own+foreign]
        after['modes'][a.FC][pid-1]['packet']['observations']=[visible(o) for o in own+foreign[:1]]
    corpus=distribution['neutral_repository_source_hidden'];corpus.update(schema='deepswe_neutral_repository_source_v2',background_language='go',language_policy=a.POLICIES['go'])
    return case


def test_new_observation_policy_is_intentional_not_claimed_byte_identical(case_v2):
    result=a.verify(*case_v2,read_source=reader_for(case_v2))
    assert result['fullcontext_observation_policy']=='max_one_allocated_foreign_owner'
    assert all(not p['all_observations_exact_previous_packet'] for p in result['participants'])
    assert all(p['owned_observations_exact'] for p in result['participants'])


@pytest.mark.parametrize('change',['two_foreign_same_owner','missing_owned','unallocated_fact','hidden_owner_metadata','wrong_language_policy','changed_raw_projection'])
def test_revised_policy_rejects_hidden_or_overallocated_observation_facts(case_v2,change):
    before,after,candidate,discovered,distribution=case_v2;packet=after['modes'][a.FC][0]['packet'];kwargs={}
    if change=='two_foreign_same_owner':packet['observations'].append({k:candidate['observations'][3][k] for k in ('argv','stdout','stderr','exit_code')})
    elif change=='missing_owned':packet['observations'].pop(0)
    elif change=='unallocated_fact':packet['observations'].append({'argv':['echo','new'],'stdout':'new','stderr':'','exit_code':0})
    elif change=='hidden_owner_metadata':packet['observations'][0]['owner']='riu-01'
    elif change=='wrong_language_policy':distribution['neutral_repository_source_hidden']['language_policy']={'extensions':['.go'],'excluded_filename_globs':[],'excluded_directory_names':[]}
    elif change=='changed_raw_projection':
        prior=json.loads(json.dumps(distribution['neutral_repository_source_hidden']));prior['files']={};kwargs['prior_corpus']=prior
    with pytest.raises(AssertionError):a.verify(*case_v2,read_source=reader_for(case_v2),**kwargs)


@pytest.mark.parametrize('language,source,testpath',[('go','src/a.go','src/a_test.go'),('python','pkg/a.py','pkg/test_a.py'),('typescript','src/a.ts','src/a.spec.ts'),('javascript','src/a.mjs','src/a.test.mjs'),('rust','src/a.rs','tests/a.rs')])
def test_fresh_git_reader_independently_applies_declared_language_test_policy(tmp_path,language,source,testpath):
    import subprocess,sys
    for path in (source,testpath,'hotspot.'+source.rsplit('.',1)[1]):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('public source\n')
    run=lambda *args:subprocess.check_output(args,cwd=tmp_path,text=True)
    run('git','init','--quiet');run('git','add','.');run('git','-c','user.name=Audit','-c','user.email=a@invalid','commit','--quiet','-m','fixture')
    commit=run('git','rev-parse','HEAD').strip();excluded=['hotspot.'+source.rsplit('.',1)[1]]
    request={'commit':commit,'paths':[source],'excluded_paths':excluded,'language_policy':a.POLICIES[language]}
    p=subprocess.run([sys.executable,'-I','-c',a.SOURCE_SCRIPT],input=json.dumps(request),capture_output=True,text=True,cwd=tmp_path,check=True)
    value=json.loads(p.stdout)
    assert [row['path'] for row in value['eligible_files']]==[source]
    assert set(value['files'])=={source}
    assert value['clean_before'] is True and value['clean_after'] is True
