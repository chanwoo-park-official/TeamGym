"""Independent exact-source and information-boundary audit of neutral FC packets.

Default execution independently reads pinned Git blobs in a fresh networkless
container. It never calls a model, applies a patch, or rewrites any input.
"""
from pathlib import Path, PurePosixPath
import argparse
import fnmatch
import hashlib
import json
import subprocess
import time

SA='single_agent_all_piu';PIU='multi_agent_piu_only';FC='multi_agent_full_context'

POLICIES={
 'go':{'extensions':['.go'],'excluded_directory_names':[],'excluded_filename_globs':['*_test.go']},
 'python':{'extensions':['.py'],'excluded_directory_names':['test','tests','__tests__'],'excluded_filename_globs':['test_*.py','*_test.py','*_tests.py','conftest.py']},
 'typescript':{'extensions':['.ts','.tsx'],'excluded_directory_names':['test','tests','__tests__'],'excluded_filename_globs':['*.test.ts','*.test.tsx','*.spec.ts','*.spec.tsx']},
 'javascript':{'extensions':['.js','.jsx','.mjs','.cjs'],'excluded_directory_names':['test','tests','__tests__'],'excluded_filename_globs':['*.test.js','*.test.jsx','*.test.mjs','*.test.cjs','*.spec.js','*.spec.jsx','*.spec.mjs','*.spec.cjs']},
 'rust':{'extensions':['.rs'],'excluded_directory_names':['test','tests','__tests__'],'excluded_filename_globs':['test_*.rs','*_test.rs','tests.rs']}}


def eligible(path,policy):
 parts=PurePosixPath(path).parts
 return PurePosixPath(path).suffix in policy['extensions'] and not set(parts[:-1])&set(policy['excluded_directory_names']) and not any(fnmatch.fnmatchcase(parts[-1],g) for g in policy['excluded_filename_globs'])




def read(path):return json.loads(Path(path).read_text())

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


SOURCE_SCRIPT=r'''
import json,hashlib,subprocess,sys,fnmatch,pathlib
r=json.load(sys.stdin)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==r['commit']
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
eligible=[]
for raw_entry in subprocess.check_output(['git','ls-tree','-r','-z',r['commit']]).decode().split('\0'):
    if not raw_entry:continue
    metadata,path=raw_entry.split('\t',1);mode,kind,oid=metadata.split()
    if mode not in ('100644','100755') or kind!='blob' or path in r['excluded_paths']:continue
    parts=pathlib.PurePosixPath(path).parts;policy=r['language_policy']
    if pathlib.PurePosixPath(path).suffix not in policy['extensions'] or set(parts[:-1])&set(policy['excluded_directory_names']) or any(fnmatch.fnmatchcase(parts[-1],g) for g in policy['excluded_filename_globs']):continue
    eligible.append({'path':path,'git_blob_sha1':oid,'bytes':int(subprocess.check_output(['git','cat-file','-s',oid]))})
eligible.sort(key=lambda x:x['path'])
files={}
for path in r['paths']:
    entry=subprocess.check_output(['git','ls-tree',r['commit'],'--',path],text=True).split()
    assert entry[0] in ('100644','100755') and entry[1]=='blob'
    raw=subprocess.check_output(['git','show',r['commit']+':'+path])
    files[path]={'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'bytes':len(raw),'mode':entry[0]}
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
print(json.dumps({'base_commit':r['commit'],'clean_before':True,'clean_after':True,'eligible_files':eligible,'files':files}))
'''


def source_reader(image,commit,paths,excluded_paths,language_policy=None):
    command=['docker','run','--rm','-i','--pull','never','--platform','linux/amd64','--network','none',
        '--cpus','2','--memory','2g','--pids-limit','128','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--workdir','/app','--entrypoint','python3',image,'-I','-c',SOURCE_SCRIPT]
    start=time.monotonic();p=subprocess.run(command,input=json.dumps({'commit':commit,'paths':paths,'excluded_paths':excluded_paths,'language_policy':language_policy or POLICIES['go']}),capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError('Fresh base-source audit failed: '+p.stderr[-2000:])
    return json.loads(p.stdout),{'command':command,'exit_code':p.returncode,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr':p.stderr,'elapsed_seconds':round(time.monotonic()-start,3)}


def verify(before,after,candidate,discovered,distribution,*,read_source=source_reader,prior_corpus=None):
    import tiktoken
    corpus=distribution['neutral_repository_source_hidden']
    assert corpus['schema'] in ('deepswe_neutral_repository_source_v1','deepswe_neutral_repository_source_v2')
    revised=corpus['schema'].endswith('_v2')
    language=corpus['background_language'] if revised else 'go'
    assert language in POLICIES
    policy=POLICIES[language]
    if revised:assert corpus['language_policy']==policy
    execution=candidate['execution_manifest'];base=execution['source']['base_commit'];original=discovered['files']
    assert discovered['base_commit']==base and discovered['task_id']==candidate['task_id']
    assert corpus['base_commit']==base and corpus['runtime_image_id']==execution['runtime_image']['id']
    assert corpus['clean_before'] is True and corpus['clean_after'] is True
    assert corpus['excluded_discovered_paths']==sorted(original),'Must exclude ALL original discovery paths'
    neutral=corpus['files'];assert isinstance(neutral,dict) and neutral
    limits=corpus['limits'];assert type(limits['max_files']) is int and 0<len(neutral)<=limits['max_files']<=600
    catalog=corpus['eligible_files'];assert [x['path'] for x in catalog]==sorted({x['path'] for x in catalog})
    selected=[];skipped=[];selected_total=0
    for entry in catalog:
        assert type(entry['bytes']) is int and entry['bytes']>=0
        if len(selected)>=limits['max_files'] or selected_total+entry['bytes']>limits['max_total_bytes']:skipped.append(entry['path']);continue
        selected.append(entry['path']);selected_total+=entry['bytes']
    assert set(selected)==set(neutral) and skipped==corpus['excluded_by_budget']
    total=0;expected={}
    for path,row in neutral.items():
        pure=PurePosixPath(path)
        assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in path and '\x00' not in path
        assert eligible(path,policy) and path not in original,'Hotspot/test/wrong-language source in neutral corpus'
        raw=row['text'].encode();sha=hashlib.sha256(raw).hexdigest();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert row['sha256']==sha and row['git_blob_sha1']==blob and type(row['bytes']) is int and row['bytes']==len(raw)
        expected[path]={'sha256':sha,'git_blob_sha1':blob,'bytes':len(raw)};total+=len(raw)
    assert corpus['selected_bytes']==total and type(limits['max_total_bytes']) is int and total<=limits['max_total_bytes']<=12000000
    assert set(before)==set(after)
    for key in set(before)-{'modes'}:assert before[key]==after[key],('nonpacket_metadata_changed',key)
    assert after['execution_manifest']==execution
    for mode in (SA,PIU):assert before['modes'][mode]==after['modes'][mode],('sparse_condition_changed',mode)
    oldviews={v['participant_id']:v for v in before['modes'][FC]};newviews={v['participant_id']:v for v in after['modes'][FC]}
    assert set(oldviews)==set(newviews)
    observations=candidate.get('observations',[])
    obs_signature=lambda row:canonical({key:row.get(key) for key in ('argv','stdout','stderr','exit_code')})
    obs_by_id={o['id']:o for o in observations}
    owner_by_signature={}
    for index,allocation in enumerate(candidate['allocation_candidates'],1):
        for oid in allocation.get('observation_ids',[]):
            signature=obs_signature(obs_by_id[oid]);assert signature not in owner_by_signature
            owner_by_signature[signature]=f'participant_{index}'
    atoms={a['id']:a for a in candidate['code_atoms']};allocations={f'participant_{n}':row for n,row in enumerate(candidate['allocation_candidates'],1)}
    enc=tiktoken.get_encoding('cl100k_base');seen={};stats=[];required_source_paths=set(neutral)
    for pid,view in newviews.items():
        packet=view['packet'];old=oldviews[pid]['packet'];owned=[atoms[x] for x in allocations[pid]['code_atom_ids']]
        assert {k:v for k,v in packet.items() if k not in ('code_fragments','observations')}=={k:v for k,v in old.items() if k not in ('code_fragments','observations')},'Common/model facts changed'
        if not revised:
            assert packet['observations']==old['observations'],'Original v1 observation policy changed'
            foreign_counts={}
        else:
            assert all(set(o)=={'argv','stdout','stderr','exit_code'} for o in packet['observations']),'Hidden observation metadata'
            signatures=[obs_signature(o) for o in packet['observations']];assert len(signatures)==len(set(signatures)),'Duplicated observation'
            owned_signatures={obs_signature(obs_by_id[oid]) for oid in allocations[pid].get('observation_ids',[])}
            assert owned_signatures<=set(signatures),'Owned observation missing'
            assert all(signature in owner_by_signature for signature in signatures),'Unallocated observation included'
            foreign_counts={}
            for signature in signatures:
                owner=owner_by_signature[signature]
                if owner!=pid:foreign_counts[owner]=foreign_counts.get(owner,0)+1
            assert all(count<=1 for count in foreign_counts.values()) and sum(foreign_counts.values())<=3,'Foreign owner observation cap exceeded'
            assert {signature for signature in signatures if owner_by_signature[signature]==pid}==owned_signatures
        ownedlines={a['path']:set() for a in owned}
        for atom in owned:ownedlines[atom['path']].update(range(atom['line_start'],atom['line_end']+1))
        usedown={path:set() for path in ownedlines};neutralblocks=0
        for block in packet['code_fragments']:
            assert set(block)=={'path','line_start','line_end','text'},'Hidden source metadata'
            path=block['path'];start,end=block['line_start'],block['line_end'];assert type(start) is int and type(end) is int and 1<=start<=end
            assert path in original or path in neutral,'Source outside both verified corpora'
            source=(original if path in original else neutral)[path]['text'];lines=source.splitlines(keepends=True)
            assert end<=len(lines) and block['text']==''.join(lines[start-1:end]),'Packet is not exact base-source slice'
            for a,b in seen.get(path,[]):assert max(a,start)>min(b,end),'Source assigned twice across/full packets'
            seen.setdefault(path,[]).append((start,end))
            if path in original:
                visible=set(range(start,end+1));assert visible<=ownedlines.get(path,set()),'Unowned original hotspot source leaked'
                assert not usedown[path]&visible;usedown[path]|=visible
            else:neutralblocks+=1
        assert usedown==ownedlines,'Canonical owned source missing or duplicated'
        assert neutralblocks>0,'Full context lacks actual neutral source'
        material={key:packet[key] for key in ('code_fragments','observations')}
        tokens=len(enc.encode_ordinary(json.dumps(material,ensure_ascii=False)))
        assert tokens<=40000,'Private packet over fixed budget'
        stats.append({'participant_id':pid,'private_tokens':tokens,'neutral_source_blocks':neutralblocks,'canonical_source_exact':True,'owned_observations_exact':True,'all_observations_exact_previous_packet':packet['observations']==old['observations'],'foreign_owner_observation_counts':foreign_counts,'full_code_material_exact_previous_packet':packet['code_fragments']==old['code_fragments']})
    # Freshly verify every corpus file AND original canonical source file from
    # the same pinned image. This does not rely on the collector's self-report.
    for atom in atoms.values():
        path=atom['path'];raw=original[path]['text'].encode();expected[path]={'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'bytes':len(raw)}
    actual,trace=read_source(execution['runtime_image']['id'],base,sorted(expected),sorted(original),policy)
    assert actual['base_commit']==base and actual['clean_before'] is True and actual['clean_after'] is True
    assert actual['eligible_files']==catalog,'Neutral catalog omits/changes eligible base files'
    assert set(actual['files'])==set(expected)
    for path,row in actual['files'].items():
        assert row['mode'] in ('100644','100755')
        assert {key:row[key] for key in ('sha256','git_blob_sha1','bytes')}==expected[path],'Fresh pinned Git blob mismatch'
    projection=None
    if prior_corpus is not None:
        assert language=='go'
        for key in ('files','eligible_files','selected_bytes','excluded_by_budget','excluded_discovered_paths','base_commit','runtime_image_id'):assert corpus[key]==prior_corpus[key],('Go raw corpus projection changed',key)
        projection={'Go_raw_selected_file_paths_text_catalog_exact_prior':True,'prior_corpus_sha256':canonical(prior_corpus),'new_corpus_sha256':canonical(corpus),'metadata_epoch_changed':corpus!=prior_corpus}
    return {'schema':'deepswe_independent_neutral_background_audit_v2','status':'passed','SA_and_PIU_only_byte_material_unchanged':True,
        'all_fullcontext_common_unchanged':True,'all_owned_observations_preserved':True,'fullcontext_observation_policy':'max_one_allocated_foreign_owner' if revised else 'original_v1' ,'Go_raw_corpus_projection':projection,'some_fullcontext_code_material_repartitioned':any(not row['full_code_material_exact_previous_packet'] for row in stats),'all_canonical_source_preserved':True,
        'no_unallocated_original_hotspot_source':True,'exclusive_source_ranges':True,'fixed_private_budget':40000,
        'original_discovery_paths_excluded':sorted(original),'neutral_corpus_files':len(neutral),'neutral_corpus_bytes':total,
        'corpus_sha256':canonical(corpus),'bundle_sha256':canonical(after),'participants':stats,
        'fresh_base_gitblob_validation':actual,'fresh_source_reader_trace':trace,'model_calls':0,'verifier_runs':0,
        'scope':'Mechanical/source audit only, not semantic packet acceptance or participant repair outcome.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','candidate','discovered','distribution','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-corpus',type=Path)
    a=p.parse_args();args=[read(getattr(a,name)) for name in ('before','after','candidate','discovered','distribution')];result=verify(*args,prior_corpus=read(a.prior_corpus) if a.prior_corpus else None)
    result['input_file_sha256']={name:digest(getattr(a,name)) for name in ('before','after','candidate','discovered','distribution')}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'neutral_files':result['neutral_corpus_files'],'participants':result['participants']}))


if __name__=='__main__':main()
