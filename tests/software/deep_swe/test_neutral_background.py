"""Neutral repository collection and packing, using real local Git blobs offline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from teamgym.data.software.deep_swe import full_context as fc
from tests.software.deep_swe.test_full_context import sources


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, 'init')
    for name, body in {'hotspot.go': 'package main\n// excluded discovery\n',
                       'neutral/helper.go': 'package neutral\nfunc helper() string { return "é" }\n',
                       'neutral/second.go': 'package neutral\n// ordinary source\n',
                       'neutral/helper_test.go': 'package neutral\n// test\n',
                       'README.md': '# unrelated\n'}.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
    git(tmp_path,'add','.')
    git(tmp_path,'-c','user.name=Offline test','-c','user.email=test@example.invalid','commit','-qm','base')
    return tmp_path


def read_corpus(root, *, excluded=('hotspot.go',), base=None, language='go'):
    request={'base_commit':base or git(root,'rev-parse','HEAD'),'runtime_image_id':'sha256:'+'1'*64,
             'excluded_discovered_paths':sorted(excluded),'limits':fc.NEUTRAL_LIMITS,
             'selection_policy':fc.NEUTRAL_SELECTION,'background_language':language,
             'language_policy':fc.BACKGROUND_LANGUAGES[language]}
    result=subprocess.run([sys.executable,'-I','-c',fc.NEUTRAL_GIT_READER],input=json.dumps(request),
                          text=True,capture_output=True,cwd=root)
    return request,result


def validated_corpus(root, *, excluded=('hotspot.go',), language='go'):
    request,result=read_corpus(root,excluded=excluded,language=language)
    assert result.returncode==0,result.stderr
    return fc.validate_neutral_repository(json.loads(result.stdout),base_commit=request['base_commit'],
        runtime_image_id=request['runtime_image_id'],excluded_paths=request['excluded_discovered_paths'])


def test_real_git_reader_collects_only_neutral_whole_files_with_blob_identity(repository):
    corpus=validated_corpus(repository)
    assert set(corpus['files'])=={'neutral/helper.go','neutral/second.go'}
    assert corpus['clean_before'] is corpus['clean_after'] is True
    assert git(repository,'status','--porcelain')==''
    for path,row in corpus['files'].items():
        raw=(repository/path).read_bytes()
        assert row['git_blob_sha1']==git(repository,'rev-parse',f'HEAD:{path}')
        assert row['sha256']==hashlib.sha256(raw).hexdigest()
        assert row['text'].encode()==raw


def test_reader_rejects_dirty_or_wrong_base_and_no_source(repository):
    _,wrong=read_corpus(repository,base='a'*40)
    assert wrong.returncode and 'base_commit_mismatch' in wrong.stderr
    (repository/'untracked.go').write_text('package dirty\n')
    _,dirty=read_corpus(repository)
    assert dirty.returncode and 'dirty_before' in dirty.stderr
    (repository/'untracked.go').unlink()
    with pytest.raises(ValueError,match='no_eligible_source'):
        validated_corpus(repository,excluded=['hotspot.go','neutral/helper.go','neutral/second.go'])


def test_reader_skips_whole_overbudget_file_and_retains_later_complete_source(repository):
    path=repository/'a_large.go'
    path.write_bytes(b'x'*(fc.NEUTRAL_LIMITS['max_total_bytes']+1))
    git(repository,'add','a_large.go')
    git(repository,'-c','user.name=Offline test','-c','user.email=test@example.invalid','commit','-qm','large public blob')
    corpus=validated_corpus(repository)
    assert corpus['excluded_by_budget']==['a_large.go']
    assert 'a_large.go' not in corpus['files']
    assert set(corpus['files'])=={'neutral/helper.go','neutral/second.go'}
    assert corpus['selected_bytes']<fc.NEUTRAL_LIMITS['max_total_bytes']


@pytest.mark.parametrize('mutation,error',[
    (lambda c:c.update(clean_after=False),'identity_or_policy'),
    (lambda c:c.update(base_commit='a'*40),'identity_or_policy'),
    (lambda c:c['files']['neutral/helper.go'].update(text='changed'),'blob_hash'),
    (lambda c:c['files']['neutral/helper.go'].update(git_blob_sha1='a'*40),'blob_hash'),
    (lambda c:c.update(selected_bytes=0),'selection_mismatch'),
    (lambda c:c.update(excluded_by_budget=['neutral/helper.go']),'selection_mismatch'),
    (lambda c:c.update(limits={'max_total_bytes':99999999,'max_files':600}),'identity_or_policy'),
])
def test_full_corpus_validation_rejects_stale_or_mutated_cached_data(repository,mutation,error):
    corpus=validated_corpus(repository);base=corpus['base_commit'];image=corpus['runtime_image_id']
    mutation(corpus)
    with pytest.raises(ValueError,match=error):
        fc.validate_neutral_repository(corpus,base_commit=base,runtime_image_id=image,excluded_paths=['hotspot.go'])


def test_collector_uses_immutable_image_network_disabled_and_fresh_read(repository,monkeypatch):
    corpus=validated_corpus(repository);calls=[]
    candidate={'execution_manifest':{'source':{'base_commit':corpus['base_commit']},
                                     'runtime_image':{'id':corpus['runtime_image_id']}}}
    def run(command,**kw):
        calls.append((command,kw));assert json.loads(kw['input'])['excluded_discovered_paths']==['hotspot.go']
        return SimpleNamespace(returncode=0,stdout=json.dumps(corpus),stderr='')
    monkeypatch.setattr(fc.subprocess,'run',run)
    first=fc.collect_neutral_repository(candidate,['hotspot.go'],language='go');second=fc.collect_neutral_repository(candidate,['hotspot.go'],language='go')
    assert len(calls)==2  # No cross-run or mutable in-memory cache.
    command=calls[0][0];assert command[command.index('--network')+1]=='none'
    assert command[command.index('--pull')+1]=='never' and candidate['execution_manifest']['runtime_image']['id'] in command
    first['files']['neutral/helper.go']['text']='mutated consumer copy'
    assert second==corpus


def test_neutral_pack_preserves_anchors_observations_and_excludes_all_discovered_fillers(repository,monkeypatch):
    candidate,original=sources();original['hotspot.go']=(repository/'hotspot.go').read_text()
    excluded=sorted(original);corpus=validated_corpus(repository,excluded=excluded)
    candidate['execution_manifest']={'source':{'base_commit':corpus['base_commit']},'runtime_image':{'id':corpus['runtime_image_id']}}
    inputs=deepcopy((candidate,original,corpus));calls=[]
    def collect(c,paths,**kwargs):calls.append(paths);return corpus
    monkeypatch.setattr(fc,'collect_neutral_repository',collect)
    observations=[{'id':'one','argv':['probe'],'stdout':'existing','stderr':'','exit_code':0}]
    old,_=fc.pack(candidate,original,observations,1000)
    assert not calls
    packed,manifest=fc.pack(candidate,original,observations,1000,background_policy='neutral_repository',background_language='go')
    assert calls==[excluded]
    canonical=fc.build_participants(candidate)
    merged={**original,**{p:r['text'] for p,r in corpus['files'].items()}}
    seen=set()
    for i,participant in enumerate(packed):
        assert participant['allocation_hidden']==old[i]['allocation_hidden']==canonical[i]['allocation_hidden']
        assert participant['private_source_material']['observations']==old[i]['private_source_material']['observations']
        for atom in canonical[i]['private_source_material']['code_fragments']:
            assert atom in participant['private_source_material']['code_fragments']
        for block in participant['private_source_material']['code_fragments']:
            if block in canonical[i]['private_source_material']['code_fragments']:continue
            assert block['path'] not in original and block['path'] in corpus['files']
            for line in range(block['line_start'],block['line_end']+1):
                key=(block['path'],line);assert key not in seen;seen.add(key)
        fc.audit_packet(candidate,participant,merged,1000)
        assert 'neutral_repository' not in json.dumps(participant['private_source_material'])
    assert (candidate,original,corpus)==inputs
    assert manifest['neutral_repository_source_hidden']==corpus
    corpus['files']['neutral/helper.go']['text']='subsequent mutable fixture change'
    assert manifest['neutral_repository_source_hidden']!=corpus


def test_neutral_pack_keeps_canonical_budget_check_and_rejects_unknown_policy(repository,monkeypatch):
    c,files=sources();corpus=validated_corpus(repository,excluded=sorted(files))
    c['execution_manifest']={'source':{'base_commit':corpus['base_commit']},'runtime_image':{'id':corpus['runtime_image_id']}}
    monkeypatch.setattr(fc,'collect_neutral_repository',lambda *_,**kwargs:corpus)
    with pytest.raises(ValueError,match='allocated_evidence_exceeds_packet_budget'):
        fc.pack(c,files,[],1,background_policy='neutral_repository',background_language='go')
    with pytest.raises(ValueError,match='unknown_background_policy'):
        fc.pack(c,files,[],1000,background_policy='other')


@pytest.mark.parametrize('language,source,extra,test',[
    ('python','src/helper.py','src/types.py','src/test_helper.py'),
    ('typescript','src/helper.ts','src/view.tsx','src/helper.spec.ts'),
    ('javascript','src/helper.js','src/module.mjs','src/helper.test.js'),
    ('rust','src/helper.rs','src/other.rs','src/helper_test.rs'),
])
def test_language_specific_real_git_reader_excludes_public_test_patterns(repository,language,source,extra,test):
    for name in [source,extra,test,'tests/inside'+Path(source).suffix,'__tests__/inside'+Path(source).suffix]:
        p=repository/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('// public source\n')
    git(repository,'add','.')
    git(repository,'-c','user.name=Offline test','-c','user.email=test@example.invalid','commit','-qm','language corpus')
    corpus=validated_corpus(repository,language=language)
    assert corpus['background_language']==language
    assert corpus['language_policy']==fc.BACKGROUND_LANGUAGES[language]
    assert set(corpus['files'])=={source,extra}


@pytest.mark.parametrize('value,expected', [('Go','go'),('Golang','go'),('Python','python'),('TS','typescript'),('JS','javascript'),('rs','rust')])
def test_declared_language_aliases(value,expected):
    assert fc.normalize_background_language(value)==expected


def test_inference_requires_unambiguous_supported_canonical_source_extensions():
    assert fc.infer_background_language({'code_atoms':[{'path':'a.ts'},{'path':'view.tsx'}]})=='typescript'
    for paths in [[],['a.py','b.go'],['README.md'],['native.c']]:
        with pytest.raises(ValueError,match='ambiguous_language'):
            fc.infer_background_language({'code_atoms':[{'path':p} for p in paths]})
    for value in ['C++','',None,1]:
        with pytest.raises(ValueError,match='unsupported_language'):
            fc.normalize_background_language(value)


def test_corpus_language_policy_is_identity_bound(repository):
    corpus=validated_corpus(repository)
    corpus['language_policy']['excluded_filename_globs']=[]
    with pytest.raises(ValueError,match='identity_or_policy'):
        fc.validate_neutral_repository(corpus,base_commit=corpus['base_commit'],runtime_image_id=corpus['runtime_image_id'],excluded_paths=['hotspot.go'],language='go')
