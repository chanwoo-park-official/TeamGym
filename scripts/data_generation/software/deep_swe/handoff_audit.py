"""Independent strict audit of an explicit qualified DeepSWE phase handoff.

Import inside the isolated producer epoch: production imports below deliberately
resolve to that epoch. No model API is called and no producer file is mutated.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tree_receipt(root):
    root = Path(root).resolve()
    return {str(p.relative_to(root)): {'sha256': digest(p), 'symlink': str(p.readlink()) if p.is_symlink() else None}
            for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.suffix != '.pyc'}


SOURCE_HASH_SCRIPT = r'''
import hashlib,json,subprocess,sys
r=json.load(sys.stdin)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==r['commit']
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
files={}
for path in r['paths']:
    mode=subprocess.check_output(['git','ls-tree',r['commit'],'--',path],text=True).split()[0]
    assert mode in ('100644','100755')
    raw=subprocess.check_output(['git','show',r['commit']+':'+path])
    files[path]=hashlib.sha256(raw).hexdigest()
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
print(json.dumps({'base_commit':r['commit'],'clean':True,'files':files}))
'''


def _source_reader(image, commit, paths):
    from teamgym.data.software.deep_swe.stages.patch_runtime import docker_python
    return docker_python(image, SOURCE_HASH_SCRIPT, {'commit': commit, 'paths': paths}, timeout=180)


def audit_background(producer, identity, *, source_reader=None):
    from teamgym.data.software.deep_swe.evidence import usable_observations
    discovered = read(producer / 'discovered_code.json')
    base = identity['source']['base_commit']
    if discovered.get('task_id') != identity['source']['task_id'] or discovered.get('base_commit') != base:
        raise ValueError('background task/base mismatch')
    files = discovered.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('missing source background')
    for path, row in files.items():
        if (not isinstance(path, str) or PurePosixPath(path).is_absolute()
                or '..' in PurePosixPath(path).parts or '\x00' in path or not isinstance(row.get('text'), str)):
            raise ValueError('unsafe source background')
    actual = (source_reader or _source_reader)(identity['runtime_image']['id'], base, sorted(files))
    expected = {path: hashlib.sha256(row['text'].encode()).hexdigest() for path, row in files.items()}
    if actual != {'base_commit': base, 'clean': True, 'files': expected}:
        raise ValueError('source background is not exact clean pinned-base blobs')
    observations = read(producer / 'observations.validated.json')
    usable, rejected = usable_observations(observations, task_id=identity['source']['task_id'], base_commit=base)
    if rejected or len(usable) != len(observations):
        raise ValueError('unverified observation background')
    return {'source_file_count': len(files), 'base_gitblob_sha256': expected,
            'source_reader_receipt': actual, 'observations': len(usable),
            'discovered_sha256': digest(producer / 'discovered_code.json'),
            'observations_sha256': digest(producer / 'observations.validated.json'),
            'selection_policy': 'Constructor-selected public source, including later optional-round expansion; new full-packet certification required.'}


def audit_checks(producer, identity):
    checks = read(producer / 'public_checks.json')
    validation = read(producer / 'public_checks.base_validation.json')
    if not isinstance(checks, dict) or not checks or set(checks) != set(validation):
        raise ValueError('check coverage mismatch')
    base = identity['source']['base_commit']
    for key, argv in checks.items():
        row = validation[key];before = row.get('checkout_before', {})
        artifact = {'id': key, 'argv': argv, 'stdout': row.get('stdout'),
                    'stderr': row.get('stderr'), 'exit_code': row.get('exit_code')}
        audit = row.get('provenance_audit', {})
        if (not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv)
                or type(row.get('exit_code')) is not int or row['exit_code'] != 0
                or row.get('timed_out') is not False or row.get('tracked_checkout_unchanged') is not True
                or row.get('base_commit') != base or row.get('resolved_base_commit') != base
                or before.get('clean') is not True or before.get('head') != base or before.get('status') != ''
                or before != row.get('checkout_after') or audit.get('approved') is not True
                or audit.get('status') != 'reviewed' or audit.get('content_sha256') != canonical(artifact)):
            raise ValueError('check content or clean-base provenance mismatch')
    return {'count': len(checks), 'checks_sha256': digest(producer / 'public_checks.json'),
            'validation_sha256': digest(producer / 'public_checks.base_validation.json')}


def audit_episode_group(directory, *, candidate_path, checks_path, identity, label='baseline', repetitions=3):
    from teamgym.data.software.deep_swe.stages.diagnostic_runs import (EPISODE_OUTPUTS,
        episode_inputs, valid_episode_result)
    from teamgym.data.software.deep_swe.stages.patch_episode import packet_for
    rows = read(directory / 'results.json')
    rows = [r for r in rows if r.get('label') == label]
    if (len(rows) != repetitions or {r.get('rep') for r in rows} != set(range(1, repetitions + 1))
            or any(type(r.get('rep')) is not int for r in rows)):
        raise ValueError('duplicate/missing episode repetitions')
    locations = set(); receipts = []
    config = identity['config']
    candidate = read(candidate_path); checks = read(checks_path)
    for row in rows:
        episode = Path(row['output_dir']).resolve()
        if episode in locations or episode.parent != directory.resolve() or episode.name != f'{label}_rep{row["rep"]}':
            raise ValueError('episode path outside condition group')
        locations.add(episode)
        job = {**config, 'task_dir': identity['source']['source_dir'], 'candidate': str(candidate_path),
               'checks': str(checks_path), 'label': label, 'rep': row['rep'], 'model_slug': config['actor_model']}
        receipt = read(episode / 'cache_manifest.json')
        inputs = episode_inputs(job)
        if (receipt.get('schema_version') != 1 or receipt.get('inputs') != inputs
                or receipt.get('inputs_sha256') != canonical(inputs)):
            raise ValueError('episode current inputs mismatch')
        expected = {str(episode / name): digest(episode / name) for name in EPISODE_OUTPUTS}
        if receipt.get('outputs') != expected:
            raise ValueError('episode outputs changed or incomplete')
        result = read(episode / 'result.json')
        if (not valid_episode_result(result) or row.get('episode_valid') is not True
                or type(row.get('success')) is not bool or row['success'] is not result['success']
                or row.get('exit_code') != 0):
            raise ValueError('episode result invalid or summary inconsistent')
        if read(episode / 'actor_packet.json') != packet_for(candidate, checks, 'all'):
            raise ValueError('episode actor packet differs from frozen view')
        if read(episode / 'candidate.frozen.hidden.json') != candidate:
            raise ValueError('episode frozen candidate differs')
        manifest = read(episode / 'manifest.json')
        if manifest.get('actor_packet_sha256') != digest(episode / 'actor_packet.json'):
            raise ValueError('episode actor packet manifest mismatch')
        if digest(episode / 'harness.frozen.py') != identity['code']['files']['teamgym/data/software/deep_swe/stages/patch_episode.py']:
            raise ValueError('episode harness differs from producer epoch')
        receipts.append({'rep': row['rep'], 'success': row['success'], 'cache_manifest_sha256': digest(episode / 'cache_manifest.json')})
    from teamgym.data.software.deep_swe.pipeline import _episode_receipts
    _episode_receipts(directory, label=label, expected_packet=packet_for(candidate, checks, 'all'),
                      expected_execution=candidate['execution_manifest'])
    return {'directory': str(directory), 'label': label, 'runs': len(rows),
            'passes': sum(r['success'] for r in rows), 'episodes': receipts}


def preflight(producer, identity, candidate, *, source_reader=None):
    from teamgym.data.software.deep_swe.qualification import QualificationPolicy
    producer = Path(producer).resolve();candidate = Path(candidate).resolve()
    if candidate.parent != producer or not re.fullmatch(r'candidate_functional_rius_round_\d+\.json', candidate.name):
        raise ValueError('candidate must be an original producer round file')
    original = tree_receipt(producer)
    if read(producer / 'run_identity.json') != identity:
        raise ValueError('producer identity mismatch')
    selected = read(candidate)
    policy = QualificationPolicy.from_manifest(selected.get('execution_manifest', {}))
    if selected.get('execution_manifest', {}).get('code') != identity['code']:
        raise ValueError('candidate code epoch mismatch')
    background = audit_background(producer, identity, source_reader=source_reader)
    checks = audit_checks(producer, identity)
    screening = audit_episode_group(producer / 'stage4a_instruction_only',
        candidate_path=producer / 'candidate_instruction_only.json', checks_path=producer / 'public_checks.json', identity=identity)
    round_no = int(re.search(r'round_(\d+)', candidate.name)[1])
    qualification = audit_episode_group(producer / f'stage4c_qualification_round_{round_no:02d}',
        candidate_path=candidate, checks_path=producer / 'public_checks.json', identity=identity,
        repetitions=policy.repetitions)
    if (screening['passes'] != 0 or qualification['runs'] != policy.repetitions
            or qualification['passes'] < policy.minimum_passes):
        raise ValueError('screening or qualification gate failed')
    if tree_receipt(producer) != original:
        raise ValueError('producer changed during preflight')
    return {'schema': 'deepswe_independent_handoff_preflight_v1', 'status': 'passed',
            'producer': str(producer), 'producer_tree': original, 'candidate_sha256': digest(candidate),
            'background': background, 'checks': checks, 'screening': screening, 'qualification': qualification,
            'qualification_policy': policy.to_dict()}


def postflight(producer, identity, candidate, output, before):
    from teamgym.data.software.deep_swe.candidate import leave_one_out_conditions, reduce_candidate
    from teamgym.data.software.deep_swe.export import load_export, MODES, participant_view_candidates
    producer = Path(producer).resolve();output = Path(output).resolve();candidate = Path(candidate).resolve()
    if before.get('status') != 'passed' or before.get('producer') != str(producer) or tree_receipt(producer) != before['producer_tree']:
        raise ValueError('producer changed after handoff')
    if digest(candidate) != before['candidate_sha256']:
        raise ValueError('selected candidate changed')
    report = read(output / 'report.json') if (output / 'report.json').exists() else {}
    if report.get('decision') == 'packet_audit_rejected':
        return {'schema':'deepswe_independent_handoff_postflight_v1','status':'rejected',
                'producer_tree_unchanged':True,'decision':'packet_audit_rejected',
                'packet_audit_sha256':digest(output/'packet_audit.json')}
    checks = producer / 'public_checks.json'; selected = read(candidate)
    conditions = {row['label']: row for row in leave_one_out_conditions(selected, 'riu')}
    targets = {row['riu_id'] for row in selected['allocation_candidates']}
    design = read(output / 'targeted_loo/design.json')
    if set(design['conditions']) != targets or len(design['conditions']) != len(targets):
        raise ValueError('targeted LOO coverage mismatch')
    loo=[]
    for target in sorted(targets):
        condition=conditions[target]
        reduced=reduce_candidate(selected, atom_ids=condition['atom_ids'], observation_ids=condition['observation_ids'])
        path=output/'targeted_loo/packets'/f'{target}.json'
        if read(path) != reduced:raise ValueError('targeted LOO packet differs')
        loo.append(audit_episode_group(output/'targeted_loo', candidate_path=path, checks_path=checks,
            identity=identity, label=target, repetitions=identity['config']['loo_repetitions']))
    bundle=read(output/'evaluation_bundle.hidden.json');participants=[]
    for mode,pid,view in participant_view_candidates(selected,bundle):
        path=output/'participant_candidates'/mode/f'{pid}.json'
        if read(path) != view:raise ValueError('participant candidate differs')
        group=audit_episode_group(output/'participant_diagnostics'/mode/pid,
            candidate_path=path, checks_path=checks, identity=identity)
        if group['passes'] != 0:raise ValueError('participant-only gate failed')
        participants.append(group)
    loaded=load_export(output/'export/instance.json')
    for mode in MODES:
        for row in bundle['modes'][mode]:
            if json.loads(loaded.render_model_input(mode,row['participant_id'])['input'][1]['content'])['evidence'] != row['packet']:
                raise ValueError('actual export consumer changed packet')
    if tree_receipt(producer) != before['producer_tree']:raise ValueError('producer changed during postflight')
    return {'schema':'deepswe_independent_handoff_postflight_v1','status':'passed',
            'producer_tree_unchanged':True,'new_episode_groups':loo+participants,
            'export_sha256':digest(output/'export/instance.json')}
