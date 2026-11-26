"""Grounded distractor packing for private, functionally allocated source packets."""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re
import subprocess
import fnmatch
from teamgym.tokenization import count_tokens

BULK_BASE = r'''
import json, subprocess, sys, pathlib, hashlib
r=json.load(sys.stdin)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==r['commit']
assert not subprocess.check_output(['git','status','--porcelain'],text=True)
entries=subprocess.check_output(['git','ls-tree','-r','-z',r['commit']]).decode('utf-8').split('\0')
tree=[e.split('\t',1)[1] for e in entries if e and e.split(' ',1)[0] in ('100644','100755')]
required=set(r['required'])
exts={'.py','.go','.ts','.tsx','.js','.mjs','.cjs','.json','.md','.yaml','.yml','.toml','.txt'}
paths=[p for p in tree if (p in required or pathlib.Path(p).suffix in exts) and not any(s in p.lower() for s in ('package-lock','pnpm-lock','yarn.lock','snapshot','vendor/','dist/','node_modules/'))]
paths.sort(key=lambda p:(p not in required,hashlib.sha256(p.encode()).hexdigest()))
files={};size=0
batch=subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for p in paths:
    if p not in required and (size>12000000 or len(files)>600): break
    batch.stdin.write((r['commit']+':'+p+'\n').encode());batch.stdin.flush()
    header=batch.stdout.readline().decode().split()
    if header[-1]=='missing':continue
    raw=batch.stdout.read(int(header[-1]));batch.stdout.read(1)
    try:text=raw.decode('utf-8')
    except UnicodeDecodeError:continue
    if len(text)>250000 and p not in required:continue
    files[p]=text;size+=len(text)
batch.stdin.close();batch.wait()
print(json.dumps({'files':files,'tree':tree,'commit':r['commit']}))
'''

NEUTRAL_LIMITS = {'max_total_bytes': 12000000, 'max_files': 600}
NEUTRAL_SELECTION = ('Sorted pinned-base regular source files for the declared background language, '
                     'excluding fixed public test-path patterns and every original discovered path; '
                     'whole-file first-fit within fixed byte/file limits; no task, reference-patch, '
                     'hidden-test or cohort-based selection.')
BACKGROUND_LANGUAGES = {
    'go': {'extensions': ['.go'], 'excluded_directory_names': [],
           'excluded_filename_globs': ['*_test.go']},
    'python': {'extensions': ['.py'], 'excluded_directory_names': ['test', 'tests', '__tests__'],
               'excluded_filename_globs': ['test_*.py', '*_test.py', '*_tests.py', 'conftest.py']},
    'typescript': {'extensions': ['.ts', '.tsx'], 'excluded_directory_names': ['test', 'tests', '__tests__'],
                   'excluded_filename_globs': ['*.test.ts', '*.test.tsx', '*.spec.ts', '*.spec.tsx']},
    'javascript': {'extensions': ['.js', '.jsx', '.mjs', '.cjs'], 'excluded_directory_names': ['test', 'tests', '__tests__'],
                   'excluded_filename_globs': ['*.test.js', '*.test.jsx', '*.test.mjs', '*.test.cjs',
                                               '*.spec.js', '*.spec.jsx', '*.spec.mjs', '*.spec.cjs']},
    'rust': {'extensions': ['.rs'], 'excluded_directory_names': ['test', 'tests', '__tests__'],
             'excluded_filename_globs': ['test_*.rs', '*_test.rs', 'tests.rs']},
}


def normalize_background_language(language):
    aliases = {'golang': 'go', 'py': 'python', 'ts': 'typescript', 'js': 'javascript', 'rs': 'rust'}
    if not isinstance(language, str):
        raise ValueError('neutral_repository_unsupported_language')
    value = language.strip().lower()
    value = aliases.get(value, value)
    if value not in BACKGROUND_LANGUAGES:
        raise ValueError('neutral_repository_unsupported_language: ' + language)
    return value


def infer_background_language(candidate):
    extensions = {ext: language for language, policy in BACKGROUND_LANGUAGES.items()
                  for ext in policy['extensions']}
    inferred = {extensions.get(PurePosixPath(atom['path']).suffix) for atom in candidate.get('code_atoms', [])}
    if not inferred or None in inferred or len(inferred) != 1:
        raise ValueError('neutral_repository_ambiguous_language_requires_explicit_declaration')
    return inferred.pop()


def eligible_background_path(path, policy):
    parts = PurePosixPath(path).parts
    return (PurePosixPath(path).suffix in policy['extensions']
            and not set(parts[:-1]) & set(policy['excluded_directory_names'])
            and not any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in policy['excluded_filename_globs']))

# Also exercised directly in a temporary local Git repository by offline tests.
NEUTRAL_GIT_READER = r'''
import fnmatch, hashlib, json, pathlib, subprocess, sys
r=json.load(sys.stdin)
def git(*args):return subprocess.check_output(['git',*args])
def clean():return not git('status','--porcelain','--untracked-files=all')
assert git('rev-parse','HEAD').decode().strip()==r['base_commit'], 'neutral_base_commit_mismatch'
assert clean(), 'neutral_repository_dirty_before'
entries=git('ls-tree','-r','-z',r['base_commit']).decode('utf-8').split('\0')
eligible=[]
for entry in entries:
    if not entry:continue
    metadata,path=entry.split('\t',1)
    mode,kind,oid=metadata.split()
    if mode not in ('100644','100755') or kind!='blob':continue
    parts=pathlib.PurePosixPath(path).parts
    policy=r['language_policy']
    if pathlib.PurePosixPath(path).suffix not in policy['extensions']:continue
    if set(parts[:-1]) & set(policy['excluded_directory_names']):continue
    if any(fnmatch.fnmatchcase(parts[-1],pattern) for pattern in policy['excluded_filename_globs']):continue
    if path in r['excluded_discovered_paths']:continue
    assert '\n' not in path and '\r' not in path
    eligible.append((path,oid))
files={};total=0;catalog=[];skipped=[]
for path,oid in sorted(eligible):
    size=int(git('cat-file','-s',oid))
    catalog.append({'path':path,'git_blob_sha1':oid,'bytes':size})
    if len(files)>=r['limits']['max_files'] or total+size>r['limits']['max_total_bytes']:
        skipped.append(path);continue
    raw=git('cat-file','blob',oid)
    assert len(raw)==size
    assert hashlib.sha1(b'blob '+str(size).encode()+b'\0'+raw).hexdigest()==oid
    files[path]={'text':raw.decode('utf-8'),'bytes':size,'git_blob_sha1':oid,
                 'sha256':hashlib.sha256(raw).hexdigest()}
    total+=size
assert clean(), 'neutral_repository_dirty_after'
assert git('rev-parse','HEAD').decode().strip()==r['base_commit'], 'neutral_base_changed'
print(json.dumps({'schema':'deepswe_neutral_repository_source_v2',**r,
    'clean_before':True,'clean_after':True,'eligible_files':catalog,
    'excluded_by_budget':skipped,'files':files,'selected_bytes':total}))
'''


def _safe_repository_path(path):
    return (isinstance(path, str) and bool(path) and path != '.' and '\\' not in path
            and not any(ord(c) < 32 for c in path) and not PurePosixPath(path).is_absolute()
            and '..' not in path.split('/') and str(PurePosixPath(path)) == path)


def validate_neutral_repository(corpus, *, base_commit, runtime_image_id, excluded_paths, language=None):
    """Check the entire retained corpus, including Git blob identities and limits."""
    expected = {'schema', 'base_commit', 'runtime_image_id', 'excluded_discovered_paths',
                'limits', 'selection_policy', 'clean_before', 'clean_after', 'eligible_files',
                'excluded_by_budget', 'files', 'selected_bytes', 'background_language', 'language_policy'}
    if not isinstance(corpus, dict) or set(corpus) != expected:
        raise ValueError('neutral_repository_schema')
    declared = normalize_background_language(language if language is not None else corpus['background_language'])
    if (corpus['schema'] != 'deepswe_neutral_repository_source_v2'
            or corpus['base_commit'] != base_commit or corpus['runtime_image_id'] != runtime_image_id
            or corpus['excluded_discovered_paths'] != sorted(excluded_paths)
            or corpus['limits'] != NEUTRAL_LIMITS or corpus['selection_policy'] != NEUTRAL_SELECTION
            or corpus['background_language'] != declared or corpus['language_policy'] != BACKGROUND_LANGUAGES[declared]
            or corpus['clean_before'] is not True or corpus['clean_after'] is not True):
        raise ValueError('neutral_repository_identity_or_policy')
    files, catalog = corpus['files'], corpus['eligible_files']
    if not isinstance(files, dict) or not files or not isinstance(catalog, list):
        raise ValueError('neutral_repository_no_eligible_source')
    paths = [row.get('path') for row in catalog if isinstance(row, dict)]
    if len(paths) != len(catalog) or not all(_safe_repository_path(p) for p in paths):
        raise ValueError('neutral_repository_invalid_path')
    if paths != sorted(set(paths)) or set(paths) & set(excluded_paths):
        raise ValueError('neutral_repository_hotspot_or_duplicate')
    if any(not eligible_background_path(p, corpus['language_policy']) for p in paths):
        raise ValueError('neutral_repository_source_type')
    selected, skipped, total = [], [], 0
    for record in catalog:
        if (set(record) != {'path', 'git_blob_sha1', 'bytes'} or type(record['bytes']) is not int
                or record['bytes'] < 0 or not isinstance(record['git_blob_sha1'], str)
                or not re.fullmatch(r'[0-9a-f]{40}', record['git_blob_sha1'])):
            raise ValueError('neutral_repository_catalog')
        path = record['path']
        if len(selected) >= NEUTRAL_LIMITS['max_files'] or total + record['bytes'] > NEUTRAL_LIMITS['max_total_bytes']:
            skipped.append(path)
            continue
        if path not in files or not isinstance(files[path], dict) or set(files[path]) != {'text', 'sha256', 'git_blob_sha1', 'bytes'}:
            raise ValueError('neutral_repository_missing_blob')
        row = files[path]
        if not isinstance(row['text'], str):
            raise ValueError('neutral_repository_blob_text')
        raw = row['text'].encode('utf-8')
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if (row['bytes'] != len(raw) or row['bytes'] != record['bytes']
                or row['git_blob_sha1'] != blob or blob != record['git_blob_sha1']
                or row['sha256'] != hashlib.sha256(raw).hexdigest()):
            raise ValueError('neutral_repository_blob_hash')
        selected.append(path)
        total += len(raw)
    if set(files) != set(selected) or corpus['excluded_by_budget'] != skipped or corpus['selected_bytes'] != total:
        raise ValueError('neutral_repository_selection_mismatch')
    return deepcopy(corpus)


def collect_neutral_repository(candidate, excluded_paths, language=None):
    """Read fresh public Git blobs from the immutable runtime, without a cache."""
    execution = candidate.get('execution_manifest') or {}
    base = execution.get('source', {}).get('base_commit')
    image = execution.get('runtime_image', {}).get('id')
    if (not isinstance(base, str) or not re.fullmatch(r'[0-9a-f]{40}', base)
            or not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image)):
        raise ValueError('neutral_repository_requires_pinned_execution_identity')
    if not all(_safe_repository_path(p) for p in excluded_paths):
        raise ValueError('neutral_repository_invalid_excluded_path')
    language = infer_background_language(candidate) if language is None else normalize_background_language(language)
    request = {'base_commit': base, 'runtime_image_id': image,
               'excluded_discovered_paths': sorted(excluded_paths), 'limits': NEUTRAL_LIMITS,
               'selection_policy': NEUTRAL_SELECTION, 'background_language': language,
               'language_policy': deepcopy(BACKGROUND_LANGUAGES[language])}
    command = ['docker', 'run', '--rm', '-i', '--pull', 'never', '--platform', 'linux/amd64',
               '--network', 'none', '--cpus', '1', '--memory', '1024m', '--workdir', '/app',
               '--entrypoint', 'python3', image, '-I', '-c', NEUTRAL_GIT_READER]
    completed = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise ValueError('neutral_repository_read_failed: ' + completed.stderr[-1200:])
    return validate_neutral_repository(json.loads(completed.stdout), base_commit=base,
                                      runtime_image_id=image, excluded_paths=excluded_paths, language=language)


def build_participants(candidate: dict, identities: list[dict] | None = None) -> list[dict]:
    """Only owned, raw source and observations; no constructor annotations."""
    allocations = candidate["allocation_candidates"]
    validate_ownership(candidate)
    if identities is None:
        identities = [
            {"name": f"Reviewer {i + 1}", "review_scope": "Software review"}
            for i in range(len(allocations))
        ]
    if len(identities) != len(allocations):
        raise ValueError("one_identity_per_riu_required")
    atoms = {a["id"]: a for a in candidate["code_atoms"]}
    observations = {o["id"]: o for o in candidate["observations"]}
    seen_atoms, seen_observations = set(), set()
    result = []
    for index, (allocation, identity) in enumerate(zip(allocations, identities), 1):
        atom_ids, obs_ids = allocation["code_atom_ids"], allocation["observation_ids"]
        if seen_atoms.intersection(atom_ids) or seen_observations.intersection(obs_ids):
            raise ValueError("overlapping_private_allocation")
        if not atom_ids and not obs_ids:
            raise ValueError("empty_private_allocation")
        seen_atoms.update(atom_ids)
        seen_observations.update(obs_ids)
        result.append({
            "participant_id": f"participant_{index}", "name": identity["name"],
            "review_scope": identity.get("review_scope", "Software review"),
            "private_source_material": {
                "code_fragments": [{k: atoms[a][k] for k in
                                    ("path", "line_start", "line_end", "text")} for a in atom_ids],
                "observations": [{k: observations[o].get(k) for k in
                                  ("argv", "stdout", "stderr", "exit_code")} for o in obs_ids],
            },
            "allocation_hidden": {"riu_id": allocation["riu_id"],
                                  "code_atom_ids": atom_ids, "observation_ids": obs_ids},
        })
    if seen_atoms != set(atoms):
        raise ValueError("unallocated_code_atoms")
    if seen_observations != set(observations):
        raise ValueError("unallocated_observations")
    return result


def tokens(material):
    return count_tokens(json.dumps(material, ensure_ascii=False))


def unowned_spans(candidate, files, chunk_lines=60):
    """Exclude every allocation's original source before producing filler spans."""
    covered = {}
    for atom in candidate['code_atoms']:
        covered.setdefault(atom['path'], []).append((atom['line_start'], atom['line_end']))
    result = []
    for path, text in sorted(files.items()):
        lines = text.splitlines(keepends=True)
        cursor = 1
        gaps = []
        for start, end in sorted(covered.get(path, [])):
            if start > cursor:
                gaps.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= len(lines):
            gaps.append((cursor, len(lines)))
        for start, end in gaps:
            for pos in range(start, end + 1, chunk_lines):
                stop = min(pos + chunk_lines - 1, end)
                body = ''.join(lines[pos - 1:stop])
                if body.strip():
                    result.append({'path': path, 'line_start': pos, 'line_end': stop, 'text': body})
    return result


def pack(candidate, files, observation_pool, target_tokens=40000, background_exclusions=None,
         background_policy='discovered', background_language=None, observation_background_exclusions=None):
    if background_policy not in ('discovered', 'neutral_repository'):
        raise ValueError('unknown_background_policy')
    corpus = None
    if background_policy == 'neutral_repository':
        # Preserve originals for canonical grounding, but never select their
        # unallocated lines as background under this policy.
        original_paths = sorted(files)
        language = infer_background_language(candidate) if background_language is None else normalize_background_language(background_language)
        corpus = collect_neutral_repository(candidate, original_paths, language=language)
        execution = candidate['execution_manifest']
        corpus = validate_neutral_repository(corpus,
            base_commit=execution['source']['base_commit'],
            runtime_image_id=execution['runtime_image']['id'], excluded_paths=original_paths, language=language)
        filler_files = {p: row['text'] for p, row in corpus['files'].items()}
        files = {**files, **filler_files}
    else:
        filler_files = files
    participants = deepcopy(build_participants(candidate))
    if len({o["id"] for o in observation_pool}) != len(observation_pool):
        raise ValueError("duplicate_collected_observation_id")
    observation_by_id = {o["id"]: o for o in observation_pool}
    for observation in candidate["observations"]:
        collected = observation_by_id.get(observation["id"])
        if collected is None or visible_observation(collected) != visible_observation(observation):
            raise ValueError("allocated_observation_collection_mismatch")
    owner = {obs_id: index for index, allocation in enumerate(candidate["allocation_candidates"])
             for obs_id in allocation["observation_ids"]}
    if set(owner) - set(observation_by_id):
        raise ValueError("allocated_observation_missing_from_collected_pool")
    observation_manifest = {}
    for index, p in enumerate(participants):
        own = set(p["allocation_hidden"]["observation_ids"])
        other = [o for o in observation_pool if o["id"] not in own and o['id'] in owner]
        # Stable rotation avoids putting the same conspicuous output first in
        # every packet. These are existing observations, never new executions.
        if other:
            offset = index % len(other)
            other = other[offset:] + other[:offset]
        extras, extra_owners = [], set()
        for observation in other:
            foreign_owner = owner[observation['id']]
            if foreign_owner in extra_owners:
                continue
            extras.append(observation)
            extra_owners.add(foreign_owner)
            if len(extras) == 3:
                break
        p["private_source_material"]["observations"].extend(
            [{k: o.get(k) for k in ("argv", "stdout", "stderr", "exit_code")} for o in extras])
        # Put every observation back in its original collection order and strip
        # IDs/families/ownership from the model-visible packet.
        signatures = {(tuple(o.get("argv") or []), o.get("stdout"), o.get("stderr"), o.get("exit_code")):
                      order for order, o in enumerate(observation_pool)}
        p["private_source_material"]["observations"].sort(key=lambda o: signatures.get(
            (tuple(o.get("argv") or []), o.get("stdout"), o.get("stderr"), o.get("exit_code")), 10**9))
        observation_manifest[p["participant_id"]] = {
            "owned": sorted(own), "distractor": [o["id"] for o in extras],
            "distractor_foreign_piu_ids": [candidate['allocation_candidates'][owner[o['id']]]['riu_id']
                                            for o in extras]}
    atoms = candidate['code_atoms']
    for a in atoms:
        if a['path'] not in files:
            raise ValueError('allocated_source_missing_from_discovered_code')
        actual = ''.join(files[a['path']].splitlines(keepends=True)[a['line_start']-1:a['line_end']])
        if actual != a['text'] or hashlib.sha256(actual.encode()).hexdigest() != a['sha256']:
            raise ValueError('source_grounding_mismatch')
    for p in participants:
        if tokens(p['private_source_material']) > target_tokens:
            raise ValueError('allocated_evidence_exceeds_packet_budget')
    fillers = unowned_spans(candidate, filler_files)
    fillers.sort(key=lambda s: (s['path'] not in {a['path'] for a in atoms},
                                hashlib.sha256(f"{s['path']}:{s['line_start']}".encode()).hexdigest()))
    used = []
    loads = [tokens(p["private_source_material"]) for p in participants]
    for span in fillers:
        span_cost = tokens(span) + 2
        fitting = []
        for i, p in enumerate(participants):
            m = p['private_source_material']
            cost = loads[i] + span_cost
            if cost <= target_tokens:
                owned_paths = {a['path'] for a in m['code_fragments']}
                fitting.append((loads[i], span['path'] not in owned_paths, i))
        if not fitting:
            continue
        _, _, index = min(fitting)
        participants[index]['private_source_material']['code_fragments'].append(span)
        loads[index] = tokens(participants[index]['private_source_material'])
        if loads[index] > target_tokens:
            participants[index]['private_source_material']['code_fragments'].pop()
            loads[index] = tokens(participants[index]['private_source_material'])
            continue
        used.append((index, span))
        if all(load >= target_tokens - 250 for load in loads):
            break
    for p in participants:
        ordered = sorted(p['private_source_material']['code_fragments'], key=lambda a:(a['path'],a['line_start']))
        merged = []
        for a in ordered:
            if merged and a['path']==merged[-1]['path'] and a['line_start']==merged[-1]['line_end']+1:
                merged[-1]['text'] += a['text']; merged[-1]['line_end'] = a['line_end']
            else:
                merged.append(dict(a))
        p['private_source_material']['code_fragments'] = merged
        audit_packet(candidate, p, files, target_tokens)
    exclusion_ledger = exclude_background(candidate, participants, files, target_tokens,
                                          background_exclusions)
    observation_exclusion_ledger = exclude_observation_background(candidate, participants,
        observation_pool, observation_manifest, files, target_tokens, observation_background_exclusions)
    manifest = {'tokenizer':'cl100k_base','target_tokens':target_tokens,
        'packet_tokens':{p['participant_id']:tokens(p['private_source_material']) for p in participants},
        'distractor_code_spans':len(used),
        'existing_observation_distractors_per_participant': {
            p['participant_id']: len(observation_manifest[p['participant_id']]['distractor']) for p in participants},
        'observation_allocation_hidden': observation_manifest,
        'source_policy':'Existing discovered_code only; exclusive non-RIU spans.',
        'observation_policy':('Existing collected, candidate-allocated foreign observations only; stable rotation '
                              'then greedy distinct-owner selection, at most one per foreign PIU and three total; '
                              'all owned observations retained, final collection order preserved.'),
        'unallocated_observation_distractors_excluded_hidden': [o['id'] for o in observation_pool if o['id'] not in owner],
        'foreign_core_audit':'exact_original_code_ranges_excluded; semantic_equivalence_not_proven'}
    if exclusion_ledger:
        manifest['background_exclusions_hidden'] = {
            'phase': 'post_pack_no_refill_or_redistribution',
            'entries': exclusion_ledger,
            'distractor_code_spans_before_exclusions': len(used)}
    if observation_exclusion_ledger:
        manifest['observation_background_exclusions_hidden'] = {
            'phase': 'post_pack_no_refill_replacement_or_redistribution',
            'entries': observation_exclusion_ledger}
    if corpus is not None:
        manifest['source_policy'] = ('Pinned-base neutral repository declared-language source only for background; '
                                     'every original discovered path excluded; exclusive non-PIU spans.')
        manifest['background_policy'] = background_policy
        manifest['neutral_repository_source_hidden'] = corpus
    return participants, manifest


def exclude_observation_background(candidate, participants, pool, allocation, files, maximum, exclusions):
    """Remove selected foreign runtime distractors after ALL source allocation."""
    if exclusions is None:
        return []
    if not isinstance(exclusions, list):
        raise ValueError('observation_background_exclusions_must_be_list')
    by_pid = {p['participant_id']: p for p in participants}
    by_id = {o['id']: o for o in pool}
    owners = {oid: i for i, a in enumerate(candidate['allocation_candidates']) for oid in a['observation_ids']}
    prepared, seen = [], set()
    for spec in exclusions:
        if not isinstance(spec, dict) or set(spec) != {'participant_id', 'observation_id', 'reason'}:
            raise ValueError('observation_background_exclusion_schema')
        pid, oid = spec['participant_id'], spec['observation_id']
        if not isinstance(pid, str) or pid not in by_pid:
            raise ValueError('observation_background_exclusion_unknown_participant')
        if not isinstance(oid, str) or oid not in by_id:
            raise ValueError('observation_background_exclusion_unknown_observation')
        if not isinstance(spec['reason'], str) or not spec['reason'].strip():
            raise ValueError('observation_background_exclusion_reason_required')
        if (pid, oid) in seen:
            raise ValueError('observation_background_exclusion_duplicate')
        seen.add((pid, oid))
        if oid not in owners:
            raise ValueError('observation_background_exclusion_unallocated')
        if oid in by_pid[pid]['allocation_hidden']['observation_ids']:
            raise ValueError('observation_background_exclusion_owned')
        if oid not in allocation[pid]['distractor']:
            raise ValueError('observation_background_exclusion_not_selected_visible')
        visible = visible_observation(by_id[oid])
        matches = [o for o in by_pid[pid]['private_source_material']['observations'] if o == visible]
        # Identical owned/foreign representations cannot be attributed safely
        # after IDs are stripped; refuse ambiguous removal rather than guess.
        if len(matches) != 1:
            raise ValueError('observation_background_exclusion_ambiguous_or_missing_visible')
        prepared.append((deepcopy(spec), visible, owners[oid]))
    ledger = []
    for spec, visible, owner_index in prepared:
        pid, oid = spec['participant_id'], spec['observation_id']
        material = by_pid[pid]['private_source_material']
        before = tokens(material)
        material['observations'].remove(visible)
        position = allocation[pid]['distractor'].index(oid)
        allocation[pid]['distractor'].pop(position)
        allocation[pid]['distractor_foreign_piu_ids'].pop(position)
        audit_packet(candidate, by_pid[pid], files, maximum)
        ledger.append({**spec, 'owner_riu_id': candidate['allocation_candidates'][owner_index]['riu_id'],
            'visible_sha256': hashlib.sha256(json.dumps(visible, sort_keys=True, ensure_ascii=False,
                separators=(',', ':')).encode()).hexdigest(), 'before_tokens': before,
            'after_tokens': tokens(material)})
    return ledger


def exclude_background(candidate, participants, files, maximum, exclusions):
    """Remove explicit original-source intervals from an already packed view.

    Specs may cover a whole source function even if only some of it is visible;
    each must intersect visible background. The ledger records only actual
    removed intervals. Canonical PIU overlap anywhere in a spec is forbidden.
    No source is filled back in, redistributed or token-sliced.
    """
    if exclusions is None:
        return []
    if not isinstance(exclusions, list):
        raise ValueError('background_exclusions_must_be_list')
    by_pid = {p['participant_id']: p for p in participants}
    validated = []
    required = {'participant_id', 'path', 'line_start', 'line_end', 'reason'}
    for spec in exclusions:
        if not isinstance(spec, dict) or set(spec) != required:
            raise ValueError('background_exclusion_schema')
        pid, path = spec['participant_id'], spec['path']
        if not isinstance(pid, str) or pid not in by_pid:
            raise ValueError('background_exclusion_unknown_participant')
        if (not isinstance(path, str) or not path or '\\' in path
                or PurePosixPath(path).is_absolute() or '..' in path.split('/')
                or str(PurePosixPath(path)) != path or path not in files):
            raise ValueError('background_exclusion_invalid_path')
        start, end = spec['line_start'], spec['line_end']
        lines = files[path].splitlines(keepends=True)
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ValueError('background_exclusion_invalid_bounds')
        if not isinstance(spec['reason'], str) or not spec['reason'].strip():
            raise ValueError('background_exclusion_reason_required')
        for atom in candidate['code_atoms']:
            if atom['path'] == path and max(start, atom['line_start']) <= min(end, atom['line_end']):
                raise ValueError('background_exclusion_canonical_overlap')
        for old in validated:
            if (old['participant_id'] == pid and old['path'] == path
                    and max(start, old['line_start']) <= min(end, old['line_end'])):
                raise ValueError('background_exclusion_overlapping_specs')
        visible = by_pid[pid]['private_source_material']['code_fragments']
        removed = []
        for block in visible:
            if block['path'] != path:
                continue
            lo, hi = max(start, block['line_start']), min(end, block['line_end'])
            if lo <= hi:
                body = ''.join(lines[lo - 1:hi])
                removed.append({'path': path, 'line_start': lo, 'line_end': hi,
                                'text_sha256': hashlib.sha256(body.encode()).hexdigest()})
        if not removed:
            raise ValueError('background_exclusion_not_visible')
        validated.append({**deepcopy(spec), 'requested_source_sha256': hashlib.sha256(
            ''.join(lines[start - 1:end]).encode()).hexdigest(), 'removed_intervals': removed})
    # Validate the whole plan before mutating any participant.
    ledger = []
    for pid, participant in by_pid.items():
        relevant = [s for s in validated if s['participant_id'] == pid]
        if not relevant:
            continue
        material = participant['private_source_material']
        before_tokens = tokens(material)
        kept = []
        for block in material['code_fragments']:
            intervals = [(block['line_start'], block['line_end'])]
            for spec in relevant:
                if spec['path'] != block['path']:
                    continue
                next_intervals = []
                for lo, hi in intervals:
                    if spec['line_end'] < lo or spec['line_start'] > hi:
                        next_intervals.append((lo, hi))
                    else:
                        if lo < spec['line_start']:
                            next_intervals.append((lo, spec['line_start'] - 1))
                        if hi > spec['line_end']:
                            next_intervals.append((spec['line_end'] + 1, hi))
                intervals = next_intervals
            for lo, hi in intervals:
                body = ''.join(files[block['path']].splitlines(keepends=True)[lo - 1:hi])
                # Use the ordinary packet shape, without gap markers or reasons.
                if kept and kept[-1]['path'] == block['path'] and kept[-1]['line_end'] + 1 == lo:
                    kept[-1]['line_end'] = hi
                    kept[-1]['text'] += body
                else:
                    kept.append({'path': block['path'], 'line_start': lo, 'line_end': hi, 'text': body})
        material['code_fragments'] = kept
        audit_packet(candidate, participant, files, maximum)
        ledger.append({'participant_id': pid, 'before_tokens': before_tokens,
                       'after_tokens': tokens(material), 'exclusions': relevant})
    return ledger


def audit_packet(candidate, participant, files, maximum):
    material = participant['private_source_material']
    if tokens(material) > maximum:
        raise ValueError('private_packet_budget')
    own = set(participant['allocation_hidden']['code_atom_ids'])
    foreign = [a for a in candidate['code_atoms'] if a['id'] not in own]
    for block in material['code_fragments']:
        actual=''.join(files[block['path']].splitlines(keepends=True)[block['line_start']-1:block['line_end']])
        if actual != block['text']:
            raise ValueError('distractor_not_grounded')
        for a in foreign:
            if a['path']==block['path'] and max(a['line_start'],block['line_start'])<=min(a['line_end'],block['line_end']):
                raise ValueError('foreign_riu_range_leak')
    if any(k in material for k in ('riu_id','allocation_hidden','relevance','rationale','golden_patch')):
        raise ValueError('private_metadata_leak')


def visible_observation(observation):
    return {k: observation[k] for k in ("argv", "stdout", "stderr", "exit_code")}


def validate_ownership(candidate):
    """Operational collaboration minimum and complete one-each unit ownership."""
    allocations = candidate["allocation_candidates"]
    if len(allocations) < 2:
        raise ValueError("collaboration_requires_two_pius")
    atoms = candidate["code_atoms"]
    observations = candidate["observations"]
    for rows in (atoms, observations):
        ids = [r["id"] for r in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("duplicate_or_invalid_evidence_id")
    pius = [a["riu_id"] for a in allocations]
    if len(set(pius)) != len(pius) or any(not isinstance(i, str) or not i for i in pius):
        raise ValueError("duplicate_or_invalid_piu_id")
    for field, rows in (("code_atom_ids", atoms), ("observation_ids", observations)):
        known = {r["id"] for r in rows}
        assigned = []
        for allocation in allocations:
            ids = allocation[field]
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise ValueError("invalid_allocation_ids")
            if not set(ids) <= known:
                raise ValueError("unknown_allocated_evidence")
            assigned.extend(ids)
        if len(assigned) != len(set(assigned)):
            raise ValueError("overlapping_private_allocation")
        if set(assigned) != known:
            raise ValueError("unallocated_" + ("code_atoms" if field == "code_atom_ids" else "observations"))
    if any(not a["code_atom_ids"] and not a["observation_ids"] for a in allocations):
        raise ValueError("empty_private_allocation")
