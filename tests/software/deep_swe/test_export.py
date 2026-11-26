"""The repair export gate and actual consumer inputs, without API or Docker."""
from copy import deepcopy
import hashlib
import json
import itertools

import pytest

from teamgym.data.software.deep_swe.export import (
    MODES, build_evaluation_bundle, bundle_sha256, candidate_sha256,
    freeze_export, load_export, make_validation_receipt, participant_view_candidates, sha256,
    validate_execution_manifest, validate_bundle,
)
from teamgym.data.software.deep_swe.full_context import build_participants, pack
from teamgym.data.software.deep_swe.packet_audit import (
    AUDIT_SYSTEM, CATEGORIES, CATEGORY_CRITERIA, audit_evaluation_bundle,
    audit_request, parse_audit_response,
)


def execution_fixture():
    from teamgym.data.software.deep_swe.source import REQUIRED_FILES
    files = {name: {"sha256": "c" * 64, "executable": False} for name in REQUIRED_FILES}
    code = {"teamgym/data/software/deep_swe/export.py": "d" * 64,
            "teamgym/data/software/deep_swe/stages/patch_episode.py": "b" * 64}
    return {"schema": "deepswe_execution_manifest_v1", "task_id": "fixture",
        "source": {"task_id": "fixture", "base_commit": "e" * 40,
                   "test_base_commit": "e" * 40, "source_revision": "f" * 40,
                   "tree": {"files": files, "sha256": sha256(files)}},
        "runtime_image": {"id": "sha256:" + "1" * 64, "repo_digests": []},
        "verifier_image": {"id": "sha256:" + "2" * 64, "repo_digests": []},
        "actor": {"model_slug": "gpt55", "model": "gpt-5.5", "provider": "bedrock_openai",
                  "base_url": "https://example.invalid/v1", "endpoint": "/responses",
                  "api_key_env": "PLACEHOLDER_API_KEY", "reasoning_effort": "medium", "temperature": 0},
        "protocol": {"patch_turns": 5, "total_output_tokens": 32000,
                     "max_output_tokens_per_call": 16000, "max_checks_per_episode": 15},
        "code": {"files": code, "sha256": sha256(code)},
        "environment": {"python": "3.11.test", "platform": "offline-fixture", "uv_lock_sha256": "a" * 64}}


def fixture():
    files = {"a.py": "# background a\ndef a():\n    return 1\n",
             "b.py": "# background b\ndef b():\n    return 2\n"}
    atoms = []
    observations = []
    allocations = []
    for index, (path, text) in enumerate(files.items(), 1):
        source = ''.join(text.splitlines(keepends=True)[1:])
        atoms.append({"id": f"atom-{index}", "path": path, "line_start": 2, "line_end": 3,
                      "text": source, "sha256": hashlib.sha256(source.encode()).hexdigest()})
        observations.append({"id": f"observation-{index}", "argv": ["python", path],
                             "stdout": str(index), "stderr": "", "exit_code": 0,
                             "constructor_rationale": "NEVER-SHOW-THIS"})
        allocations.append({"riu_id": f"riu-{index}", "code_atom_ids": [f"atom-{index}"],
                            "observation_ids": [f"observation-{index}"]})
    candidate = {"status": "candidate_frozen", "task_id": "fixture", "instruction": "Fix both functions.",
                 "image_id": "sha256:" + "1" * 64, "code_atoms": atoms, "observations": observations,
                 "execution_manifest": execution_fixture(),
                 "allocation_candidates": allocations,
                 "constructor_metadata": {"reason": "NEVER-SHOW-THIS", "reference_patch": "PRIVATE-PATCH"}}
    participants, _ = pack(candidate, files, observations, 1000)
    bundle = build_evaluation_bundle(candidate, {"syntax": ["python", "-m", "compileall"]}, participants)
    return candidate, bundle


def clean_review(request):
    return {"views": [{"mode": row["mode"], "participant_id": row["participant_id"],
                       "assessments": {category: {"issue": False,
                           "rationale": "Reviewed complete source, runtime output and formatting."}
                           for category in CATEGORIES}} for row in request["views"]]}


_episode_ids = itertools.count()


def episode(success=False, *, execution=None):
    return {"output_dir": f"episode-{next(_episode_ids)}", "episode_valid": True, "success": success, "manifest_sha256": "a" * 64,
            "result_sha256": "b" * 64, "execution_manifest_sha256": sha256(execution or execution_fixture())}


def evidence(candidate, bundle):
    return {"screening": [episode() for _ in range(3)],
            "qualification": [episode(), episode(True), episode(True)],
            "loo": {"targets": [{"riu_id": "riu-1", "rationale": "Check runtime boundary.",
                                  "episodes": [episode(True)]}]},
            "participant_results": [{"mode": mode, "participant_id": row["participant_id"],
                 "episodes": [episode() for _ in range(3)]}
                 for mode in MODES[1:] for row in bundle["modes"][mode]],
            "packet_audit": audit_evaluation_bundle(candidate, bundle, semantic_auditor=clean_review)}


def test_export_preserves_repair_protocol_and_actual_model_input_boundary(tmp_path):
    candidate, bundle = fixture()
    receipt = make_validation_receipt(candidate, bundle, **evidence(candidate, bundle))
    export = freeze_export(tmp_path / "export.json", candidate, bundle, receipt)
    assert export.protocol.patch_turns == 5
    assert export.protocol.hidden_verifier == "once_after_termination"
    assert export.protocol.max_output_tokens_per_call == 16000
    assert export.execution_manifest == candidate["execution_manifest"]
    assert export.protocol.operations == ("replace", "write", "delete")
    for mode, rows in bundle["modes"].items():
        for row in rows:
            actual = export.render_model_input(mode, row["participant_id"])
            packet = json.loads(actual["input"][1]["content"])["evidence"]
            assert packet == row["packet"]
            assert set(packet) == {"instruction", "code_fragments", "observations", "public_check_catalog"}
            serialized = json.dumps(actual)
            for hidden in ("riu-1", "atom-1", "observation-1", "NEVER-SHOW-THIS", "PRIVATE-PATCH", "constructor_metadata", "execution_manifest", "runtime_image", "verifier_image", "uv_lock_sha256", "source_revision"):
                assert hidden not in serialized
            assert actual["tools"][0]["name"] == "perform_action"
    # Consumer returns independent packets; callers cannot mutate a loaded export.
    export.packet(MODES[0])["instruction"] = "changed"
    export.execution_manifest["actor"]["model"] = "changed"
    assert export.execution_manifest["actor"]["model"] == "gpt-5.5"
    assert load_export(tmp_path / "export.json").packet(MODES[0])["instruction"] == candidate["instruction"]


@pytest.mark.parametrize("successes", [0, 1, 2, 3])
def test_qualification_requires_two_independent_successes_in_every_order(tmp_path, successes):
    candidate, bundle = fixture()
    path = tmp_path / "export.json"
    for flags in set(itertools.permutations([True] * successes + [False] * (3 - successes))):
        raw = evidence(candidate, bundle)
        raw["qualification"] = [episode(flag) for flag in flags]
        if successes < 2:
            with pytest.raises(ValueError, match="qualification_failed"):
                make_validation_receipt(candidate, bundle, **raw)
        else:
            receipt = make_validation_receipt(candidate, bundle, **raw)
            freeze_export(path, candidate, bundle, receipt)
            assert load_export(path).task_id == candidate["task_id"]


def test_loader_rejects_insufficient_successes_even_with_consistent_hashes(tmp_path):
    candidate, bundle = fixture()
    raw = evidence(candidate, bundle)
    receipt = make_validation_receipt(candidate, bundle, **raw)
    path = tmp_path / "export.json"
    freeze_export(path, candidate, bundle, receipt)
    payload = json.loads(path.read_text())
    payload["validation"]["evidence"]["qualification"][1]["success"] = False
    payload["validation"]["evidence_sha256"] = sha256(payload["validation"]["evidence"])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="qualification_failed"):
        load_export(path)


def test_participant_candidates_preserve_exact_view_and_runner_scope():
    candidate, bundle = fixture()
    views = participant_view_candidates(candidate, bundle)
    assert len(views) == 4
    for mode, pid, reduced in views:
        view = next(r["packet"] for r in bundle["modes"][mode] if r["participant_id"] == pid)
        assert [{k: a[k] for k in ("path", "line_start", "line_end", "text")} for a in reduced["code_atoms"]] == view["code_fragments"]
        assert [{k: o[k] for k in ("argv", "stdout", "stderr", "exit_code")} for o in reduced["observations"]] == view["observations"]
        assert reduced["packet_view_hidden"]["bundle_sha256"] == bundle_sha256(bundle)
        assert reduced["execution_manifest"] == candidate["execution_manifest"]


@pytest.mark.parametrize("mutation,error", [
    (lambda e: e["screening"][0].update(success=True), "screening_failed"),
    (lambda e: e["screening"].__setitem__(1, deepcopy(e["screening"][0])), "duplicate_or_missing_episode"),
    (lambda e: e["qualification"].pop(), "incomplete_episode"),
    (lambda e: [row.update(success=False) for row in e["qualification"]], "qualification_failed"),
    (lambda e: e["qualification"][0].update(episode_valid=False), "invalid_episode"),
    (lambda e: e["qualification"][0].pop("result_sha256"), "evidence_hash"),
    (lambda e: e["loo"].update(targets=[]), "loo_missing"),
    (lambda e: e["participant_results"].pop(), "validation_incomplete"),
    (lambda e: e["participant_results"][0]["episodes"][0].update(success=True), "alone_completed"),
])
def test_acceptance_requires_complete_bound_evidence(mutation, error):
    candidate, bundle = fixture()
    raw = evidence(candidate, bundle)
    mutation(raw)
    with pytest.raises(ValueError, match=error):
        make_validation_receipt(candidate, bundle, **raw)


def test_frozen_export_rejects_changed_packet_candidate_and_evidence(tmp_path):
    candidate, bundle = fixture()
    receipt = make_validation_receipt(candidate, bundle, **evidence(candidate, bundle))
    changed = deepcopy(candidate)
    changed["instruction"] += " changed"
    with pytest.raises(ValueError, match="candidate_bundle_mismatch"):
        freeze_export(tmp_path / "x.json", changed, bundle, receipt)
    receipt["evidence"]["qualification"][2]["success"] = False
    with pytest.raises(ValueError, match="evidence_hash_mismatch"):
        freeze_export(tmp_path / "x.json", candidate, bundle, receipt)
    receipt = make_validation_receipt(candidate, bundle, **evidence(candidate, bundle))
    freeze_export(tmp_path / "x.json", candidate, bundle, receipt)
    raw = json.loads((tmp_path / "x.json").read_text())
    raw["bundle"]["protocol"]["patch_turns"] = 6
    (tmp_path / "x.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="repair_semantics_changed"):
        load_export(tmp_path / "x.json")


def test_complete_ownership_and_collaboration_minimum():
    candidate, _ = fixture()
    candidate["allocation_candidates"] = candidate["allocation_candidates"][:1]
    with pytest.raises(ValueError, match="two_pius"):
        build_participants(candidate)
    candidate, _ = fixture()
    candidate["allocation_candidates"][1]["observation_ids"] = []
    with pytest.raises(ValueError, match="unallocated_observations"):
        build_participants(candidate)
    candidate, _ = fixture()
    candidate["allocation_candidates"][0]["code_atom_ids"].append("unknown")
    with pytest.raises(ValueError, match="unknown_allocated"):
        build_participants(candidate)


def test_nonowned_observation_duplication_is_reviewed_and_not_rejected_as_policy():
    candidate, bundle = fixture()
    request = audit_request(candidate, bundle)
    duplicates = [r for r in request["exact_findings"] if r["kind"] == "duplicated_nonowned_observation"]
    assert len(duplicates) == 2
    assert all(r["policy"] == "permitted_existing_runtime_background" for r in duplicates)
    assert audit_evaluation_bundle(candidate, bundle, semantic_auditor=clean_review)["passed"] is True


def test_packet_auditor_prompt_defines_both_flag_polarities_for_each_category():
    # This is a contract on instructions sent to the external reviewer, not a
    # semantic prediction test. The original prompt left insufficiency polarity
    # implicit, causing issue=true alongside "does not support entire repair".
    assert set(CATEGORY_CRITERIA) == set(CATEGORIES)
    for category, criteria in CATEGORY_CRITERIA.items():
        assert "issue=true when" in criteria
        assert "issue=false" in criteria
        assert f"{category}: {criteria}" in AUDIT_SYSTEM
    assert "An\ninsufficiency rationale MUST NOT be marked issue=true" in AUDIT_SYSTEM
    assert "Do not assume access to any unshown repository source" in AUDIT_SYSTEM
    assert "foreign runtime behavior alone is not reconstruction" in AUDIT_SYSTEM


def test_auditor_distinguishes_common_specification_from_private_functional_support():
    candidate, bundle = fixture()
    request = audit_request(candidate, bundle)
    # Baseline information is already in the actual packet. No schema change or
    # extra constructor-supplied "baseline" is needed by a frozen validator.
    assert set(request) == {'candidate_sha256', 'bundle_sha256', 'protocol',
        'constructor_functional_boundaries', 'views', 'exact_findings', 'limitations'}
    for view in request['views']:
        assert view['packet']['instruction'] == candidate['instruction']
        assert view['packet']['public_check_catalog'] == bundle['modes'][MODES[0]][0]['packet']['public_check_catalog']
    # External-review prompt contract: shared requirements cannot become private
    # leakage, while real mixed reconstruction and whole-repair risks still apply.
    assert 'The public task instruction is not itself a PIU' in AUDIT_SYSTEM
    assert 'Do not double-count a common specification as that private increment' in AUDIT_SYSTEM
    assert 'owned baseline is not a blanket exemption' in AUDIT_SYSTEM
    assert 'restore the missing functional boundary still require issue=true' in AUDIT_SYSTEM
    assert 'does not\nexclude common information from complete_repair_possible' in AUDIT_SYSTEM


@pytest.mark.parametrize("category", CATEGORIES)
def test_each_positive_packet_finding_still_blocks_even_with_contradictory_rationale(category):
    candidate, bundle = fixture()
    def contradictory_review(request):
        response = clean_review(request)
        response["views"][0]["assessments"][category] = {
            "issue": True,
            "rationale": "The view does not support the entire repair."}
        return response
    audit = audit_evaluation_bundle(candidate, bundle, semantic_auditor=contradictory_review)
    assert audit["passed"] is False
    raw = evidence(candidate, bundle)
    raw["packet_audit"] = audit
    with pytest.raises(ValueError, match="semantic_audit_failed"):
        make_validation_receipt(candidate, bundle, **raw)


def test_foreign_duplicate_and_mixed_evidence_risk_are_checked_at_packet_level():
    candidate, bundle = fixture()
    # Same content at an unrelated source location evades range-only checking.
    packet = bundle["modes"][MODES[2]][0]["packet"]
    foreign = candidate["code_atoms"][1]
    packet["code_fragments"].append({"path": "copied.py", "line_start": 1, "line_end": 2, "text": foreign["text"]})
    request = audit_request(candidate, bundle)
    assert any(f["kind"] == "exact_foreign_code_content" for f in request["exact_findings"])
    def mixed_risk(request):
        response = clean_review(request)
        response["views"][2]["assessments"]["code_observation_reconstruction"] = {
            "issue": True, "rationale": "Code plus duplicated output reconstructs a foreign boundary."}
        return response
    audit = audit_evaluation_bundle(candidate, bundle, semantic_auditor=mixed_risk)
    assert audit["passed"] is False
    raw = evidence(candidate, bundle)
    raw["packet_audit"] = audit
    with pytest.raises(ValueError, match="semantic_audit_failed"):
        make_validation_receipt(candidate, bundle, **raw)


@pytest.mark.parametrize("response", [{"views": []}, {"passed": True}, {"views": "true"}])
def test_semantic_auditor_output_must_cover_actual_views(response):
    candidate, bundle = fixture()
    with pytest.raises(ValueError, match="semantic_packet_audit"):
        audit_evaluation_bundle(candidate, bundle, semantic_auditor=lambda _: response)


def test_rehashed_audit_of_different_inputs_is_rejected():
    candidate, bundle = fixture()
    raw = evidence(candidate, bundle)
    audit = raw["packet_audit"]
    audit["request"]["views"][0]["packet"]["instruction"] = "Different packet"
    audit["request_sha256"] = sha256(audit["request"])
    with pytest.raises(ValueError, match="reviewed_different_input"):
        make_validation_receipt(candidate, bundle, **raw)


@pytest.mark.parametrize('mutation,error', [
    (lambda m: m.pop('runtime_image'), 'schema'),
    (lambda m: m['source'].update(test_base_commit='a' * 40), 'source_base_mismatch'),
    (lambda m: m['source'].update(source_revision=None), 'source_revision_missing'),
    (lambda m: m['source']['tree'].update(sha256='0' * 64), 'source_tree_hash_mismatch'),
    (lambda m: m['runtime_image'].update(id='mutable:tag'), 'runtime_image_id_missing'),
    (lambda m: m['verifier_image'].update(id=None), 'verifier_image_id_missing'),
    (lambda m: m['actor'].update(env_file='/private/credentials.env'), 'actor_configuration_invalid'),
    (lambda m: m['code'].update(sha256='0' * 64), 'code_hash_mismatch'),
    (lambda m: m['environment'].pop('uv_lock_sha256'), 'environment_or_lock_hash_missing'),
    (lambda m: m['protocol'].update(max_output_tokens_per_call=0), 'protocol_budget_invalid'),
])
def test_execution_manifest_requires_reproducible_source_images_actor_code_and_lock(mutation, error):
    manifest = execution_fixture()
    mutation(manifest)
    with pytest.raises(ValueError, match=error):
        validate_execution_manifest(manifest, task_id='fixture')


def test_candidate_runtime_and_all_four_budgets_must_match_execution_manifest():
    candidate, bundle = fixture()
    validate_execution_manifest(candidate['execution_manifest'], task_id='fixture',
                                protocol=bundle['protocol'], candidate_image_id=candidate['image_id'])
    with pytest.raises(ValueError, match='candidate_runtime_mismatch'):
        validate_execution_manifest(candidate['execution_manifest'], task_id='fixture',
                                    candidate_image_id='sha256:' + '3' * 64)
    changed = deepcopy(bundle)
    changed['protocol']['max_output_tokens_per_call'] -= 1
    with pytest.raises(ValueError, match='protocol_budget_mismatch'):
        validate_bundle(changed)


def test_model_change_cannot_reuse_old_execution_receipts_even_with_rehashed_bundle():
    candidate, bundle = fixture()
    raw = evidence(candidate, bundle)
    candidate['execution_manifest']['actor']['model'] = 'different-model'
    bundle['execution_manifest'] = deepcopy(candidate['execution_manifest'])
    bundle['candidate_sha256'] = candidate_sha256(candidate)
    validate_bundle(bundle)
    with pytest.raises(ValueError, match='episode_execution_manifest_mismatch'):
        make_validation_receipt(candidate, bundle, **raw)


def test_export_v2_requires_hidden_manifest_and_rejects_stale_manifest_change(tmp_path):
    candidate, bundle = fixture()
    receipt = make_validation_receipt(candidate, bundle, **evidence(candidate, bundle))
    path = tmp_path / 'export.json'
    freeze_export(path, candidate, bundle, receipt)
    original = json.loads(path.read_text())
    assert original['schema'] == 'teamgym_deepswe_repair_export_v2'
    missing = deepcopy(original)
    missing['bundle'].pop('execution_manifest')
    path.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match='invalid_bundle_schema'):
        load_export(path)
    changed = deepcopy(original)
    changed['bundle']['execution_manifest']['actor']['model'] = 'changed'
    changed['bundle_sha256'] = bundle_sha256(changed['bundle'])
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='stale_validation_receipt'):
        load_export(path)
