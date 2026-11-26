"""Frozen DeepSWE repair exports and their explicit, typed consumer boundary.

This format describes atomic patch actions and official verification. It is not
convertible to a scalar-answer QA example without losing the repair protocol.
Constructor evidence and validation receipts remain outside model inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .full_context import build_participants, validate_ownership
from .qualification import DEFAULT_QUALIFICATION_POLICY, QualificationPolicy

SCHEMA = "teamgym_deepswe_repair_export_v2"
QUALIFICATION_RUNS = DEFAULT_QUALIFICATION_POLICY.repetitions
MIN_QUALIFICATION_PASSES = DEFAULT_QUALIFICATION_POLICY.minimum_passes
MODES = ("single_agent_all_piu", "multi_agent_piu_only", "multi_agent_full_context")
CODE_FIELDS = ("path", "line_start", "line_end", "text")
OBS_FIELDS = ("argv", "stdout", "stderr", "exit_code")


def validate_execution_manifest(manifest: dict, *, task_id: str, protocol: dict | None = None,
                                candidate_image_id: str | None = None) -> None:
    """Validate portable hidden execution identity without reading local files.

    Hashes bind the source and software snapshot; they are not signatures from
    an independent verifier. No constructor/reference bodies are model inputs.
    """
    def fail(reason):
        raise ValueError("execution_manifest_" + reason)
    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if not isinstance(manifest, dict) or set(manifest) - {'qualification'} != {
            "schema", "task_id", "source", "runtime_image", "verifier_image", "actor", "protocol", "code", "environment"}:
        fail("missing_or_invalid_schema")
    QualificationPolicy.from_manifest(manifest)
    if manifest["schema"] != "deepswe_execution_manifest_v1" or manifest["task_id"] != task_id:
        fail("task_identity_mismatch")
    source = manifest["source"]
    if not isinstance(source, dict) or source.get("task_id") != task_id:
        fail("source_task_mismatch")
    base = source.get("base_commit")
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", base) or source.get("test_base_commit") != base:
        fail("source_base_mismatch")
    if not isinstance(source.get("source_revision"), str) or not re.fullmatch(r"[0-9a-fA-F]{40}", source["source_revision"]):
        fail("source_revision_missing")
    tree = source.get("tree")
    if not isinstance(tree, dict) or not isinstance(tree.get("files"), dict) or not tree["files"]:
        fail("source_hashes_missing")
    from .source import REQUIRED_FILES
    if not set(REQUIRED_FILES) <= set(tree["files"]):
        fail("source_artifact_hashes_missing")
    if any(not isinstance(path, str) or not isinstance(row, dict) or not digest(row.get("sha256"))
           for path, row in tree["files"].items()):
        fail("source_file_hash_invalid")
    if tree.get("sha256") != sha256(tree["files"]):
        fail("source_tree_hash_mismatch")
    for kind in ("runtime_image", "verifier_image"):
        image = manifest[kind]
        if not isinstance(image, dict) or not isinstance(image.get("id"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image["id"]):
            fail(kind + "_id_missing")
        if not isinstance(image.get("repo_digests"), list) or not all(isinstance(v, str) for v in image["repo_digests"]):
            fail(kind + "_digests_invalid")
    if candidate_image_id is not None and candidate_image_id != manifest["runtime_image"]["id"]:
        fail("candidate_runtime_mismatch")
    actor = manifest["actor"]
    actor_fields = {"model_slug", "model", "provider", "base_url", "endpoint", "api_key_env", "reasoning_effort", "temperature"}
    if not isinstance(actor, dict) or set(actor) != actor_fields:
        fail("actor_configuration_invalid")
    if any(not isinstance(actor.get(k), str) or not actor[k] for k in actor_fields - {"reasoning_effort", "temperature"}):
        fail("actor_configuration_invalid")
    if actor["reasoning_effort"] is not None and not isinstance(actor["reasoning_effort"], str):
        fail("actor_reasoning_invalid")
    if actor["temperature"] is not None and type(actor["temperature"]) not in (int, float):
        fail("actor_temperature_invalid")
    declared = manifest["protocol"]
    budgets = {"patch_turns", "total_output_tokens", "max_output_tokens_per_call", "max_checks_per_episode"}
    if not isinstance(declared, dict) or set(declared) != budgets or any(type(declared[k]) is not int or declared[k] < 1 for k in budgets) or declared["patch_turns"] != 5:
        fail("protocol_budget_invalid")
    if protocol is not None and any(declared[k] != protocol.get(k) for k in budgets):
        fail("protocol_budget_mismatch")
    code = manifest["code"]
    if not isinstance(code, dict) or not isinstance(code.get("files"), dict) or not code["files"] or any(not isinstance(k, str) or not digest(v) for k, v in code["files"].items()):
        fail("code_hashes_invalid")
    if code.get("sha256") != sha256(code["files"]):
        fail("code_hash_mismatch")
    environment = manifest["environment"]
    if not isinstance(environment, dict) or not digest(environment.get("uv_lock_sha256")) or not all(isinstance(environment.get(k), str) and environment[k] for k in ("python", "platform")):
        fail("environment_or_lock_hash_missing")


def sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def candidate_sha256(candidate: dict) -> str:
    return sha256(candidate)


def bundle_sha256(bundle: dict) -> str:
    return sha256(bundle)


def _material(atoms, observations):
    return {"code_fragments": [{k: a[k] for k in CODE_FIELDS} for a in atoms],
            "observations": [{k: o[k] for k in OBS_FIELDS} for o in observations]}


def build_evaluation_bundle(candidate: dict, checks: dict, full_context_participants: list[dict],
                            *, patch_turns: int = 5, total_output_tokens: int = 32000,
                            max_checks_per_episode: int = 15, max_output_tokens_per_call: int = 16000) -> dict:
    validate_ownership(candidate)
    validate_execution_manifest(candidate.get("execution_manifest"), task_id=candidate["task_id"],
                                candidate_image_id=candidate["image_id"])
    if patch_turns != 5 or type(total_output_tokens) is not int or total_output_tokens < 1:
        raise ValueError("invalid_repair_budget")
    if type(max_checks_per_episode) is not int or max_checks_per_episode < 1:
        raise ValueError("invalid_public_check_budget")
    if not checks or not all(isinstance(k, str) and k and isinstance(v, list) and v
                             and all(isinstance(a, str) for a in v) for k, v in checks.items()):
        raise ValueError("invalid_public_check_catalog")
    sparse = build_participants(candidate)
    expected = {p["participant_id"]: p for p in sparse}
    if len(full_context_participants) != len(expected) or {
            p["participant_id"] for p in full_context_participants} != set(expected):
        raise ValueError("full_context_participant_coverage")
    seen_ranges = {}
    for participant in full_context_participants:
        if participant["allocation_hidden"] != expected[participant["participant_id"]]["allocation_hidden"]:
            raise ValueError("full_context_ownership_mismatch")
        _validate_material(participant["private_source_material"])
        own = expected[participant["participant_id"]]["private_source_material"]
        actual = participant["private_source_material"]
        for block in actual["code_fragments"]:
            for start, end in seen_ranges.get(block["path"], []):
                if max(start, block["line_start"]) <= min(end, block["line_end"]):
                    raise ValueError("full_context_code_not_exclusive")
            seen_ranges.setdefault(block["path"], []).append((block["line_start"], block["line_end"]))
        for fragment in own["code_fragments"]:
            if not any(block["path"] == fragment["path"] and block["line_start"] <= fragment["line_start"]
                       and block["line_end"] >= fragment["line_end"]
                       and ''.join(block["text"].splitlines(keepends=True)[
                           fragment["line_start"] - block["line_start"]:
                           fragment["line_end"] - block["line_start"] + 1]) == fragment["text"]
                       for block in actual["code_fragments"]):
                raise ValueError("owned_code_missing_from_full_context")
        if any(o not in actual["observations"] for o in own["observations"]):
            raise ValueError("owned_observation_missing_from_full_context")
    def view(pid, material):
        return {"participant_id": pid, "packet": {
            "instruction": candidate["instruction"], **deepcopy(material),
            "public_check_catalog": [{"id": k, "argv": list(v)} for k, v in checks.items()]}}
    bundle = {"task_id": candidate["task_id"], "candidate_sha256": candidate_sha256(candidate),
              "execution_manifest": deepcopy(candidate["execution_manifest"]),
              "protocol": {"kind": "atomic_repository_repair", "patch_turns": patch_turns,
                  "operations": ["replace", "write", "delete"], "early_finish": True,
                  "total_output_tokens": total_output_tokens,
                  "max_output_tokens_per_call": max_output_tokens_per_call,
                  "max_checks_per_episode": max_checks_per_episode,
                  "public_check_feedback": "status_only", "hidden_verifier": "once_after_termination",
                  "source_restricted": True, "test_edits_allowed": False},
              "modes": {MODES[0]: [view("single_agent", _material(candidate["code_atoms"], candidate["observations"]))],
                  MODES[1]: [view(p["participant_id"], p["private_source_material"]) for p in sparse],
                  MODES[2]: [view(p["participant_id"], p["private_source_material"]) for p in full_context_participants]}}
    validate_bundle(bundle)
    assert_model_input_boundary(candidate, bundle)
    return bundle


def _validate_material(material):
    if set(material) != {"code_fragments", "observations"}:
        raise ValueError("unexpected_source_material_field")
    for row in material["code_fragments"]:
        if set(row) != set(CODE_FIELDS) or not isinstance(row["path"], str) or not isinstance(row["text"], str):
            raise ValueError("invalid_visible_code")
        if type(row["line_start"]) is not int or type(row["line_end"]) is not int or not 0 < row["line_start"] <= row["line_end"]:
            raise ValueError("invalid_visible_code_range")
    for row in material["observations"]:
        if set(row) != set(OBS_FIELDS) or not isinstance(row["argv"], list) or not all(isinstance(a, str) for a in row["argv"]):
            raise ValueError("invalid_visible_observation")
        if not isinstance(row["stdout"], str) or not isinstance(row["stderr"], str) or type(row["exit_code"]) is not int:
            raise ValueError("invalid_visible_observation")


def validate_bundle(bundle):
    if set(bundle) != {"task_id", "candidate_sha256", "execution_manifest", "protocol", "modes"} or set(bundle["modes"]) != set(MODES):
        raise ValueError("invalid_bundle_schema")
    protocol = bundle["protocol"]
    if set(protocol) != {"kind", "patch_turns", "operations", "early_finish", "total_output_tokens", "max_output_tokens_per_call",
                         "max_checks_per_episode", "public_check_feedback", "hidden_verifier",
                         "source_restricted", "test_edits_allowed"}:
        raise ValueError("unexpected_repair_protocol_field")
    if protocol.get("early_finish") is not True or any(type(protocol.get(k)) is not int or protocol[k] < 1
            for k in ("total_output_tokens", "max_output_tokens_per_call", "max_checks_per_episode")):
        raise ValueError("invalid_repair_budget")
    if protocol.get("kind") != "atomic_repository_repair" or protocol.get("patch_turns") != 5 or protocol.get("operations") != ["replace", "write", "delete"] or protocol.get("hidden_verifier") != "once_after_termination" or protocol.get("public_check_feedback") != "status_only" or protocol.get("source_restricted") is not True or protocol.get("test_edits_allowed") is not False:
        raise ValueError("repair_semantics_changed")
    validate_execution_manifest(bundle["execution_manifest"], task_id=bundle["task_id"], protocol=protocol)
    piu_ids = [r["participant_id"] for r in bundle["modes"][MODES[1]]]
    if len(piu_ids) < 2 or len(set(piu_ids)) != len(piu_ids):
        raise ValueError("collaboration_requires_two_participants")
    if len(bundle["modes"][MODES[0]]) != 1 or {r["participant_id"] for r in bundle["modes"][MODES[2]]} != set(piu_ids) or len(bundle["modes"][MODES[2]]) != len(piu_ids):
        raise ValueError("mode_participant_mismatch")
    common = None
    for views in bundle["modes"].values():
        for row in views:
            if set(row) != {"participant_id", "packet"}:
                raise ValueError("unexpected_view_field")
            packet = row["packet"]
            if set(packet) != {"instruction", "code_fragments", "observations", "public_check_catalog"}:
                raise ValueError("unexpected_model_input_field")
            _validate_material({k: packet[k] for k in ("code_fragments", "observations")})
            if not isinstance(packet["instruction"], str) or not isinstance(packet["public_check_catalog"], list) or not packet["public_check_catalog"]:
                raise ValueError("invalid_common_input")
            check_ids = []
            for check in packet["public_check_catalog"]:
                if set(check) != {"id", "argv"} or not isinstance(check["id"], str) or not check["id"] or not isinstance(check["argv"], list) or not check["argv"] or not all(isinstance(a, str) for a in check["argv"]):
                    raise ValueError("invalid_public_check_catalog")
                check_ids.append(check["id"])
            if len(set(check_ids)) != len(check_ids):
                raise ValueError("duplicate_public_check_id")
            frozen_common = (packet["instruction"], packet["public_check_catalog"])
            if common is None:
                common = frozen_common
            if common != frozen_common:
                raise ValueError("condition_common_input_mismatch")


def assert_model_input_boundary(candidate, bundle):
    """Inspect serialized consumer packets, not just internal field names."""
    validate_bundle(bundle)
    hidden_ids = [r["riu_id"] for r in candidate["allocation_candidates"]]
    hidden_ids += [r["id"] for r in candidate["code_atoms"] + candidate["observations"]]
    for views in bundle["modes"].values():
        for row in views:
            text = json.dumps(row["packet"], ensure_ascii=False)
            if any(re.search(r"(?<![\w-])" + re.escape(identifier) + r"(?![\w-])", text) for identifier in hidden_ids):
                raise ValueError("hidden_evidence_identifier_in_model_input")


def participant_view_candidates(candidate: dict, bundle: dict) -> list[tuple[str, str, dict]]:
    """Internal runner adapters; each source-limited edit scope is its exact view."""
    if bundle["candidate_sha256"] != candidate_sha256(candidate):
        raise ValueError("candidate_bundle_mismatch")
    result = []
    for mode in MODES[1:]:
        for view in bundle["modes"][mode]:
            packet = view["packet"]
            atoms = [{"id": f"view-code-{i}", **deepcopy(block),
                      "sha256": hashlib.sha256(block["text"].encode()).hexdigest()}
                     for i, block in enumerate(packet["code_fragments"], 1)]
            observations = [{"id": f"view-observation-{i}", **deepcopy(o)}
                            for i, o in enumerate(packet["observations"], 1)]
            reduced = {"schema": "deepswe_participant_view_candidate_v1", "task_id": candidate["task_id"],
                "instruction": packet["instruction"], "image_id": candidate["image_id"],
                "execution_manifest": deepcopy(candidate["execution_manifest"]),
                "code_atoms": atoms, "observations": observations,
                "allocation_candidates": [{"riu_id": "internal-view", "code_atom_ids": [a["id"] for a in atoms],
                                           "observation_ids": [o["id"] for o in observations]}],
                "packet_view_hidden": {"mode": mode, "participant_id": view["participant_id"],
                                       "bundle_sha256": bundle_sha256(bundle)}}
            result.append((mode, view["participant_id"], reduced))
    return result


def _episodes(rows, count=None, *, execution_sha256=None):
    if not isinstance(rows, list) or (count is not None and len(rows) != count) or not rows:
        raise ValueError("incomplete_episode_evidence")
    locations = [row.get("output_dir") for row in rows]
    if any(not isinstance(location, str) or not location for location in locations) or len(set(locations)) != len(locations):
        raise ValueError("duplicate_or_missing_episode_reference")
    for row in rows:
        if row.get("episode_valid") is not True or type(row.get("success")) is not bool:
            raise ValueError("unavailable_or_invalid_episode")
        for key in ("manifest_sha256", "result_sha256"):
            if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("missing_episode_evidence_hash")
        if execution_sha256 is not None and row.get("execution_manifest_sha256") != execution_sha256:
            raise ValueError("episode_execution_manifest_mismatch")


def _validate_receipt(bundle, receipt):
    if receipt.get("candidate_sha256") != bundle["candidate_sha256"] or receipt.get("bundle_sha256") != bundle_sha256(bundle):
        raise ValueError("stale_validation_receipt")
    evidence = receipt["evidence"]
    if receipt.get("evidence_sha256") != sha256(evidence):
        raise ValueError("validation_evidence_hash_mismatch")
    execution_sha256 = sha256(bundle["execution_manifest"])
    qualification = QualificationPolicy.from_manifest(bundle["execution_manifest"])
    _episodes(evidence["screening"], 3, execution_sha256=execution_sha256)
    _episodes(evidence["qualification"], qualification.repetitions, execution_sha256=execution_sha256)
    if any(r["success"] for r in evidence["screening"]):
        raise ValueError("instruction_only_screening_failed")
    if sum(r["success"] for r in evidence["qualification"]) < qualification.minimum_passes:
        raise ValueError("complete_evidence_qualification_failed")
    targets = evidence["loo"].get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("targeted_loo_missing")
    for target in targets:
        if not isinstance(target.get("riu_id"), str) or not str(target.get("rationale", "")).strip():
            raise ValueError("targeted_loo_identity_or_rationale_missing")
        _episodes(target["episodes"], execution_sha256=execution_sha256)
    expected = {(mode, r["participant_id"]) for mode in MODES[1:] for r in bundle["modes"][mode]}
    rows = evidence["participant_results"]
    if len(rows) != len(expected) or {(r["mode"], r["participant_id"]) for r in rows} != expected:
        raise ValueError("participant_only_validation_incomplete")
    for row in rows:
        _episodes(row["episodes"], 3, execution_sha256=execution_sha256)
        if any(r["success"] for r in row["episodes"]):
            raise ValueError("participant_alone_completed_repair")
    from .packet_audit import validate_audit_receipt
    validate_audit_receipt(bundle, evidence["packet_audit"])


def _validate_candidate_evidence(candidate, bundle, evidence):
    from .packet_audit import audit_request
    validate_execution_manifest(candidate.get("execution_manifest"), task_id=candidate["task_id"],
                                protocol=bundle["protocol"], candidate_image_id=candidate["image_id"])
    if candidate["execution_manifest"] != bundle["execution_manifest"]:
        raise ValueError("execution_manifest_candidate_bundle_mismatch")
    expected_request = audit_request(candidate, bundle)
    if evidence["packet_audit"].get("request_sha256") != sha256(expected_request):
        raise ValueError("packet_audit_candidate_evidence_mismatch")
    targets = evidence["loo"].get("targets", [])
    ids = [target.get("riu_id") for target in targets]
    known = {row["riu_id"] for row in candidate["allocation_candidates"]}
    if len(set(ids)) != len(ids) or not set(ids) <= known:
        raise ValueError("unknown_or_duplicate_targeted_loo")


def make_validation_receipt(candidate, bundle, *, screening, qualification, loo,
                            participant_results, packet_audit):
    if candidate_sha256(candidate) != bundle["candidate_sha256"]:
        raise ValueError("candidate_bundle_mismatch")
    evidence = deepcopy({"screening": screening, "qualification": qualification, "loo": loo,
                         "participant_results": participant_results, "packet_audit": packet_audit})
    receipt = {"candidate_sha256": candidate_sha256(candidate), "bundle_sha256": bundle_sha256(bundle),
               "evidence_sha256": sha256(evidence), "evidence": evidence}
    _validate_receipt(bundle, receipt)
    _validate_candidate_evidence(candidate, bundle, evidence)
    return receipt


@dataclass(frozen=True)
class DeepSWERepairProtocol:
    patch_turns: int
    total_output_tokens: int
    max_checks_per_episode: int
    max_output_tokens_per_call: int
    operations: tuple[str, ...] = ("replace", "write", "delete")
    hidden_verifier: str = "once_after_termination"


@dataclass(frozen=True)
class DeepSWEExport:
    task_id: str
    protocol: DeepSWERepairProtocol
    bundle_sha256: str
    _bundle_json: str

    @property
    def execution_manifest(self) -> dict:
        """A fresh copy of hidden source/images/actor identity for execution."""
        return json.loads(self._bundle_json)["execution_manifest"]

    def packet(self, mode: str, participant_id: str | None = None) -> dict:
        views = json.loads(self._bundle_json)["modes"][mode]
        if participant_id is None and len(views) == 1:
            return views[0]["packet"]
        for view in views:
            if view["participant_id"] == participant_id:
                return view["packet"]
        raise ValueError("participant_id_required_or_unknown")

    def render_model_input(self, mode: str, participant_id: str | None = None) -> dict:
        from .stages.patch_episode import SYSTEM, TOOL
        return {"input": [{"role": "developer", "content": SYSTEM},
                          {"role": "user", "content": json.dumps({"evidence": self.packet(mode, participant_id)}, ensure_ascii=False)},
                          {"role": "user", "content": json.dumps({"remaining": {
                              "patch_turns": self.protocol.patch_turns,
                              "output_tokens": self.protocol.total_output_tokens,
                              "public_checks": self.protocol.max_checks_per_episode}})}],
                "tools": [deepcopy(TOOL)], "tool_choice": {"type": "function", "name": "perform_action"},
                "parallel_tool_calls": False}


def freeze_export(path: Path, candidate: dict, bundle: dict, receipt: dict) -> DeepSWEExport:
    if candidate.get("status") not in ("candidate_frozen", "validation_passed", "accepted"):
        raise ValueError("candidate_not_frozen")
    if candidate_sha256(candidate) != bundle["candidate_sha256"]:
        raise ValueError("candidate_bundle_mismatch")
    assert_model_input_boundary(candidate, bundle)
    _validate_receipt(bundle, receipt)
    _validate_candidate_evidence(candidate, bundle, receipt["evidence"])
    payload = {"schema": SCHEMA, "status": "accepted", "bundle": bundle,
               "bundle_sha256": bundle_sha256(bundle), "validation": receipt}
    from .stages.patch_runtime import save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, payload)
    return load_export(path)


def load_export(path: Path) -> DeepSWEExport:
    value = json.loads(Path(path).read_text())
    if set(value) != {"schema", "status", "bundle", "bundle_sha256", "validation"} or value["schema"] != SCHEMA or value["status"] != "accepted":
        raise ValueError("not_an_accepted_deepswe_export")
    bundle = value["bundle"]
    validate_bundle(bundle)
    if value["bundle_sha256"] != bundle_sha256(bundle):
        raise ValueError("export_bundle_hash_mismatch")
    _validate_receipt(bundle, value["validation"])
    protocol = bundle["protocol"]
    return DeepSWEExport(bundle["task_id"], DeepSWERepairProtocol(
        protocol["patch_turns"], protocol["total_output_tokens"], protocol["max_checks_per_episode"],
        protocol["max_output_tokens_per_call"]),
        value["bundle_sha256"], json.dumps(bundle, ensure_ascii=False))
