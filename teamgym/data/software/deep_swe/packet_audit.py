"""Whole-packet reconstruction and salience review, including mixed evidence.

Exact matches are findings, not a proof of functional leakage: a copied runtime
observation is explicitly permitted by the Full-Context distribution policy.
The semantic reviewer must assess meaningful evidence and complete-repair risk.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

from .export import (MODES, assert_model_input_boundary, bundle_sha256,
                     candidate_sha256, sha256, validate_bundle)
from .full_context import build_participants

CATEGORIES = ("foreign_core_equivalence", "code_observation_reconstruction",
              "format_salience", "complete_repair_possible")

CATEGORY_CRITERIA = {
    "foreign_core_equivalence": """issue=true when visible code supplies the
substantive functional core of a withheld non-owned PIU, including an equivalent
implementation at another path. Identify the actual visible code and the
withheld functional boundary it restores. issue=false for shared boilerplate,
punctuation, adjacent helpers or isolated facts that do not restore that core.
Review combinations involving tests or observations under the next category.""",
    "code_observation_reconstruction": """issue=true when the packet's combined
source, tests and/or observations recover a withheld non-owned functional PIU's
substantive behavior or implementation contract sufficiently to replace that
missing functional evidence. Identify the recovered boundary and explain how
the visible combination supplies it. This need not enable the entire repair.
issue=false when the finding is only that an already collected non-owned
observation exposes its recorded base-repository behavior, or adds isolated
foreign facts without reconstructing the withheld functional boundary. Such
observations are explicitly allowed Full-Context distractors. Duplication is
not an exemption if the combined packet actually reconstructs a withheld PIU;
equally, knowledge of some foreign runtime behavior alone is not reconstruction.""",
    "format_salience": """issue=true when labels, ordering, grouping or style
provide artificial clues about PIU identity, ownership, criticality or the
foreign functional boundary beyond the ordinary source content. Identify the
specific presentation cue and what it discloses. issue=false for ordinary path
and line metadata, neutral source order, uniformly formatted observations, or
task-relevant selection of source/tests/probes alone. Construction intentionally
selects relevant evidence; relevance by itself is not a formatting defect.""",
    "complete_repair_possible": """issue=true when common input plus the ENTIRE
actual participant packet plausibly supports a verifier-passing repair of the
whole task under the supplied source-restricted bounded protocol. Describe a
concrete credible route covering all required functional boundaries. issue=false
when the packet does not support the entire repair, for example because a
required parser, evaluator or assignment boundary remains unavailable. An
insufficiency rationale MUST NOT be marked issue=true. Uncertainty or inability
to prove impossibility alone is not a positive finding; conversely, a failed
solver attempt alone does not establish that complete repair is impossible.""",
}

AUDIT_SYSTEM = """Audit DeepSWE participant packets against their privately owned functional PIU.
Review every supplied view and all four categories using the complete actual
packet, including combinations of source, tests, observations and formatting.
Each issue flag reports a BLOCKING CONCERN: issue=true means the named risk IS
present; issue=false means that risk is not supported by the packet. Keep the
flag and rationale consistent. Never invert this polarity for insufficiency.

Only the view's packet is visible to that participant. owned_evidence and foreign
functional evidence are constructor-only comparison material, not extra solver
input. Do not assume access to any unshown repository source, adjacent files,
reference patch, hidden tests or verifier internals. Apply the supplied edit
turn/check budgets and allowed operations exactly; a public check does not grant
an unrestricted repository-read tool or reveal hidden expected outputs.

DeepSWE PIUs represent coherent functional behaviors/contracts, not a particular
reference patch. Full-Context explicitly permits selected previously collected
non-owned runtime observations to be duplicated as distractors. This is allowed
even when their recorded base behavior is useful; assess substantive functional
reconstruction rather than ownership of each isolated runtime fact. Apply the
same complete-repair risk check to both collaborative modes. Do not infer
acceptance from candidate status, constructor claims or prior failed runs.

Counterfactual common-input baseline for functional reconstruction:
packet.instruction, packet.public_check_catalog and the fixed public protocol
are common to every condition. The public task instruction is not itself a PIU.
Desired syntax, examples, expected behavior and error text already stated there
must not be counted as private foreign-PIU disclosure. First identify what the
common input plus the participant's owned evidence already supplies. Then name
the incremental visible private source, tests or observations that materially
replace a coherent missing functional PIU beyond that common-plus-owned baseline.
Explain which previously missing boundary this PRIVATE increment restores, not
merely which public requirements the packet repeats or confirms. Removing the
claimed increment should remove the identified additional functional support.
Do not double-count a common specification as that private increment.

For code_observation_reconstruction, common desired behavior plus permitted
isolated base-observation facts is not by itself evidence that a missing PIU
has been replaced. A positive finding must identify substantive private support
for that substitution. Actual combinations of source/tests/observations that
restore the missing functional boundary still require issue=true, even when the
instruction already names the target behavior and even without solution code.
The owned baseline is not a blanket exemption for intrinsically overlapping
PIUs: if owned private evidence itself restores a foreign functional core,
identify that concrete private overlap explicitly rather than attributing the
public specification to private disclosure. This baseline comparison does not
exclude common information from complete_repair_possible: that category still
evaluates the entire actual packet plus common input under the bounded protocol.

Category-specific decision rules:
""" + "\n\n".join(f"{name}: {CATEGORY_CRITERIA[name]}" for name in CATEGORIES) + """

Return JSON only:
{\"views\":[{\"mode\":\"...\",\"participant_id\":\"...\",\"assessments\":{
\"foreign_core_equivalence\":{\"issue\":false,\"rationale\":\"specific evidence\"},
\"code_observation_reconstruction\":{\"issue\":false,\"rationale\":\"specific evidence\"},
\"format_salience\":{\"issue\":false,\"rationale\":\"specific evidence\"},
\"complete_repair_possible\":{\"issue\":false,\"rationale\":\"specific evidence and uncertainty\"}}}]}.
Any issue=true requires revision or a new audit; there is no majority vote or
automatic exception. Report uncertainty in the rationale without reversing the
flag's meaning. Do not provide
patches, hidden tests or reference solution text in the result.
"""


def audit_request(candidate: dict, bundle: dict) -> dict:
    """Constructor-only request; never include this in participant input."""
    if candidate_sha256(candidate) != bundle["candidate_sha256"]:
        raise ValueError("candidate_bundle_mismatch")
    assert_model_input_boundary(candidate, bundle)
    owners = {p["participant_id"]: p for p in build_participants(candidate)}
    atoms = {a["id"]: a for a in candidate["code_atoms"]}
    observations = {o["id"]: o for o in candidate["observations"]}
    views, findings = [], []
    for mode in MODES[1:]:
        for view in bundle["modes"][mode]:
            pid = view["participant_id"]
            owner = owners[pid]["allocation_hidden"]
            foreign_atoms = [a for aid, a in atoms.items() if aid not in owner["code_atom_ids"]]
            foreign_obs = [o for oid, o in observations.items() if oid not in owner["observation_ids"]]
            packet = view["packet"]
            for atom in foreign_atoms:
                needle = atom["text"].strip()
                for index, block in enumerate(packet["code_fragments"]):
                    if needle and needle in block["text"]:
                        findings.append({"mode": mode, "participant_id": pid,
                            "kind": "exact_foreign_code_content", "foreign_atom_id": atom["id"],
                            "visible_block_index": index, "requires_semantic_review": True})
            for observation in foreign_obs:
                visible = {k: observation[k] for k in ("argv", "stdout", "stderr", "exit_code")}
                if visible in packet["observations"]:
                    findings.append({"mode": mode, "participant_id": pid,
                        "kind": "duplicated_nonowned_observation", "foreign_observation_id": observation["id"],
                        "policy": "permitted_existing_runtime_background", "requires_semantic_review": True})
            views.append({"mode": mode, "participant_id": pid, "packet": deepcopy(packet),
                "owned_evidence": deepcopy(owners[pid]["private_source_material"]),
                "foreign_functional_code": deepcopy(foreign_atoms),
                "foreign_functional_observations": deepcopy(foreign_obs)})
    return {"candidate_sha256": candidate_sha256(candidate), "bundle_sha256": bundle_sha256(bundle),
            "protocol": deepcopy(bundle["protocol"]), "constructor_functional_boundaries":
            deepcopy(candidate.get("constructor_metadata", {})), "views": views,
            "exact_findings": findings,
            "limitations": "Finite packet review and model repair runs do not prove information-theoretic necessity."}


def parse_audit_response(response: dict, expected_views: set[tuple[str, str]]) -> dict:
    if not isinstance(response, dict) or set(response) != {"views"} or not isinstance(response["views"], list):
        raise ValueError("invalid_semantic_packet_audit")
    rows = response["views"]
    if len(rows) != len(expected_views):
        raise ValueError("incomplete_semantic_packet_audit")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"mode", "participant_id", "assessments"}:
            raise ValueError("invalid_semantic_packet_audit_view")
        key = (row["mode"], row["participant_id"])
        if key not in expected_views or key in seen:
            raise ValueError("unknown_or_duplicate_semantic_packet_audit_view")
        seen.add(key)
        if not isinstance(row["assessments"], dict) or set(row["assessments"]) != set(CATEGORIES):
            raise ValueError("missing_semantic_packet_audit_category")
        for assessment in row["assessments"].values():
            if not isinstance(assessment, dict) or set(assessment) != {"issue", "rationale"} or type(assessment["issue"]) is not bool or not isinstance(assessment["rationale"], str) or not assessment["rationale"].strip():
                raise ValueError("invalid_semantic_packet_audit_assessment")
    return deepcopy(response)


def audit_evaluation_bundle(candidate: dict, bundle: dict, *, semantic_auditor: Callable[[dict], dict]) -> dict:
    request = audit_request(candidate, bundle)
    expected = {(v["mode"], v["participant_id"]) for v in request["views"]}
    response = parse_audit_response(semantic_auditor(deepcopy(request)), expected)
    return {"candidate_sha256": candidate_sha256(candidate), "bundle_sha256": bundle_sha256(bundle),
            "request_sha256": sha256(request), "response_sha256": sha256(response),
            "request": request, "response": response,
            "passed": not any(a["issue"] for v in response["views"] for a in v["assessments"].values())}


def validate_audit_receipt(bundle: dict, receipt: dict):
    validate_bundle(bundle)
    if receipt.get("candidate_sha256") != bundle["candidate_sha256"] or receipt.get("bundle_sha256") != bundle_sha256(bundle):
        raise ValueError("stale_packet_audit")
    request, response = receipt["request"], receipt["response"]
    if receipt.get("request_sha256") != sha256(request) or receipt.get("response_sha256") != sha256(response):
        raise ValueError("packet_audit_evidence_hash_mismatch")
    if request.get("bundle_sha256") != bundle_sha256(bundle) or request.get("candidate_sha256") != bundle["candidate_sha256"]:
        raise ValueError("stale_packet_audit_request")
    if request.get("protocol") != bundle["protocol"]:
        raise ValueError("packet_audit_reviewed_different_protocol")
    expected = {(mode, r["participant_id"]) for mode in MODES[1:] for r in bundle["modes"][mode]}
    parse_audit_response(response, expected)
    if {(v["mode"], v["participant_id"]) for v in request["views"]} != expected or len(request["views"]) != len(expected):
        raise ValueError("incomplete_packet_audit_request")
    for view in request["views"]:
        actual = next(v["packet"] for v in bundle["modes"][view["mode"]] if v["participant_id"] == view["participant_id"])
        if view["packet"] != actual:
            raise ValueError("packet_audit_reviewed_different_input")
    passed = not any(a["issue"] for v in response["views"] for a in v["assessments"].values())
    if not passed or receipt.get("passed") is not True:
        raise ValueError("packet_semantic_audit_failed")
