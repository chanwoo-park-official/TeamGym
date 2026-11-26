"""Candidate packets: assembly, reduction, and the unit vocabulary.

A *fragment pool* is the raw material stage 2 extracts from a pinned base commit.
A *candidate* is what one solver episode actually receives: selected fragments
renumbered as ``atom-NN``, the observations collected in stage 3, and an
allocation that groups them into RIUs. Every later stage -- qualification,
leave-one-out, contribution measurement, the three-condition sweep -- works by
producing a reduced candidate and running it, so reduction lives here rather than
being reimplemented per script.
"""
from __future__ import annotations

import copy
from typing import Iterable, Sequence


def assemble_instruction_only_candidate(task) -> dict:
    """Create the screening packet without running RIU discovery or extraction."""

    return {
        "schema": "deepswe_riu_candidate_v1",
        "task_id": task.task_id,
        "status": "instruction_only",
        "instruction": task.instruction,
        "image_id": task.runtime.docker_image,
        "repo_digests": [],
        "code_atoms": [],
        "observations": [],
        "allocation_candidates": [],
        "unallocated_background_observation_candidates": [],
        "allocation_notes": "instruction-only screening",
        "audits": {},
    }


def selectable(pool: dict) -> list[dict]:
    """Return all fragments, including declarations above the token target."""
    return list(pool["fragments"]) + list(pool.get("oversized", []))


def allocated_observation_ids(candidate: dict) -> set[str]:
    """Observations an actor can actually receive, i.e. owned by some RIU."""
    return {i for riu in candidate["allocation_candidates"] for i in riu["observation_ids"]}


def unit_ids(candidate: dict) -> tuple[list[str], list[str]]:
    """The candidate's atoms and its allocated observations, as two id lists."""
    allocated = allocated_observation_ids(candidate)
    return ([atom["id"] for atom in candidate["code_atoms"]],
            [o["id"] for o in candidate["observations"] if o["id"] in allocated])


def reduce_candidate(candidate: dict, *, atom_ids: Iterable[str] = (),
                     observation_ids: Iterable[str] = ()) -> dict:
    """Return a copy with the named units removed from atoms, observations, and RIUs.

    A RIU left holding nothing is dropped, so participant count follows the
    reduced packet rather than the original allocation.
    """
    drop_atoms = set(atom_ids)
    drop_observations = set(observation_ids)
    out = copy.deepcopy(candidate)
    out["code_atoms"] = [a for a in out["code_atoms"] if a["id"] not in drop_atoms]
    out["observations"] = [o for o in out["observations"] if o["id"] not in drop_observations]
    kept = []
    for riu in out["allocation_candidates"]:
        riu["code_atom_ids"] = [i for i in riu["code_atom_ids"] if i not in drop_atoms]
        riu["observation_ids"] = [i for i in riu["observation_ids"] if i not in drop_observations]
        if riu["code_atom_ids"] or riu["observation_ids"]:
            kept.append(riu)
    out["allocation_candidates"] = kept
    return out


def leave_one_out_conditions(candidate: dict, granularity: str) -> list[dict]:
    """Removal conditions for a leave-one-out sweep, plus a baseline arm.

    The baseline removes nothing and is not optional. A packet that passes only
    two runs in three would otherwise make every removal look harmless.
    """
    rows = [{"label": "baseline", "atom_ids": set(), "observation_ids": set()}]
    if granularity == "riu":
        for riu in candidate["allocation_candidates"]:
            rows.append({"label": riu["riu_id"],
                         "atom_ids": set(riu["code_atom_ids"]),
                         "observation_ids": set(riu["observation_ids"])})
        return rows
    atoms, observations = unit_ids(candidate)
    for atom_id in atoms:
        rows.append({"label": atom_id, "atom_ids": {atom_id}, "observation_ids": set()})
    for observation_id in observations:
        rows.append({"label": observation_id, "atom_ids": set(),
                     "observation_ids": {observation_id}})
    return rows


def group_key(atom: dict, group_by: str) -> str:
    if group_by == "file":
        return atom["path"]
    if group_by == "kind":
        return atom["kind"]
    if group_by == "one":
        return "all"
    return atom["id"]


def assemble_functional_candidate(
    pool: dict,
    *,
    rius: Sequence[dict],
    observations: Sequence[dict] = (),
    uncovered: Sequence[dict] = (),
) -> dict:
    """Build a candidate from constructor-defined functional RIU boundaries.

    Constructor metadata stays in ``constructor_metadata``. Actor packet
    builders consume only the atom and observation identifiers owned by each
    allocation, so rationales, hidden-test coverage, and functional labels are
    never solver-visible.
    """

    by_fragment = {row["id"]: row for row in selectable(pool)}
    by_observation = {row["id"]: dict(row) for row in observations}
    seen_fragments: set[str] = set()
    seen_observations: set[str] = set()

    normalized: list[dict] = []
    for index, raw in enumerate(rius, 1):
        fragment_ids = list(dict.fromkeys(raw.get("code_fragment_ids") or []))
        observation_ids = list(dict.fromkeys(raw.get("observation_ids") or []))
        unknown_fragments = [item for item in fragment_ids if item not in by_fragment]
        unknown_observations = [item for item in observation_ids if item not in by_observation]
        if unknown_fragments or unknown_observations:
            raise KeyError(
                f"unknown functional RIU units: fragments={unknown_fragments}, "
                f"observations={unknown_observations}"
            )
        duplicate_fragments = seen_fragments.intersection(fragment_ids)
        duplicate_observations = seen_observations.intersection(observation_ids)
        if duplicate_fragments or duplicate_observations:
            raise ValueError(
                "functional RIUs must own disjoint units: "
                f"fragments={sorted(duplicate_fragments)}, "
                f"observations={sorted(duplicate_observations)}"
            )
        if not fragment_ids and not observation_ids:
            raise ValueError("a functional RIU cannot be empty")
        seen_fragments.update(fragment_ids)
        seen_observations.update(observation_ids)
        normalized.append({
            "riu_id": f"riu-{index:02d}",
            "functional_boundary": str(raw.get("functional_boundary") or "").strip(),
            "rationale": str(raw.get("rationale") or "").strip(),
            "coverage": list(raw.get("coverage") or []),
            "fragment_ids": fragment_ids,
            "observation_ids": observation_ids,
        })

    ordered_fragments = [
        fragment_id
        for riu in normalized
        for fragment_id in riu["fragment_ids"]
    ]
    atom_by_fragment: dict[str, str] = {}
    atoms = []
    for index, fragment_id in enumerate(ordered_fragments, 1):
        row = by_fragment[fragment_id]
        atom_id = f"atom-{index:02d}"
        atom_by_fragment[fragment_id] = atom_id
        atoms.append({
            "id": atom_id,
            "path": row["path"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "text": row["text"],
            "sha256": row["sha256"],
            "pool_fragment_id": fragment_id,
            "kind": row["kind"],
            "name": row["name"],
        })

    selected_observations = [
        by_observation[observation_id]
        for riu in normalized
        for observation_id in riu["observation_ids"]
    ]
    allocation = []
    for index, riu in enumerate(normalized, 1):
        allocation.append({
            "riu_id": riu["riu_id"],
            "role_name": f"Reviewer {index}",
            "code_atom_ids": [atom_by_fragment[item] for item in riu["fragment_ids"]],
            "observation_ids": list(riu["observation_ids"]),
        })

    return {
        "schema": "deepswe_functional_riu_candidate_v1",
        "task_id": pool["task_id"],
        "status": "functional_riu_selection",
        "instruction": pool["instruction"],
        "image_id": pool["image_id"],
        "repo_digests": [],
        "code_atoms": atoms,
        "observations": selected_observations,
        "allocation_candidates": allocation,
        "unallocated_background_observation_candidates": [],
        "allocation_notes": "constructor-defined functional boundaries",
        "constructor_metadata": {
            "rius": normalized,
            "uncovered": list(uncovered),
        },
        "audits": {
            "grounding": "pool_verified",
            "leakage": "pending",
            "boundary": "structurally_disjoint",
            "single_hidden_test_qualification": "not_run",
            "necessity_loo": "deferred",
        },
    }
