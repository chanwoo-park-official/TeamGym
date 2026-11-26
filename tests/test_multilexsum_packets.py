from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from teamgym.data.legal.multilexsum import generation, grounding, packets, short_answer
from teamgym.core.schema import validate_common_datapoint


@pytest.fixture
def source_packet() -> tuple[dict, dict]:
    documents = {}
    for index in range(3):
        document_id = f"CASE-0001-{index:04d}"
        documents[document_id] = {
            "doc_id": document_id,
            "doc_type": "Order",
            "doc_title": f"Procedural order {index}",
            "doc_text": " ".join(
                f"Document {index} sentence {sentence} records an order governing relief and subsequent proceedings."
                for sentence in range(50)
            ),
        }
    case = {"case_id": "CASE-0001", "case_documents": list(documents), "summary/long": "SECRET SUMMARY ANSWER"}
    packet = packets.build_packet(case, documents, chunk_words=40, max_chunks_per_document=12)
    return packet, documents


def build_task(packet: dict) -> dict:
    source = generation.prepare_source(packet)
    source_ids = [packet["reference_information_units"][index]["rui_id"] for index in (0, 2, 4)]
    atoms = []
    for index in range(8):
        source_id = source_ids[index % 3]
        sentence_id = next(sid for sid, value in source["sentence_lookup"].items() if value["rui_id"] == source_id)
        atoms.append({
            "atom_id": f"atom_{index:02d}", "source_rui_id": source_id, "sentence_ids": [sentence_id],
            "proposition": f"The original document records controlling procedural fact number {index} relevant to the final legal conclusion.",
            "inferential_role": "This premise distinguishes the controlling consequence from a competing legal disposition.",
        })
    outcome = "The court maintained interim relief while reserving a final determination pending further development of the relevant proceedings and record."
    generated = generation.normalize_generated({
        "suitable": True,
        "question": "This anonymized collection requires a procedural conclusion about the relationship between the requested relief, the court orders, and the subsequent legal proceedings.",
        "correct_outcome": outcome,
        "canonical_rationale": "The original court records connect the earlier order to the later proceedings and explain the controlling consequence for the requested relief.",
        "atoms": atoms,
        "distractors": [{"outcome": outcome + f" Alternative {index}.", "distinguished_by_atom_ids": [atom["atom_id"] for atom in atoms]} for index in range(3)],
        "reasoning_steps": [
            {"uses": [atom["atom_id"] for atom in atoms[index:index + 2]],
             "claim": "The court order and the subsequent procedural events jointly establish the controlling legal consequence."}
            for index in range(0, 8, 2)
        ],
    }, source)
    return generation.convert(packet, source, generated, 0, 20260712)


def test_streaming_reader_handles_boundaries_and_original_text(tmp_path: Path) -> None:
    documents = {"case/α": {"doc_text": 'Quoted "value" with } and { inside text.\nNew line.'}, "other": {"doc_text": "Second document."}}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(documents, ensure_ascii=False))
    assert dict(packets.iter_source_documents(path, read_size=1)) == documents
    assert packets.load_documents(path, {"case/α"}) == {"case/α": documents["case/α"]}
    with pytest.raises(ValueError, match="Missing source documents"):
        packets.load_documents(path, {"absent"})


def test_streaming_reader_rejects_truncated_source_json(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text('{"document": {"doc_text": "unfinished')
    with pytest.raises(ValueError, match="Invalid source JSON"):
        list(packets.iter_source_documents(path, read_size=3))


def test_source_window_retains_short_sentences_between_indexed_sentences() -> None:
    text = (
        "The parties submitted their initial motions. No. The court denied the requested relief. "
        "The parties continued to litigate the remaining claims. The case later went to trial."
    )
    source = generation.prepare_source({"reference_information_units": [{"rui_id": "excerpt", "text": text}]})

    window = generation.source_window(["excerpt_s02"], source["sentence_lookup"])

    assert window in text
    assert "motions. No. The court" in window


def test_packets_copy_original_offsets_and_exclude_summary(source_packet: tuple[dict, dict]) -> None:
    packet, documents = source_packet
    assert "SECRET SUMMARY ANSWER" not in json.dumps(packet)
    assert packet["packet_policy"]["summary_usage"] == "eligibility_only"
    for unit in packet["private_context_units"]:
        metadata = unit["metadata"]
        source = documents[metadata["source_doc_id"]]["doc_text"]
        assert unit["text"] == source[metadata["source_char_start"]:metadata["source_char_end"]]
    assert len({unit["source_ref"]["source_doc_id"] for unit in packet["reference_information_units"]}) == 3


def test_expansion_preserves_task_and_ownership_and_keeps_more_than_24_chunks(source_packet: tuple[dict, dict]) -> None:
    packet, _documents = source_packet
    task = build_task(packet)
    task["generation_metadata"].update(qualification_source="earlier_run", participant_leakage_deletion_funnel={"converged": True})
    task["dataset_specific"]["qualification_survivor"] = True
    task_before = copy.deepcopy(task)
    expanded = packets.expand_packet(task, packet)
    assert task == task_before
    assert packets.expand_packet(task, packet) == expanded
    assert validate_common_datapoint(expanded)["valid"]
    for field in ["task_description", "oracle", "decision_set", "decision_option_definitions", "reference_information_units"]:
        assert expanded[field] == task[field]
    used = {unit["metadata"]["source_rui_id"] for unit in task["private_context_units"] if unit.get("hidden_contains_rui_ids")}
    background = [unit for unit in expanded["private_context_units"] if not unit.get("hidden_contains_rui_ids")]
    assert len(background) > 24
    assert {unit["context_unit_id"] for unit in background} == {unit["context_unit_id"] for unit in packet["private_context_units"]} - used
    assert expanded["generation_metadata"]["packet_contexts_require_qualification"] is True
    assert "qualification_survivor" not in expanded["dataset_specific"]
    assert "participant_leakage_deletion_funnel" not in expanded["generation_metadata"]
    assert "qualification_source" not in expanded["generation_metadata"]
    assert expanded["generation_metadata"]["prior_qualification_source"] == "earlier_run"
    for participant in expanded["participants"]:
        assert participant["owned_rui_ids"] == next(p["owned_rui_ids"] for p in task["participants"] if p["participant_id"] == participant["participant_id"])


def test_expansion_rejects_invented_passages_and_unrelated_packets(source_packet: tuple[dict, dict]) -> None:
    packet, _documents = source_packet
    task = build_task(packet)
    window = next(unit for unit in task["private_context_units"] if unit.get("hidden_contains_rui_ids"))
    window["text"] = "A passage invented outside the original source documents."
    with pytest.raises(ValueError, match="not present in its original excerpt"):
        packets.expand_packet(task, packet)
    other = copy.deepcopy(packet)
    other["instance_id"] = "another_collection"
    with pytest.raises(ValueError, match="does not match task"):
        packets.expand_packet(task, other)


def test_expansion_restores_preserved_packets_without_changing_source_text(source_packet: tuple[dict, dict]) -> None:
    packet, _documents = source_packet
    full = packets.expand_packet(build_task(packet), packet)
    thin = copy.deepcopy(full)
    thin["private_context_units"] = [unit for unit in thin["private_context_units"] if unit.get("hidden_contains_rui_ids")]
    restored = packets.expand_packet(thin, full)
    assert {unit["context_unit_id"]: unit["text"] for unit in restored["private_context_units"]} == {unit["context_unit_id"]: unit["text"] for unit in full["private_context_units"]}
    assert restored["oracle"] == full["oracle"]


def audit_for(task: dict) -> dict:
    return {
        "instance_id": task["instance_id"], "fully_grounded": True, "canonical_fully_supported": True,
        "unsupported_rui_ids": [],
        "atom_audits": [{"rui_id": unit["rui_id"], "supported": True} for unit in task["reference_information_units"]],
    }


@pytest.mark.parametrize("corruption", ["missing_piu", "duplicate_piu", "string_boolean", "unsupported_canonical"])
def test_grounding_selection_rejects_incomplete_or_ambiguous_audits(source_packet: tuple[dict, dict], corruption: str) -> None:
    task = build_task(source_packet[0])
    audit = audit_for(task)
    if corruption == "missing_piu":
        audit["atom_audits"].pop()
    elif corruption == "duplicate_piu":
        audit["atom_audits"][-1] = audit["atom_audits"][0]
    elif corruption == "string_boolean":
        audit["atom_audits"][0]["supported"] = "false"
    else:
        audit["canonical_fully_supported"] = False
    assert grounding.select_grounded([task], [audit]) == ([], [task["instance_id"]])


def test_grounding_selection_requires_one_audit_per_task(source_packet: tuple[dict, dict]) -> None:
    task = build_task(source_packet[0])
    audit = audit_for(task)
    assert grounding.select_grounded([task], [audit]) == ([task], [])
    with pytest.raises(ValueError, match="exactly one grounding audit"):
        grounding.select_grounded([task], [])
    with pytest.raises(ValueError, match="unique"):
        grounding.select_grounded([task], [audit, audit])


def test_grounded_selection_records_audit_provenance(
    source_packet: tuple[dict, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = build_task(source_packet[0])
    assert task["dataset_specific"]["original_source_grounded"] is False
    (tmp_path / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "audits.jsonl").write_text(json.dumps(audit_for(task)) + "\n")
    monkeypatch.setattr(sys, "argv", ["select-grounded", "--input-root", str(tmp_path / "tasks.jsonl"),
                                    "--audits", str(tmp_path / "audits.jsonl"), "--output-root", str(tmp_path / "selected")])
    grounding.select_main()
    selected = json.loads((tmp_path / "selected/common_datapoints.jsonl").read_text())
    report = json.loads((tmp_path / "selected/selection_report.json").read_text())
    assert selected["dataset_specific"]["original_source_grounded"] is True
    assert selected["generation_metadata"]["requires_source_grounding_audit"] is False
    assert selected["generation_metadata"]["source_grounding_audits_sha256"] == report["grounding_audits_sha256"]


def test_grounding_auditor_does_not_accept_string_false(
    source_packet: tuple[dict, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = build_task(source_packet[0])
    (tmp_path / "tasks.jsonl").write_text(json.dumps(task) + "\n")

    class AuditClient:
        def __init__(self, _config: object) -> None:
            pass

        def generate_json(self, **_kwargs: object) -> dict:
            return {"atom_audits": audit_for(task)["atom_audits"], "canonical_fully_supported": "false"}

    monkeypatch.setattr(grounding, "OpenAIJsonClient", AuditClient)
    monkeypatch.setattr(sys, "argv", ["audit-grounding", "--input-root", str(tmp_path / "tasks.jsonl"),
                                    "--output-root", str(tmp_path / "audit")])
    grounding.main()
    audit = json.loads((tmp_path / "audit/grounding_audits.jsonl").read_text())
    assert audit["fully_grounded"] is False
    assert audit["canonical_fully_supported"] is False


def test_prepared_source_to_expanded_short_answer_without_model_calls(
    source_packet: tuple[dict, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _packet, documents = source_packet
    raw = tmp_path / "raw"
    raw.mkdir()
    case = {"case_id": "CASE-0001", "case_documents": list(documents), "summary/long": "eligibility " * 300}
    for split in ["train", "dev", "test"]:
        (raw / f"{split}.json").write_text(json.dumps(case) + "\n" if split == "train" else "")
    (raw / "sources.json").write_text(json.dumps(documents))
    monkeypatch.setattr(sys, "argv", ["prepare-packets", "--input-root", str(raw), "--output-root", str(tmp_path / "prepared"), "--limit", "1"])
    packets.prepare_main()
    prepared = json.loads((tmp_path / "prepared/source_packets.jsonl").read_text())
    task = build_task(prepared)

    class RewriteClient:
        def generate_json(self, **_kwargs: object) -> dict:
            return {
                "question": "Determine how the earlier procedural order and the subsequent proceedings affect the requested relief and explain the controlling legal conclusions from the original documents.",
                "required_conclusions": [
                    {"conclusion": "The original order remains effective while the subsequent proceedings continue before the court.", "why_required": "This establishes the procedural status controlling the requested relief."},
                    {"conclusion": "The later proceedings require a further determination before final relief can be ordered.", "why_required": "This explains the remaining condition governing the final disposition."},
                ],
            }

    rewritten = short_answer.convert(task, RewriteClient(), 0)
    expanded = packets.expand_packet(rewritten, prepared)
    assert expanded["decision_set"] == ["free_response"]
    assert expanded["source"]["collection_id"] == case["case_id"]
    assert validate_common_datapoint(expanded)["valid"]
    assert grounding.select_grounded([expanded], [audit_for(expanded)])[0] == [expanded]
