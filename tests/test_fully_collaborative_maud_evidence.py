from __future__ import annotations

from teamgym.data.legal.maud.evidence import (
    match_evidence_block,
    split_maud_row_text,
)


def test_maud_evidence_strips_page_citation_and_matches_offsets() -> None:
    contract_text = (
        "The Company Board may change its recommendation after consultation with outside counsel "
        "if failing to do so would be inconsistent with fiduciary duties."
    )
    blocks, dropped = split_maud_row_text(contract_text + " (Pages 63-64)")
    assert blocks == [contract_text]
    assert dropped == []

    match = match_evidence_block(blocks[0], contract_text)
    assert match["matched_text"] == contract_text
    assert match["match_status"] == "exact_raw"
    assert match["agreement_start"] == 0
    assert match["agreement_end"] == len(contract_text)


def test_maud_split_drops_dangling_fragments_but_keeps_short_full_row() -> None:
    blocks, dropped = split_maud_row_text(
        "The Company must give Parent notice before changing its recommendation. "
        "<omitted> from the date of such notice shall be required) ("
    )

    assert blocks == ["The Company must give Parent notice before changing its recommendation."]
    assert dropped == ["from the date of such notice shall be required) ("]

    short_blocks, short_dropped = split_maud_row_text("All Cash")
    assert short_blocks == ["All Cash"]
    assert short_dropped == []


def test_maud_split_treats_double_bracket_omitted_as_boundary() -> None:
    blocks, dropped = split_maud_row_text(
        "The Company shall provide Parent with four business days of prior written notice. "
        "[[omitted]] "
        "The Company shall negotiate in good faith during the matching-right period."
    )

    assert blocks == [
        "The Company shall provide Parent with four business days of prior written notice.",
        "The Company shall negotiate in good faith during the matching-right period.",
    ]
    assert dropped == []


def test_maud_evidence_aggressive_normalization_recovers_offsets() -> None:
    contract_text = (
        "The Company may terminate the merger agreement if Parent breaches Section 8.1(b)(iii) "
        "and the breach remains uncured for thirty days after written notice."
    )
    row_text = (
        "The Company may terminate the merger agreement if Parent breaches Section 8.1 ( b ) ( iii ) "
        "and the breach remains uncured for thirty days after written notice."
    )

    match = match_evidence_block(row_text, contract_text)
    assert match["matched_text"] == contract_text
    assert match["match_status"] == "exact_aggressive"
    assert match["agreement_start"] == 0
    assert match["agreement_end"] == len(contract_text)
