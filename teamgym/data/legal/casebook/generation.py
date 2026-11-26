#!/usr/bin/env python
"""Create a 20-example short-answer judged eCFR casebook seed."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from teamgym.core.io import write_json, write_jsonl


DEFAULT_OUTPUT_ROOT = "generated/long_context_reasoning_short_answer_casebook20_20260619"
HEADER = "CASEBOOK TRANSACTION MEMOS FOR 20-EXAMPLE SHORT-ANSWER TASKS"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    raw_dir = output_root / "raw"
    seed_dir = output_root / "seed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)

    base_source = load_base_rules(args.base_source)
    rules = build_rule_rius(base_source)
    rows, memo_texts = build_rows(base_source=base_source, rules=rules)

    source_path = raw_dir / "ecfr_title12_parts1_3_casebook20_source.txt"
    source_text = base_source.rstrip() + "\n\n" + HEADER + "\n\n" + "\n\n".join(memo_texts) + "\n"
    source_path.write_text(source_text, encoding="utf-8")
    source_path_for_seed = path_for_seed(source_path)

    for row in rows:
        memo = row.pop("_memo_text")
        row["source_path"] = source_path_for_seed
        row["source_text"] = base_source.rstrip() + "\n\n" + HEADER + "\n\n" + memo + "\n"

    seed_path = seed_dir / "ecfr_casebook20_short_answer_seed.jsonl"
    write_jsonl(seed_path, rows)
    report = {
        "schema": "long_context_short_answer_casebook20_seed_report_v1",
        "count": len(rows),
        "base_source": str(args.base_source),
        "source_path": str(source_path),
        "seed_path": str(seed_path),
        "instance_ids": [row["instance_id"] for row in rows],
    }
    write_json(output_root / "seed_build_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_base_rules(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    marker = "CASEBOOK TRANSACTION MEMOS"
    if marker in text:
        return text.split(marker, 1)[0].rstrip()
    return text.rstrip()


def build_rule_rius(base: str) -> dict[str, dict[str, Any]]:
    return {
        "type_i_def": rule("rui_type_i_definition", span(base, "(j) Type I security means:", "(k) Type II security means"), "Type I includes U.S. obligations, full-faith-credit agency obligations, certain loan-pool obligations, and general obligations.", "definition"),
        "type_i_limit": rule("rui_type_i_no_percentage_limit", span(base, "(a) Type I securities.", "(b) Type II securities."), "Type I securities are not limited to a specified percentage of capital and surplus.", "limit_rule"),
        "type_ii_def": rule("rui_type_ii_definition", span(base, "(k) Type II security means", "(l) Type III security means"), "Type II includes qualifying housing, university, international development bank, and other listed obligations.", "definition"),
        "type_ii_limit": rule("rui_type_ii_10pct_limit", span(base, "(b) Type II securities.", "(c) Type III securities."), "Type II securities from one obligor may not exceed 10 percent of capital and surplus, counting commitments.", "threshold_rule"),
        "type_iii_def": rule("rui_type_iii_definition", span(base, "(l) Type III security means", "(m) Type IV security means"), "Type III is an investment security that does not qualify as Type I, II, IV, or V.", "definition"),
        "type_iii_limit": rule("rui_type_iii_10pct_limit", span(base, "(c) Type III securities.", "(d) Type II and III securities; other investment securities limitations."), "Type III securities from one obligor may not exceed 10 percent of capital and surplus, counting commitments.", "threshold_rule"),
        "type_ii_iii_exception": rule("rui_type_ii_iii_separate_issue_exception", span(base, "(d) Type II and III securities; other investment securities limitations.", "(e) Type IV securities."), "The 10 percent limit may be applied separately to qualifying real-estate lease financing issues.", "exception"),
        "type_iv_def": rule("rui_type_iv_definition", span(base, "(m) Type IV security means:", "(n) Type V security means"), "Type IV includes certain small business-related, commercial mortgage-related, and residential mortgage-related securities.", "definition"),
        "type_iv_limit": rule("rui_type_iv_no_percentage_limit", span(base, "(e) Type IV securities.", "(f) Type V securities."), "Type IV securities are not limited to a specified percentage of capital and surplus.", "limit_rule"),
        "type_v_def": rule("rui_type_v_definition", span(base, "(n) Type V security means", "[61 FR 63982, Dec. 2, 1996, as amended at 66 FR 34791"), "Type V requires investment grade, marketability, not Type IV, and qualifying loan-pool security features.", "definition"),
        "type_v_limit": rule("rui_type_v_25pct_limit", span(base, "(f) Type V securities.", "(g) Securitization."), "Type V securities from one issuer may not exceed 25 percent of capital and surplus, counting commitments.", "threshold_rule"),
        "type_iii_v_aggregation": rule("rui_type_iii_v_aggregation", span(base, "(d) Calculation of Type III and Type V securities holdings", "(e) Limit on investment company holdings"), "Type III and Type V holdings aggregate obligations related through common control or credit enhanced by the same entity.", "aggregation_rule"),
        "estimate_treatment": rule("rui_estimate_treatment", span(base, "(i) Securities held based on estimates of obligor's performance.", "(2) The aggregate par value of securities treated"), "A marketable debt security may be treated as an investment security based on reliable estimates of obligor performance.", "rule"),
        "estimate_limit": rule("rui_estimate_5pct_limit", span(base, "(2) The aggregate par value of securities treated", "[61 FR 63982, Dec. 2, 1996, as amended at 64 FR 60098"), "Estimate-based investment securities may not exceed 5 percent of capital and surplus.", "threshold_rule"),
        "credit_records": rule("rui_credit_records", span(base, "§ 1.5 Safe and sound banking practices; credit information required.", "§ 1.6 Convertible securities."), "The bank must determine adequate evidence or reasonable belief and maintain retrievable records.", "recordkeeping_rule"),
        "dpc_acquisition": rule("rui_dpc_acquisition_rule", span(base, "(a) Securities held in satisfaction of debts previously contracted.", "(b) Holding period."), "DPC restrictions do not apply to securities acquired through foreclosure, compromise, or to avoid loss on a debt previously contracted.", "exception"),
        "dpc_holding": rule("rui_dpc_holding_period", span(base, "(b) Holding period.", "(c) Accounting treatment."), "DPC securities may be held up to five years unless the OCC extends the period.", "time_limit"),
        "dpc_purpose": rule("rui_dpc_non_speculative", span(base, "(d) Non-speculative purpose.", "§ 1.8 Nonconforming investments."), "DPC securities may not be held for speculative purposes.", "purpose_rule"),
        "nonconforming_status": rule("rui_nonconforming_status", span(base, "(a) A national bank's investment in securities that no longer conform", "(b) A bank shall exercise reasonable efforts"), "A security that conformed when made but later fails for listed reasons is nonconforming rather than a violation.", "status_rule"),
        "nonconforming_efforts": rule("rui_nonconforming_efforts", span(base, "(b) A bank shall exercise reasonable efforts", "§ 1.100 Indirect general obligations."), "A bank must use reasonable efforts to bring a nonconforming investment into conformity unless unsafe or unsound.", "duty_rule"),
        "convertible": rule("rui_convertible_prohibition", span(base, "§ 1.6 Convertible securities.", "§ 1.7 Securities held"), "A national bank may not purchase securities convertible into stock at the option of the issuer.", "prohibition"),
        "pooled_investment": rule("rui_pooled_investment_company_rule", span(base, "(h) Pooled investments", "(2) Other issuers."), "Investment company shares are allowed only if the portfolio consists of permissible assets and holdings comply with section 1.4(e).", "rule"),
        "investment_company_limit": rule("rui_investment_company_limit", span(base, "(e) Limit on investment company holdings", "§ 1.5 Safe and sound banking practices"), "A bank generally combines its pro rata share of portfolio securities with direct holdings for applicable limits.", "limit_rule"),
        "prerefunded_type_i": rule("rui_prerefunded_type_i", span(base, "§ 1.120 Prerefunded or escrowed bonds", "§ 1.130 Type II securities"), "An obligation secured by an escrow of U.S. or general obligations with sufficient interest earnings qualifies as Type I.", "classification_rule"),
        "university_type_ii": rule("rui_university_type_ii", span(base, "§ 1.130 Type II securities; guidelines", "(2) An obligation that finances the construction"), "A qualifying obligation for university facilities or study loans may be Type II.", "classification_rule"),
    }


def build_rows(*, base_source: str, rules: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    cases = [
        type_v_case("TV-26", "type_v_committed_exceeds", 18, 9, 1, False, rules),
        type_v_case("TV-18", "type_v_committed_within", 12, 8, 2, True, rules),
        type_iii_case("T3-AGG", "type_iii_common_control_exceeds", 6, 5, True, False, rules),
        type_iii_case("T3-LIME", "type_iii_same_obligor_within", 4, 3, False, True, rules),
        type_ii_case("T2-OVER", "type_ii_same_obligor_exceeds", 7, 4, False, rules),
        type_ii_exception_case("RE-LEASE", "type_ii_separate_issue_exception_applies", 8, 7, True, rules),
        type_ii_exception_case("RE-MIXED", "type_ii_separate_issue_exception_fails", 6, 6, False, rules),
        estimate_case("EST-45", "estimate_based_approve", 3.4, 1.1, True, True, True, rules),
        estimate_case("EST-CAP", "estimate_based_cap_exceeds", 4.6, 0.8, True, True, False, rules),
        estimate_case("EST-FILE", "estimate_based_records_missing", 2.2, 1.4, True, False, False, rules),
        dpc_case("DPC-57", "dpc_within_non_speculative", "four years and nine months", False, False, True, rules),
        dpc_case("DPC-LATE", "dpc_past_five_no_extension", "five years and four months", False, False, False, rules),
        dpc_case("DPC-SPEC", "dpc_speculative_purpose", "three years and two months", False, True, False, rules),
        nonconforming_case("NC-CTRL", "nonconforming_common_control_efforts_required", "two issuers became related directly or indirectly through common control after the investment was made", False, False, rules),
        nonconforming_case("NC-MERGE", "nonconforming_merger_safe_sound_exception", "two obligors merged after the investment was made", True, True, rules),
        type_i_case("T1-GO", "type_i_no_percentage_limit", 64, rules),
        type_iv_case("T4-SB", "type_iv_no_percentage_limit", 38, rules),
        convertible_case("CONV-ISS", "convertible_issuer_option_prohibited", rules),
        investment_company_case("FUND-LOOK", "investment_company_lookthrough_exceeds", 6, 5, rules),
        prerefunded_case("ESCROW-I", "prerefunded_type_i_no_percentage_limit", 42, rules),
    ]
    rows = [make_row(base_source=base_source, **case) for case in cases]
    return rows, [case["_memo_text"] for case in cases]


def type_v_case(memo_id: str, slug: str, current: int, purchase: int, sale: int, approve: bool, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = current + purchase - sale
    facts = [
        f("characteristics", "The proposed tranche is investment grade, marketable, not a Type IV security, and fully secured by interests in a pool of loans to numerous obligors in which a national bank could invest directly.", "The proposed tranche satisfies the Type V characteristics."),
        f("current", f"The bank currently holds Type V securities from Issuer {memo_id.replace('-', '')} with aggregate par value equal to {current} percent of capital and surplus.", f"Current Type V holdings equal {current} percent."),
        f("purchase", f"The bank is legally committed to purchase additional Type V securities from the same issuer equal to {purchase} percent of capital and surplus.", f"The committed Type V purchase adds {purchase} percent."),
        f("sale", f"The bank is legally committed to sell Type V securities from the same issuer equal to {sale} percent of capital and surplus.", f"The committed Type V sale subtracts {sale} percent."),
    ]
    conclusion = (
        f"The proposed purchase may proceed under the Type V issuer cap because {current} + {purchase} - {sale} = {total} percent, not above 25 percent."
        if approve
        else f"The proposed purchase should not proceed under the Type V issuer cap because {current} + {purchase} - {sale} = {total} percent, above 25 percent."
    )
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether the proposed purchase can proceed under the Type V issuer concentration rule.",
        question=f"For Memo {memo_id}, write the Type V compliance conclusion.",
        rules=[rules["type_v_def"], rules["type_v_limit"]],
        facts=facts,
        steps=[
            step(1, ["rui_type_v_definition", rid(memo_id, "characteristics")], "Classify the proposed tranche as Type V."),
            step(2, ["rui_type_v_25pct_limit", rid(memo_id, "current"), rid(memo_id, "purchase"), rid(memo_id, "sale")], f"Calculate the issuer amount as {current} + {purchase} - {sale} = {total} percent."),
            step(3, ["rui_type_v_25pct_limit"], f"Compare {total} percent to the 25 percent Type V issuer cap."),
        ],
        conclusion=conclusion,
    )


def type_iii_case(memo_id: str, slug: str, current: int, purchase: int, common_control: bool, approve: bool, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = current + purchase
    relation = "The two obligors are related directly or indirectly through common control." if common_control else "The existing holding and proposed purchase are issued by the same obligor."
    conclusion = (
        f"The purchase may proceed because the Type III amount is {total} percent, not above the 10 percent limit."
        if approve
        else f"The purchase should not proceed because the Type III amount is {total} percent, above the 10 percent limit."
    )
    facts = [
        f("classification", "The proposed corporate bond is marketable, investment grade, not predominately speculative, and does not qualify as Type I, Type II, Type IV, or Type V.", "The proposed bond is Type III."),
        f("current", f"The bank currently holds Type III securities from the relevant obligor group equal to {current} percent of capital and surplus.", f"Current Type III holdings equal {current} percent."),
        f("purchase", f"The proposed Type III purchase would add {purchase} percent of capital and surplus.", f"The proposed Type III purchase adds {purchase} percent."),
        f("relation", relation, "The exposures must be treated together for the limit calculation."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether the Type III purchase can proceed.",
        question=f"For Memo {memo_id}, write the Type III limit conclusion.",
        rules=[rules["type_iii_def"], rules["type_iii_limit"], rules["type_iii_v_aggregation"]],
        facts=facts,
        steps=[
            step(1, ["rui_type_iii_definition", rid(memo_id, "classification")], "Classify the proposed bond as Type III."),
            step(2, ["rui_type_iii_v_aggregation", rid(memo_id, "relation")], "Determine which obligor exposures must be counted together."),
            step(3, ["rui_type_iii_10pct_limit", rid(memo_id, "current"), rid(memo_id, "purchase")], f"Calculate {current} + {purchase} = {total} percent and compare it to the 10 percent Type III limit."),
        ],
        conclusion=conclusion,
    )


def type_ii_case(memo_id: str, slug: str, current: int, purchase: int, approve: bool, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = current + purchase
    facts = [
        f("classification", "The obligation is issued by a State agency for university housing purposes and otherwise qualifies as an investment security.", "The proposed obligation is Type II."),
        f("current", f"The bank currently holds Type II securities from the same obligor equal to {current} percent of capital and surplus.", f"Current Type II holdings equal {current} percent."),
        f("purchase", f"The proposed Type II purchase would add {purchase} percent of capital and surplus.", f"The proposed purchase adds {purchase} percent."),
    ]
    conclusion = (
        f"The purchase may proceed because {current} + {purchase} = {total} percent, not above the 10 percent Type II obligor limit."
        if approve
        else f"The purchase should not proceed because {current} + {purchase} = {total} percent, above the 10 percent Type II obligor limit."
    )
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether the Type II purchase can proceed.",
        question=f"For Memo {memo_id}, write the Type II limit conclusion.",
        rules=[rules["type_ii_def"], rules["type_ii_limit"]],
        facts=facts,
        steps=[
            step(1, ["rui_type_ii_definition", rid(memo_id, "classification")], "Classify the proposed obligation as Type II."),
            step(2, ["rui_type_ii_10pct_limit", rid(memo_id, "current"), rid(memo_id, "purchase")], f"Calculate {current} + {purchase} = {total} percent."),
            step(3, ["rui_type_ii_10pct_limit"], f"Compare {total} percent to the 10 percent Type II same-obligor limit."),
        ],
        conclusion=conclusion,
    )


def type_ii_exception_case(memo_id: str, slug: str, alpha: int, beta: int, qualifies: bool, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tenant_sentence = "The proceeds of each issue will be used to acquire and lease real estate and related facilities to economically and legally separate industrial tenants." if qualifies else "The proceeds of the two issues will be used for related facilities leased to the same industrial tenant."
    lien_sentence = "Each issue is payable solely from and secured by a first lien on revenues to be derived from rentals paid under net noncancellable leases." if qualifies else "Only one issue is secured by a first lien on revenues to be derived from rentals paid under a net noncancellable lease."
    conclusion = (
        f"The 10 percent limit may be applied separately because the lease-financing exception conditions are met; Issue Alpha is {alpha} percent and Issue Beta is {beta} percent."
        if qualifies
        else f"The 10 percent limit may not be applied separately because the lease-financing exception conditions are not met; the combined {alpha + beta} percent exceeds 10 percent."
    )
    facts = [
        f("same_obligor", "The municipal desk is reviewing two Type II securities issued by the same obligor.", "The two securities are Type II and share one obligor."),
        f("amounts", f"Issue Alpha would equal {alpha} percent of the bank's capital and surplus. Issue Beta would equal {beta} percent of the bank's capital and surplus.", f"The issue amounts are {alpha} percent and {beta} percent."),
        f("tenants", tenant_sentence, "The tenant-use condition is stated for the exception."),
        f("lien", lien_sentence, "The first-lien net lease condition is stated for the exception."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether the 10 percent investment limitation may be applied separately to each issue.",
        question=f"For Memo {memo_id}, write the separate-issue limit conclusion.",
        rules=[rules["type_ii_limit"], rules["type_ii_iii_exception"]],
        facts=facts,
        steps=[
            step(1, ["rui_type_ii_10pct_limit", rid(memo_id, "same_obligor"), rid(memo_id, "amounts")], "Identify the same-obligor Type II limit issue."),
            step(2, ["rui_type_ii_iii_separate_issue_exception", rid(memo_id, "tenants"), rid(memo_id, "lien")], "Apply the lease-financing separate-issue exception conditions."),
        ],
        conclusion=conclusion,
    )


def estimate_case(memo_id: str, slug: str, existing: float, add: float, reliable: bool, records: bool, approve: bool, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = round(existing + add, 1)
    reliable_sentence = "The credit group reasonably believes, based on reliable estimates, that the obligor will be able to satisfy its obligations under the security." if reliable else "The credit group does not have reliable estimates supporting that the obligor will be able to satisfy its obligations under the security."
    records_sentence = "The credit file contains retrievable evidence supporting the estimate-based conclusion." if records else "The credit file does not contain retrievable evidence supporting the estimate-based conclusion."
    if approve:
        conclusion = f"The estimate-based treatment may be used because the qualitative conditions and records are satisfied and {existing} + {add} = {total} percent, within the 5 percent cap."
    elif total > 5:
        conclusion = f"The estimate-based treatment should not be used because {existing} + {add} = {total} percent, above the 5 percent cap."
    else:
        conclusion = "The estimate-based treatment should not be used on this record because the required retrievable credit evidence or reliable-estimate support is missing."
    facts = [
        f("marketable", "The security is marketable.", "The security is marketable."),
        f("reliable", reliable_sentence, "The reliable-estimates condition is addressed."),
        f("amounts", f"Before the proposed purchase, the aggregate par value of securities treated under the reliable-estimates provision equals {existing} percent of the bank's capital and surplus. The proposed purchase would add {add} percent of capital and surplus.", f"The aggregate after purchase would be {total} percent."),
        f("records", records_sentence, "The recordkeeping condition is addressed."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether the proposed estimate-based treatment can be used.",
        question=f"For Memo {memo_id}, write the estimate-based treatment conclusion.",
        rules=[rules["estimate_treatment"], rules["estimate_limit"], rules["credit_records"]],
        facts=facts,
        steps=[
            step(1, ["rui_estimate_treatment", rid(memo_id, "marketable"), rid(memo_id, "reliable")], "Apply the marketability and reliable-estimates treatment conditions."),
            step(2, ["rui_estimate_5pct_limit", rid(memo_id, "amounts")], f"Calculate {existing} + {add} = {total} percent and compare it to the 5 percent cap."),
            step(3, ["rui_credit_records", rid(memo_id, "records")], "Apply the supporting-record condition."),
        ],
        conclusion=conclusion,
    )


def dpc_case(memo_id: str, slug: str, age: str, extension: bool, speculative: bool, approve: bool, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ext_sentence = "The OCC has granted an extension of the holding period." if extension else "The OCC has not granted an extension of the holding period."
    purpose_sentence = "The bank is holding the securities for speculation." if speculative else "The bank is holding the securities for workout and collection purposes, not for speculation."
    if approve:
        conclusion = "The bank may continue holding the securities for now because the DPC acquisition fits the exception, the five-year period has not expired, and the purpose is non-speculative."
    elif "five years" in age and not extension:
        conclusion = "The bank may not continue holding the securities without an OCC extension because the five-year DPC holding period has expired."
    else:
        conclusion = "The bank may not continue holding the securities because DPC securities may not be held for speculative purposes."
    facts = [
        f("acquired", "The workout group acquired securities through foreclosure on collateral to avoid loss in connection with a debt previously contracted.", "The securities were acquired through foreclosure to avoid loss on a DPC."),
        f("age", f"Ownership of the securities was originally transferred to the bank {age} ago.", f"The holding period age is {age}."),
        f("extension", ext_sentence, "The OCC extension status is stated."),
        f("purpose", purpose_sentence, "The holding purpose is stated."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The workout group asks whether it may continue holding the securities.",
        question=f"For Memo {memo_id}, write the DPC holding conclusion.",
        rules=[rules["dpc_acquisition"], rules["dpc_holding"], rules["dpc_purpose"]],
        facts=facts,
        steps=[
            step(1, ["rui_dpc_acquisition_rule", rid(memo_id, "acquired")], "Determine whether the acquisition fits the DPC provision."),
            step(2, ["rui_dpc_holding_period", rid(memo_id, "age"), rid(memo_id, "extension")], "Apply the five-year holding period and OCC extension status."),
            step(3, ["rui_dpc_non_speculative", rid(memo_id, "purpose")], "Apply the non-speculative purpose condition."),
        ],
        conclusion=conclusion,
    )


def nonconforming_case(memo_id: str, slug: str, reason: str, safe_exception: bool, approve_no_efforts: bool, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    safe_sentence = "Bringing the investment into conformity would be inconsistent with safe and sound banking practices." if safe_exception else "Bringing the investment into conformity would not be inconsistent with safe and sound banking practices."
    conclusion = (
        "The bank may treat the investment as nonconforming and is not required to take conformity efforts now because doing so would be inconsistent with safe and sound banking practices."
        if approve_no_efforts
        else "The investment is nonconforming rather than an immediate violation, but the bank may not simply keep it without reasonable conformity efforts."
    )
    facts = [
        f("when_made", "The security conformed to Part 1 when the bank purchased it.", "The security conformed when purchased."),
        f("reason", f"The security no longer conforms because {reason}.", "The nonconformity resulted from a listed post-purchase event."),
        f("efforts", "The bank has made no reasonable efforts to bring the investment into conformity.", "The bank has made no reasonable conformity efforts."),
        f("safe", safe_sentence, "The safe-and-sound exception status is stated."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether the bank may keep the investment without taking reasonable conformity efforts.",
        question=f"For Memo {memo_id}, write the nonconforming-investment conclusion.",
        rules=[rules["nonconforming_status"], rules["nonconforming_efforts"]],
        facts=facts,
        steps=[
            step(1, ["rui_nonconforming_status", rid(memo_id, "when_made"), rid(memo_id, "reason")], "Determine that the investment is nonconforming rather than an immediate violation."),
            step(2, ["rui_nonconforming_efforts", rid(memo_id, "efforts"), rid(memo_id, "safe")], "Determine whether reasonable conformity efforts are required."),
        ],
        conclusion=conclusion,
    )


def type_i_case(memo_id: str, slug: str, exposure: int, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    facts = [
        f("classification", "The obligation is a general obligation of a State supported by the full faith and credit of an obligor possessing general powers of taxation.", "The obligation fits a Type I category."),
        f("exposure", f"The bank's holdings from the issuer would equal {exposure} percent of capital and surplus after the proposed purchase.", f"Holdings would equal {exposure} percent of capital and surplus."),
        f("risk", "The bank has completed its safe-and-sound risk review for the proposed activity.", "The safe-and-sound review has been completed."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether a percentage-of-capital limit bars the purchase.",
        question=f"For Memo {memo_id}, write the Type I percentage-limit conclusion.",
        rules=[rules["type_i_def"], rules["type_i_limit"], rules["credit_records"]],
        facts=facts,
        steps=[
            step(1, ["rui_type_i_definition", rid(memo_id, "classification")], "Classify the obligation as Type I."),
            step(2, ["rui_type_i_no_percentage_limit", rid(memo_id, "exposure")], "Apply the Type I rule that has no specified percentage limit."),
            step(3, ["rui_credit_records", rid(memo_id, "risk")], "Confirm the general safe-and-sound review condition is addressed."),
        ],
        conclusion=f"No specified percentage-of-capital limit bars the purchase because the obligation is Type I, even though exposure would be {exposure} percent.",
    )


def type_iv_case(memo_id: str, slug: str, exposure: int, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    facts = [
        f("classification", "The security is a small business-related security fully secured by interests in a pool of loans to numerous obligors.", "The security fits the Type IV definition."),
        f("exposure", f"The bank's Type IV holdings after the purchase would equal {exposure} percent of capital and surplus.", f"Type IV holdings would equal {exposure} percent."),
        f("records", "The credit file contains retrievable evidence for the bank's risk review.", "The risk-review records are retrievable."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether a percentage-of-capital limit bars the Type IV purchase.",
        question=f"For Memo {memo_id}, write the Type IV percentage-limit conclusion.",
        rules=[rules["type_iv_def"], rules["type_iv_limit"], rules["credit_records"]],
        facts=facts,
        steps=[
            step(1, ["rui_type_iv_definition", rid(memo_id, "classification")], "Classify the security as Type IV."),
            step(2, ["rui_type_iv_no_percentage_limit", rid(memo_id, "exposure")], "Apply the Type IV no-specified-percentage-limit rule."),
            step(3, ["rui_credit_records", rid(memo_id, "records")], "Confirm the supporting risk-review record is present."),
        ],
        conclusion=f"No specified percentage-of-capital limit bars the Type IV purchase, even though Type IV holdings would equal {exposure} percent.",
    )


def convertible_case(memo_id: str, slug: str, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    facts = [
        f("convertible", "The proposed security is convertible into stock at the option of the issuer.", "The proposed security is issuer-option convertible."),
        f("marketable", "The proposed security is marketable and investment grade.", "The security otherwise has marketable and investment-grade features."),
        f("records", "The desk has retrievable credit information for the obligor.", "The desk has retrievable credit information."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether it may purchase the security.",
        question=f"For Memo {memo_id}, write the convertible-security conclusion.",
        rules=[rules["convertible"], rules["credit_records"]],
        facts=facts,
        steps=[
            step(1, ["rui_convertible_prohibition", rid(memo_id, "convertible")], "Apply the prohibition on securities convertible into stock at the issuer's option."),
            step(2, ["rui_credit_records", rid(memo_id, "marketable"), rid(memo_id, "records")], "Recognize that other credit or marketability facts do not override the convertible-security prohibition."),
        ],
        conclusion="The bank may not purchase the security because it is convertible into stock at the option of the issuer.",
    )


def investment_company_case(memo_id: str, slug: str, direct: int, pro_rata: int, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = direct + pro_rata
    facts = [
        f("portfolio", "The investment company portfolio consists exclusively of assets that a national bank may purchase and sell for its own account.", "The portfolio consists only of bank-permissible assets."),
        f("direct", f"The bank directly holds Type III securities from Issuer Lark equal to {direct} percent of capital and surplus.", f"Direct Type III holdings are {direct} percent."),
        f("prorata", f"The bank's pro rata share of the same Type III security in the investment company's portfolio equals {pro_rata} percent of capital and surplus.", f"The pro rata investment company share is {pro_rata} percent."),
        f("not_diversified", "The bank has not elected the alternate diversified investment company limit.", "The alternate diversified limit is not used."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether the investment company share purchase may proceed.",
        question=f"For Memo {memo_id}, write the investment-company look-through conclusion.",
        rules=[rules["pooled_investment"], rules["investment_company_limit"], rules["type_iii_limit"]],
        facts=facts,
        steps=[
            step(1, ["rui_pooled_investment_company_rule", rid(memo_id, "portfolio")], "Confirm the investment company portfolio consists of bank-permissible assets."),
            step(2, ["rui_investment_company_limit", rid(memo_id, "direct"), rid(memo_id, "prorata"), rid(memo_id, "not_diversified")], f"Combine direct holdings and pro rata portfolio exposure as {direct} + {pro_rata} = {total} percent."),
            step(3, ["rui_type_iii_10pct_limit"], f"Compare {total} percent to the 10 percent Type III limit."),
        ],
        conclusion=f"The investment company share purchase should not proceed because look-through exposure is {total} percent, above the 10 percent Type III limit.",
    )


def prerefunded_case(memo_id: str, slug: str, exposure: int, rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    facts = [
        f("escrow", "The refunding obligation is secured by an escrow fund consisting of obligations of the United States.", "The escrow consists of U.S. obligations."),
        f("earnings", "The escrowed obligations produce interest earnings sufficient for the full and timely payment of interest on, and principal of, the obligation.", "The escrow interest is sufficient for full and timely payment."),
        f("exposure", f"The bank's holdings after purchase would equal {exposure} percent of capital and surplus.", f"Holdings would equal {exposure} percent."),
    ]
    return case(
        memo_id=memo_id,
        slug=slug,
        ask="The desk asks whether a percentage-of-capital limit bars the purchase.",
        question=f"For Memo {memo_id}, write the prerefunded Type I conclusion.",
        rules=[rules["prerefunded_type_i"], rules["type_i_limit"]],
        facts=facts,
        steps=[
            step(1, ["rui_prerefunded_type_i", rid(memo_id, "escrow"), rid(memo_id, "earnings")], "Determine that the escrowed obligation qualifies as Type I."),
            step(2, ["rui_type_i_no_percentage_limit", rid(memo_id, "exposure")], "Apply the Type I no-specified-percentage-limit rule."),
        ],
        conclusion=f"No specified percentage-of-capital limit bars the purchase because the prerefunded obligation qualifies as Type I, even though holdings would equal {exposure} percent.",
    )


def case(
    *,
    memo_id: str,
    slug: str,
    ask: str,
    question: str,
    rules: list[dict[str, Any]],
    facts: list[dict[str, str]],
    steps: list[dict[str, Any]],
    conclusion: str,
) -> dict[str, Any]:
    fact_sentences = [fact["source_span"] for fact in facts]
    memo_text = f"Memo {memo_id}. " + " ".join(fact_sentences + [ask])
    memo_key = memo_id.lower().replace("-", "_")
    ruis = [dict(item) for item in rules] + [fact_rui(memo_id, fact) for fact in facts]
    required_ids = [str(rui["rui_id"]) for rui in ruis]
    canonical = " ".join([str(row["claim"]) for row in steps] + [f"Therefore, {conclusion[0].lower() + conclusion[1:]}"])
    return {
        "source_family": "ecfr_casebook20",
        "source_id": "ecfr_title12_parts1_3_casebook20_20260619",
        "source_title": "12 CFR Parts 1 and 3 casebook 20-example short-answer source",
        "answer_type": "short_answer_judged",
        "response_format_note": "Write a short compliance conclusion with the controlling rule/status check, every required condition or calculation, threshold or exception comparison, and final action.",
        "instance_id": f"ecfr_casebook20_{slug}_short_answer",
        "question": f"{question} Write the answer as a short compliance conclusion with a few decision steps.",
        "decision_set": ["free_response"],
        "decision_option_definitions": {"free_response": "Submit a written compliance conclusion with the required reasoning steps."},
        "answer": "free_response",
        "short_answer": canonical,
        "canonical_response": canonical,
        "rius": ruis,
        "required_rui_ids": required_ids,
        "reasoning_steps": steps,
        "required_decision_steps": steps + [
            {
                "step": max(int(row.get("step") or 0) for row in steps) + 1,
                "uses": required_ids,
                "claim": f"State the final compliance conclusion: {conclusion}",
                "judge_focus": "The final answer must reach this conclusion without relying on a compatibility label.",
            }
        ],
        "judge_rubric": judge_rubric(),
        "disqualifying_errors": disqualifying_errors(memo_text),
        "primary_answer_field": "written_answer",
        "fallback_answer_fields": ["rationale", "answer"],
        "dataset_variant": "casebook20_few_step_short_answer_judged",
        "why_sufficient": "A correct written answer must combine the distributed rule, threshold or exception, transaction facts, and final action.",
        "_memo_text": memo_text,
        "_memo_key": memo_key,
    }


def rule(rui_id: str, source_span: str, claim: str, rui_type: str) -> dict[str, Any]:
    return {"rui_id": rui_id, "source_span": source_span, "normalized_claim": claim, "rui_type": rui_type}


def fact_rui(memo_id: str, fact: dict[str, str]) -> dict[str, Any]:
    return {
        "rui_id": rid(memo_id, fact["key"]),
        "source_span": fact["source_span"],
        "normalized_claim": fact["normalized_claim"],
        "rui_type": "transaction_fact",
    }


def f(key: str, source_span: str, normalized_claim: str) -> dict[str, str]:
    return {"key": key, "source_span": source_span, "normalized_claim": normalized_claim}


def rid(memo_id: str, key: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", memo_id.lower()).strip("_")
    return f"rui_{cleaned}_{key}"


def step(step_id: int, uses: list[str], claim: str) -> dict[str, Any]:
    return {"step": step_id, "uses": uses, "claim": claim, "judge_focus": "The answer should substantively include this decision check."}


def judge_rubric() -> dict[str, Any]:
    return {
        "passing_score": 0.75,
        "criteria": [
            {"name": "final_conclusion", "weight": 0.35, "description": "Reaches the canonical final compliance action."},
            {"name": "required_steps", "weight": 0.4, "description": "Covers the required rule, fact, calculation, exception, or condition checks."},
            {"name": "source_grounding", "weight": 0.15, "description": "Uses only task/source-supported facts."},
            {"name": "clarity", "weight": 0.1, "description": "Concise but not merely a bare label."},
        ],
    }


def disqualifying_errors(text: str) -> list[str]:
    errors = [
        "The answer is only a compatibility label, option id, or bare yes/no statement.",
        "The answer reaches the opposite final compliance action from the canonical response.",
        "The answer omits the controlling rule or status check.",
        "The answer relies on material facts that were not present in the task information or public discussion.",
    ]
    if "percent" in text:
        errors.append("The answer uses the wrong arithmetic or threshold comparison.")
    return errors


def span(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise ValueError(f"Start marker not found: {start}")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise ValueError(f"End marker not found after {start}: {end}")
    return text[start_index:end_index].strip()


def path_for_seed(path: Path) -> str:
    cwd = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(cwd))
    except ValueError:
        return str(path)


def make_row(base_source: str, **case_payload: Any) -> dict[str, Any]:
    row = case_payload
    row["source_text"] = base_source
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-source",
        required=True,
        help="Frozen prepared text of 12 CFR Parts 1 and 3 used to author the casebook.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
