#!/usr/bin/env python
"""Create a source-excerpt MSLR seed with neutral PIUs.

Each PIU is an exact excerpt from the anonymized case description, analysis, or
final judgment. Canonical answers and required reasoning steps use the selected
source excerpts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = "generated/mslr_source_excerpt_casebook50_20260709"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing the anonymized MSLR entry_*.json source files.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--candidate-scan-limit", type=int, default=1000)
    parser.add_argument("--min-source-chars", type=int, default=1200)
    parser.add_argument("--min-rui-chars", type=int, default=50)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser()
    if not input_root.is_dir():
        parser.error(f"--input-root must name an existing directory: {input_root}")
    input_paths = sorted((path for path in input_root.glob("entry_*.json") if path.is_file()), key=entry_index)
    if not input_paths:
        parser.error(f"--input-root contains no entry_*.json source files: {input_root}")

    output_root = Path(args.output_root).expanduser()
    seed_dir = output_root / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in input_paths[: args.candidate_scan_limit]:
        if len(rows) >= args.limit:
            break
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            row = convert_case(raw, path=path, min_source_chars=args.min_source_chars, min_rui_chars=args.min_rui_chars)
        except ValueError as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        rows.append(row)

    seed_path = seed_dir / "mslr_source_excerpt_casebook50_seed.jsonl"
    with seed_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema": "mslr_source_excerpt_casebook50_seed_report_v1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed_path": str(seed_path),
        "requested_limit": args.limit,
        "written_count": len(rows),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:50],
        "construction_policy": {
            "rui_policy": "Each RIU is an exact source excerpt, not a structured field table.",
            "public_question_policy": "Question asks for a short legal conclusion without enumerating the required chain slots.",
            "label_policy": "Canonical answer is a source-grounded synthesis of the selected exact RIU excerpts.",
            "not_for_release_until": "single all-RIU and support checks pass on the generated common datapoints",
        },
        "sample": [
            {
                "instance_id": row["instance_id"],
                "question": row["question"],
                "source_chars": len(row["source_text"]),
                "rui_count": len(row["rius"]),
                "rui_chars": {rui["rui_id"]: len(rui["source_span"]) for rui in row["rius"]},
            }
            for row in rows[:5]
        ],
    }
    (seed_dir / "seed_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def convert_case(raw: dict[str, Any], *, path: Path, min_source_chars: int, min_rui_chars: int) -> dict[str, Any]:
    case_info = dict_or_empty(raw.get("案件信息"))
    insider = dict_or_empty(raw.get("内幕交易信息的认定"))
    party_info = dict_or_empty(raw.get("当事人信息"))
    party_base = dict_or_empty(party_info.get("当事人基础信息"))
    party_trading = dict_or_empty(party_info.get("当事人的内幕交易认定"))
    party_penalty = dict_or_empty(party_info.get("当事人处罚结果"))

    party_name = clean(party_base.get("姓名")) or infer_party_name(raw) or "当事人"
    stock_name = clean(insider.get("内幕交易的股票名称")) or "相关证券"
    if not is_single_party_name(party_name):
        raise ValueError("multi-party target skipped")

    analysis = clean(raw.get("案件分析"))
    description = clean(raw.get("案件描述"))
    final_judgment = clean(raw.get("最终判决"))
    legal_doc = clean(raw.get("法律文书原文"))
    source_text = build_source_text(
        description=description,
        analysis=analysis,
        final_judgment=final_judgment,
        legal_doc=legal_doc,
    )
    if len(source_text) < min_source_chars:
        raise ValueError("source text too short")
    if not analysis and not description:
        raise ValueError("missing analysis and description")
    if not final_judgment:
        raise ValueError("missing final judgment")

    fact_text = legal_doc or description or analysis
    legal_text = legal_doc or " ".join(part for part in [description, analysis] if part)
    penalty_text = legal_doc or final_judgment

    used_excerpts: list[str] = []
    event_excerpt = pick_event_excerpt(fact_text, insider, stock_name, exclude=used_excerpts)
    used_excerpts.append(event_excerpt)
    timing_excerpt = pick_timing_excerpt(fact_text, insider, exclude=used_excerpts)
    used_excerpts.append(timing_excerpt)
    knowledge_excerpt = pick_knowledge_excerpt(fact_text, party_name, party_trading, exclude=used_excerpts)
    used_excerpts.append(knowledge_excerpt)
    trading_excerpt = pick_trading_excerpt(fact_text, party_name, stock_name, party_trading, exclude=used_excerpts)
    used_excerpts.append(trading_excerpt)
    legal_excerpt = pick_legal_excerpt(legal_text, exclude=used_excerpts)
    used_excerpts.append(legal_excerpt)
    penalty_excerpt = pick_penalty_excerpt(penalty_text, party_name, party_penalty, exclude=[])
    ruis = [
        make_rui(
            "rui_01",
            source_span=event_excerpt,
            source_section="case_analysis_or_description",
        ),
        make_rui(
            "rui_02",
            source_span=timing_excerpt,
            source_section="case_analysis_or_description",
        ),
        make_rui(
            "rui_03",
            source_span=knowledge_excerpt,
            source_section="case_analysis_or_description",
        ),
        make_rui(
            "rui_04",
            source_span=trading_excerpt,
            source_section="case_analysis_or_description",
        ),
        make_rui(
            "rui_05",
            source_span=legal_excerpt,
            source_section="case_analysis_or_description",
        ),
        make_rui(
            "rui_06",
            source_span=penalty_excerpt,
            source_section="final_judgment",
        ),
    ]
    if any(not rui["source_span"] for rui in ruis):
        missing = [rui["rui_id"] for rui in ruis if not rui["source_span"]]
        raise ValueError(f"missing exact source excerpts: {missing}")
    if any(len(rui["source_span"]) < min_rui_chars for rui in ruis):
        short = [rui["rui_id"] for rui in ruis if len(rui["source_span"]) < min_rui_chars]
        raise ValueError(f"source excerpts too short: {short}")
    if len({normalize_space(rui["source_span"]) for rui in ruis}) != len(ruis):
        raise ValueError("duplicate source excerpts")
    for rui in ruis:
        if rui["source_span"] not in source_text:
            raise ValueError(f"rui not found in source_text: {rui['rui_id']}")
    if source_ranges_overlap(source_text, [rui["source_span"] for rui in ruis]):
        raise ValueError("source excerpts overlap")

    canonical = source_grounded_canonical_response(party_name=party_name, stock_name=stock_name, ruis=ruis)
    required_rui_ids = [rui["rui_id"] for rui in ruis]
    instance_id = f"mslr_source_excerpt_{entry_index(path):04d}_short_answer"
    question = (
        f"请基于分散的案件材料，为{party_name}就{stock_name}相关交易行为写出简短行政处罚结论。"
        "答案应说明足以支持结论的关键事实、法律认定和最终处理。"
    )
    reasoning_steps = [
        {
            "step": 1,
            "uses": ["rui_01", "rui_02"],
            "claim": source_step_claim(
                "Identify the material nonpublic information and the relevant timing from the source excerpts.",
                event_excerpt,
                timing_excerpt,
            ),
            "judge_focus": "The answer should identify the material nonpublic information and its relevant timing.",
        },
        {
            "step": 2,
            "uses": ["rui_03"],
            "claim": source_step_claim(
                "Explain why the party knew, was treated as knowing, or had access to the information.",
                knowledge_excerpt,
            ),
            "judge_focus": "The answer should explain why the party knew or is legally treated as knowing the information.",
        },
        {
            "step": 3,
            "uses": ["rui_04"],
            "claim": source_step_claim("State the relevant trading conduct.", trading_excerpt),
            "judge_focus": "The answer should state the trading conduct and connect it to the relevant period.",
        },
        {
            "step": 4,
            "uses": ["rui_05"],
            "claim": source_step_claim("State the legal characterization or legal basis for liability.", legal_excerpt),
            "judge_focus": "The answer should summarize the legal conclusion or legal basis for liability.",
        },
        {
            "step": 5,
            "uses": ["rui_06"],
            "claim": source_step_claim("State the final administrative handling or penalty.", penalty_excerpt),
            "judge_focus": "The answer should state the final administrative handling or penalty.",
        },
        {
            "step": 6,
            "uses": required_rui_ids,
            "claim": f"State the final legal conclusion: {canonical}",
            "judge_focus": "The final answer must combine the source excerpts into the party-specific conclusion.",
        },
    ]

    return {
        "instance_id": instance_id,
        "source_family": "mslr_insider_trading_source_excerpt",
        "source_id": f"mslr_anonymized_{entry_index(path):04d}",
        "source_title": f"MSLR anonymized insider-trading case {entry_index(path)}",
        "source_path": str(path),
        "answer_type": "short_answer_judged",
        "answer": "free_response",
        "decision_set": ["free_response"],
        "decision_option_definitions": {
            "free_response": "Submit a concise written legal conclusion grounded in the distributed case excerpts.",
        },
        "question": question,
        "response_format_note": "Write a concise legal conclusion with the supporting facts and final handling.",
        "short_answer": canonical,
        "canonical_response": canonical,
        "source_text": source_text,
        "rius": ruis,
        "required_rui_ids": required_rui_ids,
        "reasoning_steps": reasoning_steps[:-1],
        "required_decision_steps": reasoning_steps,
        "why_sufficient": "The source excerpts jointly support the event/timing, knowledge, trading, legal conclusion, and penalty.",
        "structured_answer_label": {
            "schema": "mslr_source_excerpt_answer_label_v1",
            "target_party": party_name,
            "stock_name": stock_name,
            "source_language": "zh",
            "source_entry": path.name,
            "label_policy": "Answer labels are source-grounded summaries of the selected exact RIU excerpts.",
        },
        "judge_rubric": {
            "passing_score": 0.75,
            "criteria": [
                {
                    "name": "final_conclusion",
                    "weight": 0.30,
                    "description": "Reaches the party-specific insider-trading and penalty conclusion.",
                },
                {
                    "name": "required_steps",
                    "weight": 0.45,
                    "description": "Covers material nonpublic information, awareness, trading conduct, legal basis, and final handling.",
                },
                {
                    "name": "source_grounding",
                    "weight": 0.15,
                    "description": "Uses facts supported by the source excerpts and does not invent material facts.",
                },
                {
                    "name": "clarity",
                    "weight": 0.10,
                    "description": "Presents a concise legal conclusion.",
                },
            ],
        },
        "disqualifying_errors": [
            "The answer is only a bare label.",
            "The answer reaches the opposite liability or penalty conclusion from the canonical response.",
            "The answer omits the party-awareness step and the trading-conduct step.",
            "The answer invents a material trading date, amount, legal basis, or penalty not supported by the source excerpts.",
        ],
        "dataset_variant": "mslr_source_excerpt_casebook50_short_answer",
    }


def make_rui(rui_id: str, *, source_span: str, source_section: str) -> dict[str, Any]:
    return {
        "rui_id": rui_id,
        "source_span": source_span,
        "normalized_claim": source_span,
        "rui_type": "source_excerpt",
        "source_ref": {"source_section": source_section},
    }


def pick_event_excerpt(text: str, insider: dict[str, Any], stock_name: str, *, exclude: list[str]) -> str:
    terms = [clean(insider.get("内幕信息内容")), stock_name, "内幕信息"]
    return pick_excerpt(
        text,
        required_any=[terms[0], stock_name],
        prefer=["内幕信息", "重大", "属于", "公开前"],
        avoid=["申辩", "辩称", "认为其不", "不属于内幕信息"],
        max_sentences=2,
        min_chars=80,
        exclude=exclude,
    )


def pick_timing_excerpt(text: str, insider: dict[str, Any], *, exclude: list[str]) -> str:
    formation = clean(insider.get("内幕信息形成时间"))
    disclosure = clean(insider.get("内幕信息公开时间"))
    return pick_excerpt(
        text,
        required_any=[formation, disclosure, "敏感期"],
        prefer=[formation, disclosure, "公开", "形成", "敏感期", "公告"],
        avoid=["申辩", "辩称", "并无不妥"],
        max_sentences=2,
        min_chars=70,
        exclude=exclude,
    )


def pick_knowledge_excerpt(text: str, party_name: str, party_trading: dict[str, Any], *, exclude: list[str]) -> str:
    knowledge_time = clean(party_trading.get("当事人知悉内幕信息时间"))
    knowledge_way = clean(party_trading.get("当事人知悉内幕信息的方式（原文）"))
    return pick_excerpt(
        text,
        required_any=[knowledge_way, knowledge_time, "知悉", "告知", "通话", "关系密切"],
        required_all=[party_name] if party_name and party_name in text else [],
        prefer=[party_name, knowledge_time, "知悉", "告知", "通话", "联系", "关系", "内幕信息"],
        avoid=["申辩", "辩称", "不知悉", "不知晓", "不是内幕信息知情人"],
        max_sentences=2,
        min_chars=80,
        exclude=exclude,
    )


def pick_trading_excerpt(text: str, party_name: str, stock_name: str, party_trading: dict[str, Any], *, exclude: list[str]) -> str:
    earliest = clean(party_trading.get("最早买入时间"))
    latest = clean(party_trading.get("最晚买入时间"))
    trade_time = clean(party_trading.get("买入时间"))
    direction = clean(party_trading.get("买入/卖出"))
    illegal_gain = clean(party_trading.get("违法所得（元）"))
    return pick_excerpt(
        text,
        required_any=[trade_time, earliest, latest, illegal_gain, direction],
        required_all=[party_name] if party_name and party_name in text else [],
        prefer=[party_name, stock_name, trade_time, earliest, latest, illegal_gain, "买入", "卖出", "交易", "成交", "违法所得"],
        avoid=["申辩", "辩称", "不符合内幕交易", "正常交易", "与利用内幕信息", "无关", "并非"],
        max_sentences=2,
        min_chars=80,
        exclude=exclude,
    )


def pick_legal_excerpt(text: str, *, exclude: list[str]) -> str:
    return pick_excerpt(
        text,
        required_any=["我会认为", "违反", "构成内幕交易", "内幕交易行为"],
        prefer=["我会认为", "违反", "构成内幕交易", "内幕交易行为", "证券法", "处罚"],
        avoid=["申辩", "辩称", "不构成内幕交易", "不符合内幕交易"],
        max_sentences=2,
        min_chars=80,
        exclude=exclude,
    )


def pick_penalty_excerpt(final_judgment: str, party_name: str, party_penalty: dict[str, Any], *, exclude: list[str]) -> str:
    fine = normalize_amount(clean(party_penalty.get("罚款数额（元）")))
    confiscation = normalize_amount(clean(party_penalty.get("没收违法所得金额（元）")))
    required = [party_name] if party_name and party_name in final_judgment else [fine, confiscation, "罚款", "没收", "决定"]
    return pick_excerpt(
        final_judgment,
        required_any=required,
        prefer=[party_name, fine, confiscation, "没收", "罚款", "决定", "证券法"],
        avoid=["复议", "诉讼期间", "不停止执行"],
        max_sentences=2,
        min_chars=80,
        exclude=exclude,
    )


def pick_excerpt(
    text: str,
    *,
    required_any: list[str],
    prefer: list[str],
    avoid: list[str],
    max_sentences: int,
    min_chars: int,
    exclude: list[str],
    required_all: list[str] | None = None,
) -> str:
    text = clean(text)
    if not text:
        return ""
    sentences = sentence_spans(text)
    clean_required = [term for term in (clean(term) for term in required_any) if term]
    clean_required_all = [term for term in (clean(term) for term in (required_all or [])) if term]
    clean_prefer = [term for term in (clean(term) for term in prefer) if term]
    clean_avoid = [term for term in (clean(term) for term in avoid) if term]
    clean_exclude = [normalize_space(item) for item in exclude if normalize_space(item)]
    best: tuple[float, int, int] | None = None
    for index, (_start, _end, sentence) in enumerate(sentences):
        if excluded_sentence(sentence, clean_exclude):
            continue
        if clean_required_all and not all(term in sentence for term in clean_required_all):
            continue
        if clean_required and not any(term in sentence for term in clean_required):
            continue
        score = sum(3 for term in clean_required_all if term and term in sentence)
        score += sum(2 for term in clean_required if term and term in sentence)
        score += sum(1 for term in clean_prefer if term and term in sentence)
        score -= sum(4 for term in clean_avoid if term and term in sentence)
        score -= max(0, len(sentence) - 850) / 850
        if best is None or score > best[0]:
            best = (score, index, index)
    if best is None:
        return ""
    center = best[1]
    start_i = center
    end_i = center
    while end_i - start_i + 1 < max_sentences:
        start = sentences[start_i][0]
        end = sentences[end_i][1]
        if len(text[start:end].strip()) >= min_chars:
            break
        candidates: list[tuple[float, int, str]] = []
        for direction, candidate_i in (("left", start_i - 1), ("right", end_i + 1)):
            if candidate_i < 0 or candidate_i >= len(sentences):
                continue
            sentence = sentences[candidate_i][2]
            if excluded_sentence(sentence, clean_exclude):
                continue
            if clean_avoid and any(term in sentence for term in clean_avoid):
                continue
            score = sum(2 for term in clean_required if term and term in sentence)
            score += sum(1 for term in clean_prefer if term and term in sentence)
            candidates.append((score, candidate_i, direction))
        if not candidates:
            break
        _score, candidate_i, direction = max(candidates, key=lambda item: (item[0], item[2] == "right"))
        if direction == "left":
            start_i = candidate_i
        else:
            end_i = candidate_i
    start = sentences[start_i][0]
    end = sentences[end_i][1]
    excerpt = text[start:end].strip()
    if len(excerpt) > 1800:
        excerpt = sentences[center][2].strip()
    return excerpt


def excluded_sentence(sentence: str, excluded: list[str]) -> bool:
    normalized = normalize_space(sentence)
    return any(normalized in item or item in normalized for item in excluded)


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[。！？；;]\s*", text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            spans.append((start, end, sentence))
        start = end
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            spans.append((start, len(text), sentence))
    return spans or [(0, len(text), text)]


def build_source_text(*, description: str, analysis: str, final_judgment: str, legal_doc: str) -> str:
    if legal_doc:
        return "【法律文书原文】\n" + legal_doc
    parts = []
    if description:
        parts.append("【案件描述】\n" + description)
    if analysis:
        parts.append("【案件分析】\n" + analysis)
    if final_judgment:
        parts.append("【最终判决】\n" + final_judgment)
    return "\n\n".join(parts)


def source_ranges_overlap(source_text: str, excerpts: list[str]) -> bool:
    ranges: list[tuple[int, int]] = []
    for excerpt in excerpts:
        start = source_text.find(excerpt)
        if start < 0:
            return True
        end = start + len(excerpt)
        for prev_start, prev_end in ranges:
            if max(start, prev_start) < min(end, prev_end):
                return True
        ranges.append((start, end))
    return False


def source_grounded_canonical_response(*, party_name: str, stock_name: str, ruis: list[dict[str, Any]]) -> str:
    by_id = {rui["rui_id"]: rui["source_span"] for rui in ruis}
    parts = [
        f"对{party_name}就{stock_name}相关交易行为的行政处罚结论应综合以下来源材料作出。",
        f"内幕信息及时间：{by_id.get('rui_01', '')} {by_id.get('rui_02', '')}",
        f"知悉或接触内幕信息的依据：{by_id.get('rui_03', '')}",
        f"交易行为：{by_id.get('rui_04', '')}",
        f"法律认定：{by_id.get('rui_05', '')}",
        f"最终处理：{by_id.get('rui_06', '')}",
    ]
    return " ".join(clean(part) for part in parts if clean(part))


def source_step_claim(instruction: str, *excerpts: str) -> str:
    evidence = " ".join(clean(excerpt) for excerpt in excerpts if clean(excerpt))
    return f"{instruction} Source evidence: {evidence}"


def normalize_amount(value: str) -> str:
    text = clean(value).replace(" ", "")
    if not text:
        return ""
    if set(text) <= {"-", "—", "－"}:
        return "-"
    stripped = text.replace(",", "")
    try:
        number = float(stripped)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number)


def infer_party_name(raw: dict[str, Any]) -> str:
    for field in (raw.get("案件描述"), raw.get("案件分析"), raw.get("最终判决")):
        text = clean(field)
        match = re.search(r"当事人[:：]?\s*(\[[^\]]+\])", text)
        if match:
            return match.group(1)
    return ""


def is_single_party_name(value: str) -> bool:
    text = clean(value)
    if not text:
        return False
    return not any(separator in text for separator in ["、", ",", "，", "；", ";", "和", "及"])


def entry_index(path: Path) -> int:
    match = re.search(r"entry_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else 10**9


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


if __name__ == "__main__":
    main()
