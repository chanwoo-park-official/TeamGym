# Dataset interface for construction, evaluation, and RL

This guide describes the existing normalized datapoint format and its consumers.
Legal, Finance, Medical, and SWE Atlas use this interface for common datapoints.
[DeepSWE](../data/software/deep_swe/README.md) uses a dedicated repository-repair
format and verifier.

The [construction registry](../data/README.md) selects a module to run. Its domain
prefixes describe commands; they do not rename stored datasets or instance IDs,
convert records, select a scorer, or add an RL training regime.

## Shared utilities

[io.py](io.py) provides `read_jsonl`, `write_json`, `write_jsonl`, `prompt_text`,
`render_template`, `estimate_tokens`, and `compact_text` for all domains. Prompt
paths are relative to the package's `prompts/` directory in both source checkouts
and installed wheels. Importing these helpers uses only the standard library;
token estimation tries the optional tokenizer when called and retains its
character-count fallback.

[information_allocation.py](information_allocation.py) provides
`assign_reference_units`, `validate_reference_allocation`, and
`owners_by_reference_unit`. Its existing modes are `one_each`, `paired`, `overlap`,
`redundant_player`, `common_anchor`, and `bottleneck`; `auto` uses the existing
weights and eligibility rules. Pass a seeded `random.Random` for repeatable
assignments. `ReferenceUnitAllocation.to_metadata()` records private and common
assignments, ordering, and validation.

[participant_roles.py](participant_roles.py) provides reference-evidence hashes,
feature and audit normalization, private-evidence assignment lookup, role-name
previews, and duplicate-name detection. It retains the existing feature checks,
stopwords, and role rendering, including exclusion of common evidence from private
role assignments. Legal corpus selection, legal prompt versions, and legal
reference-answer lookup live in `data.legal.participant_role_augmentation`.
Finance uses the shared helpers directly and keeps its additional feature checks, collision numbering, and private-feature
preservation in its own augmentation module. These utilities use only the standard
library and do not call a model.

## Records and identity

[schema.py](schema.py) provides `build_common_datapoint()` and
`validate_common_datapoint()`. [dataset.py](dataset.py) loads UTF-8 JSONL with one
JSON object per physical line. It accepts a task row or a construction record
with a task under `common_datapoint`. Intermediate construction records use
`teamgym_common_datapoint_v2`.

Final [standard-evaluation task files](../../scripts/data_generation/RECONSTRUCTION.md#accepted-task-format)
and the Hugging Face release use `teamgym_release_datapoint_v1` with PIU field
names. Construction stages write this schema through [release.py](release.py).
The loader translates those names for evaluation and judging.

The loader skips records with unrecognized versions, annotates accepted records
with `_input_path` and `_input_line_number`, and performs no structural validation.
Directory inputs recursively load every `.jsonl` file in sorted path order. Use an
explicit release file to select the intended benchmark tasks and allocations.

| Fields | Meaning and consumer expectations |
| --- | --- |
| `schema_version`, `instance_id`, `dataset` | Format version, task ID, and dataset label. Keep `evidence_condition` and `distribution_key` when the same task appears in multiple release views. |
| `source`, `scenario_family`, `generation_metadata`, `dataset_specific` | Source provenance, construction settings, and dataset-specific metadata. Preserve existing task/source identities and allocation metadata for splitting and reproduction. |
| `task_description` | Actor-visible question or objective, response instructions, and answer type. Use `answer_type: "short_answer_judged"` explicitly for written answers so all current runners recognize them. |
| `decision_set`, `decision_option_definitions` | Nonempty list of decision IDs and a nonempty text definition for every ID. Written tasks retain the compatibility decision ID described below. |
| `initial_common_information` | Information intentionally shared with actors at the start, including public facts and any `common_reference_units`. |
| `participants`, `participant_directory` | Participant records and the corresponding directory of IDs. The builder creates the directory from `participants`; the current validator requires at least two participants. |
| `private_context_units` | Source packets with `context_unit_id`, nonempty `text`, and participant IDs in `assigned_to`. Each participant lists its packets in `private_context_unit_ids`; keep both sides consistent. |
| `reference_information_units` | Pivotal information units (PIUs) with stable `rui_id` values and evidence text. Participant `owned_rui_ids` describe PIU ownership. Source packets may carry `hidden_contains_rui_ids` to locate that evidence. |
| `oracle` | Reference decision ID, supporting evidence sets, and, for written tasks, the reference response, required reasoning, and grading rubric. |
| `format_validation` | Saved result of the format helper. Recompute after conversion; the builder returns a report even when `valid` is false. |

Keep participant, context-unit, and PIU IDs unique within a datapoint. Every
participant context reference and assignee must exist. For the current
information-flow evaluation, provide nonempty evidence sets in
`oracle.smallest_sufficient_rui_sets`, each with `rui_ids`. The format helper checks
that references exist and that no participant's `owned_rui_ids`, combined with
common reference IDs, contain a whole sufficient set. It does not establish
semantic necessity or verify that assignment lists agree in both directions.

## Information visible to actors

The standard [actor prompt helpers](../interaction/llm_actors.py) select fields
from the complete datapoint:

| Information | Visibility in the ordinary full-context team run |
| --- | --- |
| Task description, decision options, initial common information | Shared with actors. |
| Participant directory | Participant IDs; generated role names are hidden by default. |
| Private source text | A participant sees the packets listed in its own `private_context_unit_ids`, within the configured context policy. A participant making the final decision also receives its own packets. |
| Public conversation | Available according to the configured communication structure and public-reading policy. The coordinator receives public task and conversation information. |
| Oracle, hidden PIU ownership/annotations, construction metadata | Used internally for evaluation and preparation; omitted from ordinary actor prompt payloads. |

These are field-selection rules, not a generic redactor: put reference answers and
grading rubrics in evaluator metadata and keep actor-visible fields free of answer
keys. Custom actors receive the full datapoint internally and must construct their
visible payloads with the same separation.

Evidence baselines intentionally change the source view: `multi_riu` replaces each
participant's packets with its owned reference evidence, `single_all_riu` supplies
all reference evidence to one actor, and `single_full` supplies the ordered union
of participant-assigned source packets. The [evaluation guide](../../evaluation/README.md)
defines their budgets. Keep the task and reference answer matched across views.

## Answer formats and authoritative scores

| Task kind | Dataset fields | Answer and correctness result |
| --- | --- | --- |
| Finite choice | Actual decision IDs and definitions; `oracle.answer` is one ID. | `final_decision.answer` is compared directly with the oracle. Actor `eval_results.jsonl` records `primary_eval.task_success`; `eval_summary.json` aggregates it. |
| Judged written answer | `task_description.answer_type: "short_answer_judged"`, `decision_set: ["free_response"]`, a definition for that ID, and `oracle.answer: "free_response"`. Supply `oracle.canonical_response`, required decision steps, evidence, and the task's rubric. | The response text is in `final_decision.written_answer`. A separate judge produces `judgments.jsonl` with `judgment.passed`, `conclusion_correct`, and step scores, plus aggregates in `judge_report.json` when reporting is enabled. |
| Numeric written answer | Same written-answer fields, with a numeric reference in `oracle.short_answer` when the existing currency rule applies. | The current short-answer judge incorporates deterministic final-currency-amount checks into its reasoning judgment. This rule is specialized: it extracts currency amounts and uses absolute error strictly less than one dollar. Other units or tolerances require an explicit scoring implementation. |

For written tasks, the episode evaluator records `primary_eval.decision_label_matches`
and `decision_label_matches_with_evidence`. Use the written-answer judgments for
substantive correctness.

The [short-answer scorer](../scoring/short_answers.py) reads reference responses,
`required_decision_steps` (or `reasoning_steps`), `judge_rubric`, and
`disqualifying_errors`, with existing fallbacks to `dataset_specific`. Preserve
the released task's pass policy. The judge's decision and the strict rubric checks
determine pass/fail. Finite numeric scores are reported separately. Report judged and error counts along
with scores. There is currently no registry that dispatches arbitrary scorer names.

The [judge command](../data/long_context_reasoning/judge.py) matches saved answers
to the original input by instance ID and available variant fields. For trajectories it reads `written_answer`,
then `rationale`, then `answer`. Keep the same instance IDs and reference file
through construction, actor evaluation, and judging. Commands and reference
settings are in the [evaluation guide](../../evaluation/README.md).

## Structural example

This small in-memory example illustrates the interface. It is a development
example with two source packets; it is not a benchmark task or paper recipe.

```python
from teamgym.core.schema import build_common_datapoint

datapoint = build_common_datapoint(
    instance_id="toy_notice_allocation_1",
    dataset="toy_notice",
    source={"source_id": "toy_source"},
    scenario_family="notice_reasoning",
    task_description={
        "question": "Was the notice timely?",
        "answer_type": "finite_multiple_choice",
    },
    decision_set=["timely", "late"],
    decision_option_definitions={"timely": "Within the deadline.", "late": "After it."},
    initial_common_information={"task": "Combine the supplied rule and facts."},
    participants=[
        {"participant_id": "p1", "private_context_unit_ids": ["c1"], "owned_rui_ids": ["r1"]},
        {"participant_id": "p2", "private_context_unit_ids": ["c2"], "owned_rui_ids": ["r2"]},
    ],
    private_context_units=[
        {"context_unit_id": "c1", "text": "Notice is due within 30 days.",
         "assigned_to": ["p1"], "hidden_contains_rui_ids": ["r1"]},
        {"context_unit_id": "c2", "text": "Notice was submitted after 20 days.",
         "assigned_to": ["p2"], "hidden_contains_rui_ids": ["r2"]},
    ],
    reference_information_units=[
        {"rui_id": "r1", "text": "Notice is due within 30 days."},
        {"rui_id": "r2", "text": "Notice was submitted after 20 days."},
    ],
    oracle={
        "answer": "timely",
        "smallest_sufficient_rui_sets": [{"rui_ids": ["r1", "r2"]}],
    },
    generation_metadata={"development_example": True},
)
assert datapoint["format_validation"]["valid"]
```

For an illustrative written variant, retain the participants and evidence and
replace the answer fields. This example uses a fresh identity for the variant:

```python
from copy import deepcopy
from teamgym.core.schema import validate_common_datapoint

written = deepcopy(datapoint)
written["instance_id"] = "toy_notice_written_allocation_1"
written["task_description"]["answer_type"] = "short_answer_judged"
written["decision_set"] = ["free_response"]
written["decision_option_definitions"] = {"free_response": "Write the answer and reasoning."}
written["oracle"]["answer"] = "free_response"
written["oracle"]["canonical_response"] = "The notice was timely: 20 days is within 30 days."
written["oracle"]["required_decision_steps"] = [
    {"step": "compare", "claim": "20 days is within the 30-day deadline.", "uses": ["r1", "r2"]},
]
written["oracle"]["judge_rubric"] = {"require_all_steps_for_pass": True}
written["format_validation"] = validate_common_datapoint(written)
assert written["format_validation"]["valid"]
```

These assertions check structural validity only. A reference answer, rubric, and
compatible format still need their domain's substantive construction checks.

## RL preparation is a separate handoff

Evaluation consumes normalized datapoints directly. RL consumes prepared split
rows that embed the normalized record as a JSON string in
`teamgym_datapoint_json`. [TeamGym data preparation](../../teamgym/rl/data.py)
also records canonical task/source identities, allocation, split membership, and
datapoint checksums, and writes JSONL, parquet, and VERL parquet views. The VERL
view wraps the prepared row in `extra_info`.

Allocation variants of the same task and examples sharing an underlying source
need consistent split grouping. Preserve explicit canonical task and source-group
metadata through conversion. Source-group splitting requires explicit group IDs.
Task-only splits can still use historical task IDs when source groups are unavailable.

The retained legal preparation also authors written-answer training tasks, so
frozen standard-evaluation files alone do not reproduce those training examples.
Keep the released prepared training splits and their manifests/checksums. A new
domain's evaluation adapter does not automatically supply a reward or training
recipe. The current paper entry point provides legal and finance regimes; see
the [training guide](../rl/README.md).

## Connecting an incoming domain

1. Compare its output fields and answer/scoring rules with this guide. Reuse
   matching records directly; add a small explicit converter for differences.
   Preserve existing legal IDs, labels, records, and scoring behavior.
2. Map public information, participant packet assignments, and reference metadata.
   Preserve source provenance and allocation-independent task identities.
3. Declare the answer type and reference rubric. Identify the authoritative score
   and any additional domain scorer needed before claiming evaluation support.
   A task without reference evidence or sufficient sets needs an explicit decision
   about information-flow metrics; the current validator expects those fields.
4. Register the construction command following the [registry guide](../data/README.md).
   Keep optional domain dependencies inside its implementation.
5. Document any RL preparation, split mapping, and reward separately. After
   combining domain code, validate construction and evaluation on the intended
   datasets, then validate training in the RL environment before claiming
   end-to-end support.
