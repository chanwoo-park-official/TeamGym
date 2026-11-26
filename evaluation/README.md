# Standard Evaluation

Pass a JSONL path with `--input-root` to select a corpus and evidence view.
Use a file from the [public TeamGym dataset](https://huggingface.co/datasets/TeamGym/teamgym-datasets)
or an accepted output from dataset construction.

The [release inventory](../README.md#data-flow-and-release-inventory) identifies
the frozen input files and their current download status. Prepared RL splits
are separate artifacts.

The [dataset interface guide](../teamgym/core/README.md)
defines normalized input fields, information visibility, and the answer/scoring
conventions for connecting additional domains.

Evaluation through model APIs needs the [lightweight environment](../requirements/README.md)
and API credentials. It does not require a local GPU. Dataset construction and
RL training have separate [construction](../scripts/data_generation/README.md)
and [training](../teamgym/rl/README.md) guides.

## Commands and Settings

`run.sh` supplies the standard budgets to the existing Python evaluators. Run
it from a source checkout with the installed lightweight dependencies:

```bash
bash evaluation/run.sh --help
bash evaluation/run.sh multi_full --help
```

| Mode | Public input file | Evaluation | Context budget | Turn budget |
| --- | --- | --- | --- | --- |
| `single_all_riu` | `single_all_piu.jsonl` | One actor with all PIUs | 40,000 tokens | Single answer |
| `multi_riu` | `multi_piu_only.jsonl` | Team with PIUs only | 40,000 per participant | 16 |
| `multi_full` | `multi_full_context.jsonl` | Team with full participant source packets | 40,000 per participant | 16 |
| `single_full` | `multi_full_context.jsonl` | One actor with the union of participant source packets | 200,000 tokens | Single answer |
| `judge` | The input used for the actor run | Semantic scoring of written answers | Uses the frozen reference rubric | Per saved answer |

The three evidence conditions retain their model-matrix settings: temperature
0, four request attempts, a 3,600-token completion cap, 600-token public
participant/coordinator messages, 1,200-token private participant/coordinator
reasoning, a 1,200-token final rationale, and 1,800-token final reasoning.
Single-agent modes use the same final-answer budgets. Request timeouts are 600
seconds, or 900 seconds for `single_full`. The judge uses a 1,800-token
completion cap and a 600-second timeout.

The full-context single-agent evaluator preserves the complete ordered source
union; its context budget does not truncate that source text. Provider context
limit failures are recorded.

Each command records failures and continues. Prompt tracing and HTML rendering
are disabled in the presets. The judge also disables request logging. Pass
`--trace-prompts`, `--render-html` (actors), or `--render-reports` and
`--request-log` (judge) when those artifacts are wanted. Later scalar arguments
override the preset values. `TEAMGYM_PYTHON` selects the Python executable.

## Actor Evaluation

### Run one task

From the repository root, download the pinned 152-task Legal full-context
subset. Set `OPENAI_API_KEY` in your environment and replace `YOUR_MODEL` with a model
available to your account on the OpenAI Responses API.

```bash
curl --fail --location --output multi_full_context_subset.jsonl \
  https://huggingface.co/datasets/TeamGym/teamgym-datasets/resolve/ea0b5210dd9509131bd8f771ed278c74acce2258/legal/multi_full_context_subset.jsonl

bash evaluation/run.sh multi_full \
  --input-root multi_full_context_subset.jsonl \
  --output-dir outputs/quickstart \
  --communication-structure coordinator_select_one__coordinator_decision \
  --model YOUR_MODEL \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --endpoint /responses \
  --api-key-env OPENAI_API_KEY \
  --limit 1
```

The evaluator writes `run_manifest.json`, `eval_results.jsonl`, and trajectory
JSON files in a timestamped run directory under `outputs/quickstart/`.
The limit selects one task. A team can make several model calls for it.
Use the [written-answer judge](#written-answer-judging) when the selected task
requires a semantic score.

### Paper reference settings

This example uses the Bedrock Responses endpoint configuration recorded with
the reference runs:

```bash
actor_args=(
  --model gpt-5.4
  --provider bedrock_openai
  --base-url https://bedrock-mantle.us-east-2.api.aws/openai/v1
  --endpoint /responses
  --api-key-env AWS_BEARER_TOKEN_BEDROCK
)

bash evaluation/run.sh multi_full "${actor_args[@]}" \
  --input-root /path/to/multi_full_context.jsonl \
  --output-dir outputs/legal/multi_full \
  --communication-structure coordinator_select_one__coordinator_decision \
  --limit 1
```

Remove `--limit 1` to evaluate the complete input. Choose the corresponding
public input file for each baseline:

```bash
bash evaluation/run.sh multi_riu "${actor_args[@]}" \
  --input-root /path/to/multi_piu_only.jsonl \
  --communication-structure coordinator_select_one__coordinator_decision \
  --output-dir outputs/legal/multi_piu_only
bash evaluation/run.sh single_all_riu "${actor_args[@]}" \
  --input-root /path/to/single_all_piu.jsonl \
  --output-dir outputs/legal/single_all_riu
bash evaluation/run.sh single_full "${actor_args[@]}" \
  --input-root /path/to/multi_full_context.jsonl \
  --output-dir outputs/legal/single_full
```

`configs/models.tsv` records the eight reference actor configurations. These
are historical model/endpoint identifiers; configure the credentials and
endpoint available for a new run. Pass the corresponding `--model`,
`--provider`, `--endpoint`, and `--base-url` values explicitly.

The five reference communication schedules are in `configs/schedules.tsv`.
For the schedule comparison, override the fixed model-matrix budget:

- `one_per_participant`: `--turn-budget N+1`, where `N` is the number of
  participants in that task;
- `single_round`: `--turn-budget 2`.

Pass the computed integer and the selected `--communication-structure`.
Evaluate tasks with different participant counts separately, using
`--num-shards TASK_COUNT --shard-index TASK_INDEX` to select a single row.
Omitting an explicit structure is an error unless
`--all-communication-structures` is passed for a debug sweep.

## Written-Answer Judging

Finite-choice tasks are scored deterministically by the actor evaluator.
Written tasks retain the compatibility label `free_response`; matching that label
in actor results does not establish written-answer correctness. Use the separate
written-answer judgments for substantive correctness and retain the task's rubric
and pass policy.

Judge short-answer trajectories in a separate step:

```bash
bash evaluation/run.sh judge \
  --input-root /path/to/multi_full_context.jsonl \
  --trajectories-root outputs/legal/multi_full \
  --output-root outputs/legal/multi_full_judge \
  --model gpt-5.5 \
  --provider bedrock_openai \
  --base-url https://bedrock-mantle.us-east-2.api.aws/openai/v1 \
  --endpoint /responses \
  --api-key-env AWS_BEARER_TOKEN_BEDROCK
```

Use the judge selected for the experiment: the model/evidence runs used
GPT-5.5; the centralized full-context baseline used GPT-5.4. Judgments report
the conclusion and required reasoning steps separately.

The updated written grader records `long_context_short_answer_judgment_v2`
on new judgments. Strict success requires an overall pass and full coverage of
the required steps. The report shows step coverage separately. The cache keys
are distinct from v1, so a shared cache directory
can hold results from both versions. Use the saved paper judgments for the
reported scores. New judgments from this checkout use the grading rule above.

## Outputs and Layout

Actor outputs include `run_manifest.json`, `eval_results.jsonl`,
`eval_summary.json`, and `trajectories/*.json`. Each run gets a timestamped
subdirectory. Prompt records and `interaction_report.html` are written when
enabled. The judge writes `judgments.jsonl` and `judge_report.json`.

Actor run manifests and summaries include `input_diagnostics` with the
normalized row count, number selected after sharding and limits, and skipped
records grouped by file and physical line number. Blank lines are ignored.
Malformed JSON aborts loading.

New interaction artifacts use unversioned format IDs such as `teamgym_episode`
and `teamgym_single_agent_eval_run` in the existing `schema_version` field.
The report tool reads saved trajectories with either the TeamGym IDs or the
historical IDs.

For direct team evaluation, use `teamgym-evaluate` with
`--context-variant full_context` or `--context-variant piu_only`.
The single-agent module, `teamgym.interaction.run_single_agent_rui_evaluation`,
accepts `--context-variant all_rius` or `--context-variant full_source`.
Its default output directory is `outputs/teamgym/evaluation/<baseline>`.
Run `python -m teamgym.interaction.run_single_participant_evaluation` for
single-participant evaluation.

- `run.sh`: portable presets for actors and judging;
- `configs/`: reference model configurations and communication schedules;
- `reporting/`: numeric rescoring for saved answers;
- `tools/`: model endpoint inspection utilities.

Evaluation algorithms remain in `teamgym.interaction`;
judging remains in `teamgym.data.long_context_reasoning`.
The installed `teamgym-evaluate` and `teamgym-judge` commands expose those
runtimes directly. Dataset inputs, saved trajectories, and model weights are
distributed separately from the code.

To render a standard report for one saved actor run, use its timestamped run
directory:

```bash
python -m teamgym.visualization.interaction_trajectory_report \
  --input outputs/legal/multi_full/RUN_DIRECTORY
```

For saved judgments, `rescore_numeric_short_answers.py` applies the current
deterministic numeric checks. The `visualize` module shows dataset context and PIU
allocations. Both commands provide input and output options in `--help`.

```bash
python evaluation/reporting/rescore_numeric_short_answers.py --help
python -m teamgym.data.long_context_reasoning.visualize --help
```
