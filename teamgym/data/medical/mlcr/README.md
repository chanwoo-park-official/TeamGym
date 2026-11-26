# MLCR construction

For distribution-specific audits and exports, add `--allocation-modes all` to
the native command, or use the [Medical allocation workflow](../README.md#distribution-specific-audits-and-exports)
to redistribute a certified construction without regenerating its task/PIUs.
Named distributions keep their intended ownership and do not apply automatic
ownership merges.

Download an immutable public source revision and convert the original Parquet
files to inspectable JSONL without rewriting text:

```python
from teamgym.data.medical.mlcr.source import acquire_inspectable
acquire_inspectable('/path/to/raw-source', revision='<full-upstream-commit>')
```

The acquisition manifest records original Parquet and derived JSONL hashes.
`load_inspectable` also accepts independently prepared JSONL with the documented
case-summary/question fields.

```sh
uv run --no-sync python -m teamgym.data.medical.mlcr.cli \
  --source-dir /path/to/raw-source --source-revision <immutable-upstream-commit> \
  --case-id <case-id> --prompt-id <source-question-id> \
  --prompt-id <another-source-question-id> --output-root /path/to/new-run \
  --model gpt-5.5 --audit-model gpt-5.5 \
  --participant-token-budget 40000 --max-input-tokens 180000 --max-output-tokens 12000 \
  --env-file /path/to/credentials.env
```

The CLI defaults to strict one PIU per participant. Explicitly specify
`--allow-ownership-merges` only for the admissible merged distribution; such an
output is distinguished from strict one-each. The reusable `run_case` API keeps
its general merge-policy option. `--min-source-questions N` remains configurable
for every integer N >= 1; no small PIU-set preference is imposed.

Only reconstructable eligible QAs contribute candidates. Mapping uses exact
source atoms within `--mapping-source-tokens` (default 120000); if the complete
case exceeds the budget, recorded question-based lexical retrieval supplies
exact atoms and adjacent scope. No text is truncated. An unreconstructable
mapping from a subset does not establish absence of evidence in the full case.
Model input is checked against `--max-input-tokens` (default 180000), separately
from response length (`--max-output-tokens`). `--audit-model` selects all
solver/judge, grounding and leakage checks; it defaults to `--model` and also
accepts the existing `--verification-model` spelling.

Acceptance requires grounding, ALL/NONE/LOO correctness and actual-evidence
checks, semantic packet audits with final repair recheck, and complete
participant-packet final-answer checks. Finite model checks are not proofs.
Exact offsets/checksums, source/QA/config cache identities, content-derived
instance IDs and ownership policy remain in the report/manifest. Accepted runs
produce `common_datapoints.jsonl`; the three conditions expose the same public
task and exact source evidence, without hidden rubric or PIU identifiers.

The common export preserves the audited neutral `Source file: ...` grouping in
visible text. Participant certification uses the actual shared consumer renderer
at `--participant-token-budget` (default 40000) and rejects any material change or truncation.
Exports compare the rendered text with the recorded live-audit text, rather than
checking only that hidden metadata was removed. Changing the observation budget
requires another material comparison and any affected model certification.
Accepted reports include a certification binding the complete report and run
manifest. Export requires boolean `true` for every mandatory audit, complete
ALL/NONE/LOO coverage and exactly one successful final-answer audit per
participant at the same budget. Missing audits or edits after certification are
rejected; earlier uncertified outputs require a fresh construction run.

`run_manifest.json` v2 and export metadata contain `execution_identity`, including
shared LLM clients, tokenizer, core schema/io, actual interaction consumers,
package versions, tokenizer vocabulary hash, lockfile, and domain code/prompts.
The response cache binds this dependency identity and inference configuration;
code drift is checked before each model call and before acceptance.

The common grading contract exports every original required claim and clinical
scope/error rule, numeric `required_decision_steps`, and an explicit all-required
pass policy. A hidden grading-metadata re-export must record its exporter/grader
version separately from the original model-gated producer, and prove that actual
actor requests and all source/public/private material are unchanged.
