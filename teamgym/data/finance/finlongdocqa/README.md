# FinLongDocQA construction

Generate free-response annual-report tasks from source QA annotations and an
original report Markdown ZIP. Source reasoning and code are excluded from model
inputs. Source pages are parsed into exact paragraphs and tables; overlapping
providers are consolidated before task generation. Final tasks pass blind
ALL/NONE/PIU leave-one-out checks, whole-packet foreign-evidence review and direct
public-plus-local complete-answer checks for both collaborative input modes.
These are model-based operational checks, not mathematical proofs or independent
verification of the financial oracle.

The source index retains exact report-wide declarations such as "Tabular amounts
are in U.S. Dollars in thousands, except share amounts, unless otherwise noted."
Their byte offsets and hashes are recorded. Only the declaration sentence is
restored as common source context, without adjacent financial results. This
context is included in mapping, ALL/NONE/LOO qualification, actual participant
inputs and the PIU-only condition. Table-specific unit exceptions still apply.
Judges must ground monetary scales in the displayed source context, rather than
infer them from the oracle or financial plausibility. The v4 source parser
requires old indexes and constructions to be regenerated for redistribution.

```bash
uv run --no-sync python -m teamgym.data.finance.finlongdocqa.pipeline \
  --stage all --qa-path /path/dataset_qa.jsonl --reports-zip /path/reports.zip \
  --report-ids A_2022 --output-root /path/fresh-output \
  --allocation-mode one_each --generation-attempts 3 --packet-repair-rounds 3 \
  --packet-audit-max-chars 1000000 --participant-context-token-budget 40000 \
  --model openai.gpt-5.5 --reasoning-effort medium \
  --env-file /path/.env
```

`run_manifest.json` binds the original QA/archive/member hashes, deterministic
parser/index, selected reports, construction code/prompts, model and budgets.
Generation reparses every selected source member and rejects modified indexes,
missing annotated pages and unsupported empty source formats. A changed
fingerprint requires a new output directory. A compatible accepted report can
resume; failed reports retain their traces and receive a new attempt on the next
invocation.

Atomic `_commits/<report>/<attempt>.json` files are the source of truth.
`attempts.jsonl`, `accepted.jsonl` and `common_datapoints.jsonl` are rebuilt from
those commits, including after an interruption. Uncommitted, damaged or
incompatible checkpoints cannot establish acceptance.

Each final PIU uses one mapped block. The table exception permits retaining exact
original rows while keeping one table as one PIU. Headers, source row numbers,
absolute row spans and original table hash are retained. Canonical table repairs
trigger full packet re-audit and repeat ALL/NONE/LOO grounding checks. Non-table
canonical overlaps remain rejected. A complete participant packet must fit the
explicit whole-packet audit budget; candidate shortlists cannot certify an
unseen remainder.

Accepted common datapoints use the standard TeamGym loader and source-only PIU
renderer. Internal summaries, functional labels and PIU identifiers are not
rendered as participant source text. The constructor records complete raw
consumer public/private inputs and hashes for the local-answer checks in
`generation_metadata.initial_view_audits`. Final local-check results and the
reference rubric remain evaluator metadata.

A complete-view review can report multiple independent reconstruction routes;
every route's support and removable contexts are validated before their union
is removed. The final clean packet is still reviewed in full. Within one repair
run, an unchanged complete rendered prompt can reuse its earlier review, with
its input hash and original trace recorded. Such reuse is not counted as an
independent model repetition.

The finance protocol defaults to a 40,000-token initial source cap per
participant. `--participant-context-token-budget` accepts 1 through 40000.
Complete canonical PIU blocks are reserved first; additional
complete blocks are kept in original source order within the fixed allocation.
The manifest lists budget exclusions and source hashes. Oversized canonical
source is rejected instead of silently truncated. Final packet and salience
audits, public-plus-local checks and actual consumer captures use the configured
budget and must preserve byte-identical source material.

## Export a completed producer run for evaluation

Run generation for one report with `--stage all --report-ids REPORT_ID` and the
40,000-token participant context budget. After the report is accepted, run
these commands from the matching producer code checkout. Set `RUN` to the
generation output directory and use the same QA JSONL and report ZIP that
produced it. Choose new directories for `PRODUCER` and `EXPORT`.

```bash
RUN=/path/to/completed-run
QA=/path/to/dataset_qa.jsonl
REPORTS=/path/to/reports.zip
PRODUCER=/path/to/new-producer-bundle
EXPORT=/path/to/new-evaluator-export

python scripts/data_generation/finance/finlongdocqa/assemble_producer_bundle.py \
  --run-dir "$RUN" --qa-path "$QA" --reports-zip "$REPORTS" --output-dir "$PRODUCER"

python -m teamgym.data.finance.finlongdocqa.evaluation \
  --producer-run "$PRODUCER" --output-dir "$EXPORT"

python - "$EXPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
receipt = json.loads((output / "release_receipt.json").read_text())
assert receipt["status"] == "verified_deterministic_evaluator_export"
assert receipt["model_observations_byte_equal"]
for name, expected in receipt["outputs"].items():
    assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, name
print(output / "common_datapoints.jsonl")
PY
```

The assembly command copies the completed run to `generated/`, stores the
original sources under `source/`, and copies the producer's hashed dependencies
to `code_snapshot/`. It verifies the accepted atomic commit and qualification
records, then captures the original row's consumer inputs offline in
`consumer_inputs.json`. The exporter compares those inputs with the inputs
rendered after adding the generic grader metadata. Pass
`$EXPORT/common_datapoints.jsonl` to standard evaluation.

## Redistribute one frozen task

Use an unchanged native `common_datapoints.jsonl` backed by accepted atomic
commits and the matching run manifest. Each distribution reruns complete packet
repair, exact consumer-budget checks, salience audits, and full-context plus
PIU-only final-answer checks. Generic grader metadata is exported only after
those checks, with actual model observations unchanged. A new empty output
directory is required; old acceptance flags cannot silently skip new checks.

```sh
uv run --no-sync python -m teamgym.data.finance.finlongdocqa.generate_frozen_allocation_modes \
  --input /path/accepted/common_datapoints.jsonl \
  --output-root /path/new-allocation-run --mode one_each --mode redundant_player \
  --audit-model openai.gpt-5.5 --env-file /path/credentials.env \
  --participant-context-token-budget 40000 --max-items 1
```

All five names are supported. The default requests the four additional modes.
Model, credentials path, endpoint and token budget are command-line options.
The native constructor and frozen-allocation command default to GPT-5.5.
