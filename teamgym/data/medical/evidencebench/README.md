# EvidenceBench construction

For distribution-specific audits and exports, add `--allocation-modes all` to
the native command, or use the [Medical allocation workflow](../README.md#distribution-specific-audits-and-exports)
to redistribute a certified construction without regenerating its task/PIUs.

Use a pinned original test-split JSON release and record the upstream revision.
The pipeline preserves one candidate per annotated anchor, exact source text,
and only contiguous context within two neighboring sentences. Candidate count
follows the source evidence; construction does not prefer a small PIU set.

```sh
uv run --no-sync python -m teamgym.data.medical.evidencebench.pipeline \
  --source-path /path/to/original-split-jsons \
  --source-revision <immutable-upstream-revision> \
  --instance-id <source-instance-id> --output-root /path/to/new-run \
  --model openai.gpt-5.5 --audit-model openai.gpt-5.5 \
  --participant-token-budget 40000 --max-input-tokens 180000 --max-output-tokens 8000 \
  --env-file /path/to/credentials.env
```

Acceptance requires candidate/task grounding checks, ALL/NONE/LOO boundaries,
full participant-packet leakage audits, and participant final-answer checks.
`--leakage-rounds 3` allows three removals and a final recheck. The same source
and exact packets are used in the three exported conditions. Source file and
record hashes, model settings and code/prompt hashes are recorded in
`run_manifest.json`; accepted summaries and common exports retain source lineage.

An accepted run writes `common_datapoints.jsonl` in the Hugging Face schema.
Model-visible PIU evidence contains exact source text. Contribution
labels, rubric links and constructor summaries remain hidden. Independent
solver/judge checks are finite evidence rather than proofs of necessity or
nonreconstructability. Use a new output directory for a fresh construction;
local response-cache keys bind rendered input and inference configuration.

The final participant solver receives the actual shared consumer's complete text
at `--participant-token-budget` (default 40,000). The input preflight ceiling is
`--max-input-tokens` (default 180,000), and response length is separately set with
`--max-output-tokens`. `--audit-model` selects the solver/judge, grounding and
leakage-check model independently of `--model`; its default follows `--model`.
`--verification-model` remains an alias. Export rechecks byte equality with that live
certification text and refuses truncation or material changes. Run-manifest v2
and common-export metadata retain `execution_identity`, covering shared LLM
clients, tokenizer/vocabulary, core schema/io, actual consumers, package versions,
lockfile and domain code/prompts. Cache reuse also binds this identity; dependency
drift is checked before calls and before acceptance. Accepted runs include an
export certification binding the summary, task, exact packets, all audit files
and manifest. Export requires successful complete ALL/NONE/LOO and participant
audits, and refuses missing or changed artifacts. Earlier uncertified outputs
require a fresh construction run.

The boundary judge receives answer criteria and clinical scope without the
condition, omitted candidate, proposed support sets or source-to-candidate map.
It evaluates semantic support from the actual displayed material, including
equivalent passages. Declared support ownership is used only afterward to
identify which claims a removal was intended to test.

Common exports carry numeric required-decision steps, the full hidden rubric and
error rules, and an explicit all-required pass policy. Grading-only metadata
releases keep the original producer identity and require unchanged actual actor
requests and source/public/private material; they are not new generation runs.
