# Medical construction

`evidencebench/` builds tasks from annotated paper sentences.
`mlcr/` builds tasks from external clinical summaries and source questions.
Each package exports accepted builds through `evaluation.py`.

The MLCR source directory contains `cases_summaries.jsonl` with `case_uuid`,
`summary_file`, and `summary_text`, and `questions.jsonl` with `case_uuid`,
`prompt_id`, `question`, and `answer`. The loader creates checksum-backed
sentence and paragraph spans. See the
[construction guide](../../../scripts/data_generation/NONLEGAL_CONSTRUCTION.md) for build commands.

MLCR accepts `--min-source-questions N` to set the minimum number of original
QAs used in a generated task. The default is 2 and N must be at least 1.
Set N to 1 to allow a task to use multiple evidence spans from one source QA.
The run report records the configured minimum.

## Distribution-specific audits and exports

The two Medical commands accept `--allocation-modes all`, or a list such as
`--allocation-modes overlap common_anchor`. Construction and the
ALL/NONE/LOO tests run once, producing `qualified_construction.json`. The task,
PIUs, grounding/boundary audits and original source are frozen before any
distribution is audited. The default command without this option retains its
existing one-each pipeline.

The shared allocator provides the five conditions from Appendix D, including `one_each`:

| Mode | Assignment | Minimum PIUs for this collaboration protocol |
|---|---|---|
| `one_each` | One private PIU per participant | 2 |
| `overlap` | Neighboring participants share some PIUs | 3 |
| `redundant_player` | Add a participant with no assigned PIUs | 2 |
| `common_anchor` | One PIU is common; the remainder is private | 3 |
| `bottleneck` | One participant receives all but one PIU | 2 |

`all` requests all five modes, including the base. An inadmissible distribution is
recorded as `skipped`; it never falls back to another mode. The modes may have
different participant counts. The same public task, answer criteria, PIU texts
and upstream source are retained across modes.

Redistribute a qualified construction, or a previously accepted certified run,
without generating another task:

```sh
uv run --no-sync python -m teamgym.data.medical.allocations \
  --dataset mlcr --source-run /path/to/qualified-run \
  --output-root /path/to/new-allocation-run --mode all \
  --audit-model gpt-5.5 --env-file /path/to/credentials.env \
  --participant-token-budget 40000 --max-input-tokens 180000 \
  --max-output-tokens 12000 --leakage-rounds 3 --seed 20260913
```

`--construction-only` on either native command creates the qualified task/PIUs
without running any distribution. An older failed packet run without a
construction certification must be reconstructed before redistribution. A
certified task whose one-each packet failed may still be redistributed: the
failed packet verdict is not reused.

Each distribution gets its own `run_manifest.json`, model cache, audit traces,
and `report.json`. Foreign-PIU audits exclude both owned and common PIUs, allowing
intentional overlap. Only noncanonical filler may be removed, with a final
recheck after the last allowed repair. Unallocated information recoverable from
protected evidence rejects that distribution; the allocator never changes the
requested distribution by merging owners.

Common anchor evidence uses the runtime's actual common-information channel.
Participant final-answer checks and exports retain snapshots of the shared
consumer's common and private material. Both channels count toward the declared
participant evidence budget. A participant with no private PIUs is preserved.
The standard PIU-only transform preserves private ownership and common PIUs.

Only a distribution that passes leakage and every participant final-answer audit
gets `MODE/common_datapoints.jsonl`. Its export certification binds the complete
report, source snapshot and run manifest, and export verifies the live consumer
material again. The root `allocation_results.json` lists accepted, rejected,
skipped and execution-error outcomes; the root `common_datapoints.jsonl` contains
accepted modes only. Use a new empty output directory for a fresh audit run.

The audit model is configurable with `--audit-model` (also spelled
`--verification-model`); source generation uses the native command's `--model`.
All model checks are finite evidence rather than proofs over all possible answers.
