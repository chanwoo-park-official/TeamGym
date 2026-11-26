# Legal Post-qualification Audit

This submission converts row-level qualification results into canonical task-level evidence. It is deliberately
separate from dataset generation and does not make additional model calls.

## Why this stage exists

The six partition modes are alternate allocations of the same task. Single-agent all-PIU and leave-one-PIU-out
checks should therefore be interpreted at the canonical task level, not as unrelated one-shot judgments for each
partition. The final experimental dataset also needs matched partition-mode coverage.

## Outputs

The audit writes to `postqualification_audit/` under the construction run root.
The default is `generated/legal_construction` under the calling directory.
Set `TEAMGYM_DATAGEN_ROOT` to audit an existing run such as
`generated/0712_datagen_final`. The submission uses the active Python
environment or `TEAMGYM_PYTHON`.

- `summary.json`: provenance, profile definitions, and aggregate counts.
- `task_evidence.jsonl`: one row per canonical task with partition coverage and qualification stability.
- `rui_evidence.jsonl`: one row per canonical task and PIU with repeated ablation outcomes.
- `profile_sensitivity.tsv`: strict, robust, and screening profile counts by dataset.
- `report.md`: compact human-readable summary.

Profiles are sensitivity analyses, not automatic labels:

- `strict`: complete six-mode grid, 6/6 all-PIU passes, and 6/6 ablation failures for every PIU.
- `robust`: complete grid, at least 5/6 all-PIU passes, at least three observations per PIU, and at least 80% ablation failures.
- `screening`: complete grid, at least 4/6 all-PIU passes, at least two observations per PIU, and at least two-thirds ablation failures.

## Submit

```bash
bash scripts/data_generation/slurm/legal_postqualification_audit/submit.sh
```

After reviewing the report, the next submission should fill only missing task/PIU repeats and then freeze one
matched six-mode manifest. This avoids rerunning the 2,699 judgments already available.
