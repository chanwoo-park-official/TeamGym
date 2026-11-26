# FinDVer construction

The free-response constructor groups the selected source QAs by filing, maps complete native contexts, consolidates overlapping evidence, constructs a new task and hidden rubric, and independently audits the public objective for answer disclosure before checking grounding, public-only insufficiency, full-evidence sufficiency, and PIU removal boundaries. It distributes source blocks, audits collective reconstruction, removes leaking surrounding blocks once, repeats the semantic audit, then attempts the final task from each participant's actual budgeted local context plus common information. An accepted record requires all applicable gates.

Native QA knowledge, labels and explanations remain construction-only. Blind solvers receive source text without PIU IDs or generated feature labels. Judges receive the same public/common information and source evidence as the solver, with the hidden oracle supplied separately.

Use a new empty output directory every time. Unverified resume or overwrite is refused. The manifest records source/config/model/code/prompt hashes, and complete JSON/JSONL files are replaced atomically. Interrupted runs retain `running` status and do not certify completed construction. The CLI exits 2 when no scenario is accepted; inspect `candidate_results.jsonl` for rejected/error outcomes.

```bash
uv run --no-sync teamgym-generate finance/findver \
  --qa-path /path/to/FinDVer/data/testmini.json \
  --reports-root /path/to/FinDVer/financial_reports \
  --source-split testmini --subset all \
  --output-root /path/to/new-output \
  --env-file /path/to/credentials.env \
  --model gpt-5.5 \
  --candidate-limit 1 --plan-attempts 3 --audit-attempts 3
```

`--subset` selects QA categories; `--source-split` selects the split. The default split is `testmini`. The default participant context budget is 20,000 tokens and is recorded in the manifest. Models perform finite audits and solver attempts; these do not constitute a proof for all possible readers or algorithms. Numeric values still require the model grounding gate and case-specific independent review; a generic formal arithmetic verifier is not claimed.

Each run yields at most one accepted task per filing. The source evidence determines the PIU count.
Use `common_datapoints.jsonl` for evaluation, or `MODE/common_datapoints.jsonl` after
redistribution. Both use the Hugging Face schema.

The objective-disclosure gate distinguishes legitimate scope years/dates and item numbering from requested source values, conclusions and answer cardinalities. Explicit answer-with-unit/cardinality disclosures are also rejected deterministically, including word-form counts.

## Redistribute one frozen task

The frozen-allocation command accepts a native accepted `scenarios.jsonl` whose
hash matches its constructor `manifest.json`. Use a new empty output directory.
It repeats whole-packet review after every removal, then attempts the complete
answer from every participant's actual full-context and PIU-only observations.
Common and deliberately duplicated PIUs are allowed. Only unallocated PIUs count
as leakage. Accepted distributions receive distinct instance IDs and retain the
original task, oracle, source links and the new audit records.

```sh
uv run --no-sync python -m teamgym.data.finance.findver.rematerialize_allocations \
  --input /path/accepted/scenarios.jsonl --reports-root /path/financial_reports \
  --output-root /path/new-allocation-run --mode one_each --mode redundant_player \
  --audit-model openai.gpt-5.5 --env-file /path/credentials.env \
  --participant-context-token-budget 40000 --packet-repair-rounds 3
```

All five allocation names are accepted. Omitting `--mode` requests the four
additional distributions. The native constructor and redistribution default to
GPT-5.5. The audit model and participant context budget remain explicit options.
The original packet pass may remove surrounding blocks once;
`--packet-repair-rounds` bounds additional repair passes after that initial
removal. Every last removal is followed by a complete recheck.
