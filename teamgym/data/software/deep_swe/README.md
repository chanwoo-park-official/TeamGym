# DeepSWE construction

The public constructor loads an upstream task, validates public checks, screens
instruction-only repair attempts, collects base code and runtime observations,
and constructs functional PIUs. It then qualifies the evidence, audits the
distributed packets, and exports a typed repository-repair instance.

Run from the repository root. Install the locked environment, then download
the upstream source and build the unmodified official verifier.

```bash
uv sync --locked --python 3.11 --extra construction --extra data --extra test
uv run --no-sync python -m teamgym.data.software.deep_swe.prepare \
  --source-dir .external_datasets/deep-swe-source \
  --task-id TASK_ID --platform linux/amd64 \
  --receipt artifacts/deepswe-preparation.json
```

The preparation receipt records the upstream commit, task-file hashes and image
identities. Pass its `verifier_tag` to `--verifier-image` below. Omitting the option
selects the legacy image tag. Pass `--revision COMMIT_SHA` to prepare
the same source revision again. An existing source directory must have the
expected remote and a clean working tree; symlink destinations are rejected.

```bash
DEEPSWE_VERIFIER_CONCURRENCY=1 uv run --no-sync teamgym-generate swe/deep-swe \
  --tasks-root /path/to/deep-swe/tasks \
  --task-id TASK_ID \
  --verifier-image VERIFIER_IMAGE \
  --env-file /path/to/.env \
  --qualification-min-passes 2 --qualification-repetitions 3 \
  --concurrency 1 \
  --output-root /path/to/new-output
```

Build the verifier image using the selected upstream task's `tests/Dockerfile`.
Docker must be running and both base and verifier images must be available.
On Apple Silicon, record the actual `linux/amd64` emulation configuration.
Model credentials and source datasets remain external inputs; another local
research checkout is not required. Pass configuration explicitly with
`--constructor-model`, `--actor-model`, `--provider`, `--base-url`,
`--api-key-env`, `--reasoning-effort`, and `--env-file`. The default qualification
actor is GPT-5.5 (`gpt55`). `--runtime-image` is an explicit, recorded override;
an implicit runtime-image environment override is not used.

Run this command from the repository root. The current pipeline records the
root `uv.lock` and uses that working directory for its stage subprocesses.
The serial example limits memory pressure on an 8 GiB Docker VM; the CLI's
general diagnostic-concurrency default is two.

For the Responses actor, the adapter's configured temperature is not forwarded
by the repair request builder. Reproduce the captured request and pinned adapter,
rather than interpreting the configuration value as a transmitted parameter.
Retain configured settings, actual request fields and response-reported values
separately; a response-reported temperature does not establish effective sampling.

## Protocol and acceptance

Each repair episode allows five atomic `replace`, `write`, or `delete` edit
transactions, optional status-only checks from the frozen public catalog, and
early finish. Rejected edits consume a turn. Test files and test-runner
configuration cannot be edited. The official hidden verifier runs once after
termination; matching the reference patch is not required.

1. **Instruction-only:** three valid attempts must all fail before construction
   continues. Infrastructure failures do not count as failed repairs.
2. **Complete evidence:** `--qualification-repetitions N` sets the number of
   valid repair attempts per bundle (default 3), without failure-based early
   stopping. `--qualification-min-passes K` sets the required official verifier
   successes (default 2). Require `1 <= K <= N`. The policy is recorded in the
   frozen execution manifest and enforced again by the export loader. These
   settings do not change the three instruction-only or participant-only attempts.
3. **Revision:** the improvement target defaults to K. A higher target may be
   requested with `--revision-target-passes`; it must lie between K and N.
   An earlier qualifying bundle is
   retained if later completed revisions perform worse. All attempts remain recorded.
4. **Functional contribution:** targeted leave-one-PIU-out runs are executed and
   recorded. Their failure is not a formal necessity proof and is not a universal
   acceptance threshold.
5. **Distribution:** at least two functional PIUs are assigned one per participant.
   Full-Context adds exclusive non-PIU base code and may duplicate previously
   collected non-owned observations, at most one per foreign PIU and three total.
6. **Packet checks:** complete views are reviewed for reconstruction and format
   clues. Each participant's public-plus-private view is also tested in both
   collaborative modes. A participant successfully completing the whole repair
   prevents collaborative acceptance.
7. **Export:** only a fully validated bundle receives `decision=accepted` and
   an `export/instance.json`. The loader verifies bound validation receipts and
   reconstructs three model-input conditions without private PIU identifiers,
   generated summaries, or constructor rationales.

The constructor may inspect reference patches and hidden-test feedback. Actor
evidence contains exact pinned base code and independently reviewed, repeatable
base execution results. Probe commands and outputs require public provenance;
execution on the base checkout alone does not establish non-disclosure.

## Results and reproducibility

`report.json` distinguishes screening, qualification, packet-audit and
participant-only rejection from execution errors and acceptance. Normal process
termination can report a valid rejection; it does not certify an instance.
Confirmed context-window failures are valid zero outcomes. Provider/container
failures without a valid outcome are recorded separately.

The semantic packet auditor defines `issue=true` as a blocking risk in every
category, including plausible completion of the entire repair. A rationale
that the packet cannot support the whole repair belongs to `issue=false` for
that category. The reviewer sees constructor-only comparison evidence but must
assess only the actual participant packet and the declared source-restricted
protocol; it must not assume an unrestricted read of unseen repository files.
Allowed duplicated runtime observations may reveal their recorded base behavior.
They become a reconstruction concern when the combined visible evidence restores
the substantive missing functional PIU, rather than merely revealing isolated
foreign facts. Neutral source formatting and relevance alone are not artificial
salience cues. Every positive category still rejects the audit: contradictory
flags are not automatically flipped or excused by the receipt parser. A revised
auditor requires a separately versioned review; earlier rejected results remain
unchanged.

`run_identity.json` records source, code/prompts, resolved image identities,
model configuration, budgets, and the environment lock. The run directory also
contains exact source provenance, qualification/LOO/participant episodes, model
requests and responses, patches, verifier results, and final consumer inputs.
Release the accompanying manifest and verification artifacts with the export.

`--skip-existing` resumes only matching content-bound stages. Changed immutable
inputs archive the previous run; missing or modified stage products invalidate
subsequent shortcuts. Diagnostic episodes separately validate their invocation
and complete artifact receipts. Mid-run input drift prevents acceptance.

The dedicated loader preserves repair semantics rather than converting the
task into a scalar-answer QA record:

```python
from pathlib import Path
from teamgym.data.software.deep_swe.export import load_export

instance = load_export(Path("/path/to/export/instance.json"))
request = instance.render_model_input("single_agent_all_piu")
```

Finite model checks support sufficiency and functional contribution under this
fixed protocol. They do not prove necessity over every possible repair. Source
visibility also affects which edits are admissible; LOO outcomes should retain
that interpretation. Internal `riu` fields remain for artifact compatibility.

Background packing can use `background_policy='discovered'` (the compatible
default) or `'neutral_repository'`. The latter reads whole public source files
for a declared Go, Python, TypeScript, JavaScript or Rust background language
from a fresh, network-disabled container at the immutable runtime image and
pinned base commit. Fixed extension and public test-file/path patterns are bound
in the corpus metadata. Native construction supplies the task's declared
language; direct pack calls may set `background_language` or infer it only from
unambiguous supported canonical-source extensions. Unsupported or ambiguous
language fails explicitly. The collector excludes **every original discovery
path**, including all unallocated lines of those files, before packing any
background. Selection uses sorted paths and fixed limits of 12 MB and 600 files;
it does not consult the instruction, reference patch, hidden tests or task cohort.
Canonical PIUs still use their exact original discovered source. Every owned
observation is retained. Distractors use only collected, candidate-allocated
foreign observations: stable rotation followed by greedy selection takes at
most one observation per foreign PIU and three total. Unallocated pool outputs
are ineligible. Final visible ordering follows the original collection; selected
foreign owners and excluded pool IDs remain hidden. This preserves allowed
duplication without aggregating a foreign PIU's entire runtime signature.
An explicit `observation_background_exclusions` plan can remove a particular
remaining foreign distractor after all source allocation. Specs contain only
`participant_id`, `observation_id` and a nonempty `reason`. Owned or unallocated
observations, nonvisible extras, duplicate specs and ambiguous visible matches
are rejected. No replacement, refill or source redistribution follows removal;
the exact removed output hash, foreign owner and token changes are hidden
provenance. This is a versioned packet revision requiring fresh final audits.

The collector checks Git HEAD and clean status before and after reading, and
verifies each full file against its Git blob identity and SHA256. There is no
cross-run corpus cache. The hidden distribution manifest retains the complete
selected raw corpus, excluded discovery paths, selection limits and policy,
file/blob hashes and skipped-file ledger. None of those provenance fields enters
the private source packet. Empty eligible source is an explicit failure, with no
fallback to discovery hotspots. Corpus metadata binds language and test-path policy.
The 40k packet cap and exact-source audit still
apply. The policy name describes source selection; it does not certify semantic
safety. Every resulting actual packet still requires the normal whole-packet
semantic review and participant-only repair checks before accepted export.

## Execution map

The public command enters [`teamgym.data.cli`](../../cli.py), resolves
`swe/deep-swe` through [`registry.py`](../../registry.py), and invokes
[`pipeline.main`](pipeline.py). The pipeline launches packaged stage modules
with `python -m`; it forwards the selected model, provider, credentials-file
path, runtime image and budgets explicitly.

The stage numbers used in filenames are internal identifiers. In particular,
instruction-only screening runs before source discovery.

| Order | Implementation | Input and work | Result or gate |
|---|---|---|---|
| 0. Resolve inputs | `pipeline._main`, `source.load_task`, `integrity.task_identity/image_identity/code_identity` | Read the task instruction/configuration and declared base commit. Resolve runtime and verifier tags to immutable local image IDs. Bind code, prompts, lockfile and options. | `run_identity.json`; invalid or missing inputs stop execution. |
| 1. Prepare public checks | `checks.plan_public_checks` or `validate_public_checks` | Plan from the public instruction, tracked filenames and public build/test manifests. Run checks on clean base containers and audit their provenance. | `public_checks.json` and base-validation receipts; checks must pass on the base. |
| 2. Instruction-only screening | `candidate.assemble_instruction_only_candidate`, `diagnostic_runs`, `patch_episode` | Supply the public instruction and common check catalog without code fragments or runtime observations. | Three valid independent failures are required; a success rejects screening. |
| 3. Discover and extract base code | `discovery.main`, `stages.build_deepswe_fragment_pool`, `fragments.extract_complete` | The constructor uses private reference/test materials to locate relevant files and dependencies. Deterministic extraction then produces exact line spans, verifies hashes and preserves readable spans for large declarations. | `discovered_code.json`, `fragment_pool.json`; an empty source pool stops construction. |
| 4. Collect observations | `observation_proposals.main`, `observations.collect_observation`, `provenance.audit_solver_visible`, `evidence.usable_observations` | Propose commands, verify their public source references, execute each twice on the base, compare results and checkout identity, and audit the visible command/output. Repair broken probe setup within the configured attempts. | Validated observations; setup errors, non-reproducible output and private solution material cannot become actor evidence. |
| 5. Construct functional PIUs | `construction.build_constructor_request/construct_in_stages/validate_plan`, `candidate.assemble_functional_candidate` | Read exact fragments and constructor-only evidence in bounded stages; group selected fragments/observations by coherent functional contract. Validate IDs, nonempty units and exclusive ownership. | A frozen candidate with constructor metadata separated from actor-visible material. |
| 6. Qualify and revise | `pipeline._main`, `diagnostic_runs`, `construction.qualification_feedback`, `qualification.QualificationPolicy` | Flatten all selected evidence and run N repair attempts. If the improvement target is unmet, use private verifier/patch feedback to revise the evidence and, when needed, expand source discovery. | Require at least K successes from N valid attempts (defaults K=2, N=3). Completed rounds are compared by success count; an optional higher improvement target cannot lower K. |
| 7. Distribute and review packets | `distribution.finish_distribution`, `full_context.pack`, `packet_audit.audit_evaluation_bundle` | Construct All-PIU, PIU-Only and Full-Context views. Review every complete private view against hidden ownership/comparison material. | Any blocking semantic or presentation finding prevents acceptance. The CLI defaults to neutral-repository background packing. |
| 8. Diagnose missing/private evidence | `candidate.leave_one_out_conditions/reduce_candidate`, `diagnostic_runs`, `pipeline._episode_receipts` | Run each targeted PIU removal and three standalone attempts for each participant view in each collaborative mode. | LOO is diagnostic. A participant completing the full repair rejects collaborative acceptance. |
| 9. Freeze and consume | `export.make_validation_receipt/freeze_export/load_export/render_model_input` | Bind all completed validation groups to the candidate, bundle and execution identity. Reconstruct the model input for each information condition. | `export/instance.json` only for accepted instances; consumer evidence must equal the validated packet. |

The native pipeline runs these steps itself. The standalone finalizer and
handoff/delta auditors under `scripts/data_generation/` are optional tools for
explicitly completing or checking content-bound phases, not required steps in
every construction. An execution error is reported as an error. If a revision
is interrupted, the native command does not automatically export an earlier
round before returning; resume or an explicit validated finalization is needed.

This constructor creates collaborative input views and tests each participant
view in isolation. Running the complete coordinator-led team dialogue and its
final repair is a separate downstream evaluation step.

## Prompt map and information boundary

[`prompts.load_prompt`](prompts.py) resolves text templates relative to the
installed package. User payloads are assembled as JSON in Python; they are not
additional static user-template files.

| Purpose | System/developer instructions | User payload assembled in |
|---|---|---|
| Public-check planning | [`public_check_planner_system.txt`](../../../prompts/software/deep_swe/public_check_planner_system.txt) | `checks.plan_public_checks`: public instruction, language, base commit, public filenames/manifests, repair feedback. |
| Code discovery | [`code_discovery_system.txt`](../../../prompts/software/deep_swe/code_discovery_system.txt) | `discovery.main`: instruction, private reference patch/test patch/test IDs, repository tree and discovered code. |
| Observation proposals | [`observation_proposer_system.txt`](../../../prompts/software/deep_swe/observation_proposer_system.txt) | `observation_proposals.main`: instruction, private reference, exact base fragments/inventory and probe-repair feedback. |
| Initial PIU construction | [`functional_riu_constructor_system.txt`](../../../prompts/software/deep_swe/functional_riu_constructor_system.txt) plus [`functional_riu_staged_system.txt`](../../../prompts/software/deep_swe/functional_riu_staged_system.txt) | `construction.build_constructor_request/construct_in_stages`: private construction materials, source inventory, exact requested evidence and remaining budgets. |
| PIU revision | [`functional_riu_revision_system.txt`](../../../prompts/software/deep_swe/functional_riu_revision_system.txt) plus the staged protocol | The same constructor, with prior candidate and private qualification/patch feedback. Regrouping an unchanged flattened evidence set is rejected as a revision. |
| Public provenance review | `provenance.AUDIT_SYSTEM`; public-check role adds `PUBLIC_BASELINE_AUDIT_SYSTEM` | `provenance.audit_solver_visible`: public grounding and proposed/observed visible artifact; private reference/test data is comparison material only. |
| Whole-packet review | `packet_audit.AUDIT_SYSTEM` and its four category criteria | `packet_audit.audit_request`: the actual private packet, common protocol, and hidden owned/foreign comparison evidence. |
| Repair actor | `stages.patch_episode.SYSTEM` and `TOOL` | `patch_episode.packet_for/main`: visible evidence, remaining edit/output/check budgets, prior actions and status-only check feedback. The export consumer uses the same base system/tool definitions. |

`model_actions.py` supplies the provider-neutral `ModelActions` transport.
Its `resolve_actor_config` is also used by diagnostic execution identities,
so recorded model/provider/endpoint defaults match the client configuration.
Registered Anthropic models default to Bedrock's native API, registered chat
models use Chat Completions, and GPT/custom model names use Responses.
Explicit provider and URL options take precedence. Unsupported repair-actor
providers fail before a client or model request is constructed.

There are six text templates and three modules with additional inline audit or
actor instructions. No constructor rationale or hidden answer is supplied to a
repair actor. Ordinary actor packets contain exactly:

```json
{
  "instruction": "public task text",
  "code_fragments": [
    {"path": "relative/source", "line_start": 1, "line_end": 2, "text": "exact base source"}
  ],
  "observations": [
    {"argv": ["program", "argument"], "stdout": "real output", "stderr": "", "exit_code": 0}
  ],
  "public_check_catalog": [
    {"id": "public-check-id", "argv": ["program", "argument"]}
  ]
}
```

The actor calls `perform_action` with `edit` or `finish`. An edit transaction
contains `replace`, `write` or `delete` operations plus optional public-check
IDs. The runtime enforces source visibility, unique replacement anchors,
protected test paths, atomic rollback and edit/check budgets. Public-check
stdout/stderr is recorded for diagnostics but only status is returned to the
actor. Optional evidence-citation instrumentation is a separate explicit
diagnostic mode; it is disabled in the native construction path.

The packet reviewer checks four risks: foreign functional-core equivalence,
reconstruction from combined code/observations, artificial formatting clues,
and credible completion of the entire repair. The shared public instruction
is common input, so repeating its specification is not private disclosure.
Each `issue=true` is blocking; uncertainty is recorded in the rationale.

## Docker runtime and model execution

Model API requests run in the host Python process. Containers execute source
reads, public checks, observation commands and the hidden verifier. The pipeline
does not automatically pull/build task images: prepare them before generation.
The upstream task defines the base environment and `tests/Dockerfile` defines
the verifier image. Inspect that Dockerfile's `COPY` paths to choose its build
context. Pass the verifier image explicitly; pass `--runtime-image` when using
an explicitly prepared runtime override. Exact resolved image IDs enter the
execution manifest.

All task containers request `linux/amd64` and have network disabled. On an Apple
Silicon host the Docker VM may be ARM64 with amd64 emulation; record the VM and
emulator configuration separately from the task image architecture.

| Container use | Resource limits in current code | Source |
|---|---|---|
| Public baseline checks / observation collection | CPU and memory from the source task runtime declaration; working directory `/app` | `checks.py`, `observations.py` |
| Source reads / patch public checks | 2 CPUs, 8 GiB memory, 256 PIDs, dropped Linux capabilities and no-new-privileges | `stages.patch_runtime.docker_python` |
| Official verifier execution | 2 CPUs, 8 GiB memory, 256 PIDs, dropped Linux capabilities and no-new-privileges | `stages.patch_runtime.verify` |
| Neutral background source collection | 1 CPU, 1 GiB memory, `/app`, no pull | `full_context.collect_neutral_repository` |

Each public-check run applies the submitted patch in a fresh base container.
For final verification the patch is mounted read-only and the verifier log
directory is mounted writable; the entrypoint is `/tests/test.sh`. Containers
are removed after execution. The CLI does not currently expose the hardcoded
2-CPU/8-GiB patch/verifier limits as parameters; they are distinct from task
runtime-declared limits used by observation collection.

`--concurrency` controls diagnostic jobs and defaults to two.
`DEEPSWE_VERIFIER_CONCURRENCY` independently limits concurrent verifier runs
with host file locks; unset/zero adds no slot limit. On an 8 GiB Docker VM use
`--concurrency 1` and `DEEPSWE_VERIFIER_CONCURRENCY=1` as in the command above.
This changes scheduling, not the per-episode evidence/action budgets. Prepare
and record a larger VM if the chosen task requires more resources.

The optional `runtime_repairs/*.Dockerfile` recipes are not selected or built
automatically. `numba.Dockerfile` restores a clean base source file;
`igel.Dockerfile` pins NumPy/SciPy; `verifier.Dockerfile` installs
`pytest-timeout` and adds a 60-second timeout to matching pytest commands in
`test.sh`. That last recipe changes verifier execution policy even though the
test patch and IDs are copied unchanged. An explicitly chosen repair image
must be documented as such and bound by image identity; do not describe its
execution settings as identical to an unmodified upstream verifier.

## Defaults, outputs, and repository boundary

| Setting | Native pipeline default |
|---|---|
| Constructor/auditor | `gpt-5.4` |
| Repair actor | `gpt55` / GPT-5.5 |
| Provider / endpoint | Bedrock OpenAI-compatible Responses API |
| Reasoning effort | `medium` |
| Edit turns | 5, fixed by the repair protocol |
| Complete-evidence attempts | 3 by default, set by `--qualification-repetitions` |
| Required complete-evidence successes | 2 by default, set by `--qualification-min-passes`. The pipeline and export enforce the recorded policy. |
| Repair output budget / per-call output cap | 32,000 / 16,000 tokens |
| Public-check budget | 15 selections per repair episode |
| Private source packet cap | 40,000 tokens |
| Proposed observation count / collection repetitions | 10 / 2 |
| Maximum construction rounds / LOO repetitions | 3 / 1 |

Constructor staged-reading budgets are separate from the repair actor's
budgets: see `construction.construct_in_stages`. The complete source pool is
not silently equated to the subset read in one API request.

New outputs live below the explicit `--output-root/TASK_ID/`, including public
checks, source/observation artifacts, frozen candidates, qualification and LOO
episodes, participant packet audits, raw model requests/responses, patches,
verifier results, `report.json`, and accepted exports. A normal exit can still
mean screening/qualification/packet rejection; inspect `report.json.decision`.

The contribution consists of source modules, prompt templates, tests, README
files, shared client/input/scoring dependencies and environment definitions.
Git ignores generated output directories, downloaded `.external_datasets`,
local `.env` credentials, `.venv` environments, and Python build products.
Place `--output-root` under an ignored directory such as `outputs/`, or outside
the checkout; a custom directory is not ignored merely because it contains
generated JSON.
Docker images and the local Docker/Colima VM are external runtime resources;
they are not transferred by `git push`.
