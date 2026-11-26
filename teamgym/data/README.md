# Construction registry

The paper's four domains have stable command prefixes:

| Key | Domain |
| --- | --- |
| `legal` | Legal |
| `finance` | Finance |
| `medical` | Medical |
| `swe` | Software Engineering |

`registry.py` stores construction metadata as strings and imports no pipeline
modules. The command imports the selected module when it is invoked. Root
help and listing therefore work without construction, model, or GPU dependencies.

```bash
teamgym-generate --list
teamgym-generate legal/cuad --help
teamgym-generate legal/sara finalize --help
```

This checkout registers CUAD, MAUD, Casebook, MSLR, SARA, and MultiLexSum under
`legal/`. Their existing aliases (`cuad`, `maud`, `casebook`, `mslr`, `sara`,
`multilexsum`) remain available. Finance registers `finance/findver` and
`finance/finlongdocqa`. Medical registers `medical/evidencebench` and
`medical/mlcr`. Software Engineering registers `swe/swe-atlas` and
`swe/deep-swe`. Each also accepts its dataset name as a short alias.
The [construction guide](../../scripts/data_generation/README.md#finance-medical-and-software-construction)
describes their source inputs. Finance RL preparation and training live in
`teamgym/rl/`.

## Adding a construction pipeline

1. Add the domain's construction module, conventionally under
   `data/DOMAIN/DATASET/pipeline.py`. Its callable `main()` reads its arguments
   from `sys.argv`, as the current legal pipelines do. An adapter can delegate
   to an existing domain implementation.
2. Add a `PipelineSpec` to `PIPELINE_SPECS` in `registry.py`. The qualified
   command is derived from the domain and dataset fields. The existing CUAD
   registration has this form:

   ```python
   PipelineSpec(
       domain="legal",
       dataset="cuad",
       module="teamgym.data.legal.cuad.pipeline",
       aliases=("cuad",),
   )
   ```

3. Supply aliases when preserving an existing command name. Dataset names can
   repeat across domains through qualified names; aliases must be unique
   across the registry. Duplicate commands and unknown domains are rejected
   before any construction module is imported.

The dispatcher forwards pipeline arguments unchanged and restores `sys.argv`
after execution. The registry controls command routing. Datapoint schemas,
source dataset identifiers, scoring, and RL settings remain in their existing
modules. Keep domain-specific dependencies inside the selected construction
module so registry discovery stays lightweight.

## Connecting construction to evaluation

Follow the [dataset interface guide](../core/README.md) for output fields,
participant information visibility, and the distinction between finite-choice
scores and written-answer judgments. It also describes how normalized evaluation
records connect to prepared RL splits. Registering a construction command supplies
command routing; domain integration also needs the appropriate record mapping
and scoring rules.
