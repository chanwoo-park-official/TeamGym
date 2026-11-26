# Installation environments

One repository provides the shared schema, construction, evaluation, and RL
integration. Select optional dependencies for the task; importing the core
dataset loader requires only Python's standard library.

The reference pins target CPython 3.11 on Linux x86_64. They record exact
third-party versions, including transitive dependencies. Other platforms or
Python versions can use the extras in `pyproject.toml`, but have not been
verified with these pins. The distribution keeps the name `rllm`; install it
from this checkout and install the modified VERL fork from `./verl`.

## Dataset use, construction, and API evaluation

### uv setup

The project includes `.python-version` for Python 3.11 and a committed `uv.lock`.
The lock uses the versions from `lightweight-py311.txt` for the construction and
CPU-test dependencies. From the repository root, install the environment and
inspect the available constructors.

```bash
uv sync --locked --extra construction --extra data --extra test
uv run --no-sync teamgym-generate --list
uv run --no-sync teamgym-generate finance/findver --help
```

`uv sync` creates or updates `.venv`. Run commands with `uv run --no-sync` after
that installation to use the prepared environment. The `construction` extra
installs the generation dependencies, `data` adds source-table and Parquet
tools, and `test` adds the CPU test toolchain.

```bash
uv run --no-sync python -m pytest -q tests/finance tests/medical tests/software
```

To refresh the lock from the existing reference pins, run the following command
from the repository root. Review the lockfile diff before committing it.

```bash
python3 - <<'PY'
from pathlib import Path
import subprocess

command = ["uv", "lock", "--python", "3.11", "--default-index", "https://pypi.org/simple"]
for line in Path("requirements/lightweight-py311.txt").read_text().splitlines():
    requirement = line.strip()
    if requirement and not requirement.startswith("#"):
        command.extend(["--upgrade-package", requirement])
subprocess.run(command, check=True)
PY
```

### pip setup

Reading the provided JSONL files needs only Python's `json` module. To use the
project loader, install `.`. The optional `data` extra adds pandas, PyArrow,
and Hugging Face downloads. The `llm` extra adds the OpenAI-compatible API
client and tokenizer. Neither extra installs PyTorch, CUDA, vLLM, or Ray.

From the repository root, create a fresh environment:

```bash
python3.11 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -c requirements/lightweight-py311.txt pip setuptools wheel

# Choose the installation for your task:
python -m pip install --no-build-isolation -c requirements/lightweight-py311.txt .
# API evaluation:
python -m pip install --no-build-isolation -c requirements/lightweight-py311.txt ".[llm]"
# Dataset reconstruction and API evaluation:
python -m pip install --no-build-isolation -c requirements/lightweight-py311.txt ".[data,llm,construction]"
python -m pip check
```

The constraints file also pins the CPU test toolchain; a constraints entry
does not cause that package to be installed. The basic loader installation
therefore stays free of optional runtime dependencies.

The default tests use the lightweight environment and mocked model clients:

```bash
python -m pip install --no-build-isolation -c requirements/lightweight-py311.txt ".[data,llm,construction,test]"
python -m pytest -q
```

This runs `tests/`, including an offline wheel build and fresh installation.
It makes no paid model calls and launches no GPU work.

The portable shell-command tests invoke `bash` and `jq`. The CPU suite was also
checked in the `python:3.11-slim` Linux image after installing `jq`.
On macOS, the system Bash can fail these checks on empty arrays under `set -u`.
Use a newer Bash on `PATH` when running the full shell-command suite there.

Run a smaller smoke test with the following command.

```bash
python -m pytest -q tests/test_core_dataset.py \
  tests/test_fully_collaborative_interaction.py tests/test_shared_written_answers.py
```

Installed command checks also run without contacting model providers:

```bash
teamgym-generate cuad --help
teamgym-generate sara finalize --help
teamgym-evaluate --help
teamgym-judge --help
teamgym-validate-frozen --help
teamgym-train --help
```

Actual generation, model interaction, and semantic judging require model
credentials and can incur API costs. Source datasets and generated outputs
are separate inputs. SARA's `prepare` stage additionally uses Apptainer and
a SWI-Prolog container image; see the
[SARA construction instructions](../scripts/data_generation/slurm/sara_target_sliced/README.md).

## RL environment

Use a separate environment for the pinned PyTorch 2.8.0 / CUDA 12.8 /
vLLM 0.11.0 stack. The CUDA runtime dependencies are included in
`rl-cu128-py311.txt`. Running GPU jobs requires a compatible NVIDIA driver;
building FlashAttention also requires the matching CUDA toolkit with `nvcc`.

```bash
python3.11 -m venv .venv-rl
source .venv-rl/bin/activate
python -m pip install -r requirements/rl-cu128-py311.txt
python -m pip install --no-build-isolation --no-deps -r requirements/rl-flash-attn.txt
python -m pip install --no-build-isolation --no-deps ./verl .
python -m pip check
```

The extension pin is separate so a resolver check never attempts a CUDA
source build. The generic upstream VERL requirements cover more environments;
these files pin the retained TeamGym path, including pandas/parquet, plotting,
the hosted judge client, and the test toolchain.

Both training entry points can preview plans without GPUs, local model files,
dataset reads, or API calls. The generic command accepts constructed datasets:

```bash
teamgym-train --prepared-data-dir /path/to/prepared-data \
  --profile short_context_smoke --reward-config written_holistic_score \
  --training-mode joint --model /path/to/local-model \
  --run-dir outputs/teamgym/custom-preview --dry-run
```

`teamgym-train --check-data` requires the RL preparation dependencies, including
CPU PyTorch and pandas/parquet support, but no GPU or model. It accepts the same
profile, reward, and split arguments without requiring `--model` or `--run-dir`.
The paper entry point retains its fixed recipes:

```bash
teamgym-paper --regime legal-joint \
  --prepared-data-dir /absolute/path/to/prepared/legal/full_context \
  --model /absolute/path/to/qwen-snapshot \
  --run-dir outputs/teamgym/preview --dry-run
```

In an environment with the RL dependencies, the following checks also avoid
training and model calls:

```bash
python -m teamgym.rl.train --cfg=job
python -m pytest -q tests teamgym/rl/tests
```

The [training guide](../teamgym/rl/README.md) describes actual data
checks, training, and checkpoint evaluation. The paper recipe uses four H200
GPUs. A GPU smoke test must still verify rollouts, reward calls, optimizer
updates, and checkpoint save/reload in the new environment.

## Provenance and updating the pins

[`reference-environment.json`](reference-environment.json) records the
observed runtime versions and local code revisions used for CPU verification.
It is an observation, not a claim that the historical environment resolves
cleanly. The RL profile preserves the main framework versions while repairing
the NumPy, Click, and Rich constraints, selecting an OpenCV version compatible
with NumPy, and filling missing audio-extra dependencies. GPU training under
the repaired profile remains unverified until the smoke test above.

The fresh lightweight environment passed `pip check` and all 213 default
tests with no PyTorch, vLLM, Ray, VERL, or Transformers installed. An isolated
environment that reused the existing GPU libraries and applied the profile's
dependency changes matched all 210 pins, passed dependency-metadata checks,
and passed all 318 CPU tests. This verifies the adjustments without replacing
the original environment; a complete CUDA installation from scratch and an
actual training run have not been tested.

The `.in` files select the local project extras and RL packages. The `.txt`
files pin their resolved third-party dependencies; local RLLM/VERL packages
are installed from the checkout. To reproduce the resolution, use uv 0.13.8
in a separate tooling environment and run from the repository root:

```bash
uv pip compile requirements/lightweight.in \
  -c requirements/lightweight-py311.txt --python .venv-data/bin/python \
  --python-platform x86_64-unknown-linux-gnu --no-python-downloads \
  --default-index https://pypi.org/simple --no-build-isolation \
  --no-emit-package rllm --no-annotate -o requirements/lightweight-py311.txt

uv pip compile requirements/rl.in \
  -c requirements/rl-cu128-py311.txt --python .venv-data/bin/python \
  --python-platform x86_64-unknown-linux-gnu --no-python-downloads \
  --default-index https://pypi.org/simple --no-build-isolation \
  --no-emit-package rllm --no-emit-package verl --no-annotate \
  --exclude-newer 2026-11-16 \
  -o requirements/rl-cu128-py311.txt
```

To change versions, update the direct inputs and remove the affected old
constraints before resolving. Recheck the lightweight installation and CPU
tests; changes to the RL stack also need a GPU smoke test. The current pins
are version constraints rather than a distribution archive with wheel hashes.
