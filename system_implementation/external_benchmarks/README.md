# External benchmark evaluation

Each benchmark owns its native environment and evaluation. Each participant owns
its prompts, planning and execution. ReAct, SAGE and DS-IA do not import MAS logic.

## Configure once

Use `../experiment_config.toml` from this directory, or `experiment_config.toml`
when running commands from `system_implementation/`.

- `model`: shared model and budgets; model names are `ollama:MODEL` or `openai:MODEL`.
- `mas`: MAS switches; `mas_models`: per-role model selection.
- `react`, `sage`, `dsia`: participant overrides.
- `benchmarks`: dataset paths and Python executables; simulator endpoint/timeouts.
- `judge`: SimuHome evaluation model; credentials stay in the environment.
- `experiments` and `runs`: run selection, repetitions, output and ablations.

CLI settings override TOML. Existing `LLM_MODEL`, `REACT_MODEL`, `SAGE_MODEL`,
`DSIA_MODEL`, provider/participant endpoint variables and benchmark path variables
remain supported. `--env-file .env` explicitly loads credentials for external
participants. Environment entries take precedence over TOML; explicit participant
CLI/JSON overrides take precedence over environment model settings.

All paths in the TOML are relative to `system_implementation/`. Python executable
names on PATH are accepted. Empty Python fields use the calling interpreter.
`--config` selects a different central TOML; older flat participant TOMLs also work.

## Extended DEXA

The project benchmark now uses the same `Episode` / `run_episode` boundary for
MAS, ReAct, DS-IA and SAGE. Run `python run_experiments.py --benchmark dexa --system react`
from `system_implementation/`. Its independent state/trace/lexical evaluator requires
no LLM judge. See `../../benchmark/README.md`. The historical 60-case reference
retains its direct MAS execution path. Native external-benchmark scoring is unchanged.

## HomeBench

Use an existing [HomeBench](https://github.com/BITHLP/HomeBench) checkout. The
configured directory must contain `test_data.jsonl` and `home_status_method.jsonl`.
No automatic download is performed.

```bash
python -m venv .venv-react
.venv-react/bin/python -m pip install -r external_benchmarks/requirements_react.txt
.venv-react/bin/python run_experiments.py --system react --benchmark homebench --limit 10 --output-dir logs/homebench/react-smoke
```

ReAct, DS-IA and SAGE's parent process need only these lightweight dependencies
for HomeBench. MAS additionally needs the project's `requirements.txt`.
HomeBench scores command strings using its released Counter evaluator. Feedback
uses a local device state model, not a physical simulator. Repeated/invalid
attempts and explicit rejected clauses enter the score. RGB tuples retain their
native formatting.

## SimuHome

Install the original [SimuHome](https://github.com/holi-lab/SimuHome) checkout in
its native Python 3.13+ environment, following its instructions. Start its server:

```bash
uv run simuhome server-start
```

From `system_implementation/`, install participant dependencies in that environment:

```bash
uv pip install --python ../SimuHome/.venv/bin/python -r external_benchmarks/requirements_react.txt
```

For MAS, additionally install `requirements.txt`. Set the dataset path and Python
executable under `benchmarks.simuhome`. `SIMULATOR_API_BASE_URL` can override the
configured endpoint. Run one experiment at a time per simulator process.

QT1 and all infeasible cases use the native LLM judge panel. Configure `judge.model`,
`judge.api_base`, temperature and seeds; the default panel has three votes.
Set `SIMUHOME_JUDGE_API_KEY` or `OPENAI_API_KEY`. A local compatible server can use
a nonempty placeholder key if it does not authenticate. Missing judges leave
scores pending. Judge calls are reported separately from participant calls.

Native tools exclude simulator administration (`add_device`, `remove_device`,
`set_tick_interval`). Documentation search may require upstream embeddings setup.
Evaluator read-back is not counted as an agent action. Accepted workflows do not
by themselves prove future physical success.

## Home Assistant

Use [Home Assistant Datasets](https://github.com/allenporter/home-assistant-datasets)
and its native pytest/Synthetic Home runtime. From `system_implementation/`:

```bash
python3.14 -m venv .venv-homeassistant
.venv-homeassistant/bin/python -m pip install -e ../home-assistant-datasets
.venv-homeassistant/bin/python -m pip install -r external_benchmarks/requirements_react.txt -r external_benchmarks/requirements_homeassistant.txt
git clone https://github.com/allenporter/home-assistant-synthetic-home.git ../home-assistant-datasets/home-assistant-synthetic-home
```

Reuse an existing custom-component checkout. For MAS, also install `requirements.txt`.
Set the dataset and Python paths under `benchmarks.homeassistant`. Each sentence
gets a fresh native home. Device/room context, exposed state and native Assist
instructions are supplied to the participant. Calls run on Home Assistant's event
loop; native state/response/tool-call checks determine success. Empty expected
change sets additionally require unchanged state for every participant.

## Original SAGE

```bash
git clone https://github.com/SAIC-MONTREAL/SAGE.git ../SAGE
git -C ../SAGE checkout 5f43df0a7de2383e4f082a69c591d56d613d1b2f
python3.10 -m venv .venv-sage
.venv-sage/bin/python -m pip install -r external_benchmarks/requirements_sage_worker.txt
```

Set `sage.sage_path` and `sage.sage_python`. The worker's old dependencies belong
in its separate environment. No SmartThings account or MongoDB is needed here.
The original coordinator, API planner, grounding loop and command tools make the
decisions; a service bridge forwards their calls to native benchmark tools.
Personal memory, photographs and unrelated external services are unavailable.

The original command tool sends successful POSTs twice; both calls are retained
and recorded. HomeBench can penalize this repetition. Label results **original
SAGE core with native benchmark backend, cold memory, no images/personal services**.
This does not reproduce the original SmartThings benchmark or its published score.

## DS-IA adaptation

`--system dsia` uses the original intent prompt and early-rejection rule, followed
by independent generation of one ordered command batch. Analysis is not forwarded
to generation. Grounding checks native names/schemas; direct SimuHome accesses
also check devices, endpoints, capabilities and writability. Native backends
validate values, Home Assistant targets and nested workflows.

SimuHome first reads a public inventory and device structures. These reads and
their latency are recorded. The batch executes once without repair/replanning.
A final native-benchmark model call answers from actual observations and errors;
it cannot generate more actions. Unavailable parts become explicit rejections.

JSON batches, native validation and response synthesis replace the original
HomeBench string/few-shot interface. This is an architecture adaptation based on
[DS-IA's code](https://github.com/xxscy/DS-IA-smart/blob/main/code/model_test_ia.py),
not execution of its original Transformers script or reproduction of paper scores.
Values discovered only during execution cannot be substituted into later calls.

## Run and resume

After configuring paths, environments and models:

```bash
python run_experiments.py --system dsia --benchmark all --output-dir logs/comparison/dsia
python run_experiments.py --system dsia --benchmark all --output-dir logs/comparison/dsia --resume
```

Replace `dsia` with `react`, `sage` or `mas`. Omit `--system` to execute the configured
`runs` matrix. Set `experiments.system` to choose a default participant instead.
`--limit 10` limits each benchmark; `0` means the complete selected dataset.
Category/case-ID filters apply only to a single benchmark. DEXA is selected
separately with `--benchmark dexa`.

`--benchmark all` runs the three external benchmarks in separate sequential
processes. Paths/Python executables can also be overridden using
`--homebench-path`, `--simuhome-path`, `--homeassistant-path` and matching
`--*-python` flags. `--benchmark-path` applies to a single benchmark.

The standalone entry point remains available:

```bash
python -m external_benchmarks.run_benchmark --system react --benchmark homebench --model ollama:gemma2 --limit 10 --output-dir logs/homebench/react-smoke
```

Each run records effective settings, source/data hashes, model usage, traces,
errors and scores. Comparisons keep benchmark-specific metrics separate. Backend
failure stops the affected run after saving; other benchmarks still run.
Exit status 2 indicates setup/backend failure; 3 indicates incomplete evaluation.
Resume reuses completed cases and refuses changed settings, data or source code.
Gold actions and hidden evaluation goals never enter participant prompts.
