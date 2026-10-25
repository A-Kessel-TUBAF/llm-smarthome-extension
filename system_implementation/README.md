# Smart-home systems and evaluation

MAS with optional intent analysis, domain agents, contracts, Lean admission and repair.
External comparisons: ReAct, original SAGE core and a DS-IA architecture adaptation.

## Configuration

Edit **`experiment_config.toml`**. No Python edits are needed for normal runs.

| Section | Settings |
| --- | --- |
| `model` | Shared model, endpoint, generation limits, seed and timeouts |
| `mas_models` | Shared/per-role MAS models |
| `mas` | Execution, checks, repair, intent analysis and Lean executable |
| `assurance` | QED_LEAF audit before runs with the Lean checker (`off`, `record`, `require`) |
| `react`, `sage`, `dsia` | Participant overrides; SAGE checkout and worker |
| `experiments` | System, benchmark, category, case limit, repetitions and output folder |
| `benchmarks` | Dataset paths, Python environments and native runtime settings |
| `judge` | SimuHome evaluation model, endpoint and seeds |
| `assurance` | QED_LEAF audit before MAS runs with the Lean checker: `off`, `record`, `require` |
| `runs` | MAS ablation matrix; each entry overrides only its differences |

Paths in this file are relative to `system_implementation/`. A Python executable
can also be a command such as `python3.14`. Empty benchmark Python paths use the
calling interpreter. Each selected environment needs the participant's dependencies.

`model` settings apply to all participants. Add the same keys to `react`, `sage`
or `dsia` for participant-specific values. MAS uses `mas.recursion_limit`
instead of the external participants' `episode_timeout`. Set `mas_models.same_for_all = false`
and add entries under `mas_models.roles` for separate MAS role models.

CLI flags override configuration. Existing model/path environment variables remain
supported and override TOML values; explicit participant CLI/JSON settings win.
`--config another.toml` or `SMARTHOME_CONFIG` selects another central file.
Older flat participant TOMLs remain accepted.

Keep API keys in the environment or a local `.env`; see `.env.example`.
MAS reads this project's `.env`. Other participants load it only with
`--env-file .env`. Remove old model/endpoint entries from `.env` when you want
the TOML to control them.

## Run

From `system_implementation/`, after installing `requirements.txt`:

```bash
python smart_home_langgraph.py
python run_experiments.py --system react --benchmark homebench --limit 10
python run_experiments.py --system dsia --benchmark all
```

Configure benchmark paths and Python environments first. SimuHome needs its
running native server and, for QT1/infeasible cases, a configured judge.
SAGE also needs its original checkout and separate Python 3.10 environment.
Installation and evaluation details: [external_benchmarks/README.md](external_benchmarks/README.md).

With no `--system`, the runner executes the `runs` matrix. Set `experiments.system`
to select one system by default, or leave it empty for the matrix. `--limit 0`
runs all selected cases. `--repetitions` overrides the configured count.
Use a new output directory for a new run; repeat the same command and output
directory with `--resume` to continue an external benchmark run.

For the planner-only ablation on all external benchmarks:

```bash
python run_experiments.py --system mas --benchmark all --settings-json '{"domain_agents_enabled":false,"intent_analysis_enabled":false,"scheduler_mode":"python"}' --output-dir logs/comparison/planner-only
```

The default matrix includes this as `planner_only`. For a single benchmark, the
standalone CLI also accepts `--no-domain-agents`. `mas.domain_agents_enabled`
controls the same switch in TOML; leave its default enabled for matrix comparisons.

For the complete extended DEXA benchmark (automatic evaluation, no judge):

```bash
python run_experiments.py --system mas --benchmark dexa
```

`--benchmark all` runs all four benchmarks: DEXA, HomeBench, SimuHome and Home Assistant.
DEXA runs only for MAS; ReAct, SAGE and DS-IA skip it. `--dataset original|extension|all` and
`--dimension` select the DEXA cases, also with `--benchmark all`. DEXA has no automatic
task-success score; its `summary.json` holds the execution metrics. Resume is supported for the
three external benchmarks; a finished DEXA run is kept and an unfinished one is repeated. Native
scores from different benchmarks are reported separately in `comparison.json` and `comparison.csv`.

### Lean checker and QED_LEAF

```bash
(cd lean && lake build)                                   # checker, proofs, axiom audit
python run_experiments.py --system mas --benchmark all --settings-json '{"checker":"lean"}'
```

The default matrix contains `lean`, the full MAS with `checker = "lean"`. Before every
MAS run with the Lean checker, `assurance.qed_leaf` audits the checker with QED_LEAF
(see [assurance/qed_leaf/README.md](assurance/qed_leaf/README.md)):

| Mode | Behaviour |
| --- | --- |
| `off` | No audit |
| `record` (default) | Reuse the audit result for unchanged Lean sources, policy, reference and QED_LEAF revision, or run the audit again; store it as `qed_leaf.json` in the run folder and in the `qed_leaf` column of `comparison.csv` |
| `require` | As `record`, but refuse the run unless the audit passed |

The first audit clones the private QED_LEAF repository, so it needs Git access to it; without
access the result is `inconclusive`. The audit takes a few minutes; later runs reuse its result.

### Windows and local Ollama

- `model.model` must name an installed tag (`ollama list`). The default `ollama:gemma2:latest` is the
  same model as `gemma2:9b`; `ollama cp gemma2:9b gemma2:latest` adds the missing tag, or set `LLM_MODEL`.
- Thinking models such as Gemma 4 reason before answering by default. `model.think = false` turns
  that off for MAS, ReAct, SAGE and DS-IA (about 5x fewer tokens per call); see `experiment_config.dgx.toml`.
- If every model call fails with `tokenize error ... Invalid API Key`, the environment variable
  `LLAMA_API_KEY` is set. Ollama's internal llama.cpp server then requires that key from Ollama
  itself. Start Ollama without it, e.g. `env -u LLAMA_API_KEY ollama serve` in Git Bash.
- Without a dedicated GPU, Ollama runs on the CPU. `OLLAMA_IGPU_ENABLE=1` lets it use an
  integrated GPU. A 9B model then still needs minutes per MAS call, so use `--limit` for
  local smoke tests and an API model or a GPU machine for full runs.

## Files

| File/directory | Responsibility |
| --- | --- |
| `configuration.py` | Shared TOML loading; no agent dependencies |
| `smart_home_langgraph.py` | MAS planning, validation, scheduling and execution |
| `agent_registry.py`, `home_defaults.py` | Default capabilities, contracts and home data |
| `llm_config.py` | MAS model clients and usage |
| `model_client.py` | Shared HTTP client, HTTP-500 retries and usage aggregation |
| `run_experiments.py` | Sequential experiment processes and comparison results |
| `run_benchmark.py` | Extended DEXA runner, shared episodes and automatic evaluation |
| `../benchmark/` | Independent DEXA environment, reference predicates and evaluator |
| `external_benchmarks/` | Native benchmark adapters and independent system wrappers |
| `lean/` | Allocation checker; build with `lake build` |
| `assurance/qed_leaf/` | Optional proof audit; see its README |
| `original/` | Preserved reference implementation |


### Installation

If you don't have Ollama installed:

1. **Install Ollama:**
```bash
   brew install ollama
   # or download from https://ollama.com/download
```

2. **Download model:**
```bash
   ollama pull gemma2
```

3. **Install Python dependencies:**
```bash
   pip install -r requirements.txt
```

### Using an online API instead of Ollama (optional)

The model is chosen in `llm_config.py` via env vars (default: local `ollama:gemma2`).
Copy `.env.example` to `.env` in this folder and set e.g.:
```bash
   LLM_MODEL=openai:gpt-4o-mini
   OPENAI_API_KEY=sk-...
```
Any `provider:model` supported by LangChain's `init_chat_model` works 
(`anthropic:...`, 
`openai:...` + `OPENAI_BASE_URL` 
for OpenAI-compatible services, ...).


### Real devices through Home Assistant (optional)

`turn_on` and `turn_off` can switch real devices through a [Home Assistant](https://www.home-assistant.io)
instance; all other operations, and all devices you do not list, stay simulated. Create a
long-lived access token on your Home Assistant profile page and add to `.env`:
```bash
   HA_URL=http://homeassistant.local:8123
   HA_TOKEN=...
   HA_ENTITIES=tv_lr=media_player.living_room_tv,light_lr_ceiling=light.living_room
```
`HA_ENTITIES` maps device IDs of `home_defaults.py` to Home Assistant entities (`media_player`,
`light` or `switch`). Then run
```bash
   python smart_home_langgraph.py --home-assistant
```
For every request, the power state of the listed devices is read from Home Assistant. After
switching, the state Home Assistant reports is recorded, so a device that does not switch within
`HA_TIMEOUT` seconds (default 15) fails the usual outcome check. In Python, use
`run_request(text, **home_assistant_setup(HomeAssistant.from_env()))`. Many TVs cannot be switched
on over the network once fully off; Home Assistant then needs Wake-on-LAN or HDMI-CEC for that TV.
`python -m unittest test_home_assistant` tests the connection against a fake Home Assistant.

### Evaluation

The extended DEXA benchmark automatically scores all selected cases through the
shared `run_episode` interface. MAS, ReAct, DS-IA and SAGE use the same environment,
fault injector and evaluator. No manual scoring or LLM judge is needed.

```bash
python run_experiments.py --benchmark dexa --system mas --suite core --output-dir logs/dexa/core
python run_experiments.py --benchmark dexa --system react --suite robustness --output-dir logs/dexa/robustness
```

See [`../benchmark/README.md`](../benchmark/README.md) for selection filters,
metric definitions, automatic text checks and the unchanged historical reference.

## System Architecture

### Components

1. **Intent Analyzer**
   - Extracts user intents from natural language
   - Identifies key modifiers (time, location, manner, etc.)
   - Calculates complexity score

2. **Task Planner**
   - Decomposes intents into device-level tasks
   - Creates task queue with detailed actions
   - Does not predict collaboration in task queue

3. **Device Agents** (8 specialized agents):
   - **clock** - Alarms, timers, stopwatch
   - **calendar** - Add schedule and appointment, Schedule Information (time, location, etc.)
   - **search_engine** - Information retrieval, recipes, weather
   - **tv_display** - Visual content display
   - **fridge** - Food inventory tracking
   - **lighting** - Light control and atmosphere creation
   - **thermostat** - Temperature control and atmosphere creation
   - **audio_system** - Music playback and volume control, atmosphere creation


## Technical Details

### Software Stack
* **Language:** Python 3.11+
* **LLM Engine:** Ollama 0.13.5+
* **Model:** `google/gemma-2-9b`
* **Orchestration:** LangGraph 1.0.5
* **Chain Framework:** LangChain 1.2.0 + langchain-ollama 1.0.1

### Hardware Environment (Tested)
* **Device:** MacBook Air (M4)
* **RAM:** 16GB Unified Memory

---

## Development

### Running in Jupyter Notebook

Use `smart_home_langgraph.ipynb` for interactive development:

1. Open in Jupyter:
```bash
   jupyter notebook smart_home_langgraph.ipynb
```

2. The first cell handles dependency installation:
```python
   %%capture --no-stderr
   %pip install -r requirements.txt
```

3. Run cells sequentially to execute the system

### Adding New Agents

To add a new device agent:

1. **Define agent function** in `smart_home_langgraph.py`:
```python
   def new_device_agent(state: SmartHomeState) -> Command:
       # Agent logic here
       return Command(     
       )
```

2. **Add to StateGraph:**
```python
   builder.add_node("new_device_agent", new_device_agent)
```

3. **Update agent registry** to recognize new device type

## Related Documentation

- [Benchmark Documentation](../benchmark/README.md) - Test cases and evaluation criteria
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview) - Framework reference
- [Ollama Documentation](https://ollama.ai/docs) - Model serving
- [Gemma Documentation](https://ai.google.dev/gemma) - Model details


