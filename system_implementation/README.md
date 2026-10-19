
# Smart Home Multi-Agent System Implementation

Multi-agent system for smart home scenarios using LangGraph and google/gemma-2-9B.


## Experiment configuration

Execution defaults are plain values in `experiment_config.toml` (Python 3.11+).
Loading, overrides and recovery behavior live in `smart_home_langgraph.py`.
`run_experiments.py` contains the seven experiment runs in its `RUNS` list:
system file and only the settings that differ from the defaults.
There is no `--variant` argument. Model selection and credentials remain in
`llm_config.py` / `.env`.

The project's `run_benchmark.py` handles only our DEXA benchmark. External
adapters and their device catalogues live in `external_benchmarks/`.
System selection and experiment metadata are handled in `run_experiments.py`.
The small standalone CLIs reuse argument definitions from `smart_home_langgraph.py`.

See [EXPERIMENTS.md](EXPERIMENTS.md) for commands, outputs, additional systems,
and the limitations of the original implementation for HomeBench and SimuHome.

```bash
python smart_home_langgraph.py --no-repair
python external_benchmarks/run_benchmark.py --benchmark homebench --benchmark-path ../HomeBench/dataset --limit 10
python run_experiments.py --benchmark homebench --benchmark-path ../HomeBench/dataset --limit 10
```

The current seven-run matrix targets HomeBench and uses a minimal sequential
baseline instead of the original. SimuHome and the Home Assistant Dataset will
be integrated later; the project benchmark remains available separately.

The DEXA dataset from the original archive is included in `benchmark/`.
HomeBench data must be downloaded separately. Lean sources are included;
retain your existing compiled `lean/.lake/` directory or run `cd lean && lake build`.

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

## Running the System

### Interaction
```bash
   jupyter notebook smart_home_langgraph.ipynb
```
or 
```bash
   python smart_home_langgraph.py
```
Type commands like "Set a 20-minute timer and dim the lights". 
Type `quit` to exit.

### Evaluation

For detailed benchmark evaluation instructions, see [`../benchmark/README.md`](../benchmark/README.md).

#### Quick Evaluation Workflow

1. **Run benchmark:**
   ```bash
   python run_benchmark.py simple
   python run_benchmark.py moderate  
   python run_benchmark.py complex
   ```
   or
   ```bash
   python run_benchmark.py all
   ```
2. **Review logs:**
   - Check `logs/` directory
   - Each test case has its own log file

3. **Evaluate according to criteria:**
   - Intent Recognition & Device Activation
   - Collaboration Success
   - Overall Completion

4. **Calculate metrics:**
```
   Success Rate = (Complete cases / Total cases) × 100%
```
Logs are saved in `logs/` directory.

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
## Replacing the previous package

Remove the old `adapters/homebench.py` and top-level `agent_registry_homebench.py`
when adopting this folder layout. They moved to `external_benchmarks/`; no aliases
are needed. An old `experiment_config.py` is also unused. Preserve your `.env`
and the compiled contents of `lean/.lake/`.

If you copied an earlier generated package, delete its obsolete `experiment_support.py`.
