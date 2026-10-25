
# Smart Home Multi-Agent System Implementation

Multi-agent system for smart home scenarios using LangGraph and google/gemma-2-9B.

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

## Running the System

### Interaction
```bash
   jupyter notebook smart_home_langgraph.ipynb
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

3. **Update task planner** to recognize new device type

## Related Documentation

- [Benchmark Documentation](../benchmark/README.md) - Test cases and evaluation criteria
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview) - Framework reference
- [Ollama Documentation](https://ollama.ai/docs) - Model serving
- [Gemma Documentation](https://ai.google.dev/gemma) - Model details

