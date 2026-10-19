# External benchmark integrations

This directory separates external comparison benchmarks from our own DEXA
benchmark and the MAS implementation.

| File | Role |
|---|---|
| `run_benchmark.py` | Selects an external benchmark and invokes its adapter |
| `adapters/homebench.py` | Imports native HomeBench houses/cases, maps actions and evaluates output |
| `agent_registry_homebench.py` | HomeBench-specific device grouping, contracts and local state effects |

These Python files are our integration code. The HomeBench dataset, native method
definitions and evaluation protocol come from [BITHLP/HomeBench](https://github.com/BITHLP/HomeBench).
The command comparison follows its published evaluator. The local state model
here is not an official physical simulator or an additional benchmark score.

Download the dataset separately, then run from the project root:

```bash
python external_benchmarks/run_benchmark.py --benchmark homebench --benchmark-path ../HomeBench/dataset --limit 10
```

Or execute the configurations in `run_experiments.py`:

```bash
python run_experiments.py --benchmark homebench --benchmark-path ../HomeBench/dataset --limit 10
```

There is no SimuHome adapter yet. The unchanged original is supported on our DEXA
inputs only; it cannot be scored on HomeBench through this interface.
