"""Run the systems/settings listed below sequentially; no variant CLI."""
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "experiment_config.toml"
OUTPUT_DIR = ROOT / "logs/experiments"
REPETITIONS = 1
# Add more systems here. Relative paths are resolved against this script.
RUNS = [
    {"name": "minimal_sequential", "system": "smart_home_langgraph.py",
     "settings": {"dependency_graph": False, "parallel_agents": False, "parallel_actions": False,
                  "auto_prepare": False, "repair_enabled": False, "intent_analysis_enabled": False,
                  "validate_contracts": False, "check_outcomes": False,
                  "include_operations": False, "include_conditions": False,
                  "checker": "python", "scheduler_mode": "python", "max_workers": 1}},
    {"name": "extended", "system": "smart_home_langgraph.py"},
    {"name": "sequential", "system": "smart_home_langgraph.py",
     "settings": {"parallel_agents": False, "parallel_actions": False}},
    {"name": "llm_scheduler", "system": "smart_home_langgraph.py",
     "settings": {"scheduler_mode": "llm"}},
    {"name": "without_preparation_repair", "system": "smart_home_langgraph.py",
     "settings": {"auto_prepare": False, "repair_enabled": False}},
    {"name": "intent_analyser", "system": "smart_home_langgraph.py",
     "settings": {"intent_analysis_enabled": True}},
    {"name": "lean", "system": "smart_home_langgraph.py", "settings": {"checker": "lean"}},
]


def run_experiment(system_path, *, benchmark="homebench", benchmark_path=None, category="all",
                   limit=0, case_ids=(), output_dir=None, config_path=CONFIG_PATH, settings=None):
    output = Path(output_dir or OUTPUT_DIR / "single")
    output.mkdir(parents=True, exist_ok=True)
    system_path = Path(system_path).resolve()
    settings = settings or {}
    try:
        # Import the runner before adding the selected system's directory:
        # original/ also contains an old run_benchmark.py with a different CLI.
        if benchmark == "dexa":
            from run_benchmark import run_benchmark as run
        else:
            from external_benchmarks.run_benchmark import run_benchmark as run
        sys.path.insert(0, str(system_path.parent))
        spec = importlib.util.spec_from_file_location("_benchmark_system", system_path)
        system = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = system
        spec.loader.exec_module(system)
        model_config = sys.modules.get("llm_config")
        metadata = {"benchmark": benchmark, "system": str(system_path), "overrides": settings,
                    "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in system_path.parent.glob("*.py")},
                    "config_toml": Path(config_path).read_text(encoding="utf-8"),
                    "models": {key: getattr(model_config, key, None) for key in
                               ("DEFAULT_MODEL", "SAME_LLM_FOR_ALL_AGENTS", "ROLE_MODELS")},
                    "python": sys.version}
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if benchmark not in system.SUPPORTED_BENCHMARKS:
            raise NotImplementedError(f"{system_path} does not support {benchmark}; see EXPERIMENTS.md")
        options = {"benchmark": benchmark} if benchmark != "dexa" else {}
        return run(category, benchmark_path, system=system, config_path=config_path, limit=limit,
                   case_ids=case_ids, output_dir=output, **options, **settings)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}",
            "status": "unsupported" if isinstance(exc, NotImplementedError) else "failed"}, indent=2), encoding="utf-8")
        raise


def run_experiments(*, benchmark="homebench", benchmark_path=None, category="all", limit=0, case_ids=(),
                    runs=None, repetitions=REPETITIONS, config_path=CONFIG_PATH, output_dir=OUTPUT_DIR):
    runs = RUNS if runs is None else runs
    output = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / "experiment_config.toml")
    (output / "runs.json").write_text(json.dumps({"runs": runs, "repetitions": repetitions}, indent=2, default=str), encoding="utf-8")
    rows = []
    for repetition in range(1, repetitions + 1):
        for run in runs:
            destination = output / f"{run['name']}_{repetition}"
            destination.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--benchmark", benchmark,
                       "--system", str((ROOT / run["system"]).resolve()), "--config", str(output / "experiment_config.toml"),
                       "--category", category, "--limit", str(limit), "--output-dir", str(destination),
                       "--settings-json", json.dumps(run.get("settings", {}))]
            if benchmark_path:
                command += ["--benchmark-path", str(Path(benchmark_path).resolve())]
            for case_id in case_ids:
                command += ["--case-id", case_id]
            print(f"Running {run['name']} ({repetition})", flush=True)
            (destination / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            with (destination / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = {"run": run["name"], "repetition": repetition, "benchmark": benchmark,
                   "status": "completed" if result.returncode == 0 else "failed", "returncode": result.returncode}
            if (destination / "failure.json").exists():
                row.update(json.loads((destination / "failure.json").read_text(encoding="utf-8")))
            if (destination / "summary.json").exists():
                row.update(json.loads((destination / "summary.json").read_text(encoding="utf-8"))["overall"])
                if row.get("errors") and row["status"] == "completed":
                    row["status"] = "completed_with_errors"
            rows.append(row)
            # Persist after every system, including failures. JSON is authoritative.
            (output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  {row['status']}; details: {destination}", flush=True)
    print(f"Comparison: {output / 'comparison.csv'}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="homebench")
    parser.add_argument("--benchmark-path")
    parser.add_argument("--category", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--system", type=Path, help="Run only this system instead of the RUNS list")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--settings-json", default="{}", help=argparse.SUPPRESS)
    args = parser.parse_args()
    options = dict(benchmark=args.benchmark, benchmark_path=args.benchmark_path,
                   category=args.category, limit=args.limit, case_ids=args.case_ids,
                   config_path=args.config)
    if args.system:
        try:
            run_experiment(args.system, output_dir=args.output_dir, settings=json.loads(args.settings_json), **options)
        except Exception as exc:
            parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    else:
        run_experiments(output_dir=args.output_dir or OUTPUT_DIR, **options)
