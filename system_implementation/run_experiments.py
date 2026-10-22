"""Run one system or the configured matrix on selected benchmarks."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from external_benchmarks.common import BENCHMARKS, SYSTEMS, write_json
from configuration import CONFIG_PATH, activate_config, config_path as configured_path, load_config

ROOT = Path(__file__).resolve().parent


def run_experiment(system_path, *, benchmark="homebench", benchmark_path=None, category="all",
                   limit=0, case_ids=(), output_dir=None, config_path=None, settings=None, resume=False):
    activate_config(config_path)
    output = Path(output_dir or Path(configured_path(load_config()["experiments"]["output_dir"])) / "single")
    output.mkdir(parents=True, exist_ok=True)
    settings = settings or {}
    if benchmark == "dexa" and resume:
        raise ValueError("Resume is supported only for external benchmarks.")
    try:
        if benchmark != "dexa":
            from external_benchmarks.common import create_system
            from external_benchmarks.run_benchmark import run_benchmark
            name = str(system_path)
            if name in SYSTEMS or (":" in name and not name.endswith(".py")):
                participant = create_system(name, settings, config_path)
            else:
                path = Path(system_path).resolve()
                sys.path.insert(0, str(path.parent))
                spec = importlib.util.spec_from_file_location("_benchmark_system", path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                if benchmark not in getattr(module, "SUPPORTED_BENCHMARKS", set()):
                    raise NotImplementedError(f"{path} does not support {benchmark}.")
                participant = create_system(module, settings, config_path)
            summary = run_benchmark(category, benchmark_path, benchmark=benchmark, system=participant,
                limit=limit, case_ids=case_ids, output_dir=output, resume=resume)
            if (output / "failure.json").exists():
                (output / "failure.json").unlink()
            return summary
        system_path = ROOT / "smart_home_langgraph.py" if str(system_path) == "mas" else Path(system_path).resolve()
        config_path = config_path or os.getenv("SMARTHOME_CONFIG") or CONFIG_PATH
        # Load this runner before original/ can shadow it.
        from run_benchmark import run_benchmark as run
        sys.path.insert(0, str(system_path.parent))
        spec = importlib.util.spec_from_file_location("_benchmark_system", system_path)
        system = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = system
        spec.loader.exec_module(system)
        model_config = sys.modules.get("llm_config")
        metadata = {"benchmark": benchmark, "system": str(system_path), "overrides": settings,
                    "source_sha256": {**{p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in system_path.parent.glob("*.py")},
                                      **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (ROOT / "external_benchmarks").rglob("*.py")}},
                    "config_toml": Path(config_path).read_text(encoding="utf-8"),
                    "models": {key: getattr(model_config, key, None) for key in
                               ("DEFAULT_MODEL", "SAME_LLM_FOR_ALL_AGENTS", "ROLE_MODELS")},
                    "python": sys.version}
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if benchmark not in system.SUPPORTED_BENCHMARKS:
            raise NotImplementedError(f"{system_path} does not support {benchmark}; see external_benchmarks/README.md")
        return run(category, benchmark_path, system=system, config_path=config_path, limit=limit,
                   case_ids=case_ids, output_dir=output, **settings)
    except Exception as exc:
        if not isinstance(exc, (FileExistsError, ValueError)):
            (output / "failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}",
                "status": "unsupported" if isinstance(exc, NotImplementedError) else "failed"}, indent=2), encoding="utf-8")
        raise


def run_experiments(*, benchmark="homebench", benchmark_path=None, category="all", limit=0, case_ids=(),
                    runs=None, repetitions=None, config_path=None, output_dir=None,
                    benchmark_paths=None, benchmark_pythons=None, resume=False, timestamped=True):
    if runs is None or any(run.get("config") is None for run in runs):
        activate_config(config_path)
    defaults = load_config()["experiments"]
    repetitions = defaults["repetitions"] if repetitions is None else repetitions
    runs = load_config()["runs"] if runs is None else runs
    if limit < 0 or repetitions < 1:
        raise ValueError("limit must be nonnegative and repetitions must be positive.")
    if benchmark == "all" and (benchmark_path or category != "all" or case_ids):
        raise ValueError("For --benchmark all use --homebench-path/--simuhome-path/--homeassistant-path; "
                         "--category and --case-id are only supported for a single benchmark.")
    output = Path(output_dir or configured_path(defaults["output_dir"])).resolve()
    if timestamped:
        output /= datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=resume)
    benchmarks = BENCHMARKS if benchmark == "all" else (benchmark,)
    configured = load_config()["benchmarks"]
    paths = {b: os.getenv({"homebench": "HOMEBENCH_PATH", "simuhome": "SIMUHOME_PATH", "homeassistant": "HOMEASSISTANT_DATASET_PATH"}[b])
             or configured_path(configured[b]["path"]) for b in BENCHMARKS}
    paths.update(benchmark_paths or {})
    if benchmark_path:
        paths[benchmark] = benchmark_path
    selected_pythons = {b: os.getenv(b.upper() + "_PYTHON") or configured_path(configured[b]["python"], executable=True)
                        for b in BENCHMARKS if os.getenv(b.upper() + "_PYTHON") or configured[b]["python"]}
    selected_pythons.update(benchmark_pythons or {})
    pythons = {}
    for name, executable in selected_pythons.items():
        found = shutil.which(os.path.expanduser(str(executable)))
        # Preserve venv symlinks; resolving them selects the base interpreter.
        pythons[name] = str(Path(found).absolute()) if found else str(executable)
    manifest = {"runs": runs, "repetitions": repetitions, "benchmark": benchmark,
                "category": category, "limit": limit, "case_ids": list(case_ids),
                "benchmark_paths": {b: str(Path(paths.get(b) or benchmark_path).resolve())
                                    for b in benchmarks if paths.get(b) or benchmark_path},
                "benchmark_pythons": {b: str(pythons.get(b) or sys.executable) for b in benchmarks}}
    manifest = json.loads(json.dumps(manifest, default=str))
    if resume and (output / "runs.json").exists():
        if json.loads((output / "runs.json").read_text()) != manifest:
            raise ValueError("Resume refused: experiment selection/settings changed.")
    else:
        write_json(output / "runs.json", manifest)
    if any(run.get("config") is None and
           (str(run["system"]) == "mas" or str(run["system"]).endswith(".py")) for run in runs):
        # External participants never load or require the MAS configuration.
        source = Path(config_path or os.getenv("SMARTHOME_CONFIG") or CONFIG_PATH)
        target = output / "experiment_config.toml"
        if resume and target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("Resume refused: MAS configuration changed.")
        if source.resolve() != target:
            shutil.copyfile(source, target)
    rows = []
    for repetition in range(1, repetitions + 1):
        for selected_benchmark in benchmarks:
            for run in runs:
                folder = output / selected_benchmark if benchmark == "all" else output
                destination = folder / f"{run['name']}_{repetition}"
                destination.mkdir(parents=True, exist_ok=resume)
                name = str(run["system"])
                wrapped = name in SYSTEMS or (":" in name and not name.endswith(".py"))
                command = [str(pythons.get(selected_benchmark) or sys.executable), str(Path(__file__).resolve()), "--benchmark", selected_benchmark,
                           "--system", name if wrapped else str((ROOT / name).resolve()),
                           "--category", category, "--limit", str(limit), "--output-dir", str(destination),
                           "--repetitions", "1",
                           "--settings-json", json.dumps(run.get("settings", {}))]
                selected_config = str(Path(os.getenv("SMARTHOME_CONFIG") or CONFIG_PATH).resolve())
                participant_config = run.get("config")
                if participant_config is None and (not wrapped or name == "mas"):
                    participant_config = output / "experiment_config.toml"
                if participant_config:
                    command += ["--config", str(Path(participant_config).resolve())]
                path = paths.get(selected_benchmark) or benchmark_path
                if path:
                    command += ["--benchmark-path", str(Path(path).resolve())]
                if resume:
                    command.append("--resume")
                for case_id in case_ids:
                    command += ["--case-id", case_id]
                print(f"Running {run['name']} on {selected_benchmark} ({repetition})", flush=True)
                write_json(destination / "command.json", command)
                try:
                    with (destination / "console.log").open("a" if resume else "w", encoding="utf-8") as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                env={**os.environ, "SMARTHOME_CONFIG": selected_config})
                    row = {"run": run["name"], "repetition": repetition, "benchmark": selected_benchmark,
                           "status": {0: "completed", 3: "incomplete"}.get(result.returncode, "failed"), "returncode": result.returncode}
                except OSError as exc:
                    row = {"run": run["name"], "repetition": repetition, "benchmark": selected_benchmark,
                           "status": "failed", "error": str(exc), "returncode": 2}
                if (destination / "failure.json").exists():
                    row.update(json.loads((destination / "failure.json").read_text(encoding="utf-8")))
                if (destination / "summary.json").exists():
                    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
                    row.update(summary["overall"])
                    if (row.get("errors") or row.get("evaluation_errors")) and row["status"] == "completed":
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
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    selected, _ = pre.parse_known_args()
    activate_config(selected.config)
    defaults = load_config()["experiments"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=(*BENCHMARKS, "all", "dexa"), default=defaults["benchmark"])
    parser.add_argument("--benchmark-path")
    for name in BENCHMARKS:
        parser.add_argument(f"--{name}-path", help="Data/checkout path for " + name)
        parser.add_argument(f"--{name}-python", default=os.getenv(name.upper() + "_PYTHON"),
                            help="Python executable with this benchmark's dependencies")
    parser.add_argument("--category", default=defaults["category"])
    parser.add_argument("--limit", type=int, default=defaults["limit"])
    parser.add_argument("--repetitions", type=int, default=defaults["repetitions"])
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--system", default=defaults["system"] or None, help="mas, react, sage, dsia, module:factory, or a legacy MAS Python file")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--settings-json", default="{}", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.env_file:
        from external_benchmarks.run_benchmark import load_env
        load_env(args.env_file)
    options = dict(benchmark=args.benchmark, benchmark_path=args.benchmark_path,
                   category=args.category, limit=args.limit, case_ids=args.case_ids,
                   config_path=args.config)
    try:
        if args.resume and not args.output_dir:
            raise ValueError("--resume requires the original --output-dir.")
        settings = json.loads(args.settings_json)
        if not isinstance(settings, dict):
            raise ValueError("--settings-json must be a JSON object.")
        paths = {b: getattr(args, b + "_path") for b in BENCHMARKS if getattr(args, b + "_path")}
        pythons = {b: getattr(args, b + "_python") or configured_path(load_config()["benchmarks"][b]["python"], executable=True)
                   for b in BENCHMARKS if getattr(args, b + "_python") or load_config()["benchmarks"][b]["python"]}
        options["benchmark_path"] = paths.get(args.benchmark) or args.benchmark_path
        executable = pythons.get(args.benchmark)
        executable = shutil.which(executable) or executable if executable else None
        current_python = not executable or Path(executable).absolute() == Path(sys.executable).absolute()
        if args.system and args.benchmark != "all" and current_python and args.repetitions == 1:
            summary = run_experiment(args.system, output_dir=args.output_dir, settings=settings, resume=args.resume, **options)
            if not summary.get("complete", True):
                parser.exit(3, "Some native evaluations remain incomplete; see summary.json.\n")
        else:
            runs = [{"name": args.system if args.system in SYSTEMS else "selected", "system": args.system, "settings": settings,
                     "config": str(args.config.resolve()) if args.config else None}] if args.system else None
            options["config_path"] = None if args.system else args.config or os.getenv("SMARTHOME_CONFIG") or CONFIG_PATH
            rows = run_experiments(runs=runs, output_dir=args.output_dir or configured_path(defaults["output_dir"]), repetitions=args.repetitions,
                benchmark_paths=paths, benchmark_pythons=pythons, resume=args.resume,
                timestamped=not args.resume and (not args.system or not args.output_dir), **options)
            if any(row["status"] in {"failed", "unsupported"} for row in rows):
                parser.exit(2, "Some benchmark runs failed; see comparison.json and console.log.\n")
            if any(row["status"] == "incomplete" for row in rows):
                parser.exit(3, "Some native evaluations remain incomplete; see comparison.json.\n")
    except Exception as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
