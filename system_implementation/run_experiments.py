"""Run one system or the configured matrix on selected benchmarks."""
import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from external_benchmarks.common import SYSTEMS, write_json
from configuration import CONFIG_PATH, activate_config, config_path as configured_path, load_config, system_settings
from run_benchmark import DIMENSIONS, SUITES

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ("dexa", "simuhome", "homeassistant", "homebench")
QED_LEAF_MODES = ("off", "record", "require")


def qed_leaf_gate(settings, output, config_path=None):
    """QED_LEAF audit of the Lean checker before a MAS run with checker = "lean" ([assurance])."""
    mode = load_config(config_path).get("assurance", {}).get("qed_leaf", "record")
    if mode not in QED_LEAF_MODES:
        raise ValueError(f"assurance.qed_leaf must be one of {', '.join(QED_LEAF_MODES)}.")
    if mode == "off" or {**system_settings("mas", config_path), **settings}["checker"] != "lean":
        return None
    sys.path.insert(0, str(ROOT / "assurance/qed_leaf"))
    import run_audit
    record = run_audit.current_record()
    write_json(output / "qed_leaf.json", record)
    status = record["summary"].get("status")
    if mode == "require" and status != "passed":
        raise RuntimeError(f"QED_LEAF audit {status}; assurance.qed_leaf = \"require\" refuses Lean-checker "
                           f"runs. See {output / 'qed_leaf.json'}.")
    return record




def run_experiment(system_path, *, benchmark="homebench", benchmark_path=None, category="all",
                   limit=0, case_ids=(), output_dir=None, config_path=None, settings=None, resume=False,
                   dataset=None, dimension=None, suite="all", failure_mode=None):
    dataset = dataset or ("extended" if benchmark == "dexa" else "original")
    if benchmark != "dexa" and (dataset != "original" or dimension or suite != "all" or failure_mode):
        raise ValueError("DEXA selection filters require --benchmark dexa.")
    activate_config(config_path)
    output = Path(output_dir or Path(configured_path(load_config()["experiments"]["output_dir"])) / "single")
    output.mkdir(parents=True, exist_ok=True)
    settings = settings or {}
    try:
        if str(system_path) == "mas" or str(system_path).endswith(".py"):
            qed_leaf_gate(settings, output, config_path)
        from external_benchmarks.common import create_system
        if benchmark == "dexa":
            from run_benchmark import run_benchmark
            participant = system_path
            if str(system_path).endswith(".py") and dataset != "original":
                path = Path(system_path).resolve()
                spec = importlib.util.spec_from_file_location("_benchmark_system", path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                participant = create_system(module, settings, config_path)
            summary = run_benchmark(category, benchmark_path, system=participant, config_path=config_path,
                limit=limit, case_ids=case_ids, output_dir=output, resume=resume, dataset=dataset,
                dimension=dimension, suite=suite, failure_mode=failure_mode, **settings)
        else:
            from external_benchmarks.run_benchmark import run_benchmark
            name = str(system_path)
            if name in SYSTEMS or (":" in name and not name.endswith(".py")):
                participant = create_system(name, settings, config_path)
            else:
                path = Path(system_path).resolve()
                spec = importlib.util.spec_from_file_location("_benchmark_system", path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                if benchmark not in getattr(module, "SUPPORTED_BENCHMARKS", set()):
                    raise NotImplementedError(f"{path} does not support {benchmark}.")
                participant = create_system(module, settings, config_path)
            summary = run_benchmark(category, benchmark_path, benchmark=benchmark, system=participant,
                limit=limit, case_ids=case_ids, output_dir=output, resume=resume)
        (output / "failure.json").unlink(missing_ok=True)
        return summary
    except Exception as exc:
        if not isinstance(exc, (FileExistsError, ValueError)):
            write_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}",
                "status": "unsupported" if isinstance(exc, NotImplementedError) else "failed"})
        raise


def run_experiments(*, benchmark="all", benchmark_path=None, category="all", limit=0, case_ids=(),
                    runs=None, repetitions=None, config_path=None, output_dir=None,
                    benchmark_paths=None, benchmark_pythons=None, resume=False, timestamped=True,
                    dataset=None, dimension=None, suite="all", failure_mode=None):
    dataset = dataset or ("extended" if benchmark in {"dexa", "all"} else "original")
    if benchmark not in {"dexa", "all"} and (dataset != "original" or dimension or suite != "all" or failure_mode):
        raise ValueError("DEXA selection filters require --benchmark dexa or all.")
    if benchmark == "all" and dataset == "original":
        raise ValueError("The full matrix uses extended DEXA; select --benchmark dexa for --dataset original.")
    if runs is None or any(run.get("config") is None for run in runs):
        activate_config(config_path)
    defaults = load_config()["experiments"]
    repetitions = defaults["repetitions"] if repetitions is None else repetitions
    runs = load_config()["runs"] if runs is None else runs
    if limit < 0 or repetitions < 1:
        raise ValueError("limit must be nonnegative and repetitions must be positive.")
    if benchmark == "all" and (benchmark_path or category != "all" or case_ids):
        raise ValueError("For --benchmark all use --dexa-path/--homebench-path/--simuhome-path/--homeassistant-path; "
                         "--category and --case-id are only supported for a single benchmark.")
    output = Path(output_dir or configured_path(defaults["output_dir"])).resolve()
    if timestamped:
        output /= datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=resume)
    benchmarks = BENCHMARKS if benchmark == "all" else (benchmark,)
    configured = load_config()["benchmarks"]
    paths = {b: os.getenv({"dexa": "DEXA_PATH", "homebench": "HOMEBENCH_PATH", "simuhome": "SIMUHOME_PATH", "homeassistant": "HOMEASSISTANT_DATASET_PATH"}[b])
             or configured_path("../benchmark/benchmark_data.json" if b == "dexa" and dataset == "original"
                                else configured[b]["path"]) for b in BENCHMARKS}
    paths.update(benchmark_paths or {})
    if benchmark_path:
        paths[benchmark] = benchmark_path
    selected_pythons = {b: os.getenv(b.upper() + "_PYTHON") or configured_path(configured[b].get("python"), executable=True)
                        for b in BENCHMARKS if os.getenv(b.upper() + "_PYTHON") or configured[b].get("python")}
    selected_pythons.update(benchmark_pythons or {})
    pythons = {}
    for name, executable in selected_pythons.items():
        found = shutil.which(os.path.expanduser(str(executable)))
        # Preserve venv symlinks; resolving them selects the base interpreter.
        pythons[name] = str(Path(found).absolute()) if found else str(executable)
    manifest = {"runs": runs, "repetitions": repetitions, "benchmark": benchmark, "benchmarks": list(benchmarks),
                "category": category, "limit": limit, "case_ids": list(case_ids),
                "benchmark_paths": {b: str(Path(paths.get(b) or benchmark_path).resolve())
                                    for b in benchmarks if paths.get(b) or benchmark_path},
                "benchmark_pythons": {b: str(pythons.get(b) or sys.executable) for b in benchmarks}}
    manifest = json.loads(json.dumps(manifest, default=str))
    if "dexa" in benchmarks:
        manifest.update(dataset=dataset, dimension=dimension, suite=suite, failure_mode=failure_mode)
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
    for selected_benchmark in benchmarks:
        for repetition in range(1, repetitions + 1):
            for run in runs:
                name = str(run["system"])
                wrapped = name in SYSTEMS or (":" in name and not name.endswith(".py"))
                if selected_benchmark == "dexa" and wrapped and name != "mas":
                    continue  # DEXA runs only MAS or a legacy MAS file.
                folder = output / selected_benchmark if benchmark == "all" else output
                destination = folder / f"{run['name']}_{repetition}"
                destination.mkdir(parents=True, exist_ok=resume)
                command = [str(pythons.get(selected_benchmark) or sys.executable), str(Path(__file__).resolve()), "--benchmark", selected_benchmark,
                           "--system", name if wrapped else str((ROOT / name).resolve()),
                           "--category", category, "--limit", str(limit), "--output-dir", str(destination),
                           "--repetitions", "1",
                           "--settings-json", json.dumps(run.get("settings", {}))]
                command += ["--" + selected_benchmark + "-python", command[0]]
                selected_config = str(Path(os.getenv("SMARTHOME_CONFIG") or CONFIG_PATH).resolve())
                participant_config = run.get("config")
                if participant_config is None and (not wrapped or name == "mas"):
                    participant_config = output / "experiment_config.toml"
                if participant_config:
                    command += ["--config", str(Path(participant_config).resolve())]
                if selected_benchmark == "dexa":
                    command += ["--dataset", dataset, "--suite", suite]
                    if failure_mode:
                        command += ["--failure-mode", failure_mode]
                    if dimension:
                        command += ["--dimension", dimension]
                path = paths.get(selected_benchmark) or benchmark_path
                if path:
                    command += ["--benchmark-path", str(Path(path).resolve())]
                if resume and selected_benchmark != "dexa":
                    command.append("--resume")
                for case_id in case_ids:
                    command += ["--case-id", case_id]
                print(f"Running {run['name']} on {selected_benchmark} ({repetition})", flush=True)
                write_json(destination / "command.json", command)
                try:
                    # DEXA has no per-case resume: a finished run is kept, an unfinished one repeated.
                    if resume and selected_benchmark == "dexa" and (destination / "summary.json").exists():
                        returncode = 0
                    else:
                        with (destination / "console.log").open("a" if resume else "w", encoding="utf-8") as log:
                            returncode = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                        env={**os.environ, "SMARTHOME_CONFIG": selected_config}).returncode
                    row = {"run": run["name"], "repetition": repetition, "benchmark": selected_benchmark,
                           "status": {0: "completed", 3: "incomplete"}.get(returncode, "failed"), "returncode": returncode}
                except OSError as exc:
                    row = {"run": run["name"], "repetition": repetition, "benchmark": selected_benchmark,
                           "status": "failed", "error": str(exc), "returncode": 2}
                if (destination / "qed_leaf.json").exists():
                    row["qed_leaf"] = json.loads((destination / "qed_leaf.json").read_text(encoding="utf-8"))["summary"].get("status")
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
    parser.add_argument("--benchmark", choices=(*BENCHMARKS, "all"), default=defaults["benchmark"])
    parser.add_argument("--benchmark-path")
    for name in BENCHMARKS:
        parser.add_argument(f"--{name}-path", help="Data/checkout path for " + name)
        parser.add_argument(f"--{name}-python", default=os.getenv(name.upper() + "_PYTHON"),
                            help="Python executable with this benchmark's dependencies")
    parser.add_argument("--category", default=defaults["category"])
    parser.add_argument("--limit", type=int, default=defaults["limit"])
    parser.add_argument("--repetitions", type=int, default=defaults["repetitions"])
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--dataset", choices=("original", "extended", "extension", "all"), default=None,
                        help="DEXA: untouched original or revised extended edition (default)")
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument("--failure-mode", choices=("none", "transient", "unavailable", "no_effect", "state_changed"))
    parser.add_argument("--dimension", choices=DIMENSIONS, help="Filter DEXA extension cases")
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
                   config_path=args.config, dataset=args.dataset, dimension=args.dimension, suite=args.suite, failure_mode=args.failure_mode)
    try:
        if args.resume and not args.output_dir:
            raise ValueError("--resume requires the original --output-dir.")
        settings = json.loads(args.settings_json)
        if not isinstance(settings, dict):
            raise ValueError("--settings-json must be a JSON object.")
        paths = {b: getattr(args, b + "_path") for b in BENCHMARKS if getattr(args, b + "_path")}
        pythons = {b: getattr(args, b + "_python") or configured_path(load_config()["benchmarks"][b].get("python"), executable=True)
                   for b in BENCHMARKS if getattr(args, b + "_python") or load_config()["benchmarks"][b].get("python")}
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
