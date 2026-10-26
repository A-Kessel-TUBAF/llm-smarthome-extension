"""Run one system or the configured matrix on selected benchmarks."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from external_benchmarks.common import SYSTEMS, write_json
from configuration import CONFIG_PATH, activate_config, config_path as configured_path, load_config
from run_benchmark import DIMENSIONS

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ("dexa", "simuhome", "homeassistant", "homebench")


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


# The deadline plan is deliberately separate from --benchmark all: each stage has
# different participants and case selection, and DEXA has already been evaluated.
DEADLINE_BENCHMARKS = ("homeassistant", "homebench", "simuhome")
HOMEBENCH_STRATA = ("normal", "unexist_device", "unexist_attribute", "multi")
HOMEBENCH_WEIGHTS = (0.35, 0.25, 0.15, 0.25)


def homebench_sample(path, size=200, seed=42):
    """Stable, stratified case-ID sample; never assume the first N are balanced."""
    from external_benchmarks.adapters.homebench import load
    if not isinstance(size, int) or size < 1:
        raise ValueError("HomeBench sample size must be a positive integer.")
    cases, _, paths = load(path)
    groups = {name: [] for name in HOMEBENCH_STRATA}
    for case in cases:
        name = str(case["type"])
        stratum = "multi" if name.startswith("multi") else name
        if stratum in groups:
            groups[stratum].append(case["id"])
    if len({x for values in groups.values() for x in values}) != sum(map(len, groups.values())):
        raise ValueError("HomeBench contains duplicate case IDs.")
    counts = [int(size * weight) for weight in HOMEBENCH_WEIGHTS]
    # Assign fractional leftovers deterministically by remainder, then category order.
    for i in sorted(range(len(counts)), key=lambda i: (-(size * HOMEBENCH_WEIGHTS[i] - counts[i]), i))[:size - sum(counts)]:
        counts[i] += 1
    sample = []
    for index, name in enumerate(HOMEBENCH_STRATA):
        ids = sorted(groups[name])
        if len(ids) < counts[index]:
            raise ValueError(f"HomeBench has only {len(ids)} {name} cases, need {counts[index]}.")
        random.Random(seed + 1009 * index).shuffle(ids)
        sample.extend(ids[:counts[index]])
    # Keep a consistent, balanced order independent of the original dataset order.
    random.Random(seed).shuffle(sample)
    breakdown = dict(zip(HOMEBENCH_STRATA, counts))
    source_hash = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"case_ids": sample, "strata": breakdown, "seed": seed, "size": size, "source_sha256": source_hash}


def run_deadline_plan(*, config_path=None, output_dir=None, resume=False, dry_run=False,
                      benchmark_paths=None, benchmark_pythons=None):
    """Run Home Assistant -> fixed HomeBench sample -> judge-free SimuHome.

    Runs are independent per benchmark. An interruption can be resumed with the
    same output directory; saved case IDs and run order are immutable.
    """
    activate_config(config_path)
    config = load_config()
    plan = config.get("deadline", {})
    runs_by_name = {item["name"]: item for item in config["runs"]}
    expected = set(runs_by_name)
    stage_runs = {}
    for benchmark in DEADLINE_BENCHMARKS:
        chosen = plan.get(benchmark + "_runs")
        if not isinstance(chosen, list) or not chosen or len(chosen) != len(set(chosen)):
            raise ValueError(f"[deadline].{benchmark}_runs must list unique system names.")
        missing = set(chosen) - expected
        if missing:
            raise ValueError(f"Unknown systems in [deadline].{benchmark}_runs: {sorted(missing)}")
        stage_runs[benchmark] = [runs_by_name[name] for name in chosen]
    if {"minimal_sequential", "extended"} & set(plan["simuhome_runs"]):
        raise ValueError("The deadline plan must skip completed SimuHome variants minimal_sequential and extended.")

    bench_cfg = config["benchmarks"]
    paths = {name: (benchmark_paths or {}).get(name) or os.getenv({
        "homeassistant": "HOMEASSISTANT_DATASET_PATH", "homebench": "HOMEBENCH_PATH",
        "simuhome": "SIMUHOME_PATH"}[name]) or configured_path(bench_cfg[name]["path"])
        for name in DEADLINE_BENCHMARKS}
    pythons = benchmark_pythons or {}
    sample = homebench_sample(paths["homebench"], plan.get("homebench_cases", 200), plan.get("homebench_seed", 42))
    # Validate judge-free selection now, before launching any expensive benchmark.
    from external_benchmarks.adapters.simuhome import load as load_simuhome, needs_judge
    _, auto_cases = load_simuhome(paths["simuhome"], category="auto")
    if any(needs_judge(episode) for _, episode in auto_cases):
        raise ValueError("SimuHome auto selector unexpectedly contains judge-required cases.")
    if len(auto_cases) != 250:
        raise ValueError(f"Expected 250 automatically scored SimuHome episodes, found {len(auto_cases)}.")

    # Timestamp only the plan root, never individual stages. Resume requires --output-dir.
    root = Path(output_dir or configured_path(config["experiments"]["output_dir"])).resolve()
    if not resume and not output_dir and not dry_run:
        root /= "deadline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if resume and not output_dir:
        raise ValueError("--resume requires the original --output-dir for the deadline plan.")
    plan_manifest = {"plan": "deadline", "benchmarks": list(DEADLINE_BENCHMARKS),
        "runs": {name: stage_runs[name] for name in DEADLINE_BENCHMARKS},
        "paths": {name: str(Path(path).expanduser().resolve()) for name, path in paths.items()},
        "pythons": pythons, "homebench": sample,
        "simuhome": {"category": "auto", "cases": len(auto_cases),
                     "case_ids": [file.stem for file, _ in auto_cases]}}
    plan_manifest = json.loads(json.dumps(plan_manifest, default=str))
    if dry_run:
        print("Deadline plan (no experiments started):")
        for name in DEADLINE_BENCHMARKS:
            print(f"  {name}: " + ", ".join(run["name"] for run in stage_runs[name]))
            if name == "homebench":
                print(f"    {sample['size']} fixed cases: {sample['strata']} (seed={sample['seed']})")
            if name == "simuhome":
                print(f"    {len(auto_cases)} native-scored episodes; no LLM judge; skip first two variants")
        print(f"  Output root: {root}")
        return []
    root.mkdir(parents=True, exist_ok=resume)
    manifest_file = root / "deadline_plan.json"
    if resume:
        if not manifest_file.exists() or json.loads(manifest_file.read_text(encoding="utf-8")) != plan_manifest:
            raise ValueError("Resume refused: deadline plan (systems, paths, sample or settings) changed.")
    else:
        write_json(manifest_file, plan_manifest)
    rows = []
    for name in DEADLINE_BENCHMARKS:
        print(f"\n=== {name.upper()} ===", flush=True)
        stage = run_experiments(benchmark=name, runs=stage_runs[name], category="auto" if name == "simuhome" else "all",
            limit=0, case_ids=sample["case_ids"] if name == "homebench" else (),
            output_dir=root / name, timestamped=False, resume=resume,
            config_path=config_path, benchmark_paths=paths, benchmark_pythons=pythons)
        rows += stage
        write_json(root / "comparison.json", rows)
        with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for r in rows for key in r)))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Plan comparison: {root / 'comparison.csv'}", flush=True)
    return rows


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    selected, _ = pre.parse_known_args()
    activate_config(selected.config)
    defaults = load_config()["experiments"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=(*BENCHMARKS, "all"), default=defaults["benchmark"])
    parser.add_argument("--plan", choices=("deadline",), help="Run the configured priority order with benchmark-specific cases/systems")
    parser.add_argument("--dry-run", action="store_true", help="Print the deadline plan without running experiments")
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
    parser.add_argument("--suite", choices=("all", "core", "robustness"), default="all")
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
        if args.dry_run and not args.plan:
            raise ValueError("--dry-run requires --plan deadline.")
        if args.plan:
            if args.system or args.case_ids or args.category != defaults["category"] or args.limit != defaults["limit"] or args.repetitions != 1:
                raise ValueError("--plan uses its own ordered participants/cases; do not set --system, --case-id, --category, --limit or --repetitions.")
            paths = {b: getattr(args, b + "_path") for b in BENCHMARKS if getattr(args, b + "_path")}
            pythons = {b: getattr(args, b + "_python") for b in BENCHMARKS if getattr(args, b + "_python")}
            rows = run_deadline_plan(config_path=args.config, output_dir=args.output_dir,
                benchmark_paths=paths, benchmark_pythons=pythons, resume=args.resume, dry_run=args.dry_run)
            if any(r["status"] in {"failed", "unsupported"} for r in rows):
                parser.exit(2, "Some deadline runs failed; see comparison.json and console.log.\n")
            if any(r["status"] == "incomplete" for r in rows):
                parser.exit(3, "Some deadline runs are incomplete; see comparison.json.\n")
            parser.exit(0)
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
