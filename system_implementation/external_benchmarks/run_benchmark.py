"""Run an external benchmark independently of the selected participant."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_benchmarks.common import BENCHMARKS, SYSTEMS, write_json
from configuration import activate_config, config_path as configured_path, load_config


def run_benchmark(category="all", benchmark_path=None, *, benchmark="homebench", system=None,
                  config_path=None, limit=0, case_ids=(), output_dir=None, resume=False, **settings):
    if limit < 0:
        raise ValueError("--limit must be zero or positive.")
    if benchmark not in BENCHMARKS:
        raise NotImplementedError(f"The {benchmark} adapter is not implemented.")
    activate_config(config_path)
    if output_dir is None:
        name = system if isinstance(system, str) else getattr(system, "name", "mas")
        output_dir = Path(configured_path(load_config()["experiments"]["output_dir"])) / benchmark / name
    from importlib import import_module
    run = import_module("external_benchmarks.adapters." + benchmark).run
    return run(benchmark_path, category=category, limit=limit, case_ids=case_ids,
               output_dir=output_dir, system=system, settings=settings, config_path=config_path, resume=resume)


def load_env(path):
    """Explicit, simple KEY=VALUE files; existing environment variables take precedence."""
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        if "=" not in line:
            raise ValueError(f"Invalid environment-file entry on line {number}.")
        name, value = line.split("=", 1)
        parts = shlex.split(value, comments=True)
        if len(parts) > 1 or not name.strip().isidentifier():
            raise ValueError(f"Invalid environment-file entry on line {number}.")
        os.environ.setdefault(name.strip(), parts[0] if parts else "")


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--system", default=None)
    pre.add_argument("--config", type=Path)
    pre.add_argument("--env-file", type=Path)
    selected, _ = pre.parse_known_args(argv)
    activate_config(selected.config)
    defaults = load_config()["experiments"]
    selected.system = selected.system or defaults["system"] or "mas"
    if selected.env_file:
        load_env(selected.env_file)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default=selected.system)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--benchmark", choices=BENCHMARKS,
                        default=defaults["benchmark"] if defaults["benchmark"] in BENCHMARKS else "homebench")
    parser.add_argument("--benchmark-path")
    parser.add_argument("--category", default=defaults["category"])
    parser.add_argument("--limit", type=int, default=defaults["limit"], help="0 runs every selected case")
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--settings-json", default="{}", help="Participant-specific overrides as a JSON object")
    if selected.system == "mas":
        from external_benchmarks.systems.mas import add_settings_arguments, settings_from_args
        add_settings_arguments(parser)
    else:
        parser.add_argument("--config", type=Path, help="Central experiment_config.toml or participant settings")
        if selected.system in SYSTEMS:
            from importlib import import_module
            DEFAULTS = import_module("external_benchmarks.systems." + selected.system).DEFAULTS
            for name, default in DEFAULTS.items():
                parser.add_argument("--" + name.replace("_", "-"), dest=name, default=None,
                    **({"action": argparse.BooleanOptionalAction} if isinstance(default, bool) else
                       {"type": type(default) if default is not None else str}))
    args = parser.parse_args(argv)
    try:
        overrides = json.loads(args.settings_json)
        if not isinstance(overrides, dict):
            raise ValueError("--settings-json must be a JSON object.")
        if args.system == "mas":
            overrides.update(settings_from_args(args))
        elif args.system in SYSTEMS:
            overrides.update({name: getattr(args, name) for name in DEFAULTS if getattr(args, name) is not None})
        summary = run_benchmark(args.category, args.benchmark_path, benchmark=args.benchmark,
            system=args.system, config_path=args.config, limit=args.limit, case_ids=args.case_ids,
            output_dir=args.output_dir, resume=args.resume, **overrides)
        print(json.dumps(summary["overall"], indent=2))
        if args.output_dir and (args.output_dir / "failure.json").exists():
            (args.output_dir / "failure.json").unlink()
        return 0 if summary.get("complete", True) else 3
    except Exception as exc:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(exc, (FileExistsError, ValueError)):
                write_json(args.output_dir / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
