import sys
from pathlib import Path

# Also allow: python external_benchmarks/run_benchmark.py ...
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_benchmark(category="all", benchmark_path=None, *, benchmark="homebench", system=None,
                  config_path=None, limit=0, case_ids=(), output_dir=None, **settings):
    if benchmark != "homebench":
        raise NotImplementedError(f"The {benchmark} adapter is not implemented yet.")
    from external_benchmarks.adapters.homebench import run
    return run(benchmark_path, category=category, limit=limit, case_ids=case_ids,
               output_dir=output_dir, settings=settings, system=system, config_path=config_path)


if __name__ == "__main__":
    import argparse
    from smart_home_langgraph import add_settings_arguments, settings_from_args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="homebench")
    parser.add_argument("--benchmark-path")
    parser.add_argument("--category", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    add_settings_arguments(parser)
    args = parser.parse_args()
    run_benchmark(args.category, args.benchmark_path, benchmark=args.benchmark,
                  config_path=args.config, limit=args.limit, case_ids=args.case_ids,
                  output_dir=args.output_dir, **settings_from_args(args))
