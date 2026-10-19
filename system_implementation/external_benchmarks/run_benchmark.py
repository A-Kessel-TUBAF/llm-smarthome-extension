"""Run an external benchmark; its data/contracts stay in external_benchmarks/."""
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
    from experiment_support import run_cli
    run_cli(run_benchmark, external=True)
