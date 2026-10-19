import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_benchmark(category="all", benchmark_path=None, *, system=None,
                  config_path=None, limit=0, case_ids=(), output_dir=None, **overrides):
    if system is None:
        import smart_home_langgraph as system
    settings = system.resolve_settings(overrides, config_path=config_path) if hasattr(system, "resolve_settings") else overrides
    output = Path(output_dir or ROOT / "logs/dexa")
    output.mkdir(parents=True, exist_ok=True)
    if config_path is not None or hasattr(system, "CONFIG_PATH"):
        config = Path(config_path or system.CONFIG_PATH)
        (output / "experiment_config.toml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    path = Path(benchmark_path) if benchmark_path else ROOT / "benchmark/benchmark_data.json"
    (output / "dataset.json").write_text(json.dumps({"path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), encoding="utf-8")
    cases = json.loads(path.read_text(encoding="utf-8"))["test_cases"]
    cases = [c for c in cases if (category == "all" or c["category"] == category)
             and (not case_ids or c["id"] in case_ids)]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No DEXA cases match the selected category/IDs.")
    rows = []
    for case in cases:
        started = time.perf_counter()
        events, error, effective = [], None, settings
        try:
            for event in system.run_request(case["user_input"], **settings):
                events.append(event)
                if "experiment_config" in event:
                    effective = event["experiment_config"]["settings"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = {"id": case["id"], "input": case["user_input"], "category": case["category"],
               "latency_s": time.perf_counter() - started, "error": error, "settings": effective,
               "events": events}
        (output / f"{case['id']}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        rows.append(row)
        print(f"[{len(rows)}/{len(cases)}] {case['id']}: {row['latency_s']:.2f}s" + (f" ERROR: {error}" if error else ""))
    summary = {"overall": {"cases": len(rows), "errors": sum(r["error"] is not None for r in rows),
                            "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows) if rows else 0,
                            "score": None, "note": "DEXA logs only; no automatic task-success score."}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    from smart_home_langgraph import add_settings_arguments, settings_from_args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", nargs="?", default="all")
    parser.add_argument("benchmark_path", nargs="?")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    add_settings_arguments(parser)
    args = parser.parse_args()
    run_benchmark(args.category, args.benchmark_path, config_path=args.config,
                  limit=args.limit, case_ids=args.case_ids, output_dir=args.output_dir,
                  **settings_from_args(args))
