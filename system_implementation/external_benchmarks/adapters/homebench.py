"""HomeBench adapter. Requires the original dataset directory, not live search.

The primary score follows code/eval.py's whitespace/comma Counter comparison.
Every concrete proposal, preparation and repeated attempt is exported; failures
are not silently removed. Device writes below are a local controller model,
not an official physical simulator or an additional state-based benchmark score.
"""
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from llm_config import summarize_calls

from external_benchmarks.agent_registry_homebench import build_home, execute


def load(path=None):
    root = Path(path) if path else Path(__file__).resolve().parents[1] / "data/homebench/dataset"
    if root.is_dir() and (root / "dataset").is_dir():
        root = root / "dataset"
    tasks_path = root if root.is_file() else root / "test_data.jsonl"
    homes_path = tasks_path.parent / "home_status_method.jsonl"
    cases = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    homes = {row["home_id"]: row for row in
             (json.loads(line) for line in homes_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    return cases, homes, (tasks_path, homes_path)


def setup(home):
    config, registry = build_home(home)
    return {"home_config": config, "agent_registry": registry,
            "tool_bindings": {(agent, skill): execute for agent, card in registry.items() for skill in card["operations"]},
            "simulation_data": {"benchmark": "homebench", "search_results": {}, "search_available": False}}


def command(action, config):
    arguments = action.get("arguments", {})
    target = arguments.get("device_id", "unknown_device")
    device = config["devices"].get(target, {})
    name = device.get("homebench_name", str(target))
    skill = action.get("skill_id", "unknown_method")
    schema = device.get("parameters", {}).get(skill, {})
    # Native method parameter order, with no quotes around string values, as in the dataset.
    names = [key for key in schema if key != "device_id"] or [key for key in arguments if key != "device_id"]
    def literal(value):
        if isinstance(value, list):
            return "(" + ",".join(literal(item) for item in value) + ")"
        return str(value)
    values = ",".join(literal(arguments.get(key, "missing_argument")) for key in names)
    return f"{name}.{skill}({values})"


def prediction(report, config):
    attempts = {}
    for result in report.get("task_results", []):
        if result["status"] != "blocked":
            attempts.setdefault(result["task_id"], []).append(result)
    commands = []
    for proposal in report.get("proposed_actions", []):
        # Actual arguments include resolved references. Preserve each proposal if it
        # was rejected before execution, and preserve repeated execution attempts.
        commands.extend(command(action, config) for action in attempts.get(proposal["task_id"], [proposal]))
    commands.extend("error_input" for _ in report.get("rejected_requests", []))
    return "{" + ",".join(commands) + "}"


def score(predicted, expected):
    # Match the released evaluator, including its comma-based treatment of tuple arguments.
    import re
    predicted = ",".join(re.findall(r"\{(.*?)\}", predicted.replace(" ", "").replace("\n", "")))
    expected = expected.replace("'''", "").replace(" ", "").replace("\n", "")
    actual = Counter(value for value in predicted.split(",") if value)
    gold = Counter(value for value in expected.split(",") if value)
    return {"exact_match": actual == gold, "true_positive": sum((actual & gold).values()),
            "predicted": sum(actual.values()), "expected": sum(gold.values())}


def summarize(rows):
    count = len(rows)
    tp = sum(row["score"]["true_positive"] for row in rows)
    predicted = sum(row["score"]["predicted"] for row in rows)
    expected = sum(row["score"]["expected"] for row in rows)
    precision, recall = tp / predicted if predicted else 0, tp / expected if expected else 0
    return {"cases": count, "exact_match": sum(row["score"]["exact_match"] for row in rows) / count if count else 0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "errors": sum(row.get("error") is not None for row in rows),
            "runtime_errors": sum(row.get("exception") is not None for row in rows),
            "planning_failures": sum(any(t["kind"] == "plan" for t in row.get("report", {}).get("task_failures", [])) for row in rows),
            "action_failures": sum(any(t["kind"] in {"precondition", "tool", "output", "postcondition", "dispatch"}
                                       for t in row.get("report", {}).get("task_failures", [])) for row in rows),
            **summarize_calls([call for row in rows for call in row.get("report", {}).get("llm_calls", [])]),
            "mean_latency_s": sum(row["latency_s"] for row in rows) / count if count else 0}


def run(path=None, *, category="all", limit=0, case_ids=(), output_dir=None, settings=None, system=None, config_path=None):
    import hashlib
    import time
    if system is None:
        import smart_home_langgraph as system
    run_request = system.run_request
    from llm_config import DEFAULT_MODEL, SAME_LLM_FOR_ALL_AGENTS, ROLE_MODELS

    cases, homes, paths = load(path)
    cases = [case for case in cases if (category == "all" or case["type"] == category)
             and (not case_ids or case["id"] in case_ids)]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No HomeBench cases match the selected category/IDs.")
    settings = system.resolve_settings(settings, config_path=config_path)
    output = Path(output_dir or "logs/homebench")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"benchmark": "homebench", "defaults": system.load_settings(config_path), "settings": settings,
                "category": category, "limit": limit, "case_ids": list(case_ids),
                "models": {"default": DEFAULT_MODEL, "same_for_all": SAME_LLM_FOR_ALL_AGENTS, "roles": ROLE_MODELS},
                "data_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "output_protocol": "all concrete proposals and execution attempts; original Counter scoring"}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    for case in cases:
        environment = setup(homes[case["home_id"]])
        report, effective, error = {}, settings, None
        started = time.perf_counter()
        try:
            for event in run_request(case["input"], **environment, **settings):
                if "experiment_config" in event:
                    effective = event["experiment_config"]["settings"]
                if "episode_result" in event:
                    report = event["episode_result"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        exception = error
        error = error or report.get("error")
        try:
            predicted = prediction(report, environment["home_config"])
        except (KeyError, TypeError, ValueError) as exc:
            error = error or f"PredictionError: {exc}"
            predicted = "{}"
        row = {"id": case["id"], "home_id": case["home_id"], "category": case["type"], "input": case["input"],
               "prediction": predicted, "expected": case["output"], "score": score(predicted, case["output"]),
               "latency_s": elapsed, "error": error, "exception": exception, "settings": effective, "report": report,
               "model_usage": report.get("model_usage", summarize_calls([]))}
        # A crashed run cannot pass, even if a future empty-output task is added.
        if exception:
            row["score"]["exact_match"] = False
        rows.append(row)
        (output / f"{case['id']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{len(rows)}/{len(cases)}] {case['id']}: EM={int(row['score']['exact_match'])}, {elapsed:.2f}s"
              + (f" ERROR: {error}" if error else ""))
    summary = {"overall": summarize(rows), "by_category": {
        name: summarize([row for row in rows if row["category"] == name]) for name in sorted({row["category"] for row in rows})}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
