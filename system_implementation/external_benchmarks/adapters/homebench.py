"""Native HomeBench command scoring with local device feedback."""
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import os
from configuration import config_path as configured_path, load_config
from external_benchmarks.common import (Episode, cached_case, case_file, create_system,
    prepare_output, summarize_calls, system_report, write_json)


def load(path=None):
    root = Path(path or os.getenv("HOMEBENCH_PATH") or configured_path(load_config()["benchmarks"]["homebench"]["path"]))
    if root.is_dir() and (root / "dataset").is_dir():
        root = root / "dataset"
    tasks_path = root if root.is_file() else root / "test_data.jsonl"
    homes_path = tasks_path.parent / "home_status_method.jsonl"
    cases = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    homes = {row["home_id"]: row for row in
             (json.loads(line) for line in homes_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    return cases, homes, (tasks_path, homes_path)


class Session:
    """Native method names and a local feedback model; the score remains command-based."""
    def __init__(self, home):
        self.initial_home = deepcopy(home)
        self.home, self.calls = deepcopy(home), []
        self.methods, self.tools = {}, []
        types = {"int": {"type": "integer"}, "str": {"type": "string"},
                 "typing.Tuple[int, int, int]": {"type": "array", "items": {"type": "integer"},
                                               "minItems": 3, "maxItems": 3}}
        for method in home["method"]:
            room, device, operation = method["room_name"], method["device_name"], method["operation"]
            name = f"{device}.{operation}" if room == "None" else f"{room}.{device}.{operation}"
            props = {}
            attrs = {k.strip(): v for k, v in self.device(method).get("attributes", {}).items()}
            for parameter in method["parameters"]:
                key = parameter["name"]
                schema = deepcopy(types[parameter["type"]])
                metadata = attrs.get("cleaning_area" if key == "area" else key, {})
                if "options" in metadata:
                    schema["enum"] = metadata["options"]
                if schema.get("type") == "integer":
                    for source, target in (("lowest", "minimum"), ("highest", "maximum")):
                        if source in metadata:
                            schema[target] = int(metadata[source])
                props[key] = schema
            self.methods[name] = method
            self.tools.append({"name": name, "description": "HomeBench native device method.",
                "parameters": {"type": "object", "properties": props, "required": list(props),
                               "additionalProperties": False}})
        self.tools.append({"name": "get_home_state", "description": "Read the local HomeBench device state model.",
                           "parameters": {"type": "object", "properties": {}, "additionalProperties": False}})

    def device(self, method):
        if method["room_name"] == "None":
            return self.home["home_status"]["VacuumRobot"]
        return self.home["home_status"][method["room_name"]][method["device_name"]]

    def command(self, name, arguments):
        method = self.methods.get(name)
        keys = [p["name"] for p in method["parameters"]] if method else list(arguments)
        def literal(value):
            return "(" + ",".join(literal(v) for v in value) + ")" if isinstance(value, list) else str(value)
        values = [literal(arguments.get(k, "missing_argument")) for k in keys]
        values.extend(f"{k}={literal(v)}" for k, v in arguments.items() if k not in keys)
        return f"{name}({','.join(values)})"

    def call(self, name, arguments):
        from jsonschema import Draft202012Validator
        if name == "get_home_state":
            if arguments:
                raise ValueError("get_home_state takes no arguments.")
            return deepcopy(self.home["home_status"])
        # Score every attempt, including invalid/repeated commands.
        call = {"tool": name, "params": deepcopy(arguments), "command": self.command(name, arguments)}
        self.calls.append(call)
        try:
            tool = next((t for t in self.tools if t["name"] == name), None)
            if tool is None:
                raise ValueError("Unknown HomeBench method: " + name)
            Draft202012Validator(tool["parameters"]).validate(arguments)
            method = self.methods[name]
            device = self.device(method)
            operation = method["operation"]
            if operation.startswith("set_"):
                field = operation[4:]
                value = next(iter(arguments.values()))
                device.setdefault("attributes", {}).setdefault(field, {})["value"] = deepcopy(value)
            else:
                device["state"] = {"turn_on": "on", "turn_off": "off", "open": "open", "close": "closed",
                    "play": "playing", "pause": "paused", "stop": "stopped", "pack": "packed"}[operation]
            call["observation"] = {"applied": True, "state": deepcopy(device)}
            return call["observation"]
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def episode(self, request):
        return Episode("homebench", request, {"home_status": deepcopy(self.initial_home["home_status"])},
                       self.tools, self.call, self)

    def prediction(self, report):
        commands = [c["command"] for c in self.calls]
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
    return {"cases": count, "incomplete_cases": sum(not row["completed"] for row in rows),
            "exact_match": sum(row["score"]["exact_match"] for row in rows) / count if count else 0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "errors": sum(row.get("error") is not None for row in rows),
            "runtime_errors": sum(row.get("exception") is not None for row in rows),
            "planning_failures": sum(any(t["kind"] == "plan" for t in row.get("report", {}).get("task_failures", [])) for row in rows),
            "action_failures": sum(any(t["kind"] in {"precondition", "tool", "output", "postcondition", "dispatch"}
                                       for t in row.get("report", {}).get("task_failures", [])) for row in rows),
            **summarize_calls([call for row in rows for call in row.get("report", {}).get("llm_calls", [])]),
            "mean_latency_s": sum(row["latency_s"] for row in rows) / count if count else 0}


def run(path=None, *, category="all", limit=0, case_ids=(), output_dir=None, settings=None, system=None, config_path=None,
        resume=False):
    import hashlib
    import time
    system = create_system(system, settings, config_path)

    cases, homes, paths = load(path)
    cases = [case for case in cases if (category == "all" or case["type"] == category)
             and (not case_ids or case["id"] in case_ids)]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No HomeBench cases match the selected category/IDs.")
    settings = system.settings
    metadata = {"benchmark": "homebench", "system": system.metadata(),
                "category": category, "limit": limit, "case_ids": [c["id"] for c in cases],
                "data_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "output_protocol": "all concrete proposals and execution attempts; original Counter scoring"}
    output = prepare_output(output_dir or f"logs/homebench/{system.name}", metadata, resume)
    rows = []
    for case in cases:
        cached = cached_case(output, case["id"], resume)
        if cached is not None:
            rows.append(cached)
            continue
        session = Session(homes[case["home_id"]])
        started = time.perf_counter()
        report = system_report(system, session.episode(case["input"]))
        elapsed = time.perf_counter() - started
        error, exception = report.get("error"), report.get("exception")
        predicted = report.get("prediction", session.prediction(report))
        row = {"id": case["id"], "home_id": case["home_id"], "category": case["type"], "input": case["input"],
               "prediction": predicted, "expected": case["output"], "score": score(predicted, case["output"]),
               "latency_s": elapsed, "error": error, "exception": exception, "settings": report.get("settings", settings), "report": report,
               "native_calls": session.calls,
               "technical_error": bool(report.get("technical_error") or report.get("backend_unavailable")),
               "model_usage": report["model_usage"]}
        row["completed"] = not row["technical_error"]
        rows.append(row)
        write_json(case_file(output, case["id"]), row)
        write_json(output / "summary.json", {"overall": summarize(rows),
            "complete": len(rows) == len(cases) and all(r["completed"] for r in rows)})
        print(f"[{len(rows)}/{len(cases)}] {case['id']}: EM={int(row['score']['exact_match'])}, {elapsed:.2f}s"
              + (f" ERROR: {error}" if error else ""), flush=True)
        if report.get("backend_unavailable"):
            raise RuntimeError("Model backend unavailable; stopped after saving this case. Configure the endpoint, then --resume.")
    summary = {"overall": summarize(rows), "by_category": {
        name: summarize([row for row in rows if row["category"] == name]) for name in sorted({row["category"] for row in rows})},
        "complete": all(row["completed"] for row in rows)}
    write_json(output / "summary.json", summary)
    return summary
