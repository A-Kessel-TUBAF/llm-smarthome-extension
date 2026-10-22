"""Run DEXA benchmark cases, including extension scenarios and measurements."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from threading import RLock
import time

from configuration import config_path as configured_path, load_config

ROOT = Path(__file__).resolve().parent

BENCHMARK = Path(__file__).resolve().parent.parent / "benchmark"
DIMENSIONS = ("dependency_structure", "ambiguity", "scale", "concurrency", "failure_recovery")
AGENTS = {"search": "search_engine", "display": "tv_display", "audio": "audio_system",
          "light": "lighting", "clock": "clock", "calendar": "calendar",
          "thermostat": "thermostat", "fridge": "fridge"}
NOW = "2026-10-22T08:00:00+00:00"
EVENTS = [{"event_id": "fixture-dentist", "title": "Dentist appointment",
           "start": "2026-10-23T10:00:00+00:00", "end": "2026-10-23T11:00:00+00:00", "location": "Dentist"},
          {"event_id": "fixture-meeting", "title": "Team meeting",
           "start": "2026-10-23T13:00:00+00:00", "end": "2026-10-23T14:00:00+00:00", "location": "Office"}]
NOTES = {
    "ext_dep_03": "Device-action scheduling is not supported by the default house.",
    "ext_dep_04": "Timed thermostat changes are not supported by the default house.",
    "ext_dep_05": "The default fridge inventory has no expiry dates.",
    "ext_dep_07": "Video-completion triggers and delayed device actions are not supported.",
    "ext_dep_08": "There is no kitchen display; the flour-missing branch can still be evaluated.",
    "ext_scale_01": "The kitchen has no thermostat.",
    "ext_scale_02": "The default fridge inventory has no expiry dates.",
    "ext_scale_03": "RGB party colors and kitchen dimming are not supported.",
    "ext_scale_04": "There are no guest or kids rooms; brightness 50% is not supported on every light.",
    "ext_scale_07": "There is no study room or to-do service.",
    "ext_scale_08": "There is no dining room or kitchen display; several lights have no color control.",
    "ext_conc_04": "Kitchen dimming and bathroom heating are not supported.",
    "ext_fail_08": "Calendar contracts allow overlapping events; detecting the injected clash is a measured limitation.",
}


def load_cases(dataset, original_path):
    if dataset not in {"original", "extension", "all"}:
        raise ValueError("dataset must be original, extension or all.")
    spec = importlib.util.spec_from_file_location("_dexa_complexity", BENCHMARK / "complexity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    annotations = json.loads((BENCHMARK / "complexity_annotations.json").read_text(encoding="utf-8"))["cases"]
    paths = ([Path(original_path)] if dataset != "extension" else []) + (
        [BENCHMARK / "extension_tasks.json"] if dataset != "original" else [])
    cases = []
    for path in paths:
        for raw in json.loads(path.read_text(encoding="utf-8"))["test_cases"]:
            case = deepcopy(raw)
            ann = case.get("annotation", annotations.get(case["id"]))
            if ann is not None:
                case["annotation"] = deepcopy(ann)
            case["complexity"] = module.complexity(case, ann) if ann is not None else None
            cases.append(case)
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate DEXA case IDs.")
    return cases, paths


class Scenario:
    """Fixed extension context; expected answers and annotations never enter prompts."""
    def __init__(self, case, system):
        self.case, self.system = case, system
        self.fault = case.get("injected_failure")
        if self.fault and self.fault["mode"] not in {"unavailable", "no_effect", "state_changed"}:
            raise ValueError("Unknown injected-failure mode.")
        self.tool_calls, self.injections, self.batches = [], [], []
        self.pending, self.changed = {}, False
        self.lock = RLock()

    def snapshot(self):
        with self.lock:
            values, self.pending = self.pending, {}
            if values:
                self.record_injection("calendar", "external_update", "calendar_personal", None)
            return NOW, values

    def record_injection(self, agent, skill, target, action_id):
        with self.lock:
            self.injections.append({"mode": self.fault["mode"], "agent_id": agent, "skill_id": skill,
                "target_id": target, "action_id": action_id, "at": datetime.now(timezone.utc).isoformat()})

    def setup(self):
        observations = {"service.calendar_personal.events": self.system.make_observation(deepcopy(EVENTS), source="benchmark:fixture", observed_at=NOW)}
        if self.case["id"] == "ext_fail_05":
            observations["device.audio_lr.power"] = self.system.make_observation(True, source="benchmark:fixture", observed_at=NOW)
        bindings = {key: self.wrap(key, tool) for key, tool in self.system.TOOLS.items()}
        return {"observations": observations, "environment": self, "tool_bindings": bindings,
                "simulation_data": {"now": NOW, "reference_time": NOW, "user_location": "living_room"}}

    def wrap(self, key, tool):
        def invoke(skill, args, simulator, now, simulation):
            started = time.perf_counter()
            target = args.get("device_id", args.get("service_id"))
            entry = {"agent_id": key[0], "skill_id": skill, "target_id": target,
                     "action_id": simulation.get("_action_id"), "arguments": deepcopy(args),
                     "started_at": datetime.now(timezone.utc).isoformat(), "status": "returned"}
            try:
                mode = self.fault["mode"] if self.fault and AGENTS[self.fault["device_type"]] == key[0] else None
                if mode == "unavailable":
                    self.record_injection(key[0], skill, target, entry["action_id"])
                    raise RuntimeError("Injected benchmark failure: agent unavailable.")
                if mode == "no_effect" and skill == "play":
                    self.record_injection(key[0], skill, target, entry["action_id"])
                    shadow = self.system.Simulator(simulator.snapshot(), fixed_time=now)
                    return tool(skill, args, shadow, now, simulation)
                return tool(skill, args, simulator, now, simulation)
            except Exception as exc:
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                entry.update(ended_at=datetime.now(timezone.utc).isoformat(), latency_s=time.perf_counter() - started)
                with self.lock:
                    self.tool_calls.append(entry)
        return invoke

    def observe(self, event):
        for node, update in event.items():
            if not isinstance(update, dict):
                continue
            if node == "scheduler":
                actions = update.get("action_graph", {})
                active = update.get("active_actions", [])
                running = [i for i in active if actions.get(i, {}).get("status") == "running"]
                rejected = [r for r in update.get("task_results", []) if (r.get("failure") or {}).get("source") == "lean"]
                if running or rejected:
                    self.batches.append({"action_ids": running, "lean_rejected": bool(rejected)})
            elif node == "validate_action" and update.get("pending_actions"):
                self.batches.append({"action_ids": [update["pending_actions"][0]["task_id"]], "lean_rejected": False})
            if self.fault and self.fault["mode"] == "state_changed" and not self.changed:
                actions = list(update.get("action_graph", {}).values()) + update.get("pending_actions", [])
                if any(a.get("skill_id") == "create_event" and a.get("status", "pending") == "pending" for a in actions):
                    self.changed = True
                    self.pending["service.calendar_personal.events"] = deepcopy(EVENTS) + [{
                        "event_id": "injected-meeting", "title": "Externally added meeting",
                        "start": "2026-10-22T08:15:00+00:00", "end": "2026-10-22T08:45:00+00:00", "location": "Office"}]

    def report(self, episode, settings):
        lean = settings.get("checker") == "lean"
        accepted = sum(bool(b["action_ids"]) for b in self.batches)
        rejected = sum(b["lean_rejected"] for b in self.batches)
        calls = episode.get("llm_calls", [])
        first = min((i["at"] for i in self.injections), default=None)
        return {"tool_calls": sorted(self.tool_calls, key=lambda c: c["started_at"]),
            "injections": self.injections, "injection_status": "triggered" if first else "not_triggered" if self.fault else "not_requested",
            "metrics": {"execution_rounds": accepted, "max_batch_size": max((len(b["action_ids"]) for b in self.batches), default=0),
                "lean_batches_accepted": accepted if lean else None, "lean_batches_rejected": rejected if lean else None,
                "lean_acceptance_rate": accepted / (accepted + rejected) if lean and accepted + rejected else None,
                "llm_calls_after_injection": sum(c.get("started_at", "") >= first for c in calls) if first else None},
            "semantic_metrics": {"intent_recall": None, "dependency_correctness": None, "conflict_handling": None,
                "recovery_locality": None, "recovery_correctness": None, "failure_reported_to_user": None},
            "evaluation_note": "Semantic scores require reviewing intent-to-task matches; tool return, task status and injection are not user-goal success.",
            "compatibility_notes": [NOTES[self.case["id"]]] if self.case["id"] in NOTES else []}


def grouped_summary(rows, field):
    groups = {}
    for value in sorted({r.get(field, "original") for r in rows}):
        selected = [r for r in rows if r.get(field, "original") == value]
        groups[value] = {"cases": len(selected), "errors": sum(r["error"] is not None for r in selected),
            "mean_latency_s": sum(r["latency_s"] for r in selected) / len(selected),
            "injections_triggered": sum(r.get("extension", {}).get("injection_status") == "triggered" for r in selected)}
    return groups


def run_benchmark(category="all", benchmark_path=None, *, system=None,
                  config_path=None, limit=0, case_ids=(), output_dir=None,
                  dataset="original", dimension=None, **overrides):
    from llm_config import summarize_calls
    if system is None:
        import smart_home_langgraph as system
    settings = system.resolve_settings(overrides, config_path=config_path) if hasattr(system, "resolve_settings") else overrides
    output = Path(output_dir or ROOT / "logs/dexa")
    output.mkdir(parents=True, exist_ok=True)
    if config_path is not None or hasattr(system, "CONFIG_PATH"):
        config = Path(config_path or os.getenv("SMARTHOME_CONFIG") or system.CONFIG_PATH)
        (output / "experiment_config.toml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    if dimension and (dimension not in DIMENSIONS or dataset == "original"):
        raise ValueError("--dimension requires --dataset extension or all and a known dimension.")
    path = Path(benchmark_path or configured_path(load_config(config_path)["benchmarks"]["dexa"]["path"]))
    cases, paths = load_cases(dataset, path)
    sources = paths + [ROOT.parent / "benchmark" / name for name in ("complexity.py", "complexity_annotations.json")]
    (output / "dataset.json").write_text(json.dumps({"dataset": dataset, "dimension": dimension,
        "path": str(paths[0].resolve()), "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "sources": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}, indent=2), encoding="utf-8")
    cases = [c for c in cases if (category == "all" or c["category"] == category)
             and (not case_ids or c["id"] in case_ids) and (not dimension or c.get("dimension") == dimension)]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No DEXA cases match the selected category/IDs.")
    if any(c.get("dimension") in DIMENSIONS for c in cases) and not all(
            hasattr(system, name) for name in ("TOOLS", "Simulator", "make_observation")):
        raise NotImplementedError("DEXA extension scenarios require the current MAS simulator/tool boundary.")
    rows = []
    for case in cases:
        started = time.perf_counter()
        events, report, error, effective = [], {}, None, settings
        scenario = Scenario(case, system) if case.get("dimension") in DIMENSIONS else None
        setup = scenario.setup() if scenario else {}
        try:
            for event in system.run_request(case["user_input"], **settings, **setup):
                events.append(event)
                if scenario:
                    scenario.observe(event)
                if "episode_result" in event:
                    report = event["episode_result"]
                if "experiment_config" in event:
                    effective = event["experiment_config"]["settings"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        exception = error
        error = error or report.get("error")
        row = {"id": case["id"], "input": case["user_input"], "category": case["category"],
               "dimension": case.get("dimension", "original"), "complexity": case.get("complexity"),
               "annotation": case.get("annotation"), "expected_recovery": case.get("expected_recovery"),
               "expected_concurrency": case.get("expected_concurrency"),
               "latency_s": time.perf_counter() - started, "error": error, "exception": exception, "settings": effective,
               "events": events, "episode_result": report, "task_failures": report.get("task_failures", []),
               "model_usage": report.get("model_usage", summarize_calls([]))}
        if scenario:
            row["extension"] = scenario.report(report, effective)
        (output / f"{case['id']}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        rows.append(row)
        print(f"[{len(rows)}/{len(cases)}] {case['id']}: {row['latency_s']:.2f}s" + (f" ERROR: {error}" if error else ""))
    summary = {"overall": {"cases": len(rows), "errors": sum(r["error"] is not None for r in rows),
                            "runtime_errors": sum(r["exception"] is not None for r in rows),
                            "planning_failures": sum(any(t["kind"] == "plan" for t in r["task_failures"]) for r in rows),
                            "action_failures": sum(any(t["kind"] in {"precondition", "tool", "output", "postcondition", "dispatch"}
                                                       for t in r["task_failures"]) for r in rows),
                            **summarize_calls([call for r in rows for e in r["events"]
                                               for call in e.get("episode_result", {}).get("llm_calls", [])]),
                            "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows) if rows else 0,
                            "score": None, "note": "DEXA logs only; no automatic task-success score. Injected faults may produce expected execution errors."},
               "by_category": grouped_summary(rows, "category"), "by_dimension": grouped_summary(rows, "dimension")}
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
    parser.add_argument("--dataset", choices=("original", "extension", "all"), default="original")
    parser.add_argument("--dimension", choices=DIMENSIONS)
    add_settings_arguments(parser)
    args = parser.parse_args()
    run_benchmark(args.category, args.benchmark_path, config_path=args.config,
                  limit=args.limit, case_ids=args.case_ids, output_dir=args.output_dir,
                  dataset=args.dataset, dimension=args.dimension,
                  **settings_from_args(args))
