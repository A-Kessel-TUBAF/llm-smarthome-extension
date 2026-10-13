import json
import re
import argparse
import math
import os
import operator
import subprocess
import time
from threading import RLock
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from graphlib import CycleError, TopologicalSorter
from fractions import Fraction
from pathlib import Path
from typing import Any, Annotated, Literal, TypedDict
from uuid import uuid4
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from llm_config import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Send, interrupt
from agent_registry import AGENT_REGISTRY
from home_defaults import HOME_DEFAULTS, SEARCH_FIXTURES

SCHEDULER_MODE = os.getenv("SCHEDULER_MODE", "python")  # Central ablation switch, also read from .env.

# Live objects stay outside LangGraph checkpoints: workers share one simulator per request.
SIMULATORS = {}

class Simulator:
    def __init__(self, home):
        self.home = deepcopy(home)
        self.lock = RLock()
        self.reservations = {}

    def snapshot(self):
        with self.lock:
            return deepcopy(self.home)

    def write(self, values, now=None, *, source="simulation:tool", ref=None):
        with self.lock:
            for key, value in values.items():
                old = self.home.get(key, {})
                self.home[key] = {**make_observation(value, source=source,
                    unit=old.get("unit", UNITS.get(key.rsplit(".", 1)[-1])), observed_at=now,
                    version=old.get("version", 0) + 1), "ref": ref}

    def admit(self, action):
        # One critical section covers freshness AND reservation; tool calls run outside it.
        with self.lock:
            if time.monotonic() >= action["dispatch_until"] or any(
                self.home.get(k, {}).get("version", 0) != v for k, v in action["bound_versions"].items()
            ):
                raise ValueError("FreshDispatch: checked state is stale or authorization expired.")
            reads, writes, target = action["footprint"]
            reads, writes = set(reads), set(writes)
            if any(target == t or writes & (r | w) or reads & w
                   for r, w, t in self.reservations.values()):
                raise ValueError("FreshDispatch: resource is reserved by another action.")
            self.reservations[action["task_id"]] = (reads, writes, target)
            return deepcopy(self.home)

    def release(self, action_id):
        with self.lock:
            self.reservations.pop(action_id, None)

class Observation(TypedDict):
    value: Any
    observed_at: str
    source: str
    unit: str | None
    version: int

class Task(TypedDict):
    task_id: str
    agent_id: str
    action: str
    status: Literal["pending", "planning", "waiting", "planned", "success", "failed", "blocked"]
    depends_on: list[str]
    scope: list[str]
    constraints: list[str]
    goal: dict
    # Optional actions contains an already concrete plan; absent means domain refinement.

class Action(TypedDict):
    task_id: str
    agent_id: str
    skill_id: str
    arguments: dict
    depends_on: list[str]
    status: str

class TaskResult(TypedDict):
    task_id: str
    attempt_id: str
    agent_id: str
    skill_id: str
    arguments: dict
    status: Literal["success", "failed", "blocked", "stale"]
    output: dict
    observations: dict[str, Observation]
    error: str | None
    verified: bool

class SmartHomeState(TypedDict):
    original_user_input: str
    request_id: str
    infos: list[str]
    key_modifiers: list[str]
    task_queue: list[Task]
    tasks: dict[str, Task]
    task_results: Annotated[list[TaskResult], operator.add]
    home_config: dict
    home_state: dict[str, Observation]
    agent_registry: dict
    collaboration_request: dict
    pending_task: dict | None
    pending_actions: list[dict]
    action_graph: dict[str, Action]
    planning_task: dict
    planning_updates: Annotated[list[dict], operator.add]
    active_actions: list[str]
    task_history: Annotated[list[dict], operator.add]
    final_response: str
    settings: dict
    simulation_data: dict
    repair_context: dict

# Timestamps/source are assigned by Python, never by the LLM.
def make_observation(value, *, source, unit=None, observed_at=None, version=0):
    return {"value": deepcopy(value), "source": source, "unit": unit,
            "observed_at": observed_at or datetime.now(timezone.utc).isoformat(), "version": version}

def create_initial_state(observations=None, *, home_config=None, use_defaults=True,
                         observed_at=None, disabled_agents=(), include_operations=True,
                         include_conditions=True, simulation_data=None, validate_contracts=True,
                          dependency_graph=True, parallel_agents=True, parallel_actions=True,
                          intent_analysis_enabled=False, check_outcomes=True, max_repairs=1,
                          max_replans=1, max_workers=4, checker="python", lean_executable=None,
                          search_mode="fixtures", scheduler_mode=None):
    scheduler_mode = scheduler_mode or SCHEDULER_MODE
    if scheduler_mode not in {"python", "llm"}:
        raise ValueError("SCHEDULER_MODE must be python or llm.")
    if search_mode not in {"fixtures", "live"}:
        raise ValueError("search_mode must be fixtures or live.")
    simulation_data = deepcopy(simulation_data or {})
    simulation_data.setdefault("search_results", deepcopy(SEARCH_FIXTURES) if search_mode == "fixtures" else {})
    simulation_data["search_mode"] = search_mode
    if checker == "lean":
        executable_name = "allocation_checker.exe" if os.name == "nt" else "allocation_checker"
        lean_executable = str(Path(lean_executable or Path(__file__).parent / "lean/.lake/build/bin" / executable_name).resolve())
        if not Path(lean_executable).is_file():
            raise FileNotFoundError("Build the Lean checker first: cd lean && lake build")
        if not dependency_graph or not validate_contracts or not check_outcomes:
            raise ValueError("Lean admission requires the graph, contract validation and observed outcome checks.")
    setup = deepcopy(HOME_DEFAULTS if home_config is None else home_config)
    config, home = {}, {}
    stamp = observed_at or (simulation_data or {}).get("now") or datetime.now(timezone.utc).isoformat()
    for section, scope in (("rooms", "room"), ("devices", "device"), ("services", "service")):
        config[section] = {}
        for entity_id, entity in setup[section].items():
            # Imported rooms may simply be ID -> display name.
            config[section][entity_id] = (entity if isinstance(entity, str) else
                                         {k: v for k, v in entity.items() if k != "initial"})
            if use_defaults and isinstance(entity, dict):
                for prop, (value, unit) in entity.get("initial", {}).items():
                    home[f"{scope}.{entity_id}.{prop}"] = make_observation(
                        value, source="simulation:initial", unit=unit, observed_at=stamp)
    home.update(deepcopy(observations or {}))
    registry = deepcopy(AGENT_REGISTRY)
    for agent in disabled_agents:
        registry[agent]["enabled"] = False
    request_id = uuid4().hex
    SIMULATORS[request_id] = Simulator(home)
    return dict(original_user_input="", request_id=request_id, infos=[], key_modifiers=[],
                task_queue=[], tasks={}, task_results=[], home_config=config, home_state=home,
                agent_registry=registry, collaboration_request={}, pending_task=None, pending_actions=[],
                action_graph={}, planning_task={}, planning_updates=[], active_actions=[],
                task_history=[], final_response="", repair_context={}, simulation_data=deepcopy(simulation_data or {}),
                settings={"version": "V3" if dependency_graph else "V2" if validate_contracts else "V1",
                            "dependency_graph": dependency_graph, "parallel_agents": parallel_agents,
                            "parallel_actions": parallel_actions, "scheduler_mode": scheduler_mode,
                             "intent_analysis_enabled": intent_analysis_enabled, "check_outcomes": check_outcomes,
                             "max_repairs": max_repairs, "max_replans": max_replans, "replans": 0,
                             "max_workers": max_workers, "checker": checker, "lean_executable": lean_executable,
                            "validate_contracts": validate_contracts, "include_operations": include_operations,
                            "include_conditions": include_conditions})

def get_available_agents(state, *, include_operations=None):
    if include_operations is None:
        include_operations = state["settings"]["include_operations"]
    available = {}
    for agent, card in state["agent_registry"].items():
        if not card["enabled"]:
            continue
        entry = {"description": card["description"], "devices": {}, "services": {}}
        used = set()
        for section in ("devices", "services"):
            for entity_id, entity in state["home_config"][section].items():
                skills = sorted(set(entity["skills"]) & set(card["capabilities"]) & card["operations"].keys())
                if entity["agent_id"] == agent and skills:
                    entry[section][entity_id] = {**entity, "skills": skills}
                    used.update(skills)
        if not used:
            continue
        entry["capabilities"] = sorted(used)
        if include_operations:
            entry["operations"] = {name: deepcopy(card["operations"][name]) for name in sorted(used)}
            if not state["settings"]["include_conditions"]:
                for definition in entry["operations"].values():
                    definition.pop("preconditions")
                    definition.pop("postconditions")
        available[agent] = entry
    return available

# Tools write the live simulator directly. Their output is only a report, never the after-state.
def device_tool(skill, args, simulator, now, simulation):
    home = simulator.snapshot()
    prefix = f"device.{args['device_id']}."
    get = lambda prop: deepcopy(home[prefix + prop]["value"])
    setters = {"set_brightness": ("brightness_pct", "brightness_pct"),
               "set_color_temperature": ("color_temperature_k", "color_temperature_k"),
               "set_temperature": ("target_temperature_c", "temperature_c"),
               "set_mode": ("mode", "mode"), "set_volume": ("volume_pct", "volume_pct")}
    if skill in setters:
        field, argument = setters[skill]
        changes = {field: args[argument]}
    elif skill in {"turn_on", "turn_off"}:
        changes = {"power": skill == "turn_on"}
        if skill == "turn_off":
            if prefix + "current_content" in home:
                changes["current_content"] = None
            if prefix + "playback_status" in home:
                changes.update(playback_status="stopped", current_track=None)
    elif skill in {"play", "pause", "stop"}:
        changes = {"playback_status": {"play": "playing", "pause": "paused", "stop": "stopped"}[skill]}
        if skill != "pause":
            changes["current_track"] = args["track"] if skill == "play" else None
    elif skill == "show_content":
        changes = {"current_content": {"type": args["content_type"], "value": args["content"]}}
    elif skill == "get_inventory":
        return {"inventory": get("inventory")}
    else:
        raise ValueError(f"Unbekannter Device-Skill: {skill}")
    output = deepcopy(changes)
    if skill == "pause":
        output["current_track"] = get("current_track")
    simulator.write({prefix + prop: value for prop, value in changes.items()}, now,
                    ref=simulation.get("_action_id"))
    return output

def clock_tool(skill, args, simulator, now, simulation):
    home = simulator.snapshot()
    prefix = f"service.{args['service_id']}."
    get = lambda prop: deepcopy(home[prefix + prop]["value"])
    stamp = datetime.fromisoformat(now.replace("Z", "+00:00"))
    if skill == "get_time":
        return {"current_time": stamp.astimezone(ZoneInfo(args["timezone"])).isoformat(),
                "timezone": args["timezone"]}
    if skill in {"set_timer", "set_alarm"}:
        timer = skill == "set_timer"
        field, id_key = ("timers", "timer_id") if timer else ("alarms", "alarm_id")
        item = {id_key: uuid4().hex, "label": args["label"]}
        item.update({"duration_seconds": args["duration_seconds"],
                     "due_at": (stamp + timedelta(seconds=args["duration_seconds"])).isoformat()}
                    if timer else {"at": args["at"], "enabled": True})
        simulator.write({prefix + field: get(field) + [item]}, now, ref=simulation.get("_action_id"))
        return {k: v for k, v in item.items() if k not in {"label", "enabled"}}
    elapsed = get("stopwatch_elapsed_seconds")
    if skill == "start_stopwatch":
        simulator.write({prefix + "stopwatch_running": True, prefix + "stopwatch_started_at": now},
                        now, ref=simulation.get("_action_id"))
        return {"running": True, "started_at": now, "elapsed_seconds": elapsed}
    elapsed += (stamp - datetime.fromisoformat(get("stopwatch_started_at").replace("Z", "+00:00"))).total_seconds()
    simulator.write({prefix + "stopwatch_running": False, prefix + "stopwatch_started_at": None,
                     prefix + "stopwatch_elapsed_seconds": elapsed}, now, ref=simulation.get("_action_id"))
    return {"running": False, "elapsed_seconds": elapsed}

def calendar_tool(skill, args, simulator, now, simulation):
    key = f"service.{args['service_id']}.events"
    events = simulator.snapshot()[key]["value"]
    if skill == "list_events":
        dt = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        events = [e for e in events if (args["start"] is None or dt(e["end"]) > dt(args["start"]))
                  and (args["end"] is None or dt(e["start"]) < dt(args["end"]))]
        return {"events": sorted(events, key=lambda e: (dt(e["start"]), e["event_id"]))}
    if skill == "create_event":
        event = {"event_id": uuid4().hex, **{k: args[k] for k in ("title", "start", "end", "location")}}
        simulator.write({key: events + [event]}, now, ref=simulation.get("_action_id"))
        return {"event": event}
    simulator.write({key: [e for e in events if e["event_id"] != args["event_id"]]}, now,
                    ref=simulation.get("_action_id"))
    return {"deleted_event_id": args["event_id"]}

def fixture_results(query, simulation):
    if simulation.get("search_mode", "fixtures") == "live":
        raise NotImplementedError("Live web search is not connected yet; use search_mode='fixtures'.")
    query = " ".join(query.casefold().split())
    fixtures = {" ".join(k.casefold().split()): v for k, v in simulation.get("search_results", {}).items()}
    key = next((k for k in sorted(fixtures, key=len, reverse=True)
                if k == query or re.search(r"\b" + re.escape(k) + r"\b", query)), None)
    if key is None:
        raise ValueError(f"InformationError: no search fixture for {query!r}.")
    return fixtures[key]


def search_tool(skill, args, home, now, simulation):
    rows = fixture_results(args["query"], simulation)
    if not rows:
        raise ValueError("InformationError: search returned no usable information.")
    return {"query": args["query"], "provider": "simulation",
            "results": deepcopy(rows[:args["max_results"]])}


def freeze_goal(task, state):
    """Keep the planner's structured goal unchanged through repairs."""
    if "goal" in task:
        return task["goal"]
    operations = state["agent_registry"][task["agent_id"]]["operations"]
    requirements = []
    for action in task.get("requirements", task.get("actions", [])):
        parameters = operations.get(action["skill_id"], {}).get("parameters", {})
        arguments = {k: deepcopy(v) for k, v in action.get("arguments", {}).items()
                     if k in parameters and (isinstance(v, dict) and "from_task" in v or
                         Draft202012Validator(parameters[k], format_checker=FormatChecker()).is_valid(v))}
        requirements.append({"skill_id": action["skill_id"], "arguments": arguments})
    return deepcopy({"text": task["action"], "scope": task.get("scope", []),
                     "constraints": task.get("constraints", []), "requirements": requirements,
                     "source_field": task.get("source_field"), "literal_content": task.get("literal_content", False)})


def resolve_arguments(state, action):
    """Bind data to confirmed current outputs; output text never comes from a placeholder."""
    sources = []
    records = list(state["action_graph"].values()) if state["settings"]["dependency_graph"] else [
        {**r, "task": {"task_id": r["task_id"].rsplit(":", 1)[0]}}
        for r in state["task_results"]]
    def resolve(value):
        if not isinstance(value, dict) or "from_task" not in value:
            return deepcopy(value)
        source, field = value["from_task"], value["field"]
        provider = next((a for a in reversed(records) if (a["task_id"] == source or a["task"]["task_id"] == source)
                         and a["status"] == "success" and (not state["settings"]["check_outcomes"] or a.get("verified"))
                         and field.split('.')[0] in a.get("output", {})), None)
        if provider is None:
            raise ValueError(f"InformationError: no confirmed {field} output from {source}.")
        actual = provider["output"]
        try:
            for key in field.split('.'):
                actual = actual[int(key)] if isinstance(actual, list) else actual[key]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"InformationError: output field {field} missing from {source}.") from exc
        if field.split('.')[0] == "results" and (actual is None or isinstance(actual, (list, dict, str)) and not actual):
            raise ValueError("InformationError: referenced search results are empty.")
        sources.append(provider["task_id"])
        if "add" in value:
            if type(actual) not in {int, float} or type(value["add"]) not in {int, float}:
                raise ValueError("InformationError: numeric offset needs numeric values.")
            actual += value["add"]
        if value.get("format") == "text":
            if isinstance(actual, list) and actual and all(isinstance(x, dict) and "title" in x and ("content" in x or "snippet" in x) for x in actual):
                return "\n\n".join(x["title"] + "\n" + x.get("content", x.get("snippet", "")) for x in actual)
            return actual if isinstance(actual, str) else json.dumps(actual, ensure_ascii=False, indent=2)
        return deepcopy(actual)
    args = {key: resolve(value) for key,value in action.get("arguments", {}).items()}
    if action["skill_id"] == "show_content":
        goal = action["task"].get("goal", {})
        data_kind = goal.get("source_field")
        if data_kind and not goal.get("literal_content") and not sources:
            owners = set(action["task"].get("depends_on", [])) | {action["task"]["task_id"] + ":help"}
            providers = [a for a in records if a["task"]["task_id"] in owners
                         and a["status"] == "success" and data_kind in a.get("output", {})]
            if not providers:
                raise ValueError(f"InformationError: retrieve actual {data_kind} before displaying it.")
            args["content"] = resolve({"from_task": providers[-1]["task_id"], "field": data_kind, "format": "text"})
            args["content_type"] = "text"
        elif not isinstance(args.get("content"), str) or not args["content"].strip():
            raise ValueError("InformationError: display content must not be empty.")
    return args, list(dict.fromkeys(sources))


def check_goal(state, task, actions, *, complete=False):
    goal = task.get("goal", {})
    requirements = goal.get("requirements", [])
    if not requirements or task.get("parent"):
        return
    def matches(action, requirement):
        if action["skill_id"] != requirement["skill_id"]:
            return False
        arguments = requirement["arguments"]
        if any(isinstance(v, dict) and "from_task" in v for v in arguments.values()):
            arguments = resolve_arguments(state, {"task": task, **requirement})[0]
        return all(action["arguments"].get(k) == v or
                   k == "content" and action.get("source_actions") and not goal.get("literal_content")
                   for k, v in arguments.items())
    allowed = {r["skill_id"] for r in requirements}
    operations = state["agent_registry"][task["agent_id"]]["operations"]
    if any(c.get("equals") is True and c.get("state", "").endswith(".power")
           for skill in allowed for c in operations.get(skill, {}).get("preconditions", [])):
        allowed.add("turn_on")
    for action in actions:
        own = [r for r in requirements if r["skill_id"] == action["skill_id"]]
        if action["skill_id"] not in allowed or own and not any(matches(action, r) for r in own):
            raise ValueError("GoalError: repair changed the requested operation, target or arguments.")
    if complete and any(not any(matches(a, r) for a in actions) for r in requirements):
        raise ValueError("GoalError: original requested outcome is still missing.")


TOOLS = {
    (agent, skill): (clock_tool if agent == "clock" else calendar_tool if agent == "calendar"
                    else search_tool if agent == "search_engine" else device_tool)
    for agent, card in AGENT_REGISTRY.items() for skill in card["capabilities"]
}
UNITS = {"brightness_pct": "%", "color_temperature_k": "K", "volume_pct": "%",
         "target_temperature_c": "degC", "stopwatch_elapsed_seconds": "s"}

def action_footprint(state, action):
    """One registry-derived footprint reused by scheduling, dispatch, Lean and repair."""
    definition = state["agent_registry"][action["task"]["agent_id"]]["operations"][action["skill_id"]]
    args = action["arguments"]
    reads = set(definition["reads"])
    for condition in definition["preconditions"]:
        reads.update(condition[k] for k in ("state", "equals_state") if k in condition)
    target_key = "device_id" if "device_id" in definition["parameters"] else "service_id"
    reads = {k.format_map(args) for k in reads}
    for dep in action.get("depends_on", []):
        reads.update(state["action_graph"].get(dep, {}).get("evidence_versions", {}))
    return (reads, {k.format_map(args) for k in definition["writes"]},
            (target_key, args[target_key]))

# Only the condition vocabulary currently used in the Cards, without a generic interpreter.
def check_condition(condition, arguments, home, context):
    def dt(value):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Zeitstempel braucht eine Zeitzone.")
        return stamp

    def value(path):
        return home[path.format_map(arguments)]["value"]

    rule = condition.get("rule")
    if rule is None:
        actual = (value(condition["state"]) if "state" in condition else
                  context["output"][condition["output"]] if "output" in condition else context[condition["context"]])
        if condition.get("known") is True:
            return True  # a present None is known; a missing key raises an error
        expected = (condition["equals"] if "equals" in condition else
                    value(condition["equals_state"]) if "equals_state" in condition else
                    context["output"][condition["equals_output"]] if "equals_output" in condition else
                    context["before"][condition["equals_before"].format_map(arguments)]["value"] if "equals_before" in condition else
                    arguments[condition["equals_arg"]])
        return actual == expected and (type(actual) is type(expected)
                                      if isinstance(actual, bool) or isinstance(expected, bool) else True)
    if rule == "time_available":
        dt(context["now"])
        return True
    if rule == "valid_timezone":
        ZoneInfo(arguments[condition["argument"]])
        return True
    if rule == "future_time":
        return dt(arguments[condition["argument"]]) > dt(context["now"])
    if rule in {"start_before_end", "optional_time_window"}:
        start, end = arguments[condition["start_arg"]], arguments[condition["end_arg"]]
        start, end = None if start is None else dt(start), None if end is None else dt(end)
        return ((start is None or end is None) and rule == "optional_time_window") or (
            start is not None and end is not None and start < end)
    if rule == "unique_item":
        return sum(row[condition["id_field"]] == arguments[condition["id_arg"]]
                   for row in value(condition["state"])) == 1
    if rule == "nonempty_state":
        actual = value(condition["state"])
        return isinstance(actual, str) and bool(actual.strip())
    if rule == "valid_content":
        if arguments["content_type"] == "text":
            return True
        parsed = urlsplit(arguments["content"])
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    if rule == "track_available":
        return context["mode"] == "simulation"  # the current tools only implement simulation
    if rule == "stopwatch_ready":
        prefix = "service.{service_id}."
        elapsed = value(prefix + "stopwatch_elapsed_seconds")
        running = value(prefix + "stopwatch_running")
        return (type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0
                and running is condition["running"] and bool(dt(context["now"]))
                and (not running or dt(value(prefix + "stopwatch_started_at")) <= dt(context["now"])))
    # Outcome predicates use the observed after-state and the before-state, not a 'done' flag.
    output, before = context.get("output", {}), context.get("before", {})
    if rule == "content_matches_request":
        return value(condition["state"]) == {"type": arguments["content_type"], "value": arguments["content"]}
    if rule == "search_matches_source":
        expected = fixture_results(arguments["query"], context["simulation"])[:arguments["max_results"]]
        return output["query"] == arguments["query"] and output["provider"] == "simulation" and output["results"] == expected
    if rule in {"event_added", "event_removed", "events_selected"}:
        path = condition["state"].format_map(arguments)
        previous, current = before[path]["value"], home[path]["value"]
        if rule == "event_added":
            event = output["event"]
            return (all(event[k] == arguments[k] for k in ("title", "start", "end", "location"))
                    and (sum(e == event for e in current) == 1 if context.get("retained") else
                         current == previous + [event] and not any(e["event_id"] == event["event_id"] for e in previous)))
        if rule == "event_removed":
            return (output["deleted_event_id"] == arguments["event_id"]
                    and (not any(e["event_id"] == arguments["event_id"] for e in current) if context.get("retained") else
                         current == [e for e in previous if e["event_id"] != arguments["event_id"]]))
        expected = [e for e in current if (arguments["start"] is None or dt(e["end"]) > dt(arguments["start"]))
                    and (arguments["end"] is None or dt(e["start"]) < dt(arguments["end"]))]
        return output["events"] == sorted(expected, key=lambda e: (dt(e["start"]), e["event_id"]))
    if rule == "clock_result":
        skill = condition["operation"]
        prefix = f"service.{arguments['service_id']}."
        now = dt(context["now"])
        if skill == "get_time":
            return output["timezone"] == arguments["timezone"] and output["current_time"] == now.astimezone(ZoneInfo(arguments["timezone"])).isoformat()
        if skill in {"set_timer", "set_alarm"}:
            timer = skill == "set_timer"
            prop, id_key = ("timers", "timer_id") if timer else ("alarms", "alarm_id")
            new = [e for e in value(prefix + prop) if e[id_key] == output[id_key]]
            if len(new) != 1 or (not context.get("retained") and value(prefix + prop) != before[prefix + prop]["value"] + new):
                return False
            item = new[0]
            return (item["label"] == arguments["label"] and
                    (item["duration_seconds"] == output["duration_seconds"] == arguments["duration_seconds"]
                     and dt(item["due_at"]) == dt(output["due_at"]) == now + timedelta(seconds=arguments["duration_seconds"])
                     if timer else item["at"] == output["at"] == arguments["at"] and item["enabled"] is True))
        running = skill == "start_stopwatch"
        elapsed = before[prefix + "stopwatch_elapsed_seconds"]["value"]
        if not running:
            elapsed += (now - dt(before[prefix + "stopwatch_started_at"]["value"])).total_seconds()
        return (value(prefix + "stopwatch_running") is running and output["running"] is running
                and output["elapsed_seconds"] == elapsed
                and (output["started_at"] == value(prefix + "stopwatch_started_at") == context["now"] if running else
                     value(prefix + "stopwatch_started_at") is None and value(prefix + "stopwatch_elapsed_seconds") == elapsed))
    raise ValueError(f"Unbekannte Vertragsregel: {rule}")

def validate_action(state):
    action = deepcopy(state["pending_actions"][0])
    task, args, skill = action["task"], action.get("arguments", {}), action.get("skill_id", "")
    try:
        home = {**state["home_state"], **SIMULATORS[state["request_id"]].snapshot()}
        args, source_actions = resolve_arguments(state, action)
        action.update(arguments=args, source_actions=source_actions,
                      depends_on=list(dict.fromkeys(action.get("depends_on", []) + source_actions)))
        card = state["agent_registry"][task["agent_id"]]
        if not card["enabled"] or skill not in card["capabilities"] or skill not in card["operations"]:
            raise ValueError("Agent oder Skill ist nicht verfuegbar.")
        definition = card["operations"][skill]
        for key, schema in definition["parameters"].items():
            if key not in args and "default" in schema:
                args[key] = deepcopy(schema["default"])
        if state["settings"]["validate_contracts"]:
            schema = {"type": "object", "properties": definition["parameters"],
                      "required": list(definition["parameters"]),
                      "additionalProperties": False}
            Draft202012Validator(schema, format_checker=FormatChecker()).validate(args)
            # NaN/Infinity are Python numbers but not valid JSON arguments.
            json.dumps(args, allow_nan=False)
        elif set(definition["parameters"]) - args.keys():
            raise ValueError("Pflichtargument fehlt.")
        check_goal(state, task, [action])
        target_key = "device_id" if "device_id" in definition["parameters"] else "service_id"
        section = "devices" if target_key == "device_id" else "services"
        target = state["home_config"][section][args[target_key]]
        if target["agent_id"] != task["agent_id"] or skill not in target["skills"]:
            raise ValueError("Agent, Ziel und Skill passen nicht zusammen.")
        if target_key == "device_id" and task.get("scope"):
            if args[target_key] not in task["scope"] and target.get("room_id") not in task["scope"]:
                raise ValueError("Geraet liegt ausserhalb des zugewiesenen Auftragsbereichs.")
        now = state["simulation_data"].get("now") or datetime.now(timezone.utc).isoformat()
        context = {"now": now, "mode": "simulation",
                   "search_available": state["simulation_data"].get("search_available", True)}
        if state["settings"]["validate_contracts"]:
            for condition in definition["preconditions"]:
                if not check_condition(condition, args, home, context):
                    raise ValueError(f"Precondition nicht erfuellt: {condition}")
        action.update(arguments=args, now=now, agent_id=task["agent_id"], depends_on=action.get("depends_on", []))
        reads, writes, target = action_footprint(state, action)
        bindings = {k: home.get(k, {}).get("version", 0) for k in reads | writes}
        for dep in action.get("depends_on", []):
            bindings.update(state["action_graph"].get(dep, {}).get("evidence_versions", {}))
        action.update(footprint=(sorted(reads), sorted(writes), target), bound_versions=bindings,
                      dispatch_until=time.monotonic() + 30)
        return Command(update={"pending_actions": [action, *state["pending_actions"][1:]], "home_state": home}, goto="execute_action")
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError, ValidationError, NotImplementedError) as exc:
        result: TaskResult = dict(task_id=action["task_id"], attempt_id=uuid4().hex,
                                 agent_id=task["agent_id"], skill_id=skill, arguments=args,
                                  status="failed", verified=False, output={}, observations={},
                                  error=f"ValidationError: {exc.message if isinstance(exc, ValidationError) else exc}")
        return Command(update={"task_results": [result]}, goto="complete_round")

def execute_action(state):
    action = state["pending_actions"][0]
    task, skill, args = action["task"], action["skill_id"], action["arguments"]
    result: TaskResult = dict(task_id=action["task_id"], attempt_id=uuid4().hex,
                             agent_id=task["agent_id"], skill_id=skill, arguments=deepcopy(args),
                              status="failed", verified=False, output={}, observations={}, error=None)
    simulator = SIMULATORS[state["request_id"]]
    admitted = False
    try:
        before = simulator.admit(action)
        admitted = True
        parameters = state["agent_registry"][task["agent_id"]]["operations"][skill]["parameters"]
        target_key = "device_id" if "device_id" in parameters else "service_id"
        output = TOOLS[(task["agent_id"], skill)](
            skill, args, simulator, action["now"], {**state["simulation_data"], "_action_id": action["task_id"]})
        observed = simulator.snapshot()
        # Observe actual device/service writes; returned dictionaries cannot fabricate state changes.
        result.update(status="success", output={target_key: args[target_key], **output})
        if state["settings"]["check_outcomes"]:
            definition = state["agent_registry"][task["agent_id"]]["operations"][skill]
            schema = {"type": "object", "properties": definition["output"], "required": list(definition["output"]),
                      "additionalProperties": False}
            Draft202012Validator(schema, format_checker=FormatChecker()).validate(result["output"])
            context = {"now": action["now"], "output": result["output"], "before": before,
                       "simulation": state["simulation_data"]}
            if not all(check_condition(c, args, observed, context) for c in definition["postconditions"]):
                raise ValueError("OutcomeError: observed outcome does not match the operation contract.")
            result["verified"] = True
        result.update(before={k: before[k] for k in set(action["footprint"][0]) | set(action["footprint"][1]) if k in before},
                      evidence_versions={k: observed[k]["version"] for k in
                                         set(action["footprint"][0]) | set(action["footprint"][1]) if k in observed},
                      valid_until=time.time() + (600 if not action["footprint"][1] else 3600))
        # Also record read-only results with source/time metadata, outside the device-property namespace.
        result["observations"].update({f"result.{action['task_id']}.{key}": make_observation(
            value, source="simulation:result", observed_at=action["now"])
            for key, value in result["output"].items()})
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError, ValidationError, NotImplementedError) as exc:
        result["status"] = "failed" if admitted else "stale"
        result["error"] = f"ToolError: {type(exc).__name__}: {exc}"
    finally:
        if admitted:
            observed = simulator.snapshot()
            result["observations"].update({k: observed[k] for k in action["footprint"][1] if k in observed})
            simulator.release(action["task_id"])
    return Command(update={"task_results": [result]}, goto="complete_round")


def _ask(instructions, *, role="planner", expected=None, **context):
    llm = get_llm(role)
    result = (llm | JsonOutputParser()).invoke([
        SystemMessage(content=instructions + "\nReturn only the requested JSON object; no explanation."),
        HumanMessage(content=json.dumps(context, ensure_ascii=False, separators=(",", ":"))),
    ])
    if not isinstance(result, dict):
        raise ValueError("Das Modell muss ein JSON-Objekt liefern.")
    for field, kind in (expected or {}).items():
        if field not in result or not isinstance(result[field], kind):
            raise ValueError(f"Modellantwort: {field!r} fehlt oder ist kein {kind.__name__}.")
    return result

def human(state):
    text = interrupt("wait for user input...")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nicht leere Texteingabe erwartet.")
    return Command(update={"original_user_input": text},
                   goto="intent_analysis" if state["settings"]["intent_analysis_enabled"] else "task_planner")

def intent_analysis(state):
    result = _ask('Extract intents/facts and HOW, WHEN, WHERE, quantities and negation. '
                  'Output: {"infos": ["..."], "key_modifiers": ["..."]}.',
                  role="intent_analysis", expected={"infos": list},
                  user_message=state["original_user_input"])
    return Command(update={"infos": result["infos"], "key_modifiers": result.get("key_modifiers", [])}, goto="task_planner")

def _scope_ids(scope):
    """Room/device IDs of a planner scope; models also write {"room_id": ..., "device_id": ...} or one ID."""
    values = scope.values() if isinstance(scope, dict) else scope if isinstance(scope, list) else [scope]
    return [i for v in values for i in (v if isinstance(v, list) else [v]) if isinstance(i, str) and i]

def task_planner(state):
    repair = state["repair_context"]
    if state["task_queue"] and not repair:
        if state["settings"]["dependency_graph"]:
            return Command(goto="plan_agents")
        task = state["task_queue"][0]
        if task.get("actions"):
            actions = [{**a, "task": task, "task_id": f"{task['task_id']}:{i}"} for i, a in enumerate(task["actions"])]
            return Command(update={"pending_actions": actions}, goto="validate_action")
        return Command(goto=task["agent_id"] + "_agent")
    if (state["tasks"] or state["final_response"]) and not repair:
        return Command(goto="finish")
    available = get_available_agents(state)
    graph_mode = state["settings"]["dependency_graph"]
    graph_instruction = (' Give each task a unique task_id (e.g. t1), scope (list of room/device ID strings, '
                         'e.g. scope=["living_room"]; never an object), constraints '
                         '(conditions, timing, quantities, negation), and depends_on list of task-ID strings. '
                         'Example: depends_on=["t1", "t2"]. Never put objects or dictionaries in depends_on. '
                         'Independent goals have depends_on=[]. A task that needs earlier effects/information '
                         'depends on that predecessor. Combine related goals on the same target or order them. '
                         if graph_mode else '')
    context = dict(original_input=state["original_user_input"],
                   available_agents={agent: {"description": card["description"], "skills": card["capabilities"],
                       "devices": card["devices"], "services": card["services"],
                       "operations": {skill: {k: d[k] for k in ("parameters", "preconditions") if k in d}
                                      for skill, d in card.get("operations", {}).items()}}
                                     for agent, card in available.items()},
                   search_fixture_queries=list(state["simulation_data"].get("search_results", {})),
                   rooms=state["home_config"]["rooms"],
                   home_state={key: o["value"] for key, o in state["home_state"].items() if not key.startswith("result.")})
    if state["settings"]["intent_analysis_enabled"]:
        context["intent_analysis"] = {"infos": state["infos"], "key_modifiers": state["key_modifiers"]}
    if repair:
        context.update(repair=repair, tasks=state["tasks"], confirmed_results=state["task_history"])
    try:
        result = _ask('You are the central task planner. Plan the ENTIRE original_input, not only its first clause. '
                      'Create one high-level task per distinct requested goal, assigned to its primary domain agent. '
                      'AND or THEN may join multiple goals; preserve THEN ordering and ALL conditions, timing and locations. '
                      f'Allowed agent_id values: {json.dumps(list(available))}. '
                      'Use these exact IDs, never node names such as audio_system_agent or invented aliases. '
                      'audio_system handles music/playback. '
                      'Use actual goal text for action and explicit room/device IDs for scope. '
                      'Give requirements=[{skill_id, arguments}] for requested outcomes, with all known targets/values; '
                      'for ALL devices include one requirement per device. Prerequisites belong in actions. '
                      'For fully specified goals add actions including prerequisites and local dependencies; '
                      'they run directly through Python/Lean without a domain-agent call. Otherwise omit actions. '
                      'For predecessor data use {"from_task":"t1","field":"results.0.content"} as an argument value. '
                      'References to records/lists may add "format":"text". Declare that task in depends_on. '
                      'For information displays set source_field to events, inventory or results; '
                      'literal_content=true only for explicit literal user text/URLs. Never invent missing data. '
                      'Keep task.goal unchanged during repair; prerequisites alone do not complete its outcomes. '
                      'The original_input is authoritative; any intent_analysis is optional, fallible supporting context. '
                      'Check that EVERY requested goal is represented. Do not invent capabilities or extra goals. '
                      'When repair is present, replace every ID in repair.task_ids, keeping IDs and original goals. Do not renumber tasks. '
                      'You may add information-only tasks with kind="information" and parent=<affected task ID>. '
                      'Preserve dependencies on retained tasks; do not repeat completed unrelated goals. '
                      + graph_instruction + ' Output: {"task_queue": [{"task_id": "t1", "agent_id": "lighting", '
                      '"action": "Turn on the living room ceiling light", "actions": [{"task_id": "a1", "skill_id": "turn_on", '
                      '"arguments": {"device_id": "light_lr_ceiling"}, "depends_on": []}], '
                      '"requirements": [{"skill_id": "turn_on", "arguments": {"device_id": "light_lr_ceiling"}}]}], '
                      '"response": "reason if no tasks"}. The actions field is optional.',
                      role="planner", expected={"task_queue": list}, **context)
        queue = []
        for index, raw in enumerate(result["task_queue"]):
            if not isinstance(raw, dict) or not isinstance(raw.get("action"), str) or not raw["action"].strip():
                raise ValueError("Planner-Auftrag braucht agent_id und eine nicht leere action.")
            if raw.get("agent_id") not in available:
                raise ValueError(f"PlanError: agent_id={raw.get('agent_id')!r} is unavailable; "
                                 f"allowed IDs: {list(available)}.")
            task_id = raw.get("task_id", f"t{index + 1}") if graph_mode else f"{state['request_id']}:{index}"
            queue.append(dict(task_id=task_id, agent_id=raw["agent_id"], action=raw["action"],
                              depends_on=raw.get("depends_on", []) if graph_mode else [], status="pending",
                              scope=_scope_ids(raw.get("scope", [])), constraints=raw.get("constraints", [])))
            if raw.get("actions"):
                queue[-1]["actions"] = raw["actions"]
            queue[-1]["goal"] = (deepcopy(state["tasks"][task_id]["goal"])
                                  if repair and task_id in state["tasks"] else freeze_goal({**raw, **queue[-1]}, state))
            if repair and task_id not in repair["task_ids"]:
                if raw.get("kind") != "information" or raw.get("parent") not in repair["task_ids"]:
                    raise ValueError(f"PlanError: unexpected repair task {task_id!r}; expected IDs {repair['task_ids']}. "
                                     "Extra tasks require kind=information and parent=<affected task ID>.")
                queue[-1]["parent"] = raw["parent"]
        if graph_mode:
            for t in queue:
                refs = [v["from_task"] for a in t.get("actions", []) for v in a.get("arguments", {}).values()
                        if isinstance(v, dict) and "from_task" in v]
                t["depends_on"] = list(dict.fromkeys(t["depends_on"] + [r.split("/")[0] for r in refs if r.split("/")[0] != t["task_id"]]))
            ids = {t["task_id"] for t in queue}
            allowed = ids | (state["tasks"].keys() if repair else set())
            if len(ids) != len(queue) or any(not isinstance(t["depends_on"], list)
                    or any(not isinstance(d, str) for d in t["depends_on"])
                    or set(t["depends_on"]) - allowed for t in queue):
                raise ValueError("PlanError: doppelte Task-ID; depends_on muss eine Liste bekannter Task-ID-Strings sein.")
            merged = {**state["tasks"], **{t["task_id"]: t for t in queue}} if repair else {t["task_id"]: t for t in queue}
            tuple(TopologicalSorter({t["task_id"]: t["depends_on"] for t in merged.values()}).static_order())
            if repair:
                if not set(repair["task_ids"]).issubset(ids):
                    raise ValueError("PlanError: repair must preserve every affected task ID.")
                for t in queue:
                    if t["task_id"] not in state["tasks"]:
                        continue
                    # A repair may change routing/dependencies, never the user's assigned goal or constraints.
                    for key in ("action", "scope", "constraints", "repairs"):
                        if key in state["tasks"][t["task_id"]]:
                            t[key] = deepcopy(state["tasks"][t["task_id"]][key])
                settings = {**state["settings"], "replans": state["settings"]["replans"] + 1}
                retained = {i: a for i, a in state["action_graph"].items()
                            if a["task"]["task_id"] not in ids or a["status"] == "success"}
                for t in queue:
                    t["generation"] = state["tasks"].get(t["task_id"], {}).get("generation", 0) + 1
                return Command(update={"tasks": {**state["tasks"], **{t["task_id"]: t for t in queue}},
                                       "action_graph": retained, "repair_context": {}, "settings": settings}, goto="plan_agents")
        return Command(update={"task_queue": queue, "tasks": {t["task_id"]: t for t in queue},
                               "final_response": result.get("response", "") if not queue else ""},
                       goto=("plan_agents" if graph_mode else "task_planner") if queue else "finish")
    except (ValueError, KeyError, TypeError) as exc:
        if not repair:
            return Command(update={"final_response": json.dumps(
                {"status": "failed", "error": f"PlannerError: {exc}"}, ensure_ascii=False)}, goto="finish")
        error = f"PlannerRepairError: {exc}"
        settings = {**state["settings"], "replans": state["settings"]["replans"] + 1}
        if settings["replans"] < settings["max_replans"]:
            return Command(update={"settings": settings, "repair_context": {**repair, "reason": error}},
                           goto="task_planner")
        tasks = deepcopy(state["tasks"])
        for task_id in repair["task_ids"]:
            tasks[task_id].update(status="failed", terminal=True, error=error)
        return Command(update={"tasks": tasks, "settings": settings, "repair_context": {}}, goto="plan_agents")


def _agent(state, agent):
    available = get_available_agents(state)
    if agent not in available:
        raise ValueError("Agent ist nicht verfuegbar.")
    graph_mode = state["settings"]["dependency_graph"]
    help_request = state["collaboration_request"]
    if graph_mode:
        task = deepcopy(state["planning_task"])
        helping, pending = bool(task.get("parent")), task.get("help_done", False)
    else:
        helping = help_request.get("target") == agent
        task = ({"task_id": help_request["task_id"] + ":help", "agent_id": agent, "action": help_request["request"]}
                if helping else state["task_queue"][0])
        pending = state["pending_task"]
    graph_instruction = ('Give each action a local task_id (a1, a2, ...) and depends_on list of local action IDs. '
                         'Use [] for independent actions, and explicit predecessors for prerequisites. '
                         'Example: turn_on a1 depends_on=[], set_brightness a2 depends_on=["a1"]. '
                         'Do not request collaboration when collaboration_allowed is false. '
                         'For repair, propose only missing/corrective actions; do not repeat retained successful actions. '
                         'If repair requires changing cross-agent routing/dependencies, return actions=[] with '
                         'replan_required=true and reason. Stay within the assigned goal and its constraints. '
                         if graph_mode else '')
    task.setdefault("goal", freeze_goal(task, state))
    instructions = ('Preserve task.goal and its required targets/values; complete all outcomes after prerequisites. '
                    'Use {"from_task":"t1","field":"results.0.content"} for confirmed predecessor data; '
                    'use "format":"text" for records/lists. Retrieve missing data through collaboration. '
                    'Return ordered calls with concrete IDs using operation parameters/conditions. '
                    'For music playback on a powered-off device, plan turn_on followed by play with the requested '
                    'track; when already powered on, plan play directly. Do not invent catalog entries or genre matches. '
                    'Use current state, cards and previous confirmed results. Include prerequisite actions yourself. '
                    'Never fabricate inventory, time, appointments or search facts. For ALL devices emit separate calls. '
                    'Timers/alarms only store entries; delayed device actions are not supported. '
                    'Request one other agent only for missing information; do not ask it to execute your goal. '
                    'When requesting help, return actions=[] and the collaboration_request object. '
                    'After receiving help, complete without another collaboration request. ' + graph_instruction + 'Output: '
                    '{"actions": [{"skill_id": "<operation>", "arguments": {"<parameter>": "<value>"}}], '
                    '"collaboration_request": {"target": "fridge", "request": "specific question"} or {}}.')
    try:
        result = _ask(instructions, role=agent, expected={"actions": list},
                      original_input=state["original_user_input"],
                      search_fixture_queries=list(state["simulation_data"].get("search_results", {})) if agent == "search_engine" else [],
                      agent=agent, task=task, agent_card=available[agent], available_agents=list(available),
                      home_state={key: observation["value"] for key, observation in state["home_state"].items()
                                  if not key.startswith("result.")},
                      task_history=[{**h, "results": [r for r in h["results"]
                          if state["action_graph"].get(r["task_id"], {}).get("status") == "success"]}
                          for h in state["task_history"]] if graph_mode else state["task_history"],
                      repair={"attempt": task.get("repairs", 0), "error": task.get("error"),
                              "retained_actions": [a for a in state["action_graph"].values()
                                                   if a["task"]["task_id"] == task["task_id"] and a["status"] == "success"],
                              "previous_results": [{**r, "retained": state["action_graph"].get(r["task_id"], {}).get("status") == "success"}
                                                   for r in state["task_results"]
                                                   if r["task_id"].rsplit("/", 1)[0] == task["task_id"]]} if graph_mode else {},
                      now=state["simulation_data"].get("now") or datetime.now(timezone.utc).isoformat(),
                      collaboration_allowed=not helping and (not pending if graph_mode else pending is None))
    except Exception as exc:
        if not graph_mode:
            raise
        return Command(update={"planning_updates": [{"task_id": task["task_id"], "error": str(exc)}]}, goto="collect_plans")
    if graph_mode:
        return Command(update={"planning_updates": [{"task_id": task["task_id"], "response": result}]}, goto="collect_plans")
    request = result.get("collaboration_request") or {}
    if request.get("target"):
        if helping or pending is not None or request["target"] == agent or request["target"] not in available:
            raise ValueError("Ungueltige oder wiederholte Kollaboration.")
        return Command(update={"collaboration_request": {**request, "requester": agent, "task_id": task["task_id"]},
                               "pending_task": task}, goto=request["target"] + "_agent")
    actions = result["actions"]
    if not actions:
        raise ValueError("Agent muss mindestens einen strukturierten Tool-Aufruf liefern.")
    if any(not isinstance(action, dict) for action in actions):
        raise ValueError("Jeder Tool-Aufruf muss ein Dictionary sein.")
    if helping and any(a.get("skill_id") not in {"get_inventory", "get_time", "list_events", "search"} for a in actions):
        raise ValueError("Kollaboration dient nur zur Informationsbeschaffung.")
    actions = [{**action, "task": task, "task_id": f"{task['task_id']}:{index}"}
               for index, action in enumerate(actions)]
    return Command(update={"pending_actions": actions}, goto="validate_action")

# V3: only these coordinators update shared tasks/plans. Parallel nodes return lists.
def plan_agents(state):
    home = {**state["home_state"], **SIMULATORS[state["request_id"]].snapshot()}
    state = {**state, "home_state": home}
    tasks, graph = deepcopy(state["tasks"]), deepcopy(state["action_graph"])
    invalid, changed_keys = set(), set()
    for action_id, action in graph.items():
        if action["status"] != "success" or not action.get("verified"):
            continue
        if action.get("superseded"):
            continue  # a historical intermediate goal stays fulfilled, even if the final goal later breaks
        reads, writes, _ = action["footprint"]
        reads, writes = set(reads), set(writes)
        touched = {k for k, v in action.get("evidence_versions", {}).items()
                   if home.get(k, {}).get("version", 0) != v}
        expired = time.time() >= action.get("valid_until", float("inf"))
        if not touched and not expired:
            continue
        task_id = action["task"]["task_id"]
        needed = tasks[task_id]["status"] != "success" or any(
            a["status"] == "pending" and action_id in a["depends_on"] for a in graph.values()
        ) or any(task_id in t["depends_on"] and t["status"] not in {"success", "failed", "blocked"}
                 for t in tasks.values())
        # Historical read results need not be reacquired after all consumers have completed.
        if not writes and not needed:
            continue
        try:
            context = {"now": action["now"], "output": action["output"], "before": action["before"],
                       "simulation": state["simulation_data"], "retained": True}
            valid = bool(writes) and all(check_condition(c, action["arguments"], home, context)
                for c in state["agent_registry"][action["agent_id"]]["operations"][action["skill_id"]]["postconditions"])
        except (KeyError, ValueError, TypeError):
            valid = False
        if valid:
            action["evidence_versions"] = {k: home[k]["version"] for k in reads | writes if k in home}
            action["valid_until"] = time.time() + 3600
        else:
            invalid.add(action_id)
            changed_keys.update(touched)

    # Vaibhav's repair region: dependency descendants AND actions touching changed state.
    affected = invalid | {i for i, a in graph.items() if a["status"] in {"failed", "blocked"}}
    while True:
        previous = set(affected)
        for i in tuple(affected):
            changed_keys.update(graph[i]["footprint"][1])
        for i, a in graph.items():
            reads, writes, _ = a["footprint"]
            if affected.intersection(a["depends_on"]):
                affected.add(i)
                if a["status"] == "success" and not a.get("superseded"):
                    invalid.add(i)
            elif (set(reads) | set(writes)) & changed_keys:
                affected.add(i)
        if affected == previous:
            break
    for i in affected:
        a = graph[i]
        if a["status"] != "success" or i in invalid:
            task = tasks[a["task"]["task_id"]]
            if not task.get("terminal"):
                task.update(status="failed", error=task.get("error") or "Observed outcome or dependency is no longer valid.")
    graph = {i: a for i, a in graph.items() if i not in invalid}
    for a in graph.values():
        if a.get("superseded"):
            a["depends_on"] = [d for d in a["depends_on"] if d in graph]
    for task in list(tasks.values()):
        if task["status"] != "failed" or task.get("terminal"):
            continue
        # Every retained success was rechecked above; only the domain agent replans missing steps.
        own = {i: a for i, a in graph.items() if a["task"]["task_id"] == task["task_id"]}
        graph = {i: a for i, a in graph.items() if i not in own or a["status"] == "success"}
        if task.get("requires_replan") or task.get("repairs", 0) >= state["settings"]["max_repairs"]:
            if not task.get("parent") and state["settings"]["replans"] < state["settings"]["max_replans"]:
                affected = {task["task_id"]} | {t["task_id"] for t in tasks.values()
                                               if t["status"] == "failed" and t.get("requires_replan")}
                while True:
                    expanded = affected | {t["task_id"] for t in tasks.values() if affected.intersection(t["depends_on"])}
                    if expanded == affected:
                        break
                    affected = expanded
                return Command(update={"tasks": tasks, "action_graph": graph, "home_state": home,
                                       "repair_context": {"task_ids": sorted(affected), "reason": task.get("error", "")}},
                               goto="task_planner")
            task["terminal"] = True
            if task.get("parent"):
                tasks[task["parent"]].update(status="failed", error="Informationsbeschaffung fehlgeschlagen.")
            continue
        task.update(status="pending", repairs=task.get("repairs", 0) + 1,
                    generation=task.get("generation", 0) + 1)
        task.pop("actions", None)  # failed direct commands are repaired by the responsible domain agent
    changed = True
    while changed:
        changed = False
        for task in tasks.values():
            if task["status"] == "pending" and any(tasks[d]["status"] in {"failed", "blocked"} and tasks[d].get("terminal")
                                                   for d in task["depends_on"]):
                task.update(status="blocked", error="Vorgaenger fehlgeschlagen.")
                task["terminal"] = True
                changed = True
    ready = [t for t in tasks.values() if t["status"] == "pending"
             and all(tasks[d]["status"] == "success" for d in t["depends_on"])][:state["settings"]["max_workers"]]
    if ready:
        if not state["settings"]["parallel_agents"]:
            first_agent = next((t for t in ready if not t.get("actions")), None)
            ready = [t for t in ready if t.get("actions") or t is first_agent]
        for task in ready:
            task["status"] = "planning"
        update = {"tasks": tasks, "action_graph": graph, "home_state": home}
        direct = [{"task_id": t["task_id"], "response": {"actions": t["actions"]}} for t in ready if t.get("actions")]
        if direct:
            update.update(collect_plans({**state, **update, "planning_updates": direct}).update)
        agents = [t for t in ready if not t.get("actions")]
        if agents:
            return Command(update=update, goto=[Send(t["agent_id"] + "_agent", {**state, **update, "planning_task": t}) for t in agents])
        next_node = "validate_plan" if all(update["tasks"][t["task_id"]]["status"] == "planned" for t in ready) else "plan_agents"
        return Command(update=update, goto=next_node)
    if all(t["status"] in {"success", "failed", "blocked"} for t in tasks.values()):
        return Command(update={"tasks": tasks, "action_graph": graph, "home_state": home}, goto="finish")
    return Command(update={"tasks": tasks, "action_graph": graph, "home_state": home}, goto="validate_plan")

def collect_plans(state):
    tasks, graph = deepcopy(state["tasks"]), deepcopy(state["action_graph"])
    latest = {u["task_id"]: u for u in state["planning_updates"]}
    for task_id, report in latest.items():
        task = tasks[task_id]
        if task["status"] != "planning":
            continue
        try:
            if "error" in report:
                raise ValueError(report["error"])
            response = report["response"]
            if response.get("replan_required"):
                task.update(status="failed", requires_replan=True, error=response.get("reason", "Cross-agent repair requested."))
                continue
            request = response.get("collaboration_request") or {}
            if request.get("target"):
                target = request["target"]
                if task.get("parent") or task.get("help_done") or target == task["agent_id"] or target not in get_available_agents(state):
                    raise ValueError("Ungueltige oder wiederholte Kollaboration.")
                helper_id = task_id + ":help"
                tasks[helper_id] = dict(task_id=helper_id, agent_id=target, action=request["request"],
                                       depends_on=[], status="pending", parent=task_id, scope=[], constraints=[],
                                       goal={"text": request["request"], "request": state["original_user_input"], "requirements": []})
                task["status"] = "waiting"
                continue
            actions = response["actions"]
            if not actions or any(not isinstance(a, dict) for a in actions):
                raise ValueError("Agent muss konkrete Aktionen liefern.")
            requirements = task["goal"]["requirements"]
            if not task.get("generation") and not any(r["arguments"] for r in requirements):
                first_goal = freeze_goal({k:v for k,v in {**task, "actions": actions}.items() if k != "goal"}, state)
                task["goal"]["requirements"] = first_goal["requirements"] + [
                    r for r in requirements if r["skill_id"] not in {a["skill_id"] for a in first_goal["requirements"]}]
            local_ids = [a.get("task_id", f"a{i + 1}") for i, a in enumerate(actions)]
            if len(set(local_ids)) != len(local_ids) or any(not isinstance(i, str) or not i or "/" in i for i in local_ids):
                raise ValueError("Doppelte oder ungueltige Aktions-ID.")
            additions = {}
            predecessors = task["depends_on"] + ([task_id + ":help"] if task.get("help_done") else [])
            predecessor_actions = [i for i, a in graph.items() if a["status"] == "success" and
                                   (a["task"]["task_id"] in predecessors or a["task"]["task_id"] == task_id)]
            prefix = task_id + "/" + (f"r{task['generation']}-" if task.get("generation") else "")
            for index, action in enumerate(actions):
                if task.get("parent") and action.get("skill_id") not in {"get_inventory", "get_time", "list_events", "search"}:
                    raise ValueError("Kollaboration dient nur zur Informationsbeschaffung.")
                # Old ordered action lists remain usable: omitted depends_on means previous action.
                deps = action.get("depends_on", local_ids[index - 1:index] if index else [])
                if not isinstance(deps, list):
                    raise ValueError("depends_on muss eine Liste sein.")
                deps = [d if "/" in d else prefix + d for d in deps]
                action_id = prefix + local_ids[index]
                additions[action_id] = {**action, "task_id": action_id, "task": deepcopy(task),
                                        "agent_id": task["agent_id"], "depends_on": list(dict.fromkeys(deps + predecessor_actions)),
                                        "status": "pending"}
                reads, writes, target = action_footprint(state, additions[action_id])
                if not isinstance(target[1], str):
                    raise ValueError("Konkrete device_id/service_id muss ein String sein.")
                additions[action_id]["footprint"] = (sorted(reads), sorted(writes), target)
            candidate = {**graph, **additions}
            if any(set(a["depends_on"]) - candidate.keys() for a in additions.values()):
                raise ValueError("Unbekannter Aktions-Vorgaenger.")
            tuple(TopologicalSorter({i: a["depends_on"] for i, a in candidate.items()}).static_order())
            graph = candidate
            task["status"] = "planned"
        except (KeyError, ValueError, TypeError, CycleError) as exc:
            task.update(status="failed", error=f"PlanError: {exc}")
    return Command(update={"tasks": tasks, "action_graph": graph}, goto="plan_agents")

def validate_plan(state):
    graph = state["action_graph"]
    affected = set()
    cross_agent = False
    try:
        order = tuple(TopologicalSorter({i: a["depends_on"] for i, a in graph.items()}).static_order())
        ancestors, effects = {}, {}
        for action_id in order:
            action = graph[action_id]
            affected = {action["task"]["task_id"]}
            args, agent, skill = action.get("arguments", {}), action["agent_id"], action.get("skill_id")
            definition = state["agent_registry"][agent]["operations"][skill]
            ancestors[action_id] = set(action["depends_on"]).union(*(ancestors[d] for d in action["depends_on"]))
            desired = {}
            for condition in definition["postconditions"]:
                if "state" in condition and ("equals" in condition or condition.get("equals_arg") in args):
                    value = condition["equals"] if "equals" in condition else args[condition["equals_arg"]]
                    desired[condition["state"].format_map(args)] = json.dumps(value, sort_keys=True)
            for other_id, other in effects.items():
                if any(desired[k] != other[k] for k in desired.keys() & other.keys()) and other_id not in ancestors[action_id]:
                    affected.add(graph[other_id]["task"]["task_id"])
                    cross_agent = len(affected) > 1
                    raise ValueError(f"PlanError: widerspruechliche Ziele ohne Reihenfolge: {other_id}, {action_id}.")
            effects[action_id] = desired
        return Command(goto="scheduler")
    except (KeyError, ValueError, TypeError, CycleError) as exc:
        tasks = deepcopy(state["tasks"])
        affected = affected or {a["task"]["task_id"] for a in graph.values() if a["status"] == "pending"}
        for task_id in affected:
            tasks[task_id].update(status="failed", error=f"PlanError: {exc}", requires_replan=cross_agent)
        retained = {i: a for i, a in graph.items() if a["task"]["task_id"] not in affected or a["status"] == "success"}
        return Command(update={"tasks": tasks, "action_graph": retained}, goto="plan_agents")

def check_allocation(state, batch):
    """Vaibhav's unchanged checker over a finite registry projection; rich contracts stay in Python.
    Arguments are sent as typed values (numbers, booleans, enum strings) together with their
    declared types; the checker decodes them itself and rejects wrongly typed values."""
    if state["settings"]["checker"] != "lean":
        return
    registry, home = state["agent_registry"], state["home_state"]
    actions = {**state["action_graph"], **{a["task_id"]: a for a in batch}}
    agents = sorted(registry)
    op_keys = [(agent, skill) for agent in agents for skill in sorted(registry[agent]["operations"])]
    op_ids = {key: i for i, key in enumerate(op_keys)}
    entities = {f"{scope}.{i}": e for section, scope in (("devices", "device"), ("services", "service"))
                for i, e in state["home_config"][section].items()}
    dev_ids = {name: i for i, name in enumerate(sorted(set(entities) | {k.rsplit('.', 1)[0] for k in home}))}
    attrs = sorted({k.rsplit('.', 1)[-1] for k in home} | {"__target__"} | {
        k.rsplit('.', 1)[-1] for card in registry.values() for d in card["operations"].values()
        for k in d["reads"] + d["writes"] + [c["state"] for c in d["preconditions"] if "state" in c]})
    attr_ids = {name: i for i, name in enumerate(attrs)}
    task_ids = {name: i for i, name in enumerate(actions)}
    key = lambda path: [dev_ids[path.rsplit('.', 1)[0]], attr_ids[path.rsplit('.', 1)[-1]]]
    # Finite string equality for enum preconditions; free text/list/object values remain opaque.
    strings = sorted({c["equals"] for card in registry.values() for d in card["operations"].values()
                      for c in d["preconditions"] if isinstance(c.get("equals"), str)})
    scalar = lambda v: (int(v) if type(v) in (bool, int) and abs(v) <= 10**9 else
                        strings.index(v) + 2 if isinstance(v, str) and v in strings else 0)
    ops, fields = [], {}
    for agent, skill in op_keys:
        d = registry[agent]["operations"][skill]
        projected, bounds, types = [], [], []
        for name, schema in d["parameters"].items():
            if "enum" in schema and all(isinstance(v, str) for v in schema["enum"]):
                projected.append(name)
                bounds.append([0, len(schema["enum"]) - 1])
                types.append({"enum": schema["enum"]})
            elif schema.get("type") == "boolean":
                projected.append(name)
                bounds.append([0, 1])
                types.append("bool")
            elif schema.get("type") in ("integer", "number") and "minimum" in schema and "maximum" in schema:
                values = [schema["minimum"], schema["maximum"]] + [a["arguments"][name] for a in actions.values()
                    if (a["agent_id"], a["skill_id"]) == (agent, skill) and type(a["arguments"].get(name)) in (int, float)
                    and math.isfinite(a["arguments"][name])]
                scale = math.lcm(*(Fraction(str(v)).denominator for v in values))
                # Exact decimal scaling, never rounding; values outside the finite codec stay in Python.
                if max(abs(Fraction(str(v)) * scale) for v in values) <= 10**9:
                    projected.append(name)
                    bounds.append([int(Fraction(str(schema[k])) * scale) for k in ("minimum", "maximum")])
                    types.append({"number": scale})
        fields[(agent, skill)] = projected
        pre, reads = [], set(d["reads"])
        for c in d["preconditions"]:
            if "state" not in c:
                continue
            reads.add(c["state"])
            if c.get("known"):
                pre.append([attr_ids[c["state"].rsplit('.', 1)[-1]], "ge", -10**9])
            elif "equals" in c and type(c["equals"]) in (bool, int, str):
                pre.append([attr_ids[c["state"].rsplit('.', 1)[-1]], "eq", scalar(c["equals"])])
        ops.append(dict(id=op_ids[(agent, skill)], cap=agents.index(agent), bounds=bounds, types=types,
                        reads=sorted({attr_ids[p.rsplit('.', 1)[-1]] for p in reads}),
                        writes=sorted({attr_ids[p.rsplit('.', 1)[-1]] for p in d["writes"]}),
                        excl=[attr_ids["__target__"]], shared_reads=[], pre=pre,
                        coupled=False, load_base=0, load_per_unit=0))
    devices = [dict(id=dev_ids[name], caps=[agents.index(e["agent_id"])],
                    ops=[op_ids[(e["agent_id"], skill)] for skill in e["skills"]
                         if registry[e["agent_id"]]["enabled"] and skill in registry[e["agent_id"]]["capabilities"]])
               for name, e in entities.items()]
    ranks, tasks, evidence, commands = {}, [], [], []
    selected_ids = {a["task_id"] for a in batch}
    for aid in TopologicalSorter({i: a["depends_on"] for i, a in actions.items()}).static_order():
        a = actions[aid]
        args = a["arguments"]
        agent, skill = a["agent_id"], a["skill_id"]
        target = "device." + args["device_id"] if "device_id" in args else "service." + args["service_id"]
        # Typed values as they are; the checker decodes them with the declared types.
        vals = [args.get(name) for name in fields[(agent, skill)]]
        ranks[aid] = 1 + max((ranks[d] for d in a["depends_on"]), default=-1)
        verified = a["status"] == "success" and a.get("verified", False)
        tasks.append(dict(id=task_ids[aid], status="pending" if aid in selected_ids else
                          "verified" if verified else "failed" if a["status"] in {"failed", "blocked"} else "pending",
                          op=op_ids[(agent, skill)], device=dev_ids[target],
                          # Only the batch's tasks need their arguments (for the contract check).
                          args=[{"lit": v} for v in vals] if aid in selected_ids else [],
                          deps=[task_ids[d] for d in a["depends_on"]], rank=ranks[aid], pre=[], deadline=10**12))
        if verified:
            evidence.append(dict(task=task_ids[aid], verified=True, valid_until=int(a["valid_until"]), output=None,
                                 basis=[[*key(k), v] for k, v in a["evidence_versions"].items()]))
        if aid in selected_ids:
            commands.append(dict(task=task_ids[aid], device=dev_ids[target], op=op_ids[(agent, skill)], args=vals))
    simulator = SIMULATORS[state["request_id"]]
    with simulator.lock:
        active = [dict(reads=[key(k) for k in r], writes=[key(k) for k in w],
                       excl=[[dev_ids[("device." if t[0] == "device_id" else "service.") + t[1]], attr_ids["__target__"]]])
                  for r, w, t in simulator.reservations.values()]
    payload = dict(schema="dexa-alloc-request/1", request_id=uuid4().hex, auth_ttl=30,
        catalogue=dict(revision=1, ops=ops, devices=devices, load_attr=attr_ids["__target__"], meter_key=[0, 0], load_budget=0),
        graph=dict(revision=1, tasks=tasks),
        snapshot=dict(now=int(time.time()), worker_cap=state["settings"]["max_workers"],
                      state=[[*key(k), scalar(o["value"]), o.get("version", 0)] for k, o in home.items()],
                      evidence=evidence, active=active),
        batch=dict(graph_revision=1, catalogue_revision=1, commands=commands))
    result = subprocess.run([state["settings"]["lean_executable"], "--oneshot"], input=json.dumps(payload) + "\n",
                            capture_output=True, text=True, timeout=15, check=True)
    response = json.loads(result.stdout)
    # The checker echoes the commands with their decoded (integer) arguments.
    shape = lambda cs: [(c["task"], c["device"], c["op"], len(c["args"])) for c in cs]
    if (response.get("request_id") != payload["request_id"] or shape(response.get("commands", [])) != shape(commands)
            or response.get("verdict") != "accept"):
        raise ValueError(f"Lean allocation checker rejected the batch: {response.get('diagnostics', response.get('error'))}")
    for action in batch:
        action["dispatch_until"] = min(action["dispatch_until"], time.monotonic() + max(0, response["expires_at"] - payload["snapshot"]["now"]))

def scheduler(state):
    state = {**state, "home_state": {**state["home_state"], **SIMULATORS[state["request_id"]].snapshot()}}
    graph = deepcopy(state["action_graph"])
    blocked = []
    changed = True
    while changed:
        changed = False
        for action in graph.values():
            if action["status"] == "pending" and any(graph[d]["status"] in {"failed", "blocked"} for d in action["depends_on"]):
                action["status"] = "blocked"
                blocked.append(dict(task_id=action["task_id"], attempt_id=uuid4().hex, agent_id=action["agent_id"],
                                    skill_id=action["skill_id"], arguments=action.get("arguments", {}),
                                    status="blocked", verified=False, output={}, observations={}, error="Vorgaenger fehlgeschlagen."))
                changed = True
    if blocked:
        return Command(update={"action_graph": graph, "task_results": blocked, "active_actions": []}, goto="complete_round")
    limit = state["settings"]["max_workers"] if state["settings"]["parallel_actions"] else 1
    ready = [a for a in graph.values() if a["status"] == "pending"
             and all(graph[d]["status"] == "success" for d in a["depends_on"])]
    llm_scheduler = state["settings"]["scheduler_mode"] == "llm"
    if llm_scheduler and ready:
        ids = _ask('You schedule the next execution batch. Select 1..max_actions unique action IDs from actions. '
                   'Choose useful parallel work: no shared target and no write/write or read/write conflicts. '
                   'Keep agents and arguments unchanged. Output only {"action_ids": ["t1/a1"]}.',
                   expected={"action_ids": list}, role="scheduler", original_input=state["original_user_input"],
                   max_actions=limit, actions=[{"action_id": a["task_id"], "agent_id": a["agent_id"],
                       "goal": a["task"]["action"], "skill_id": a["skill_id"], "arguments": a["arguments"],
                       "reads": a["footprint"][0], "writes": a["footprint"][1], "target": a["footprint"][2]}
                       for a in ready])["action_ids"]
        if not ids or len(ids) > limit or len(set(ids)) != len(ids) or set(ids) - {a["task_id"] for a in ready}:
            raise ValueError("SchedulerError: choose a nonempty batch of unique ready action IDs within max_actions.")
        ready = [graph[i] for i in ids]
    selected, rejected, targets, round_reads, round_writes = [], [], set(), set(), set()
    for action in ready:
        args = action["arguments"]
        target = ("device", args["device_id"]) if "device_id" in args else ("service", args["service_id"])
        reads, writes, _ = action_footprint(state, action)
        if target in targets or writes & (round_reads | round_writes) or reads & round_writes:
            if llm_scheduler:
                raise ValueError("SchedulerError: proposed actions conflict within the batch.")
            continue  # conservative: one simultaneous action per device/service
        checked = validate_action({**state, "pending_actions": [action]})
        if checked.goto != "execute_action":
            action["status"] = "failed"
            rejected.extend(checked.update["task_results"])
            continue
        action.update(checked.update["pending_actions"][0])
        state["home_state"] = {**state["home_state"], **checked.update["home_state"]}
        targets.add(target)
        round_reads.update(reads)
        round_writes.update(writes)
        action["status"] = "running"
        selected.append(action)
        if len(selected) >= limit:
            break
    try:
        if selected:
            check_allocation(state, selected)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        for action in selected:
            action["status"] = "failed"
            rejected.append(dict(task_id=action["task_id"], attempt_id=uuid4().hex, agent_id=action["agent_id"],
                                 skill_id=action["skill_id"], arguments=action["arguments"], status="failed", verified=False,
                                 output={}, observations={}, error=f"AdmissionError: {exc}"))
        selected = []
    if not selected and not rejected:
        raise ValueError("PlanError: keine ausfuehrbare Aktion; Abhaengigkeiten pruefen.")
    return Command(update={"action_graph": graph, "task_results": rejected, "home_state": state["home_state"],
                           "active_actions": [a["task_id"] for a in selected] + [r["task_id"] for r in rejected]},
                   goto=[Send("execute_action", {**state, "pending_actions": [a]}) for a in selected] if selected else "complete_round")

def complete_round(state):
    home = {**state["home_state"], **SIMULATORS[state["request_id"]].snapshot()}
    if not state["settings"]["dependency_graph"]:
        task = state["pending_actions"][0]["task"]
        result = state["task_results"][-1]
        home.update({k: o for k, o in result["observations"].items() if k.startswith("result.")})
        if result["status"] == "stale":
            return Command(update={"home_state": home}, goto="validate_action")
        remaining = state["pending_actions"][1:] if result["status"] == "success" else []
        if remaining:
            return Command(update={"home_state": home, "pending_actions": remaining}, goto="validate_action")
        results = [r for r in state["task_results"] if r["task_id"].rsplit(":", 1)[0] == task["task_id"]]
        helping = state["collaboration_request"].get("target") == task["agent_id"]
        history = [{"agent_id": task["agent_id"], "action": task["action"],
                    "type": "collaboration" if helping else "completion", "results": results}]
        update = {"home_state": home, "pending_actions": [], "task_history": history}
        if helping:
            requester = state["collaboration_request"]["requester"]
            return Command(update={**update, "collaboration_request": {}}, goto=requester + "_agent")
        task = {**task, "status": "success" if all(r["status"] == "success" for r in results) else "failed"}
        return Command(update={**update, "tasks": {**state["tasks"], task["task_id"]: task},
                               "task_queue": state["task_queue"][1:], "pending_task": None}, goto="task_planner")
    graph, tasks = deepcopy(state["action_graph"]), deepcopy(state["tasks"])
    history = []
    for result in sorted((r for r in state["task_results"] if r["task_id"] in state["active_actions"]), key=lambda r: r["task_id"]):
        action = graph[result["task_id"]]
        action["status"] = result["status"]
        if result["status"] == "stale":
            action["dispatch_retries"] = action.get("dispatch_retries", 0) + 1
            action["status"] = "pending" if action["dispatch_retries"] <= state["settings"]["max_repairs"] + 1 else "failed"
        for key in ("output", "before", "evidence_versions", "valid_until", "verified"):
            if key in result:
                action[key] = result[key]
        if result["status"] == "success" and result["verified"]:
            written = set(action["footprint"][1])
            for previous in graph.values():
                if previous is not action and previous["status"] == "success" and not previous.get("superseded") and written.intersection(previous["footprint"][1]):
                    context = {"now": previous["now"], "output": previous["output"], "before": previous["before"],
                               "simulation": state["simulation_data"], "retained": True}
                    previous["superseded"] = not all(check_condition(c, previous["arguments"], home, context)
                        for c in state["agent_registry"][previous["agent_id"]]["operations"][previous["skill_id"]]["postconditions"])
                    if previous["superseded"]:
                        previous["evidence_versions"] = {}  # execution history, not a current state claim
        for key, observation in result["observations"].items():
            if key.startswith("result."):
                home[key] = {**observation, "version": home.get(key, {}).get("version", 0) + 1}
    for task in tasks.values():
        actions = [a for a in graph.values() if a["task"]["task_id"] == task["task_id"]]
        if task["status"] != "planned" or not actions or any(a["status"] in {"pending", "running"} for a in actions):
            continue
        task["status"] = "failed" if any(a["status"] == "failed" for a in actions) else "blocked" if any(a["status"] == "blocked" for a in actions) else "success"
        action_ids = {a["task_id"] for a in actions}
        if task["status"] == "success":
            try:
                check_goal(state, task, [a for a in actions if a["status"] == "success"], complete=True)
                task.pop("error", None)
            except ValueError as exc:
                task.update(status="failed", error=str(exc))
        else:
            task["error"] = "; ".join(r["error"] for r in state["task_results"]
                                      if r["task_id"] in action_ids and r.get("error"))
        results = [r for r in state["task_results"] if r["task_id"].rsplit("/", 1)[0] == task["task_id"]]
        history.append({"agent_id": task["agent_id"], "action": task["action"],
                        "type": "collaboration" if task.get("parent") else "completion", "results": results})
        if task.get("parent"):
            parent = tasks[task["parent"]]
            if task["status"] == "success":
                parent.update(status="pending", help_done=True)
            else:
                parent["status"] = "waiting"  # plan_agents first gives the helper its bounded repair opportunity.
    return Command(update={"home_state": home, "action_graph": graph, "tasks": tasks,
                           "active_actions": [], "task_history": history}, goto="plan_agents")

def finish(state):
    if state["settings"]["dependency_graph"]:
        summary = [{"task_id": t["task_id"], "agent_id": t["agent_id"], "action": t["action"],
                    "status": t["status"], "error": t.get("error"),
                    "results": [{"skill_id": r["skill_id"], "status": r["status"], "verified": r["verified"], "output": r["output"], "error": r["error"]}
                                for r in state["task_results"] if r["task_id"].rsplit("/", 1)[0] == t["task_id"]]}
                   for t in state["tasks"].values()]
        return Command(update={"final_response": state["final_response"] or json.dumps(summary, ensure_ascii=False)}, goto=END)
    response = state["final_response"] or json.dumps(
        [{"agent_id": h["agent_id"], "action": h["action"],
          "results": [{"skill_id": r["skill_id"], "status": r["status"], "output": r["output"], "error": r["error"]}
                      for r in h["results"]]} for h in state["task_history"]], ensure_ascii=False)
    return Command(update={"final_response": response}, goto=END)

# Eight independent graph nodes; shared implementation avoids eight code copies.
def _node(agent):
    def run(state):
        return _agent(state, agent)
    run.__name__ = agent + "_agent"
    return run

builder = StateGraph(SmartHomeState)
for name, node in {"human": human, "intent_analysis": intent_analysis, "task_planner": task_planner,
                   "validate_action": validate_action, "execute_action": execute_action,
                   "plan_agents": plan_agents, "collect_plans": collect_plans, "validate_plan": validate_plan,
                   "scheduler": scheduler,
                   "complete_round": complete_round, "finish": finish}.items():
    builder.add_node(name, node)
for agent in AGENT_REGISTRY:
    node = _node(agent)
    globals()[node.__name__] = node
    builder.add_node(node.__name__, node)
builder.add_edge(START, "human")
graph = builder.compile(checkpointer=MemorySaver())

def run_request(text, **setup):
    config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 100}
    state = create_initial_state(**setup)
    try:
        for _ in graph.stream(state, config):
            pass
        yield from graph.stream(Command(resume=text), config)
    finally:
        SIMULATORS.pop(state["request_id"], None)

def interactive(**setup):
    while True:
        text = input("\nYour command (quit to exit): ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if text:
            started = time.perf_counter()
            for event in run_request(text, **setup):
                for node, update in event.items():
                    if update and update.get("final_response"):
                        print(update["final_response"])
            print(f"Time: {time.perf_counter() - started:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent-analysis", action="store_true", help="Enable the optional intent analyser")
    parser.add_argument("--search-mode", choices=("fixtures", "live"), default="fixtures")
    parser.add_argument("--scheduler", choices=("python", "llm"), default=SCHEDULER_MODE)
    parser.add_argument("--checker", choices=("python", "lean"), default="python")
    parser.add_argument("--sequential", action="store_true", help="Disable agent/action parallelism")
    parser.add_argument("--no-operations", action="store_true", help="Omit operation definitions from LLM prompts (ablation)")
    parser.add_argument("--no-conditions", action="store_true", help="Omit operation conditions from LLM prompts (ablation)")
    args = parser.parse_args()
    interactive(intent_analysis_enabled=args.intent_analysis, checker=args.checker,
                search_mode=args.search_mode, scheduler_mode=args.scheduler,
                parallel_agents=not args.sequential, parallel_actions=not args.sequential,
                include_operations=not args.no_operations, include_conditions=not args.no_conditions)
