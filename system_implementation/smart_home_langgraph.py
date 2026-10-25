import json
import llm_config
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
from llm_config import configure_models, get_llm, model_for_role, summarize_calls
from model_client import BackendUnavailable, retry_model_call
from configuration import config_path as configured_path, system_settings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Send, interrupt
from agent_registry import AGENT_REGISTRY
from home_assistant import HomeAssistant
from home_defaults import HOME_DEFAULTS, SEARCH_FIXTURES

SUPPORTED_BENCHMARKS = {"dexa", "homebench", "simuhome", "homeassistant"}


def load_settings(path=None):
    return system_settings("mas", path)


def resolve_settings(overrides=None, *, config_path=None):
    settings = load_settings(config_path)
    settings.update(overrides or {})
    if not settings["domain_agents_enabled"]:
        settings["local_repair"] = False
    if not settings["repair_enabled"]:
        for key in ("action_retries", "dispatch_retries", "local_repair", "planner_replan"):
            settings[key] = False
    if not settings["local_repair"]:
        settings["max_repairs"] = 0
    if not settings["planner_replan"]:
        settings["max_replans"] = 0
    if settings.get("lean_executable"):
        settings["lean_executable"] = configured_path(settings["lean_executable"])
    return settings


def add_settings_arguments(parser):
    parser.add_argument("--config", type=Path, help="Central settings TOML")
    for key, value in load_settings().items():
        flag = key.replace("_", "-")
        flag = {"repair_enabled": "repair", "intent_analysis_enabled": "intent-analysis",
                "domain_agents_enabled": "domain-agents",
                "scheduler_mode": "scheduler", "recursion_limit": "max-supersteps"}.get(key, flag)
        options = dict(dest=key, default=None)
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(value)
        parser.add_argument("--" + flag, **options)
    parser.add_argument("--sequential", action="store_true")


def settings_from_args(args):
    overrides = {key: getattr(args, key) for key in load_settings() if getattr(args, key, None) is not None}
    if args.sequential:
        overrides.update(parallel_agents=False, parallel_actions=False)
    if args.lean_executable:
        overrides["lean_executable"] = configured_path(args.lean_executable)
    return overrides



# Live objects stay outside LangGraph checkpoints: workers share one simulator per request.
SIMULATORS = {}

class Simulator:
    def __init__(self, home, environment=None, fixed_time=None, user_location=None):
        self.home = deepcopy(home)
        self.environment, self.fixed_time = environment, fixed_time
        self.user_location = user_location
        self.virtual_time = None
        self.reference_time = fixed_time
        self.lock = RLock()
        self.reservations = {}
        self.llm_calls = []

    def snapshot(self):
        with self.lock:
            if self.environment is not None:
                self.virtual_time, values = self.environment.snapshot()
                for key, value in values.items():
                    old = self.home.get(key)
                    version = 0 if old is None else old["version"] + (old["value"] != value)
                    self.home[key] = make_observation(value, source="external:simulator",
                        observed_at=self.virtual_time, version=version)
            return deepcopy(self.home)

    def now(self):
        return self.virtual_time or self.fixed_time or datetime.now(timezone.utc).isoformat()

    def write(self, values, now=None, *, source="simulation:tool", ref=None):
        with self.lock:
            for key, value in values.items():
                old = self.home.get(key, {})
                self.home[key] = {**make_observation(value, source=source,
                    unit=old.get("unit", UNITS.get(key.rsplit(".", 1)[-1])), observed_at=now,
                    version=old.get("version", 0) + 1), "ref": ref}

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
    status: Literal["pending", "planning", "waiting", "planned", "success", "failed", "blocked", "rejected"]
    depends_on: list[str]
    scope: list[str]
    constraints: list[str]
    goal: dict
    failure: dict | None
    # Optional actions contains an already concrete plan; absent means domain refinement.

class Action(TypedDict):
    task_id: str
    agent_id: str
    skill_id: str
    arguments: dict
    depends_on: list[str]
    status: str
    execution_retries: int

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
    failure: dict | None
    execution_retries: int
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
    answer: str
    settings: dict
    simulation_data: dict
    repair_context: dict
    rejected_requests: list[dict]

# Timestamps/source are assigned by Python, never by the LLM.
def make_observation(value, *, source, unit=None, observed_at=None, version=0):
    return {"value": deepcopy(value), "source": source, "unit": unit,
            "observed_at": observed_at or datetime.now(timezone.utc).isoformat(), "version": version}

def create_initial_state(observations=None, *, home_config=None, use_defaults=True,
                         observed_at=None, disabled_agents=(), simulation_data=None,
                         agent_registry=None, tool_bindings=None, environment=None, config_path=None, **overrides):
    settings = resolve_settings(overrides, config_path=config_path)
    validate_contracts = settings["validate_contracts"]
    dependency_graph = settings["dependency_graph"]
    check_outcomes = settings["check_outcomes"]
    checker = settings["checker"]
    search_mode = settings["search_mode"]
    scheduler_mode = settings["scheduler_mode"]
    lean_executable = settings.get("lean_executable")
    if scheduler_mode not in {"python", "llm"}:
        raise ValueError("scheduler_mode must be python or llm.")
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
    registry = deepcopy(AGENT_REGISTRY if agent_registry is None else agent_registry)
    for agent in disabled_agents:
        registry[agent]["enabled"] = False
    request_id = uuid4().hex
    simulator = Simulator(home, environment, simulation_data.get("now"), simulation_data.get("user_location"))
    home = simulator.snapshot()
    simulator.reference_time = simulation_data.get("reference_time", simulator.now())
    simulator.tools = {**TOOLS, **(tool_bindings or {})}
    SIMULATORS[request_id] = simulator
    return dict(original_user_input="", request_id=request_id, infos=[], key_modifiers=[],
                task_queue=[], tasks={}, task_results=[], home_config=config, home_state=home,
                agent_registry=registry, collaboration_request={}, pending_task=None, pending_actions=[],
                action_graph={}, planning_task={}, planning_updates=[], active_actions=[],
                task_history=[], final_response="", answer="", repair_context={}, rejected_requests=[], simulation_data=deepcopy(simulation_data or {}),
                settings={**settings, "version": "V3" if dependency_graph else "V2" if validate_contracts else "V1",
                          "replans": 0, "lean_executable": lean_executable})

def refresh_state(state):
    simulator = SIMULATORS[state["request_id"]]
    return {**state, "home_state": {**state["home_state"], **simulator.snapshot()},
            "simulation_data": {**state["simulation_data"], "now": simulator.now()}}


def get_available_agents(state, *, include_operations=None):
    if include_operations is None:
        include_operations = state["settings"]["include_operations"]
    available = {}
    for agent, card in state["agent_registry"].items():
        if not card["enabled"]:
            continue
        entry = {"description": card["description"], "devices": {}, "services": {}}
        if "context" in card:
            entry["context"] = deepcopy(card["context"])
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

def prompt_agents(state, available):
    """Show arguments directly under each method; nest current values by entity."""
    cards, covered = deepcopy(available), set()
    for card in cards.values():
        operations = card.get("operations", {})
        for section, scope, target in (("devices", "device", "device_id"), ("services", "service", "service_id")):
            for entity_id, entity in card[section].items():
                methods = {}
                for skill in entity["skills"]:
                    definition = entity.get("parameters", {}).get(skill)
                    if definition is None:
                        agent = state["home_config"][section][entity_id]["agent_id"]
                        definition = state["agent_registry"][agent]["operations"][skill]["parameters"]
                    methods[skill] = {k: v for k, v in definition.items() if k != target}
                prefix = f"{scope}.{entity_id}."
                values = {key[len(prefix):]: value["value"] for key, value in state["home_state"].items() if key.startswith(prefix)}
                covered.update(prefix + key for key in values)
                if entity.get("method_family"):
                    card.setdefault("methods_by_type", {})[entity["method_family"]] = methods
                card[section][entity_id] = {**{k: v for k, v in entity.items()
                    if k not in {"agent_id", "skills", "parameters", "homebench_name", "method_family"}},
                    "target": {target: entity_id},
                    **({} if entity.get("method_family") else {"methods": methods}), "state": values}
        card.pop("capabilities", None)  # Exact methods remain on each entity.
        for definition in operations.values():
            definition.pop("parameters", None)  # Supplied directly by each entity's methods.
    remaining = {k: v["value"] for k, v in state["home_state"].items() if k not in covered and not k.startswith("result.")}
    return cards, remaining


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


def operation_parameters(state, agent, skill, arguments):
    definition = state["agent_registry"][agent]["operations"][skill]
    target = "device_id" if "device_id" in definition["parameters"] else "service_id"
    section = "devices" if target == "device_id" else "services"
    entity = state["home_config"][section][arguments[target]]
    if entity["agent_id"] != agent or skill not in entity["skills"]:
        raise ValueError("Agent, target and method do not match.")
    return entity.get("parameters", {}).get(skill, definition["parameters"])


def fill_service_ids(state, agent, actions):
    """Fill only explicit routing defaults; all supplied arguments still need validation."""
    operations = state["agent_registry"].get(agent, {}).get("operations", {})
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("arguments"), dict):
            continue
        schema = operations.get(action.get("skill_id"), {}).get("parameters", {}).get("service_id", {})
        if "default" in schema:
            action["arguments"].setdefault("service_id", deepcopy(schema["default"]))


def relative_arguments(state, task, action):
    """HomeBench percentages are points; prefer the original request's delta."""
    args = deepcopy(action.get("arguments", {}))
    if state["simulation_data"].get("benchmark") != "homebench" or "device_id" not in args:
        return args
    device = state["home_config"]["devices"].get(args["device_id"], {})
    room = state["home_config"]["rooms"].get(device.get("room_id"), {}).get("name", "").lower()
    kind = device.get("type", "").replace("_", " ").rstrip("s")
    goal = task.get("goal", {})
    texts = [state["original_user_input"],
             " ".join([goal.get("text", task["action"]), *goal.get("constraints", task.get("constraints", []))])]
    room_names = {name.lower().replace("_", " ") for key, value in state["home_config"]["rooms"].items()
                  for name in (key, value["name"])}
    definition = state["agent_registry"][task["agent_id"]]["operations"].get(action["skill_id"], {})
    for condition in definition.get("postconditions", []):
        name = condition.get("equals_arg")
        if not name or name not in args or "state" not in condition:
            continue
        for text in texts:
            clauses = [clause for clause in re.split(r"[,;]|\band\b", text.lower().replace("_", " "))
                       if name.replace("_", " ") in clause and re.search(r"\b" + re.escape(kind) + r"s?\b", clause)]
            matching = [clause for clause in clauses if room and room in clause]
            clauses = matching or (clauses if len(clauses) == 1 and not any(room in clauses[0] for room in room_names) else [])
            if not clauses:
                continue
            match = re.search(r"\b(increase|decrease|raise|lower|reduce)\b.*?\bby\s+(\d+(?:\.\d+)?)", clauses[0])
            if len(clauses) == 1 and match:
                delta = Fraction(match[2]) * (-1 if match[1] in {"decrease", "lower", "reduce"} else 1)
                args[name] = {"from_state": condition["state"].format_map(args),
                              "add": int(delta) if delta.denominator == 1 else float(delta)}
            break
    return args


def freeze_goal(task, state):
    """Keep the planner's structured goal unchanged through repairs."""
    if "goal" in task:
        return task["goal"]
    requirements = []
    for action in task.get("requirements", task.get("actions", [])):
        action = {**action, "arguments": relative_arguments(state, task, action)}
        parameters = operation_parameters(state, task["agent_id"], action["skill_id"], action["arguments"])
        arguments = {k: deepcopy(v) for k, v in action.get("arguments", {}).items()
                     if k in parameters and (isinstance(v, dict) and ("from_task" in v or "from_state" in v) or
                         Draft202012Validator(parameters[k], format_checker=FormatChecker()).is_valid(v))}
        requirements.append({"skill_id": action["skill_id"], "arguments": arguments})
    return deepcopy({"text": task["action"], "scope": task.get("scope", []),
                     "constraints": task.get("constraints", []), "requirements": requirements,
                     "source_field": task.get("source_field"), "literal_content": task.get("literal_content", False)})


def output_value(output, field):
    for key in field.split('.'):
        output = output[int(key)] if isinstance(output, list) else output[key]
    return output


def argument_template(action):
    return {**action.get("arguments", {}), **{
        name: {k: v for k, v in ref.items() if k in {"from_task", "from_state", "field", "add", "format"}}
        for name, ref in action.get("argument_sources", {}).items()}}


def resolve_arguments(state, action):
    """Bind data to confirmed current outputs; output text never comes from a placeholder."""
    sources, bindings = [], {}
    records = list(state["action_graph"].values()) if state["settings"]["dependency_graph"] else [
        {**r, "task": {"task_id": r["task_id"].rsplit(":", 1)[0]}}
        for r in state["task_results"]]
    def resolve(name, value):
        if isinstance(value, dict) and "from_state" in value:
            previous = action.get("argument_sources", {}).get(name, {})
            base = previous.get("base_value", state["home_state"][value["from_state"]]["value"])
            if type(base) not in (int, float) or type(value.get("add")) not in (int, float):
                raise ValueError("Relative values require a numeric observation and delta.")
            actual = Fraction(str(base)) + Fraction(str(value["add"]))
            actual = int(actual) if actual.denominator == 1 else float(actual)
            bindings[name] = {**deepcopy(value), "base_value": base, "validation": "python"}
            return actual
        if not isinstance(value, dict) or "from_task" not in value:
            return deepcopy(value)
        source, field = value["from_task"], value["field"]
        provider = next((a for a in reversed(records) if (a["task_id"] == source or a["task"]["task_id"] == source)
                         and a["status"] == "success" and (not state["settings"]["check_outcomes"] or a.get("verified"))
                         and field.split('.')[0] in a.get("output", {})), None)
        if provider is None:
            raise ValueError(f"InformationError: no confirmed {field} output from {source}.")
        try:
            actual = output_value(provider["output"], field)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"InformationError: output field {field} missing from {source}.") from exc
        if field.split('.')[0] == "results" and (actual is None or isinstance(actual, (list, dict, str)) and not actual):
            raise ValueError("InformationError: referenced search results are empty.")
        sources.append(provider["task_id"])
        bindings[name] = {**deepcopy(value), "action_id": provider["task_id"], "validation": "python"}
        if "add" in value:
            if type(actual) not in {int, float} or type(value["add"]) not in {int, float}:
                raise ValueError("InformationError: numeric offset needs numeric values.")
            # Decimal JSON values must not acquire binary rounding noise (0.1 + 0.2).
            actual = Fraction(str(actual)) + Fraction(str(value["add"]))
            actual = int(actual) if actual.denominator == 1 else float(actual)
        if value.get("format") == "text":
            if isinstance(actual, list) and actual and all(isinstance(x, dict) and "title" in x and ("content" in x or "snippet" in x) for x in actual):
                return "\n\n".join(x["title"] + "\n" + x.get("content", x.get("snippet", "")) for x in actual)
            return actual if isinstance(actual, str) else json.dumps(actual, ensure_ascii=False, indent=2)
        return deepcopy(actual)
    # Revalidate the original reference on retries, rather than its previous resolved value.
    arguments = argument_template(action)
    args = {key: resolve(key, value) for key, value in arguments.items()}
    if action["skill_id"] == "show_content":
        goal = action["task"].get("goal", {})
        data_kind = goal.get("source_field")
        if data_kind and not goal.get("literal_content") and not sources:
            owners = set(action["task"].get("depends_on", [])) | {action["task"]["task_id"] + ":help"}
            providers = [a for a in records if a["task"]["task_id"] in owners
                         and a["status"] == "success" and data_kind in a.get("output", {})]
            if not providers:
                raise ValueError(f"InformationError: retrieve actual {data_kind} before displaying it.")
            args["content"] = resolve("content", {"from_task": providers[-1]["task_id"], "field": data_kind, "format": "text"})
            args["content_type"] = "text"
        elif not isinstance(args.get("content"), str) or not args["content"].strip():
            raise ValueError("InformationError: display content must not be empty.")
    return args, list(dict.fromkeys(sources)), bindings


def check_goal(state, task, actions, *, complete=False):
    goal = task.get("goal", {})
    requirements = goal.get("requirements", [])
    if not requirements or task.get("parent"):
        return
    def matches(action, requirement):
        if action["skill_id"] != requirement["skill_id"]:
            return False
        arguments = requirement["arguments"]
        if any(isinstance(v, dict) and ("from_task" in v or "from_state" in v) for v in arguments.values()):
            arguments = resolve_arguments(state, {"task": task, **requirement,
                                                 "argument_sources": action.get("argument_sources", {})})[0]
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
def home_assistant_setup(home_assistant):
    """Start state and tools for the devices switched through Home Assistant (home_assistant.py).
    Use as run_request(text, **home_assistant_setup(HomeAssistant.from_env()))."""
    observations = {f"device.{device}.power": make_observation(power, source="home_assistant")
                    for device, power in home_assistant.power_states().items()}
    tool = home_assistant.power_tool(device_tool)
    bindings = {(agent, skill): tool for agent, card in AGENT_REGISTRY.items()
                for skill in ("turn_on", "turn_off") if TOOLS.get((agent, skill)) is device_tool}
    return dict(observations=observations, tool_bindings=bindings)

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
    reads.update(value["from_state"] for value in (*args.values(), *action.get("argument_sources", {}).values())
                 if isinstance(value, dict) and "from_state" in value)
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

def validate_action(state, home=None):
    action = deepcopy(state["pending_actions"][0])
    task, args, skill = action["task"], action.get("arguments", {}), action.get("skill_id", "")
    simulator = SIMULATORS[state["request_id"]]
    failure = {"kind": "plan"}
    try:
        with simulator.lock:
            home = {**state["home_state"], **(simulator.snapshot() if home is None else home)}
            args, source_actions, bindings = resolve_arguments(state, action)
            action.update(arguments=args, source_actions=source_actions, argument_sources=bindings,
                          depends_on=list(dict.fromkeys(action.get("depends_on", []) + source_actions)))
            card = state["agent_registry"][task["agent_id"]]
            if not card["enabled"] or skill not in card["capabilities"] or skill not in card["operations"]:
                raise ValueError("Agent oder Skill ist nicht verfuegbar.")
            definition = card["operations"][skill]
            parameters = operation_parameters(state, task["agent_id"], skill, args)
            for key, schema in parameters.items():
                if key not in args and "default" in schema:
                    args[key] = deepcopy(schema["default"])
            if state["settings"]["validate_contracts"]:
                schema = {"type": "object", "properties": parameters,
                          "required": list(parameters),
                          "additionalProperties": False}
                Draft202012Validator(schema, format_checker=FormatChecker()).validate(args)
                # NaN/Infinity are Python numbers but not valid JSON arguments.
                json.dumps(args, allow_nan=False)
            elif set(parameters) - args.keys():
                raise ValueError("Pflichtargument fehlt.")
            if state["settings"]["validate_contracts"]:
                check_goal(state, task, [action])
            target_key = "device_id" if "device_id" in definition["parameters"] else "service_id"
            section = "devices" if target_key == "device_id" else "services"
            target = state["home_config"][section][args[target_key]]
            if target["agent_id"] != task["agent_id"] or skill not in target["skills"]:
                raise ValueError("Agent, Ziel und Skill passen nicht zusammen.")
            if target_key == "device_id" and task.get("scope"):
                if args[target_key] not in task["scope"] and target.get("room_id") not in task["scope"]:
                    raise ValueError("Geraet liegt ausserhalb des zugewiesenen Auftragsbereichs.")
            now = simulator.now()
            context = {"now": now, "mode": "simulation",
                       "search_available": state["simulation_data"].get("search_available", True)}
            if state["settings"]["validate_contracts"]:
                for condition in definition["preconditions"]:
                    failure = {"kind": "precondition", "condition": condition}
                    if not check_condition(condition, args, home, context):
                        raise ValueError(f"Precondition nicht erfuellt: {condition}")
            action.update(arguments=args, now=now, agent_id=task["agent_id"], depends_on=action.get("depends_on", []))
            reads, writes, target = action_footprint(state, action)
            bindings = {k: home.get(k, {}).get("version", 0) for k in reads | writes}
            for dep in action.get("depends_on", []):
                bindings.update(state["action_graph"].get(dep, {}).get("evidence_versions", {}))
            failure = {"kind": "dispatch"}
            if any(home.get(k, {}).get("version", 0) != v for k, v in bindings.items()):
                raise ValueError("FreshDispatch: dependency evidence is stale.")
            if any(target == t or writes & (r | w) or reads & w
                   for r, w, t in simulator.reservations.values()):
                raise ValueError("FreshDispatch: resource is reserved by another action.")
            action.update(footprint=(sorted(reads), sorted(writes), target), bound_versions=bindings,
                          before={k: deepcopy(home[k]) for k in reads | writes if k in home},
                          dispatch_until=time.monotonic() + 30)
            simulator.reservations[action["task_id"]] = (reads, writes, target)
            return Command(update={"pending_actions": [action, *state["pending_actions"][1:]], "home_state": home}, goto="execute_action")
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError, ValidationError, NotImplementedError) as exc:
        result: TaskResult = dict(task_id=action["task_id"], attempt_id=uuid4().hex,
                                 agent_id=task["agent_id"], skill_id=skill, arguments=args,
                                  argument_sources=deepcopy(action.get("argument_sources", {})),
                                  status="stale" if failure["kind"] == "dispatch" else "failed",
                                  verified=False, output={}, observations={}, failure=failure,
                                  error=f"ValidationError: {exc.message if isinstance(exc, ValidationError) else exc}")
        return Command(update={"task_results": [result]}, goto="complete_round")

def execute_action(state):
    action = state["pending_actions"][0]
    task, skill, args = action["task"], action["skill_id"], action["arguments"]
    result: TaskResult = dict(task_id=action["task_id"], attempt_id=uuid4().hex,
                             agent_id=task["agent_id"], skill_id=skill, arguments=deepcopy(args),
                              argument_sources=deepcopy(action.get("argument_sources", {})),
                              status="failed", verified=False, output={}, observations={}, error=None, failure=None,
                              execution_retries=action.get("execution_retries", 0))
    simulator = SIMULATORS[state["request_id"]]
    admitted = False
    failure = {"kind": "dispatch"}
    try:
        if time.monotonic() >= action["dispatch_until"]:
            raise ValueError("FreshDispatch: authorization expired.")
        before = action["before"]
        result["before"] = before
        admitted = True
        parameters = state["agent_registry"][task["agent_id"]]["operations"][skill]["parameters"]
        target_key = "device_id" if "device_id" in parameters else "service_id"
        failure = {"kind": "tool"}
        output = simulator.tools[(task["agent_id"], skill)](
            skill, args, simulator, action["now"], {**state["simulation_data"], "_action_id": action["task_id"]})
        failure = {"kind": "output"}
        observed = simulator.snapshot()
        # Observe actual device/service writes; returned dictionaries cannot fabricate state changes.
        result.update(status="success", output={target_key: args[target_key], **output})
        if state["settings"]["check_outcomes"]:
            definition = state["agent_registry"][task["agent_id"]]["operations"][skill]
            schema = {"type": "object", "properties": definition["output"], "required": list(definition["output"]),
                      "additionalProperties": False}
            Draft202012Validator(schema, format_checker=FormatChecker()).validate(result["output"])
            json.dumps(result["output"], allow_nan=False)
            context = {"now": action["now"], "output": result["output"], "before": before,
                       "simulation": state["simulation_data"]}
            for condition in definition["postconditions"]:
                failure = {"kind": "postcondition", "condition": condition}
                if not check_condition(condition, args, observed, context):
                    raise ValueError("OutcomeError: observed outcome does not match the operation contract.")
            result["verified"] = True
        result.update(evidence_versions={k: observed[k]["version"] for k in
                                         set(action["footprint"][0]) | set(action["footprint"][1]) if k in observed},
                      valid_until=time.time() + (600 if not action["footprint"][1] else 3600))
        # Also record read-only results with source/time metadata, outside the device-property namespace.
        result["observations"].update({f"result.{action['task_id']}.{key}": make_observation(
            value, source="simulation:result", observed_at=simulator.now())
            for key, value in result["output"].items()})
    except Exception as exc:
        result["status"] = "failed" if admitted else "stale"
        result["failure"] = failure
        result["error"] = f"ToolError: {type(exc).__name__}: {exc}"
    finally:
        if admitted:
            observed = simulator.snapshot()
            result["observations"].update({k: observed[k] for k in action["footprint"][1] if k in observed})
        simulator.release(action["task_id"])
    return Command(update={"task_results": [result]}, goto="complete_round")


def _scope_ids(scope):
    """Normalize clear planner scopes without discarding invalid restrictions."""
    if isinstance(scope, dict):
        if not scope or set(scope) - {"room_id", "device_id"}:
            raise ValueError("PlanError: scope objects may contain only room_id and device_id.")
        scope = list(scope.values())
    if isinstance(scope, str):
        scope = [scope]
    if not isinstance(scope, list) or any(not isinstance(item, str) or not item.strip() for item in scope):
        raise ValueError("PlanError: scope must contain nonempty room/device ID strings.")
    return list(dict.fromkeys(scope))


def action_schema(state, agent, card):
    """Bind each call to one actual entity, method and parameter schema."""
    reference = {"type": "object", "required": ["from_task", "field"], "additionalProperties": False,
                 "properties": {"from_task": {"type": "string"}, "field": {"type": "string"},
                                "add": {"type": "number"}, "format": {"enum": ["text"]}}}
    choices = []
    for section, target in (("devices", "device_id"), ("services", "service_id")):
        for entity_id, entity in card[section].items():
            for skill in entity["skills"]:
                definition = state["agent_registry"][agent]["operations"][skill]
                parameters = deepcopy(operation_parameters(state, agent, skill, {target: entity_id}))
                required = [name for name, schema in parameters.items() if "default" not in schema]
                for name, schema in parameters.items():
                    if name != target:
                        parameters[name] = {"anyOf": [schema, reference]}
                        paths = [c["state"].format_map({target: entity_id}) for c in definition["postconditions"]
                                 if c.get("equals_arg") == name and "state" in c]
                        if paths and lean_schema(schema)[0].get("type") in ("integer", "number"):
                            parameters[name]["anyOf"].append({"type": "object", "required": ["from_state", "add"],
                                "additionalProperties": False, "properties": {
                                    "from_state": {"enum": paths}, "add": {"type": "number"}}})
                parameters[target] = {"type": "string", "const": entity_id}
                choices.append({"type": "object", "required": ["skill_id", "arguments"], "additionalProperties": False,
                    "properties": {"task_id": {"type": "string"}, "skill_id": {"const": skill},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "arguments": {"type": "object", "properties": parameters,
                                      "required": list(dict.fromkeys([target, *required])), "additionalProperties": False}}})
    return {"anyOf": choices} if choices else False


def domain_response_schema(state, agent, card):
    return {"type": "object", "required": ["actions"], "additionalProperties": False, "properties": {
        "actions": {"type": "array", "items": action_schema(state, agent, card)},
        "collaboration_request": {"type": "object", "additionalProperties": False, "properties": {
            "target": {"enum": list(get_available_agents(state))}, "request": {"type": "string"}}},
        "replan_required": {"type": "boolean"}, "reason": {"type": "string"},
        "rejected_requests": {"type": "array", "items": {"type": "object", "required": ["request", "reason"],
            "additionalProperties": False, "properties": {"request": {"type": "string"}, "reason": {"type": "string"}}}}}}


def planner_response_schema(agents, state=None, repair=None):
    """Keep task descriptions separate from executable tool calls."""
    strings = {"type": "array", "items": {"type": "string"}}
    operation = {"type": "object", "required": ["skill_id", "arguments"], "properties": {
        "task_id": {"type": "string"}, "skill_id": {"type": "string"},
        "arguments": {"type": "object"}, "depends_on": strings}}
    task = {"type": "object", "required": ["agent_id", "action"], "properties": {
        "task_id": {"type": "string"}, "agent_id": {"type": "string", "enum": list(agents)},
        "action": {"type": "string", "minLength": 1}, "depends_on": strings,
        "constraints": strings,
        "scope": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "requirements": {"type": "array", "items": operation},
        "actions": {"type": "array", "items": operation}}}
    tasks = []
    for agent in agents:
        variant = deepcopy(task)
        variant["properties"]["agent_id"] = {"const": agent}
        if state and (not state["simulation_data"].get("domain_planning_only") or not state["settings"]["domain_agents_enabled"]):
            calls = action_schema(state, agent, get_available_agents(state)[agent])
            for field in ("requirements", "actions"):
                variant["properties"][field]["items"] = calls
        tasks.append(variant)
    if repair:
        for variant in tasks:
            variant["required"].append("task_id")
            variant["properties"]["task_id"] = {"enum": repair["task_ids"]}
    return {"type": "object", "required": ["task_queue"], "properties": {
        "task_queue": {"type": "array", "items": {"anyOf": tasks},
                       **({"minItems": len(repair["task_ids"]), "maxItems": len(repair["task_ids"])} if repair else {})},
        "response": {"type": "string"},
        "rejected_requests": {"type": "array", "items": {"type": "object",
            "required": ["request", "reason"], "properties": {
                "request": {"type": "string"}, "reason": {"type": "string"}}}}}}


def _ask(instructions, *, role="planner", expected=None, request_id=None, response_schema=None, **context):
    schema_state = context.pop("schema_state", None)
    simulator = SIMULATORS.get(request_id)
    if simulator is not None:
        context.update(now=simulator.now(), reference_time=simulator.reference_time, user_location=simulator.user_location)
    schema = response_schema
    if schema is None and role == "planner" and context.get("available_agents"):
        schema = planner_response_schema(context["available_agents"], schema_state, context.get("repair"))
    if schema and context.get("domain_planning_only"):
        for task_schema in schema["properties"]["task_queue"]["items"]["anyOf"]:
            task_schema["properties"]["scope"]["minItems"] = 1
            task_schema["required"].append("scope")
            for field in ("actions", "requirements"):
                task_schema["properties"][field] = {"type": "array", "maxItems": 0}
    if schema and context.get("planner_only"):
        for task_schema in schema["properties"]["task_queue"]["items"]["anyOf"]:
            task_schema["required"].append("actions")
            task_schema["properties"]["actions"]["minItems"] = 1
    messages = [SystemMessage(content=instructions + "\nReturn one JSON object with the requested fields and actual values. "
                             "Do not return a schema, catalogue, example, or Markdown."),
                HumanMessage(content=json.dumps(context, ensure_ascii=False, separators=(",", ":")))]
    # Keep the actual request visible after the catalogue, too. This does not add new input.
    request = context.get("task", {}).get("action") or context.get("original_input", context.get("user_message"))
    if request:
        messages[-1].content += "\nUser request: " + request
    call = {"call_id": uuid4().hex, "role": role, "model": model_for_role(role), "context_time": context.get("now"),
            "reference_time": context.get("reference_time"),
            "started_at": datetime.now(timezone.utc).isoformat(), "prompt_chars": sum(len(m.content) for m in messages),
            "input_tokens": None, "output_tokens": None, "error": None}
    started = time.perf_counter()
    try:
        llm = get_llm(role)
        call["json_mode"] = getattr(llm, "format", None)
        call["configured_num_ctx"] = getattr(llm, "num_ctx", None)  # None means the server decides.
        if schema and call["model"].startswith("ollama:"):
            llm = llm.bind(format=schema)
            call.update(json_mode="schema", response_schema=schema)
        elif schema and call["model"].startswith("openai:"):
            llm = llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "mas_response", "schema": schema}})
            call.update(json_mode="schema", response_schema=schema)
        response = retry_model_call(lambda: llm.invoke(messages), call=call,
                                    retry_delays=llm_config.MODEL_SETTINGS["model"]["retry_delays"])
        metadata, usage = response.response_metadata or {}, response.usage_metadata or {}
        call.update(raw_response=response.content, response_metadata=deepcopy(metadata), usage_metadata=deepcopy(usage),
                    input_tokens=usage.get("input_tokens", metadata.get("prompt_eval_count")),
                    output_tokens=usage.get("output_tokens", metadata.get("eval_count")))
        if call["model"].startswith("ollama:"):
            call["server_timings_s"] = {key: metadata[key] / 1e9 for key in
                ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration") if metadata.get(key) is not None}
        result = JsonOutputParser().invoke(response)
        if not isinstance(result, dict):
            raise ValueError("Das Modell muss ein JSON-Objekt liefern.")
        for field, kind in (expected or {}).items():
            if field not in result or not isinstance(result[field], kind):
                raise ValueError(f"Modellantwort: {field!r} fehlt oder ist kein {kind.__name__}.")
        if role == "planner" and isinstance(result.get("task_queue"), list):
            for task in result["task_queue"]:
                if isinstance(task, dict) and "scope" in task:
                    task["scope"] = _scope_ids(task["scope"])
        if schema:
            if schema_state:
                if role == "planner":
                    for task in result.get("task_queue", []):
                        if isinstance(task, dict):
                            for field in ("actions", "requirements"):
                                fill_service_ids(schema_state, task.get("agent_id"), task.get(field, []))
                else:
                    fill_service_ids(schema_state, context["agent"], result.get("actions", []))
            Draft202012Validator(schema).validate(result)
        return result
    except Exception as exc:
        if isinstance(exc, ValidationError):
            exc = ValueError(f"Invalid model response at {list(exc.absolute_path)}: {exc.message}")
        call["error"] = f"{type(exc).__name__}: {exc}"
        call["technical_error"] = "raw_response" not in call
        raise exc
    finally:
        call["latency_s"] = time.perf_counter() - started
        simulator = SIMULATORS.get(request_id)
        if simulator is not None:
            with simulator.lock:
                simulator.llm_calls.append(call)


def human(state):
    text = interrupt("wait for user input...")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nicht leere Texteingabe erwartet.")
    return Command(update={"original_user_input": text},
                   goto="intent_analysis" if state["settings"]["intent_analysis_enabled"] else "task_planner")

def intent_analysis(state):
    state = refresh_state(state)
    result = _ask('Extract intents/facts and HOW, WHEN, WHERE, quantities and negation. '
                  'Output: {"infos": ["..."], "key_modifiers": ["..."]}.',
                  role="intent_analysis", expected={"infos": list}, request_id=state["request_id"],
                  user_message=state["original_user_input"])
    return Command(update={"infos": result["infos"], "key_modifiers": result.get("key_modifiers", [])}, goto="task_planner")


def read_rejections(result):
    rejected = result.get("rejected_requests", [])
    Draft202012Validator({"type": "array", "items": {"type": "object", "required": ["request", "reason"],
        "properties": {key: {"type": "string", "minLength": 1} for key in ("request", "reason")}}}).validate(rejected)
    return rejected


def task_planner(state):
    state = refresh_state(state)
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
    planner_only = not state["settings"]["domain_agents_enabled"]
    graph_instruction = (' Give each task a unique task_id (e.g. t1), scope, constraints '
                         '(conditions, timing, quantities, negation), and depends_on list of task-ID strings. '
                         'Example: depends_on=["t1", "t2"]. Never put objects or dictionaries in depends_on. '
                         'Independent goals have depends_on=[]. A task that needs earlier effects/information '
                         'depends on that predecessor. Combine related goals on the same target or order them. '
                         if graph_mode else '')
    cards, remaining_state = prompt_agents(state, available)
    if not planner_only:
        for card in cards.values():
            card["operations"] = {skill: {k: d[k] for k in ("preconditions",) if k in d}
                                  for skill, d in card.get("operations", {}).items()}
    context = dict(original_input=state["original_user_input"], available_agents=cards, schema_state=state,
                   search_fixture_queries=list(state["simulation_data"].get("search_results", {})),
                   rooms=state["home_config"]["rooms"], home_state=remaining_state, planner_only=planner_only)
    if state["simulation_data"].get("domain_planning_only") and not planner_only:
        # Omit native methods, but keep public context/state for resolving implicit goals.
        context.update(domain_planning_only=True, home_state={})
        context["available_agents"] = {agent: {**{k: card[k] for k in ("description", "context") if k in card},
            "devices": {key: {k: device[k] for k in ("room_id", "type", "state")}
                        for key, device in card["devices"].items()},
            "services": {key: {k: service[k] for k in ("room_id", "type", "state")}
                         for key, service in card["services"].items()}} for agent, card in cards.items()}
        context["room_states"] = {key: observation["value"] for key, observation in state["home_state"].items()
                                  if key.startswith("room.")}
    if state["settings"]["intent_analysis_enabled"]:
        context["intent_analysis"] = {"infos": state["infos"], "key_modifiers": state["key_modifiers"]}
    if state["simulation_data"].get("planning_error"):
        context["previous_error"] = state["simulation_data"]["planning_error"]
    if repair:
        context.update(repair=repair, tasks={key: state["tasks"][key] for key in repair["task_ids"]},
                       confirmed_results=state["task_history"])
    try:
        result = _ask(('Repair ONLY the listed tasks as JSON; original_input is background. ' if repair else
                       'Plan the ENTIRE original_input as JSON. ') +
                      'Resolve implicit targets and values from the public context, entity.state and explicit preferences. '
                      'Explicit targets and unique observed referents take precedence over default devices. '
                      'On the initial plan, if an essential choice remains unresolved after using the context '
                      'or user instructions conflict, put a targeted clarification question in response. '
                      'Keep all clear, independent goals in task_queue and execute them now; omit only unresolved goals '
                      'and tasks depending on their answer. Use task_queue=[] only when no independent work is possible. '
                      'Do not guess or reject a request merely because clarification is needed. '
                      'Read facts available through tools instead of asking the user for them. '
                      'This ends the current request; do not wait for a user reply or plan a follow-up dialogue. '
                      'Otherwise create one task per distinct user goal; preserve order, locations, quantities and conditions. '
                      'Use only agent IDs and device/service IDs from available_agents. '
                      'Methods are in entity.methods or the agent card methods_by_type[entity.type]. '
                      'Each methods map pairs exact skill names with their argument definitions. '
                      'Use the method KEY as skill_id and its argument names/limits, '
                      'adding device_id or service_id with the entity ID. '
                      'Current values are in each entity.state. '
                      'task.action MUST be a plain-language STRING describing the user goal, never a tool-call object. '
                      'Concrete tool calls belong only in task.actions, an ARRAY of objects. '
                      'Give scope as a list of room/device ID strings, e.g. scope=["living_room"]; never an object. '
                      'Give constraints as strings. '
                      'Give requirements=[{skill_id,arguments}] for requested outcomes with known targets/values; '
                      'include every requested device. '
                      + ('You are the sole action planner. Every task MUST include a nonempty actions array. '
                         'Catalogue groups are tool owners, not agents you can call. Never delegate. '
                         'Use read actions and task dependencies for missing information. '
                         'Use card.context control rules, current states and native units for implicit goals. '
                         'Preserve schedules with native start_time parameters when supported. '
                         if planner_only else 'Add actions only when concrete; otherwise let the domain agent refine. ') +
                      'Add prerequisite actions only when required by the listed method preconditions. '
                      'Do not add turn_on unless requested or required by those preconditions. '
                      'Prerequisites belong in actions; they do not replace required outcomes. '
                      'For missing predecessor data use {"from_task":"<task ID>","field":"<output field>"}; declare its dependency. '
                      'For displays set source_field to events, inventory or results; literal_content=true only for explicit user text/URLs. '
                      'For each invalid user subrequest add {"request":"<original clause>","reason":"<unavailable device/method/value>"} '
                      'only to rejected_requests, never as a tool call. Keep the valid tasks. '
                      'Do not invent devices or substitute another device type. '
                      'Check contradictory times against reference_time and observed appliance states; explain conflicts instead of guessing. '
                      'Reject only when evidence establishes impossibility; '
                      + ('plan read actions for uncertain state questions. ' if planner_only else
                         'route uncertain capability/state questions to a domain first. ') +
                      'Planning/execution failures are not invalid user requests. '
                      'Intent analysis is optional supporting context; original_input is authoritative. '
                      'For repair output ONLY repair.task_ids, preserving their IDs, goals and dependencies. '
                      'Never repeat the original whole plan or introduce new tasks. '
                      + graph_instruction + ' Output: {"task_queue":[],"rejected_requests":[],"response":""}. '
                      'A task has task_id, agent_id, action, requirements and '
                      + ('actions. ' if planner_only else 'optional actions. ') +
                      'An action has task_id, skill_id, arguments and depends_on. '
                      'Format example for a SEPARATE illustrative catalogue: agent thermostat, '
                      'device study_heater, methods={"set_temperature":{"temperature_c":{"type":"number"}}}. '
                      'Request: Set the study temperature to 21 degrees. Output: '
                      '{"task_queue":[{"task_id":"t1","agent_id":"thermostat",'
                      '"action":"Set the study temperature to 21 degrees",'
                      '"requirements":[{"skill_id":"set_temperature","arguments":{"device_id":"study_heater","temperature_c":21}}],'
                      '"actions":[{"task_id":"a1","skill_id":"set_temperature",'
                      '"arguments":{"device_id":"study_heater","temperature_c":21},"depends_on":[]}],'
                      '"depends_on":[]}],"rejected_requests":[],"response":""}. '
                      'Use IDs and argument names from the ACTUAL catalogue, not from this illustrative example. '
                      + ('' if planner_only else
                         'When domain_planning_only=true, output high-level task goals and scope only; '
                         'omit actions and requirements. Domain agents receive the full native method catalogue. ') +
                      ('For room-condition questions use home_information read methods when available; '
                         'for device questions use its catalogue group read methods. ' if planner_only else
                         'If home_information is available, route room-condition questions there; device questions go to the device domain. ') +
                      'Information questions require readings, never state changes. '
                      'Interpret discomfort as a directional room goal (too hot: lower temperature, too dim: more light). '
                      'Use original_input and user_location for implicit goals; do not override user perception with sensor values. '
                      'Use the controlling catalogue group with the room as scope for implicit goals. '
                      'For future device goals preserve every time relation in action/constraints; use start_time when supported. '
                      'For appliance-relative timing obtain anchor readings first and declare dependencies; include anchor/target IDs in scope. '
                      'Interpret relative-to-now times from the fixed reference_time, not a later now. '
                      'For numeric state changes use {"from_state":"device.<ID>.<field>","add":<signed delta>} '
                      'instead of calculating a final value. HomeBench percentages change percentage points.',
                      role="planner", expected={"task_queue": list}, request_id=state["request_id"], **context)
        rejected = read_rejections(result)
        queue = []
        for index, raw in enumerate(result["task_queue"]):
            if not isinstance(raw, dict) or not isinstance(raw.get("action"), str) or not raw["action"].strip():
                raise ValueError("Planner-Auftrag braucht agent_id und eine nicht leere action.")
            if raw.get("agent_id") not in available:
                raise ValueError(f"PlanError: agent_id={raw.get('agent_id')!r} is unavailable; "
                                 f"allowed IDs: {list(available)}.")
            if planner_only and not raw.get("actions"):
                raise ValueError("Planner-only requires concrete actions for every task; delegation is disabled.")
            task_id = raw.get("task_id", f"t{index + 1}") if graph_mode else f"{state['request_id']}:{index}"
            queue.append(dict(task_id=task_id, agent_id=raw["agent_id"], action=raw["action"],
                              depends_on=raw.get("depends_on", []) if graph_mode else [], status="pending",
                              scope=raw.get("scope", []), constraints=raw.get("constraints", [])))
            if raw.get("actions"):
                queue[-1]["actions"] = [{**a, "arguments": relative_arguments(state, queue[-1], a)} for a in raw["actions"]]
            queue[-1]["goal"] = (deepcopy(state["tasks"][task_id]["goal"])
                                  if repair and task_id in state["tasks"] else freeze_goal({**raw, **queue[-1]}, state))
            if repair and task_id not in repair["task_ids"]:
                raise ValueError(f"PlanError: repair must use the existing task IDs {repair['task_ids']}.")
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
                    # Repair may change routing, but keeps the assigned goal and dependencies.
                    for key in ("action", "scope", "constraints", "repairs", "depends_on"):
                        if key in state["tasks"][t["task_id"]]:
                            t[key] = deepcopy(state["tasks"][t["task_id"]][key])
                settings = {**state["settings"], "replans": state["settings"]["replans"] + 1}
                retained = {i: a for i, a in state["action_graph"].items()
                            if a["task"]["task_id"] not in ids or a["status"] == "success"}
                for t in queue:
                    t["generation"] = state["tasks"].get(t["task_id"], {}).get("generation", 0) + 1
                return Command(update={"tasks": {**state["tasks"], **{t["task_id"]: t for t in queue}},
                                       "action_graph": retained, "repair_context": {}, "settings": settings,
                                       "rejected_requests": [*state["rejected_requests"], *rejected]}, goto="plan_agents")
        return Command(update={"rejected_requests": rejected, "task_queue": queue, "tasks": {t["task_id"]: t for t in queue},
                               "final_response": result.get("response", "")},
                       goto=("plan_agents" if graph_mode else "task_planner") if queue else "finish")
    except (ValueError, KeyError, TypeError, ValidationError) as exc:
        if not repair:
            if state["settings"]["planner_replan"] and state["settings"]["replans"] < state["settings"]["max_replans"]:
                return Command(update={"settings": {**state["settings"], "replans": state["settings"]["replans"] + 1},
                    "simulation_data": {**state["simulation_data"], "planning_error": str(exc)}}, goto="task_planner")
            return Command(update={"final_response": json.dumps(
                {"status": "failed", "error": f"PlannerError: {exc}",
                 "failure": {"kind": "plan"}}, ensure_ascii=False)}, goto="finish")
        error = f"PlannerRepairError: {exc}"
        settings = {**state["settings"], "replans": state["settings"]["replans"] + 1}
        if settings["planner_replan"] and settings["replans"] < settings["max_replans"]:
            return Command(update={"settings": settings, "repair_context": {**repair, "reason": error}},
                           goto="task_planner")
        tasks = deepcopy(state["tasks"])
        for task_id in repair["task_ids"]:
            tasks[task_id].update(status="failed", terminal=True, error=error,
                                  failure={"kind": "plan"})
        return Command(update={"tasks": tasks, "settings": settings, "repair_context": {}}, goto="plan_agents")


def _agent(state, agent):
    state = refresh_state(state)
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
                         'If action a2 needs the result of a1, a2 has depends_on=["a1"]. '
                         'Do not request collaboration when collaboration_allowed is false. '
                         'For repair, propose only missing/corrective actions; do not repeat retained successful actions. '
                         'If repair requires changing cross-agent routing/dependencies, return actions=[] with '
                         'replan_required=true and reason. Stay within the assigned goal and its constraints. '
                         if graph_mode else '')
    task.setdefault("goal", freeze_goal(task, state))
    instructions = ('Fulfil ONLY task.action and task.goal; original_input is background, not an additional assignment. '
                    'Preserve task.goal and its required targets/values; complete all outcomes after prerequisites. '
                    'Use {"from_task":"t1","field":"results.0.content"} for confirmed predecessor data; '
                    'use "format":"text" for records/lists. Retrieve missing data through collaboration. '
                    'Return ordered calls with concrete IDs using operation parameters/conditions. '
                    'Add prerequisites only when required by the listed operation preconditions; never assume turn_on is required. '
                    'For plain on/off requests choose the native power method; do not add brightness or mode changes. '
                    'Use only listed operations; never turn an error message into a skill_id. Do not invent genre matches. '
                    'Use current state, cards and previous confirmed results. Include prerequisite actions yourself. '
                    'For numeric changes use {"from_state":"device.<ID>.<field>","add":<signed delta>}; '
                    'Python computes the final value. HomeBench percentages change percentage points. '
                    'Never fabricate inventory, time, appointments or search facts. For ALL devices emit separate calls. '
                    'For information questions use only read methods; obtain the requested values without changing devices. '
                    'For implicit room goals, use native environment control rules from agent_card.context when provided, '
                    'current states and the user goal to choose supported controls and required adjustments. '
                    'Power alone may be insufficient; preserve the requested direction and native units. '
                    'If a device method offers start_time, set it to a future virtual YYYY-MM-DD HH:MM:SS for delayed execution; '
                    'omit it for immediate execution. Methods without this parameter cannot schedule device actions. '
                    'Compute relative-to-now times from reference_time; preserve before/after and chained time relations. '
                    'For appliance-relative times prefer get_cycle_status.ends_at, or use observed CountdownTime (seconds) with its observation time. '
                    'cycle_duration gives native cycle seconds by selector value for planning simultaneous completion. '
                    'Read anchor state first or request information from its domain; never invent a duration or end time. '
                    'Check all timing claims together; reject contradictions and impossible completion deadlines before control writes. '
                    'Reject unavailable capabilities or unreachable goals only with catalogue/state/rule evidence. '
                    'Report unsupported subrequests only in rejected_requests, never as tool calls. '
                    'Return rejected_requests=[{"request":"<clause>","reason":"<observed limitation or conflict>"}]; '
                    'actions may be empty for a justified rejection. Keep independent achievable actions. '
                    'Execution/format errors are not evidence that the user request is impossible. '
                    'Request one other agent only for missing information; do not ask it to execute your goal. '
                    'When requesting help, return actions=[] and the collaboration_request object. '
                    'After receiving help, complete without another collaboration request. ' + graph_instruction + 'Output: '
                    '{"actions": [{"skill_id": "<operation>", "arguments": {"<parameter>": "<value>"}}], '
                    '"collaboration_request": {"target": "fridge", "request": "specific question"} or {}, '
                    '"rejected_requests": []}.')
    prompt_state = state
    scope = task.get("scope") or [r["arguments"][key] for r in task["goal"]["requirements"]
                                 for key in ("device_id", "service_id") if key in r["arguments"]]
    if scope:
        card = deepcopy(available[agent])
        card["devices"] = {key: device for key, device in card["devices"].items()
                           if key in scope or device["room_id"] in scope}
        device_types = {device["type"] for device in card["devices"].values()}
        card["services"] = {key: service for key, service in card["services"].items()
                            if key in scope or service["room_id"] in scope
                            or service.get("serves_type") in device_types
                            or service["room_id"] is None and not service.get("serves_type")}
        skills = {skill for section in ("devices", "services") for entity in card[section].values() for skill in entity["skills"]}
        card["operations"] = {skill: definition for skill, definition in card.get("operations", {}).items() if skill in skills}
        available[agent] = card
        prefixes = tuple(f"{scope}.{key}." for section, scope in (("devices", "device"), ("services", "service")) for key in card[section])
        prompt_state = {**state, "home_state": {key: value for key, value in state["home_state"].items()
                                              if key.startswith(prefixes) or key.startswith("room.")}}
    cards, remaining_state = prompt_agents(prompt_state, {agent: available[agent]})
    instructions += (' Methods are in entity.methods or agent_card.methods_by_type[entity.type]. '
                     'Method maps pair exact skill names with argument definitions. '
                     'Use the method KEY as skill_id and its argument names/limits; '
                     'Copy entity.target into arguments: services require service_id, devices require device_id. '
                     'State values are nested in each entity.state.')
    try:
        result = _ask(instructions, role=agent, expected={"actions": list}, request_id=state["request_id"],
                      schema_state=state,
                      response_schema=domain_response_schema(state, agent, available[agent]),
                      original_input=state["original_user_input"],
                      search_fixture_queries=list(state["simulation_data"].get("search_results", {})) if agent == "search_engine" else [],
                      agent=agent, task=task, agent_card=cards[agent], available_agents=list(available),
                      home_state=remaining_state,
                      task_history=[{**h, "results": [r for r in h["results"]
                          if r["status"] == "success" and state["action_graph"].get(r["task_id"], {}).get("status") == "success"]}
                          for h in state["task_history"]] if graph_mode else state["task_history"],
                      repair={"attempt": task.get("repairs", 0), "error": task.get("error"),
                              "retained_actions": [a for a in state["action_graph"].values()
                                                   if a["task"]["task_id"] == task["task_id"] and a["status"] == "success"],
                              "previous_results": [{**r, "retained": state["action_graph"].get(r["task_id"], {}).get("status") == "success"}
                                                   for r in state["task_results"]
                                                   if r["task_id"].rsplit("/", 1)[0] == task["task_id"]]} if graph_mode else {},
                      now=state["simulation_data"]["now"],
                      collaboration_allowed=not helping and (not pending if graph_mode else pending is None))
        rejected = read_rejections(result)
    except BackendUnavailable:
        raise
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
    if not actions and rejected:
        update = {"rejected_requests": [*state["rejected_requests"], *rejected],
                  "task_history": [{"agent_id": agent, "action": task["action"], "type": "rejection",
                                    "rejected_requests": rejected, "results": []}]}
        if helping:
            return Command(update={**update, "collaboration_request": {}}, goto=help_request["requester"] + "_agent")
        return Command(update={**update, "tasks": {**state["tasks"], task["task_id"]: {**task, "status": "rejected"}},
                               "task_queue": state["task_queue"][1:], "pending_task": None}, goto="task_planner")
    if not actions:
        raise ValueError("Agent muss mindestens einen strukturierten Tool-Aufruf liefern.")
    if any(not isinstance(action, dict) for action in actions):
        raise ValueError("Jeder Tool-Aufruf muss ein Dictionary sein.")
    if helping and any(state["agent_registry"][agent]["operations"][a["skill_id"]]["writes"] for a in actions):
        raise ValueError("Kollaboration dient nur zur Informationsbeschaffung.")
    actions = [{**action, "task": task, "task_id": f"{task['task_id']}:{index}"}
               for index, action in enumerate(actions)]
    return Command(update={"pending_actions": actions, "rejected_requests": [*state["rejected_requests"], *rejected]}, goto="validate_action")

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
        ) or any(task_id in t["depends_on"] and t["status"] not in {"success", "failed", "blocked", "rejected"}
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
                if a.get("failure") or not task.get("failure"):
                    task["failure"] = a.get("failure") or {
                        "kind": "postcondition" if i in invalid and a["footprint"][1] else "dispatch"}
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
        if not state["settings"]["local_repair"] or task.get("requires_replan") or task.get("repairs", 0) >= state["settings"]["max_repairs"]:
            if state["settings"]["planner_replan"] and not task.get("parent") and state["settings"]["replans"] < state["settings"]["max_replans"]:
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
            if task["status"] == "pending" and any(tasks[d]["status"] in {"failed", "blocked", "rejected"} and tasks[d].get("terminal")
                                                   for d in task["depends_on"]):
                task.update(status="rejected" if any(tasks[d]["status"] == "rejected" for d in task["depends_on"]) else "blocked",
                            error="Vorgaenger abgelehnt oder fehlgeschlagen.")
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
    if all(t["status"] in {"success", "failed", "blocked", "rejected"} for t in tasks.values()):
        return Command(update={"tasks": tasks, "action_graph": graph, "home_state": home}, goto="finish")
    return Command(update={"tasks": tasks, "action_graph": graph, "home_state": home}, goto="validate_plan")

def collect_plans(state):
    tasks, graph = deepcopy(state["tasks"]), deepcopy(state["action_graph"])
    rejected, history = list(state["rejected_requests"]), []
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
                task.update(status="failed", requires_replan=True, error=response.get("reason", "Cross-agent repair requested."),
                            failure={"kind": "plan"})
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
            fill_service_ids(state, task["agent_id"], actions)
            Draft202012Validator({"type": "array", "items": action_schema(state, task["agent_id"],
                get_available_agents(state)[task["agent_id"]])}).validate(actions)
            actions = [{**a, "arguments": relative_arguments(state, task, a)} for a in actions]
            for action in actions:
                for name, schema in operation_parameters(state, task["agent_id"], action["skill_id"], action["arguments"]).items():
                    if "default" in schema:
                        action["arguments"].setdefault(name, deepcopy(schema["default"]))
            rejected.extend(response.get("rejected_requests", []))
            if not actions and response.get("rejected_requests"):
                task.update(status="rejected", terminal=True)
                task.pop("error", None)
                task.pop("failure", None)
                history.append({"agent_id": task["agent_id"], "action": task["action"], "type": "rejection",
                                "rejected_requests": response["rejected_requests"], "results": []})
                if task.get("parent"):
                    tasks[task["parent"]].update(status="pending", help_done=True)
                continue
            if not actions or any(not isinstance(a, dict) for a in actions):
                raise ValueError("Agent muss konkrete Aktionen liefern.")
            retained = [a for a in graph.values() if a["task"]["task_id"] == task_id and a["status"] == "success"]
            retained_ids = {}
            if task.get("generation"):
                missing = []
                for index, action in enumerate(actions):
                    old = next((a for a in retained if a["skill_id"] == action["skill_id"] and
                                action["arguments"] in (a["arguments"], argument_template(a))), None)
                    if old:
                        retained_ids[action.get("task_id", f"a{index + 1}")] = old["task_id"]
                    else:
                        missing.append(action)
                actions = missing
                if not actions:
                    check_goal(state, task, retained, complete=True)
                    task["status"] = "success"
                    continue
            requirements = task["goal"]["requirements"]
            if not any(r["arguments"] for r in requirements):
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
                if task.get("parent") and state["agent_registry"][task["agent_id"]]["operations"][action["skill_id"]]["writes"]:
                    raise ValueError("Kollaboration dient nur zur Informationsbeschaffung.")
                # Old ordered action lists remain usable: omitted depends_on means previous action.
                deps = action.get("depends_on", local_ids[index - 1:index] if index else [])
                if not isinstance(deps, list):
                    raise ValueError("depends_on muss eine Liste sein.")
                deps = [retained_ids.get(d, d if "/" in d else prefix + d) for d in deps]
                action_id = prefix + local_ids[index]
                additions[action_id] = {**action, "task_id": action_id, "task": deepcopy(task),
                                        "agent_id": task["agent_id"], "depends_on": list(dict.fromkeys(deps + predecessor_actions)),
                                        "status": "pending"}
                previous = next((r for r in reversed(state["task_results"]) if r["task_id"].split("/")[0] == task_id
                                 and r["skill_id"] == action["skill_id"] and r["arguments"].get("device_id") == action["arguments"].get("device_id")
                                 and r["arguments"].get("service_id") == action["arguments"].get("service_id")), {})
                additions[action_id]["argument_sources"] = {name: ref for name, ref in previous.get("argument_sources", {}).items()
                    if "from_state" in ref and action["arguments"].get(name) == {"from_state": ref["from_state"], "add": ref["add"]}}
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
        except (KeyError, ValueError, TypeError, CycleError, ValidationError) as exc:
            task.update(status="failed", error=f"PlanError: {exc.message if isinstance(exc, ValidationError) else exc}", failure={"kind": "plan"})
    return Command(update={"tasks": tasks, "action_graph": graph, "rejected_requests": rejected,
                           "task_history": history}, goto="plan_agents")

def validate_plan(state):
    graph = deepcopy(state["action_graph"])
    home = SIMULATORS[state["request_id"]].snapshot()
    affected = set()
    cross_agent = False
    try:
        # Only explicitly approved prerequisites; preserve existing ordering and goals.
        for action_id in tuple(TopologicalSorter({i: a["depends_on"] for i, a in graph.items()}).static_order()):
            action = graph[action_id]
            if action["status"] != "pending" or not state["settings"]["validate_contracts"] or not state["settings"]["auto_prepare"]:
                continue
            affected = {action["task"]["task_id"]}
            args, agent = action["arguments"], action["agent_id"]
            operations = state["agent_registry"][agent]["operations"]
            for path, preparation in operations[action["skill_id"]].get("auto_prepare", {}).items():
                if not any(c.get("state") == path and c.get("equals") is True
                           for c in operations[action["skill_id"]]["preconditions"]):
                    continue
                order = tuple(TopologicalSorter({i: a["depends_on"] for i, a in graph.items()}).static_order())
                ancestors = {}
                for i in order:
                    ancestors[i] = set(graph[i]["depends_on"]).union(*(ancestors[d] for d in graph[i]["depends_on"]))
                key = path.format_map(args)
                writers = [i for i in order if i in ancestors[action_id] and graph[i]["status"] == "pending"
                           and key in graph[i]["footprint"][1]]
                powered = home.get(key, {}).get("value")
                if writers:
                    previous = graph[writers[-1]]
                    conditions = state["agent_registry"][previous["agent_id"]]["operations"][previous["skill_id"]]["postconditions"]
                    powered = next((c["equals"] for c in conditions if "equals" in c and
                                    c.get("state", "").format_map(previous["arguments"]) == key), None)
                if powered is True:
                    continue
                on_id = next((i for i, a in graph.items() if a["skill_id"] == preparation and
                              a["agent_id"] == agent and a["arguments"].get("device_id") == args["device_id"] and
                              a["status"] == "pending" and i not in ancestors[action_id] and
                              i != action_id and action_id not in ancestors[i] and
                              all(w in ancestors[i] for w in writers)), None)
                if on_id is None:
                    on_id = f"{action_id}:prepare-{len(graph)}"
                    graph[on_id] = dict(task_id=on_id, task=deepcopy(action["task"]), agent_id=agent,
                                        skill_id=preparation, arguments={"device_id": args["device_id"]},
                                        depends_on=list(action["depends_on"]), status="pending")
                    reads, writes, target = action_footprint({**state, "action_graph": graph}, graph[on_id])
                    graph[on_id]["footprint"] = (sorted(reads), sorted(writes), target)
                action["depends_on"].append(on_id)
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
        return Command(update={"action_graph": graph}, goto="scheduler")
    except (KeyError, ValueError, TypeError, CycleError) as exc:
        tasks = deepcopy(state["tasks"])
        affected = affected or {a["task"]["task_id"] for a in graph.values() if a["status"] == "pending"}
        for task_id in affected:
            tasks[task_id].update(status="failed", error=f"PlanError: {exc}", requires_replan=cross_agent,
                                  failure={"kind": "plan"})
        retained = {i: a for i, a in graph.items() if a["task"]["task_id"] not in affected or a["status"] == "success"}
        return Command(update={"tasks": tasks, "action_graph": retained}, goto="plan_agents")

def lean_value(value, kind):
    """Encode selected evidence exactly as Lean decodes the consuming argument."""
    if kind == "bool" and type(value) is bool:
        return int(value)
    if isinstance(kind, dict) and "enum" in kind and type(value) is str and value in kind["enum"]:
        return kind["enum"].index(value)
    if isinstance(kind, dict) and "number" in kind and type(value) in (int, float):
        encoded = Fraction(str(value)) * kind["number"]
        if encoded.denominator == 1 and abs(encoded) <= 10**9:
            return int(encoded)
    raise ValueError(f"Lean reference value {value!r} does not match {kind!r}.")


def lean_schema(schema):
    """The parameter schema Lean checks, and whether null (an omitted optional value) is allowed."""
    types = schema.get("type")
    if isinstance(types, list) and "null" in types:
        rest = [t for t in types if t != "null"]
        return ({**schema, "type": rest[0]} if len(rest) == 1 else {}), True
    options = [s for s in schema.get("anyOf", []) if s.get("type") != "null"]
    if "anyOf" in schema and len(options) == len(schema["anyOf"]) - 1:
        return (options[0] if len(options) == 1 else {}), True
    return schema, False


def check_allocation(state, batch, snapshot=None):
    """Vaibhav's unchanged checker over a finite registry projection; rich contracts stay in Python.
    Arguments are sent as typed values (numbers, booleans, enum strings) together with their
    declared types; the checker decodes them itself and rejects wrongly typed values."""
    if state["settings"]["checker"] != "lean":
        return
    if snapshot is None:  # Also supports direct checker calls in tests.
        simulator = SIMULATORS[state["request_id"]]
        with simulator.lock:
            snapshot = (deepcopy(simulator.home), deepcopy(simulator.reservations))
    home, reservations = snapshot
    registry = state["agent_registry"]
    actions = {**state["action_graph"], **{a["task_id"]: a for a in batch}}
    agents = sorted(registry)
    entities = {f"{scope}.{i}": e for section, scope in (("devices", "device"), ("services", "service"))
                for i, e in state["home_config"][section].items()}
    target_of = lambda args: "device." + args["device_id"] if "device_id" in args else "service." + args["service_id"]
    home_device = "__home__"

    def footprint(a):
        d = registry[a["agent_id"]]["operations"][a["skill_id"]]
        reads = d["reads"] + [c[k] for c in d["preconditions"] for k in ("state", "equals_state") if k in c]
        return {p.format_map(a["arguments"]) for p in reads}, {p.format_map(a["arguments"]) for p in d["writes"]}

    # Lean's operations write only their own target. A command that writes another entity's state
    # (Home Assistant's name/area tools) is checked as a command on one pseudo-device for the whole
    # home, with every key it touches as explicit state of that device; commands that only read such
    # keys read them as shared keys. These keys alone decide the conflicts (no exclusive target), so
    # whole-home commands with disjoint keys stay parallel. Lean lists at most 64 keys per operation,
    # so the keys are grouped into blocks: keys that no footprint of the batch or of a running command
    # tells apart share one block. Without such commands (DEXA, HomeBench, SimuHome) nothing changes.
    footprints = {aid: footprint(a) for aid, a in actions.items()}
    home_actions, home_keys = set(), set()
    while found := {aid for aid, (_, w) in footprints.items() if aid not in home_actions and any(
            k.rsplit('.', 1)[0] != target_of(actions[aid]["arguments"]) or k in home_keys for k in w)}:
        home_actions |= found
        home_keys |= {k for aid in found for k in footprints[aid][0] | footprints[aid][1]}
    relevant = [aid for aid in (*(a["task_id"] for a in batch), *reservations) if aid in footprints]
    groups = {}
    for k in sorted(home_keys):
        groups.setdefault(frozenset(aid for aid in relevant if k in footprints[aid][0] | footprints[aid][1]), []).append(k)
    block = {k: "|".join(ks) for ks in groups.values() for k in ks}

    def op_key(aid, a):
        # One Lean operation per device whose own schema has limits/options (external benchmarks),
        # and per set of omitted optional arguments, since Lean's arguments are positional.
        target = target_of(a["arguments"])
        own = entities.get(target, {}).get("parameters", {}).get(a["skill_id"])
        schema = own if own is not None else registry[a["agent_id"]]["operations"][a["skill_id"]]["parameters"]
        omitted = tuple(sorted(n for n, p in schema.items() if lean_schema(p)[1] and a["arguments"].get(n) is None))
        reads, writes = footprints[aid]
        at_home = aid in home_actions
        keys_ = (tuple(sorted(reads)), tuple(sorted(writes))) if at_home else tuple(sorted(reads & home_keys))
        return a["agent_id"], a["skill_id"], target if own is not None else None, omitted, at_home, keys_

    keys = {aid: op_key(aid, a) for aid, a in actions.items()}
    op_keys = sorted(set(keys.values()), key=repr)
    op_ids = {key: i for i, key in enumerate(op_keys)}
    known = sorted(set(entities) | {k.rsplit('.', 1)[0] for k in home} | ({home_device} if home_actions else set()))
    # Targets outside the home (e.g. hallucinated devices) get IDs without a catalogue entry,
    # so the checker rejects their commands instead of the projection failing with a KeyError.
    targets = {target_of(a["arguments"]) for a in actions.values()}
    dev_ids = {name: i for i, name in enumerate(known + sorted(targets - set(known)))}
    attrs = sorted({k.rsplit('.', 1)[-1] for k in home} | {"__target__"} | set(block.values()) | {
        k.rsplit('.', 1)[-1] for card in registry.values() for d in card["operations"].values()
        for k in d["reads"] + d["writes"] + [c["state"] for c in d["preconditions"] if "state" in c]})
    attr_ids = {name: i for i, name in enumerate(attrs)}
    task_ids = {name: i for i, name in enumerate(actions)}
    key = lambda path: [dev_ids[path.rsplit('.', 1)[0]], attr_ids[path.rsplit('.', 1)[-1]]]
    # State and evidence keep the real keys; footprints name whole-home keys by their block.
    keyset = lambda paths: [list(k) for k in sorted({(dev_ids[home_device], attr_ids[block[p]]) if p in home_keys
                                                     else tuple(key(p)) for p in paths})]
    # Finite string equality for enum preconditions; free text/list/object values remain opaque.
    strings = sorted({c["equals"] for card in registry.values() for d in card["operations"].values()
                      for c in d["preconditions"] if isinstance(c.get("equals"), str)})
    scalar = lambda v: (int(v) if type(v) in (bool, int) and abs(v) <= 10**9 else
                        strings.index(v) + 2 if isinstance(v, str) and v in strings else 0)
    ops, fields, int_options = [], {}, {}
    # Anything but a whole number for whole-number options is sent as null, which Lean rejects.
    wire = lambda key_, i, v: (str(v) if type(v) is int else None) if i in int_options.get(key_, ()) else v
    for op_key_ in op_keys:
        agent, skill, target, omitted, at_home, footprint_keys = op_key_
        d = registry[agent]["operations"][skill]
        if at_home and d["preconditions"]:
            raise ValueError(f"Lean: preconditions of the whole-home command {skill} cannot be checked.")
        parameters = entities[target]["parameters"][skill] if target else d["parameters"]
        projected, bounds, types = [], [], []
        for name, schema in parameters.items():
            schema, optional = lean_schema(schema)
            if optional and name in omitted:
                continue
            if "enum" in schema and all(isinstance(v, str) for v in schema["enum"]):
                projected.append(name)
                bounds.append([0, len(schema["enum"]) - 1])
                types.append({"enum": schema["enum"]})
            elif "enum" in schema and all(type(v) is int for v in schema["enum"]):
                # Lean's options are text: whole-number options travel as their decimal text.
                projected.append(name)
                bounds.append([0, len(schema["enum"]) - 1])
                types.append({"enum": [str(v) for v in schema["enum"]]})
                int_options.setdefault(op_key_, set()).add(len(projected) - 1)
            elif schema.get("type") == "boolean":
                projected.append(name)
                bounds.append([0, 1])
                types.append("bool")
            elif schema.get("type") in ("integer", "number"):
                values = [schema[k] for k in ("minimum", "maximum") if k in schema] + [a["arguments"][name] for aid, a in actions.items()
                    if keys[aid] == op_key_ and type(a["arguments"].get(name)) in (int, float)
                    and math.isfinite(a["arguments"][name])]
                for a in batch:
                    ref = a.get("argument_sources", {}).get(name)
                    if keys[a["task_id"]] == op_key_ and ref and "from_task" in ref and ref.get("format") != "text":
                        values.extend([output_value(actions[ref["action_id"]]["output"], ref["field"]), ref.get("add", 0)])
                values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
                scale = math.lcm(*(Fraction(str(v)).denominator for v in values))
                # Exact decimal scaling, never rounding; values outside the finite codec stay in Python.
                if scale <= 10**9 and all(abs(Fraction(str(v)) * scale) <= 10**9 for v in values):
                    projected.append(name)
                    bounds.append([math.ceil(Fraction(str(schema["minimum"])) * scale) if "minimum" in schema else -10**9,
                                   math.floor(Fraction(str(schema["maximum"])) * scale) if "maximum" in schema else 10**9])
                    types.append({"number": scale})
        fields[op_key_] = projected
        pre, reads = [], set(d["reads"])
        for c in d["preconditions"]:
            if "state" not in c:
                continue
            reads.add(c["state"])
            if c.get("known"):
                pre.append([attr_ids[c["state"].rsplit('.', 1)[-1]], "ge", -10**9])
            elif "equals" in c and type(c["equals"]) in (bool, int, str):
                pre.append([attr_ids[c["state"].rsplit('.', 1)[-1]], "eq", scalar(c["equals"])])
        if at_home:  # explicit whole-home keys, by block; they alone decide the conflicts
            op_reads, op_writes = (sorted({attr_ids[block[k]] for k in ks}) for ks in footprint_keys)
            shared, excl = [], []
        else:
            op_reads = sorted({attr_ids[p.rsplit('.', 1)[-1]] for p in reads})
            op_writes = sorted({attr_ids[p.rsplit('.', 1)[-1]] for p in d["writes"]})
            shared, excl = keyset(footprint_keys), [attr_ids["__target__"]]
        ops.append(dict(id=op_ids[op_key_], cap=agents.index(agent), bounds=bounds, types=types,
                        reads=op_reads, writes=op_writes, excl=excl, shared_reads=shared, pre=pre,
                        coupled=False, load_base=0, load_per_unit=0))
    devices = [dict(id=dev_ids[name], caps=[agents.index(e["agent_id"])],
                    ops=[i for (agent, skill, target, _, at_home, _), i in op_ids.items() if agent == e["agent_id"]
                         and not at_home and target in (None, name) and skill in e["skills"]
                         and registry[agent]["enabled"] and skill in registry[agent]["capabilities"]])
               for name, e in entities.items()]
    if home_actions:
        home_ops = {i: k[0] for k, i in op_ids.items() if k[4]}
        devices.append(dict(id=dev_ids[home_device], caps=sorted({agents.index(a) for a in home_ops.values()}),
                            ops=sorted(home_ops)))
    ranks, tasks, evidence, commands = {}, [], [], []
    views, references = {}, []
    selected_ids = {a["task_id"] for a in batch}
    for aid in TopologicalSorter({i: a["depends_on"] for i, a in actions.items()}).static_order():
        a = actions[aid]
        args = a["arguments"]
        agent, skill = a["agent_id"], a["skill_id"]
        target = target_of(args)
        # Typed values as they are; the checker decodes them with the declared types.
        vals = [wire(keys[aid], i, args.get(name)) for i, name in enumerate(fields[keys[aid]])]
        ranks[aid] = 1 + max((ranks[d] for d in a["depends_on"]), default=-1)
        verified = a["status"] == "success" and a.get("verified", False)
        tasks.append(dict(id=task_ids[aid], status="pending" if aid in selected_ids else
                          "verified" if verified else "failed" if a["status"] in {"failed", "blocked"} else "pending",
                          op=op_ids[keys[aid]], device=dev_ids[home_device if aid in home_actions else target],
                          # Only the batch's tasks need their arguments (for the contract check).
                          args=[{"lit": v} for v in vals] if aid in selected_ids else [],
                          deps=[task_ids[d] for d in a["depends_on"]], rank=ranks[aid], pre=[], deadline=10**12))
        if verified:
            evidence.append(dict(task=task_ids[aid], verified=True, valid_until=int(a["valid_until"]), output=None,
                                 basis=[[*key(k), v] for k, v in a["evidence_versions"].items()]))
        if aid in selected_ids:
            commands.append(dict(task=task_ids[aid], device=dev_ids[home_device if aid in home_actions else target],
                                 op=op_ids[keys[aid]], args=vals))
            for index, name in enumerate(fields[keys[aid]]):
                ref = a.get("argument_sources", {}).get(name)
                if not ref or "from_task" not in ref or ref.get("format") == "text":
                    continue
                kind = ops[op_ids[keys[aid]]]["types"][index]
                provider = ref["action_id"]
                value = lean_value(wire(keys[aid], index, output_value(actions[provider]["output"], ref["field"])), kind)
                offset = lean_value(ref.get("add", 0), kind) if isinstance(kind, dict) and "number" in kind else 0
                view = (ref["field"], kind, value)
                options = views.setdefault(task_ids[provider], [])
                if view not in options:
                    options.append(view)
                references.append((tasks[-1], index, task_ids[provider], options.index(view), offset, vals[index], ref))
    active = [dict(reads=keyset(r), writes=keyset(w), excl=[] if aid in home_actions else
                   [[dev_ids[("device." if t[0] == "device_id" else "service.") + t[1]], attr_ids["__target__"]]])
              for aid, (r, w, t) in reservations.items()]
    payload = dict(schema="dexa-alloc-request/1", request_id=uuid4().hex, auth_ttl=30,
        catalogue=dict(revision=1, ops=ops, devices=devices, load_attr=attr_ids["__target__"], meter_key=[0, 0], load_budget=0),
        graph=dict(revision=1, tasks=tasks),
        snapshot=dict(now=int(time.time()), worker_cap=state["settings"]["max_workers"],
                      state=[[*key(k), scalar(o["value"]), o.get("version", 0)] for k, o in home.items()],
                      evidence=evidence, active=active),
        batch=dict(graph_revision=1, catalogue_revision=1, commands=commands))
    # Lean has one output per source task. Check distinct field/type views separately;
    # never place ambiguous outputs for the same source in one request.
    for round_index in range(max((len(v) for v in views.values()), default=1)):
        for task, index, provider, view_index, offset, literal, ref in references:
            task["args"][index] = ({"ref": provider, "offset": offset} if view_index == round_index else
                                   {"lit": literal})
        for ev in evidence:
            options = views.get(ev["task"], [])
            ev["output"] = options[round_index][2] if round_index < len(options) else None
        payload["request_id"] = uuid4().hex
        payload["snapshot"]["now"] = int(time.time())
        result = subprocess.run([state["settings"]["lean_executable"], "--oneshot"], input=json.dumps(payload, allow_nan=False) + "\n",
                                capture_output=True, text=True, timeout=15, check=True)
        response = json.loads(result.stdout)
        if response.get("request_id") != payload["request_id"]:
            raise ValueError("Lean allocation checker answered a different request.")
        # Check the verdict first, so that a rejection reports Lean's own reason.
        if response.get("verdict") != "accept":
            raise ValueError(f"Lean allocation checker rejected the batch: {response.get('diagnostics', response.get('error'))}")
        # An accepted batch is echoed with its decoded integer arguments.
        expected = [{**c, "args": [lean_value(v, kind) for v, kind in zip(c["args"], ops[c["op"]]["types"])]}
                    for c in commands]
        if response.get("commands") != expected:
            raise ValueError("Lean allocation checker accepted different commands than were sent.")
        for action in batch:
            action["dispatch_until"] = min(action["dispatch_until"], time.monotonic() + max(0, response["expires_at"] - time.time()))
    for *_, ref in references:
        ref["validation"] = "lean"

def scheduler(state):
    state = refresh_state(state)
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
                                    status="blocked", verified=False, output={}, observations={}, error="Vorgaenger fehlgeschlagen.",
                                    failure={"kind": "dispatch", "reason": "dependency"}))
                changed = True
    if blocked:
        return Command(update={"action_graph": graph, "task_results": blocked, "active_actions": [r["task_id"] for r in blocked]}, goto="complete_round")
    limit = state["settings"]["max_workers"] if state["settings"]["parallel_actions"] else 1
    ready = [a for a in graph.values() if a["status"] == "pending"
             and all(graph[d]["status"] == "success" for d in a["depends_on"])]
    retries = [a for a in ready if a.get("execution_retries", 0)]
    if retries:
        ready = retries  # Direct retries also bypass the optional LLM scheduler.
    llm_scheduler = state["settings"]["scheduler_mode"] == "llm" and not retries
    if llm_scheduler and ready:
        ids = _ask('You schedule the next execution batch. Select 1..max_actions unique action IDs from actions. '
                   'Choose useful parallel work: no shared target and no write/write or read/write conflicts. '
                   'Keep agents and arguments unchanged. Output only {"action_ids": ["t1/a1"]}.',
                   expected={"action_ids": list}, role="scheduler", request_id=state["request_id"], original_input=state["original_user_input"],
                   max_actions=limit, actions=[{"action_id": a["task_id"], "agent_id": a["agent_id"],
                       "goal": a["task"]["action"], "skill_id": a["skill_id"], "arguments": a["arguments"],
                       "reads": a["footprint"][0], "writes": a["footprint"][1], "target": a["footprint"][2]}
                       for a in ready])["action_ids"]
        if not ids or len(ids) > limit or len(set(ids)) != len(ids) or set(ids) - {a["task_id"] for a in ready}:
            raise ValueError("SchedulerError: choose a nonempty batch of unique ready action IDs within max_actions.")
        ready = [graph[i] for i in ids]
    selected, rejected, targets, round_reads, round_writes = [], [], set(), set(), set()
    simulator = SIMULATORS[state["request_id"]]
    with simulator.lock:
        snapshot = (deepcopy(simulator.home), deepcopy(simulator.reservations))
        state = {**state, "home_state": {**state["home_state"], **snapshot[0]}}
        try:
            for action in ready:
                args = action["arguments"]
                target = ("device", args["device_id"]) if "device_id" in args else ("service", args["service_id"])
                reads, writes, _ = action_footprint(state, action)
                if target in targets or writes & (round_reads | round_writes) or reads & round_writes:
                    if llm_scheduler:
                        raise ValueError("SchedulerError: proposed actions conflict within the batch.")
                    continue  # conservative: one simultaneous action per device/service
                checked = validate_action({**state, "pending_actions": [action]}, snapshot[0])
                if checked.goto != "execute_action":
                    action["status"] = checked.update["task_results"][0]["status"]
                    rejected.extend(checked.update["task_results"])
                    continue
                action.update(checked.update["pending_actions"][0])
                state["home_state"] = {**state["home_state"], **checked.update["home_state"]}
                targets.add(target)
                round_reads.update(action["footprint"][0])
                round_writes.update(action["footprint"][1])
                action["status"] = "running"
                selected.append(action)
                if len(selected) >= limit:
                    break
        except Exception:
            for action in selected:
                simulator.release(action["task_id"])
            raise
    try:
        if selected:
            check_allocation(state, selected, snapshot)
    except Exception as exc:
        for action in selected:
            simulator.release(action["task_id"])
        if not isinstance(exc, (ValueError, OSError, subprocess.SubprocessError)):
            raise
        for action in selected:
            action["status"] = "failed"
            rejected.append(dict(task_id=action["task_id"], attempt_id=uuid4().hex, agent_id=action["agent_id"],
                                 skill_id=action["skill_id"], arguments=action["arguments"], status="failed", verified=False,
                                 argument_sources=deepcopy(action.get("argument_sources", {})),
                                 output={}, observations={}, error=f"AdmissionError: {exc}",
                                 failure={"kind": "dispatch", "source": "lean"}))
        selected = []
    if not selected and not rejected:
        raise ValueError("PlanError: keine ausfuehrbare Aktion; Abhaengigkeiten pruefen.")
    return Command(update={"action_graph": graph, "task_results": rejected, "home_state": state["home_state"],
                           "active_actions": [a["task_id"] for a in selected] + [r["task_id"] for r in rejected]},
                   goto=[Send("execute_action", {**state, "pending_actions": [a]}) for a in selected] if selected else "complete_round")

def complete_round(state):
    home = {**state["home_state"], **SIMULATORS[state["request_id"]].snapshot()}
    graph_mode = state["settings"]["dependency_graph"]
    pending = deepcopy(state["pending_actions"])
    graph = deepcopy(state["action_graph"]) if graph_mode else {pending[0]["task_id"]: pending[0]}
    active = state["active_actions"] if graph_mode else graph.keys()
    # The reducer retains all attempts; only the latest result changes an action's status.
    latest = {r["task_id"]: r for r in state["task_results"] if r["task_id"] in active}
    for result in latest.values():
        action = graph[result["task_id"]]
        action["status"] = result["status"]
        failure = result.get("failure") or {}
        operation = state["agent_registry"][result["agent_id"]]["operations"].get(result["skill_id"], {})
        if result["status"] == "stale":
            action["dispatch_retries"] = action.get("dispatch_retries", 0) + 1
            action["status"] = "pending" if state["settings"]["dispatch_retries"] and action["dispatch_retries"] <= state["settings"]["max_dispatch_retries"] else "failed"
        elif result["status"] == "failed" and failure.get("kind") in {"tool", "output", "postcondition"}:
            if state["settings"]["action_retries"] and action.get("execution_retries", 0) < operation.get("max_retries", 0):
                action["execution_retries"] = action.get("execution_retries", 0) + 1
                action["status"] = "pending"
        for key in ("output", "before", "evidence_versions", "valid_until", "verified", "error", "failure"):
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
    if not graph_mode:
        action = pending[0]
        task = action["task"]
        if action["status"] == "pending":
            return Command(update={"home_state": home, "pending_actions": pending}, goto="validate_action")
        remaining = pending[1:] if action["status"] == "success" else []
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
        final = {r["task_id"]: r for r in results}
        task = {**task, "status": "success" if all(r["status"] == "success" for r in final.values()) else "failed",
                "error": action.get("error"), "failure": action.get("failure")}
        return Command(update={**update, "tasks": {**state["tasks"], task["task_id"]: task},
                               "task_queue": state["task_queue"][1:], "pending_task": None}, goto="task_planner")
    tasks, history = deepcopy(state["tasks"]), []
    for task in tasks.values():
        actions = [a for a in graph.values() if a["task"]["task_id"] == task["task_id"]]
        if task["status"] != "planned" or not actions or any(a["status"] in {"pending", "running"} for a in actions):
            continue
        task["status"] = "failed" if any(a["status"] == "failed" for a in actions) else "blocked" if any(a["status"] == "blocked" for a in actions) else "success"
        if task["status"] == "success":
            try:
                check_goal(state, task, [a for a in actions if a["status"] == "success"], complete=True)
                task.pop("error", None)
                task.pop("failure", None)
            except ValueError as exc:
                task.update(status="failed", error=str(exc), failure={"kind": "plan"})
        else:
            failed = [a for a in actions if a["status"] == "failed"] or [a for a in actions if a["status"] == "blocked"]
            task.update(error="; ".join(a["error"] for a in failed if a.get("error")), failure=failed[0].get("failure"))
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
    answer = ""
    if state["simulation_data"].get("answer_enabled"):
        state = refresh_state(state)
        units = ('Native temperature and humidity readings are scaled by 100; lux and PM10 are unscaled. '
                 if state["simulation_data"].get("benchmark") == "simuhome" else
                 'Use the units supplied by the tools and state attributes. ')
        answer = _ask('Answer original_input using ONLY the supplied observed results and execution report. '
                      'If execution_report asks for clarification, preserve that targeted question even when tasks ran. '
                      'Report confirmed independent results, then ask about the unresolved part; do not claim it was executed '
                      'or that the whole request is complete. End after the question, without waiting for a user reply. '
                      'For information questions state every requested value and the correct room/device. '
                      + units +
                      'Clearly distinguish observations, accepted commands and achieved physical effects. '
                      'Do not infer completion from command acceptance. Explain failures or missing evidence. '
                      'For rejected requests explain the supported limitation or timing conflict, including the relevant times. '
                      'For accepted future actions confirm the schedule, never claim the action has already happened. '
                      'Return JSON: {"answer":"<natural-language answer>"}.',
                      role="answer", expected={"answer": str}, request_id=state["request_id"],
                      original_input=state["original_user_input"], results=state["task_results"],
                      tasks=state["tasks"], rejected_requests=state["rejected_requests"],
                      execution_report=state["final_response"])["answer"]
    if state["settings"]["dependency_graph"]:
        summary = [{"task_id": t["task_id"], "agent_id": t["agent_id"], "action": t["action"],
                    "status": t["status"], "error": t.get("error"), "failure": t.get("failure"),
                    "results": [{"skill_id": r["skill_id"], "status": r["status"], "verified": r["verified"], "output": r["output"], "error": r["error"],
                                 "failure": r.get("failure"), "execution_retries": r.get("execution_retries", 0)}
                                for r in state["task_results"] if r["task_id"].rsplit("/", 1)[0] == t["task_id"]]}
                   for t in state["tasks"].values()]
        return Command(update={"final_response": state["final_response"] or json.dumps(summary, ensure_ascii=False), "answer": answer}, goto=END)
    response = state["final_response"] or json.dumps(
        [{"agent_id": h["agent_id"], "action": h["action"],
          "results": [{"skill_id": r["skill_id"], "status": r["status"], "output": r["output"], "error": r["error"],
                                 "failure": r.get("failure"), "execution_retries": r.get("execution_retries", 0)}
                      for r in h["results"]]} for h in state["task_history"]], ensure_ascii=False)
    return Command(update={"final_response": response, "answer": answer}, goto=END)

# All domain nodes use the same implementation with their selected registry.
def _node(agent):
    def run(state):
        return _agent(state, agent)
    run.__name__ = agent + "_agent"
    return run

def build_graph(registry, *, domain_agents_enabled=True):
    builder = StateGraph(SmartHomeState)
    for name, node in {"human": human, "intent_analysis": intent_analysis, "task_planner": task_planner,
                       "validate_action": validate_action, "execute_action": execute_action,
                       "plan_agents": plan_agents, "collect_plans": collect_plans, "validate_plan": validate_plan,
                       "scheduler": scheduler,
                       "complete_round": complete_round, "finish": finish}.items():
        builder.add_node(name, node)
    if domain_agents_enabled:
        for agent in registry:
            node = _node(agent)
            builder.add_node(node.__name__, node)
    builder.add_edge(START, "human")
    return builder.compile(checkpointer=MemorySaver())


graph = build_graph(AGENT_REGISTRY)


def run_request(text, **setup):
    configure_models(setup.get("config_path"))
    state = create_initial_state(**setup)
    config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": state["settings"]["recursion_limit"]}
    selected_graph = (build_graph(state["agent_registry"], domain_agents_enabled=state["settings"]["domain_agents_enabled"])
                      if "agent_registry" in setup or not state["settings"]["domain_agents_enabled"] else graph)
    proposals, failure = {}, None
    try:
        yield {"experiment_config": {"settings": state["settings"]}}
        try:
            for _ in selected_graph.stream(state, config):
                pass
            for event in selected_graph.stream(Command(resume=text), config):
                for update in event.values():
                    if isinstance(update, dict):
                        actions = list(update.get("action_graph", {}).values()) + update.get("pending_actions", [])
                        for action in actions:
                            if action["task_id"] not in proposals:
                                proposals[action["task_id"]] = deepcopy(action)
                yield event
        except Exception as exc:
            failure = exc
        final = selected_graph.get_state(config).values or state
        calls = deepcopy(SIMULATORS[state["request_id"]].llm_calls)
        task_failures = [{"task_id": t["task_id"], "kind": "blocked" if t.get("status") == "blocked" else (t.get("failure") or {}).get("kind", "plan"),
                          "error": t.get("error") or t["status"]} for t in final["tasks"].values()
                         if t.get("status") in {"failed", "blocked"}]
        try:
            response = json.loads(final["final_response"])
        except (TypeError, ValueError):
            response = None
        if isinstance(response, dict) and response.get("status") == "failed":
            task_failures.append({"kind": (response.get("failure") or {}).get("kind", "plan"),
                                  "error": response.get("error") or "Planning failed"})
        error = (f"{type(failure).__name__}: {failure}" if failure else
                 "; ".join(t["error"] for t in task_failures) or None)
        yield {"episode_result": {"proposed_actions": list(proposals.values()), "task_results": final["task_results"],
                  "rejected_requests": final["rejected_requests"], "tasks": final["tasks"],
                  "final_response": final["final_response"], "error": error, "task_failures": task_failures,
                  "answer": final.get("answer", ""),
                  "llm_calls": calls, "model_usage": summarize_calls(calls)}}
        if failure is not None:
            raise failure
    finally:
        SIMULATORS.pop(state["request_id"], None)
        selected_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])


def interactive(home_assistant=None, **setup):
    while True:
        text = input("\nYour command (quit to exit): ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if text:
            started = time.perf_counter()
            # Read the real devices' power state again for every request.
            extra = home_assistant_setup(home_assistant) if home_assistant else {}
            for event in run_request(text, **setup, **extra):
                for node, update in event.items():
                    if node != "episode_result" and update and update.get("final_response"):
                        print(update["final_response"])
            print(f"Time: {time.perf_counter() - started:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart home interactive session")
    add_settings_arguments(parser)
    parser.add_argument("--home-assistant", action="store_true",
                        help="switch the devices listed in HA_ENTITIES (.env) on and off through Home Assistant")
    args = parser.parse_args()
    interactive(config_path=args.config, home_assistant=HomeAssistant.from_env() if args.home_assistant else None,
                **settings_from_args(args))
