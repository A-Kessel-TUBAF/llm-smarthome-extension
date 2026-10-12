import json
import math
import operator
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Annotated, Literal, TypedDict
from uuid import uuid4
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit
from jsonschema import Draft202012Validator, FormatChecker
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from agent_registry import AGENT_REGISTRY
from home_defaults import HOME_DEFAULTS

llm = ChatOllama(model="gemma2", temperature=0.0, format="json")

class Observation(TypedDict):
    value: Any
    observed_at: str
    source: str
    unit: str | None

class Task(TypedDict):
    task_id: str
    agent_id: str
    action: str
    status: Literal["pending", "success", "failed"]

class TaskResult(TypedDict):
    task_id: str
    attempt_id: str
    agent_id: str
    skill_id: str
    arguments: dict
    status: Literal["success", "failed"]
    output: dict
    observations: dict[str, Observation]
    error: str | None

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
    task_history: Annotated[list[dict], operator.add]
    final_response: str
    experiment: dict
    simulation_data: dict

# Timestamps/source are assigned by Python, never by the LLM.
def make_observation(value, *, source, unit=None, observed_at=None):
    return {"value": deepcopy(value), "source": source, "unit": unit,
            "observed_at": observed_at or datetime.now(timezone.utc).isoformat()}

def create_initial_state(observations=None, *, home_config=None, use_defaults=True,
                         observed_at=None, disabled_agents=(), include_operations=True,
                         include_conditions=True, simulation_data=None, validate_contracts=True):
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
    return dict(original_user_input="", request_id=uuid4().hex, infos=[], key_modifiers=[],
                task_queue=[], tasks={}, task_results=[], home_config=config, home_state=home,
                agent_registry=registry, collaboration_request={}, pending_task=None, pending_actions=[],
                task_history=[], final_response="", simulation_data=deepcopy(simulation_data or {}),
                experiment={"version": "V2" if validate_contracts else "V1",
                            "validate_contracts": validate_contracts, "include_operations": include_operations,
                            "include_conditions": include_conditions})

def get_available_agents(state, *, include_operations=None):
    if include_operations is None:
        include_operations = state["experiment"]["include_operations"]
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
            if not state["experiment"]["include_conditions"]:
                for definition in entry["operations"].values():
                    definition.pop("preconditions")
                    definition.pop("postconditions")
        available[agent] = entry
    return available

# Tools return output + property changes; they never mutate shared graph state.
def device_tool(skill, args, home, now, simulation):
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
        return {"inventory": get("inventory")}, {}
    else:
        raise ValueError(f"Unbekannter Device-Skill: {skill}")
    output = deepcopy(changes)
    if skill == "pause":
        output["current_track"] = get("current_track")
    return output, changes

def clock_tool(skill, args, home, now, simulation):
    prefix = f"service.{args['service_id']}."
    get = lambda prop: deepcopy(home[prefix + prop]["value"])
    stamp = datetime.fromisoformat(now.replace("Z", "+00:00"))
    if skill == "get_time":
        return {"current_time": stamp.astimezone(ZoneInfo(args["timezone"])).isoformat(),
                "timezone": args["timezone"]}, {}
    if skill in {"set_timer", "set_alarm"}:
        timer = skill == "set_timer"
        field, id_key = ("timers", "timer_id") if timer else ("alarms", "alarm_id")
        item = {id_key: uuid4().hex, "label": args["label"]}
        item.update({"duration_seconds": args["duration_seconds"],
                     "due_at": (stamp + timedelta(seconds=args["duration_seconds"])).isoformat()}
                    if timer else {"at": args["at"], "enabled": True})
        return {k: v for k, v in item.items() if k not in {"label", "enabled"}}, {field: get(field) + [item]}
    elapsed = get("stopwatch_elapsed_seconds")
    if skill == "start_stopwatch":
        return {"running": True, "started_at": now, "elapsed_seconds": elapsed}, {
            "stopwatch_running": True, "stopwatch_started_at": now}
    elapsed += (stamp - datetime.fromisoformat(get("stopwatch_started_at").replace("Z", "+00:00"))).total_seconds()
    return {"running": False, "elapsed_seconds": elapsed}, {
        "stopwatch_running": False, "stopwatch_started_at": None, "stopwatch_elapsed_seconds": elapsed}

def calendar_tool(skill, args, home, now, simulation):
    events = deepcopy(home[f"service.{args['service_id']}.events"]["value"])
    if skill == "list_events":
        dt = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        events = [e for e in events if (args["start"] is None or dt(e["end"]) > dt(args["start"]))
                  and (args["end"] is None or dt(e["start"]) < dt(args["end"]))]
        return {"events": sorted(events, key=lambda e: (dt(e["start"]), e["event_id"]))}, {}
    if skill == "create_event":
        event = {"event_id": uuid4().hex, **{k: args[k] for k in ("title", "start", "end", "location")}}
        return {"event": event}, {"events": events + [event]}
    return {"deleted_event_id": args["event_id"]}, {
        "events": [e for e in events if e["event_id"] != args["event_id"]]}

def search_tool(skill, args, home, now, simulation):
    # Explicit fixtures only: no fabricated weather, recipes or live search claims.
    rows = simulation.get("search_results", {}).get(args["query"], [])
    return {"query": args["query"], "provider": "simulation",
            "results": deepcopy(rows[:args["max_results"]])}, {}

TOOLS = {
    (agent, skill): (clock_tool if agent == "clock" else calendar_tool if agent == "calendar"
                    else search_tool if agent == "search_engine" else device_tool)
    for agent, card in AGENT_REGISTRY.items() for skill in card["capabilities"]
}
UNITS = {"brightness_pct": "%", "color_temperature_k": "K", "volume_pct": "%",
         "target_temperature_c": "degC", "stopwatch_elapsed_seconds": "s"}

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
        actual = value(condition["state"]) if "state" in condition else context[condition["context"]]
        if condition.get("known") is True:
            return True  # a present None is known; a missing key raises an error
        expected = condition["equals"] if "equals" in condition else arguments[condition["equals_arg"]]
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
    raise ValueError(f"Unbekannte Precondition-Regel: {rule}")

def validate_action(state):
    action = deepcopy(state["pending_actions"][0])
    task, args, skill = action["task"], action.get("arguments", {}), action.get("skill_id", "")
    try:
        card = state["agent_registry"][task["agent_id"]]
        if not card["enabled"] or skill not in card["capabilities"] or skill not in card["operations"]:
            raise ValueError("Agent oder Skill ist nicht verfuegbar.")
        definition = card["operations"][skill]
        if not isinstance(args, dict):
            raise ValueError("arguments muss ein Dictionary sein.")
        for key, schema in definition["parameters"].items():
            if key not in args and "default" in schema:
                args[key] = deepcopy(schema["default"])
        if state["experiment"]["validate_contracts"]:
            schema = {"type": "object", "properties": definition["parameters"],
                      "required": [k for k, v in definition["parameters"].items() if "default" not in v],
                      "additionalProperties": False}
            errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(args))
            # NaN/Infinity are Python numbers but not valid JSON arguments.
            json.dumps(args, allow_nan=False)
            if errors:
                raise ValueError("; ".join(f"{'.'.join(map(str, e.path)) or 'arguments'}: {e.message}" for e in errors))
        target_key = "device_id" if "device_id" in definition["parameters"] else "service_id"
        section = "devices" if target_key == "device_id" else "services"
        target = state["home_config"][section][args[target_key]]
        if target["agent_id"] != task["agent_id"] or skill not in target["skills"]:
            raise ValueError("Agent, Ziel und Skill passen nicht zusammen.")
        # V1 also needs mandatory arguments to dispatch a tool, but does not check their contracts.
        if set(definition["parameters"]) - args.keys():
            raise ValueError("Pflichtargument fehlt.")
        now = state["simulation_data"].get("now") or datetime.now(timezone.utc).isoformat()
        context = {"now": now, "mode": "simulation",
                   "search_available": state["simulation_data"].get("search_available", True)}
        if state["experiment"]["validate_contracts"]:
            for condition in definition["preconditions"]:
                if not check_condition(condition, args, state["home_state"], context):
                    raise ValueError(f"Precondition nicht erfuellt: {condition}")
        action.update(arguments=args, now=now)
        return Command(update={"pending_actions": [action, *state["pending_actions"][1:]]}, goto="execute_action")
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        result: TaskResult = dict(task_id=action["task_id"], attempt_id=uuid4().hex,
                                 agent_id=task["agent_id"], skill_id=skill, arguments=args,
                                 status="failed", output={}, observations={}, error=f"ValidationError: {exc}")
        return Command(update={"task_results": [result]}, goto="update_home_state")

def execute_action(state):
    action = state["pending_actions"][0]
    task, skill, args = action["task"], action["skill_id"], action["arguments"]
    result: TaskResult = dict(task_id=action["task_id"], attempt_id=uuid4().hex,
                             agent_id=task["agent_id"], skill_id=skill, arguments=deepcopy(args),
                             status="failed", output={}, observations={}, error=None)
    try:
        parameters = state["agent_registry"][task["agent_id"]]["operations"][skill]["parameters"]
        target_key = "device_id" if "device_id" in parameters else "service_id"
        scope = "device" if target_key == "device_id" else "service"
        output, changes = TOOLS[(task["agent_id"], skill)](
            skill, args, state["home_state"], action["now"], state["simulation_data"])
        observations = {f"{scope}.{args[target_key]}.{prop}": make_observation(
            value, source="simulation:tool", unit=UNITS.get(prop), observed_at=action["now"])
            for prop, value in changes.items()}
        result.update(status="success", output={target_key: args[target_key], **output}, observations=observations)
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        result["error"] = f"ToolError: {type(exc).__name__}: {exc}"
    return Command(update={"task_results": [result]}, goto="update_home_state")

def update_home_state(state):
    task = state["pending_actions"][0]["task"]
    result = state["task_results"][-1]
    home = {**state["home_state"], **result["observations"]}
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

def _ask(instructions, *, expected=None, **context):
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
    return Command(update={"original_user_input": text}, goto="intent_analysis")

def intent_analysis(state):
    result = _ask('Extract intents/facts and HOW, WHEN, WHERE, quantities and negation. '
                  'Output: {"infos": ["..."], "key_modifiers": ["..."]}.',
                  expected={"infos": list},
                  user_message=state["original_user_input"])
    return Command(update={"infos": result["infos"], "key_modifiers": result.get("key_modifiers", [])}, goto="task_planner")

def task_planner(state):
    if state["task_queue"]:
        return Command(goto=state["task_queue"][0]["agent_id"] + "_agent")
    if state["tasks"] or state["final_response"]:
        return Command(goto="finish")
    available = get_available_agents(state, include_operations=False)
    result = _ask('Assign each goal to one primary domain agent, in execution order. Preserve all constraints. '
                  'Use available agent IDs; let agents choose specific tools/devices and request collaboration. '
                  'Do not invent unavailable capabilities. Output: '
                  '{"task_queue": [{"agent_id": "lighting", "action": "full goal"}], "response": "reason if no tasks"}.',
                  expected={"task_queue": list},
                  original_input=state["original_user_input"], infos=state["infos"], key_modifiers=state["key_modifiers"],
                  available_agents={agent: card["capabilities"] for agent, card in available.items()},
                  rooms=state["home_config"]["rooms"],
                  home_state={key: observation["value"] for key, observation in state["home_state"].items()})
    queue = []
    for index, raw in enumerate(result["task_queue"]):
        if not isinstance(raw, dict) or not isinstance(raw.get("action"), str) or not raw["action"].strip():
            raise ValueError("Planner-Auftrag braucht agent_id und eine nicht leere action.")
        if raw.get("agent_id") not in available:
            raise ValueError("Plan enthaelt einen nicht verfuegbaren Agent.")
        queue.append(dict(task_id=f"{state['request_id']}:{index}", agent_id=raw["agent_id"], action=raw["action"], status="pending"))
    return Command(update={"task_queue": queue, "tasks": {t["task_id"]: t for t in queue},
                           "final_response": result.get("response", "") if not queue else ""},
                   goto=queue[0]["agent_id"] + "_agent" if queue else "finish")

def _agent(state, agent):
    available = get_available_agents(state)
    if agent not in available:
        raise ValueError("Agent ist nicht verfuegbar.")
    help_request = state["collaboration_request"]
    helping = help_request.get("target") == agent
    task = ({"task_id": help_request["task_id"] + ":help", "agent_id": agent, "action": help_request["request"]}
            if helping else state["task_queue"][0])
    pending = state["pending_task"]
    instructions = ('Choose concrete device/service IDs and return ordered tool calls for your assigned goal. '
                    'Use current state, cards and previous confirmed results. Include prerequisite actions yourself. '
                    'Never fabricate inventory, time, appointments or search facts. For ALL devices emit separate calls. '
                    'Timers/alarms only store entries; delayed device actions are not supported. '
                    'Request one other agent only for missing information; do not ask it to execute your goal. '
                    'When requesting help, return actions=[] and the collaboration_request object. '
                    'After receiving help, complete without another collaboration request. Output: '
                    '{"actions": [{"skill_id": "turn_on", "arguments": {"device_id": "..."}}], '
                    '"collaboration_request": {"target": "fridge", "request": "specific question"} or {}}.')
    result = _ask(instructions, expected={"actions": list},
                  agent=agent, task=task, agent_card=available[agent], available_agents=list(available),
                  home_state={key: observation["value"] for key, observation in state["home_state"].items()},
                  task_history=state["task_history"],
                  now=state["simulation_data"].get("now") or datetime.now(timezone.utc).isoformat(),
                  collaboration_allowed=not helping and pending is None)
    request = result.get("collaboration_request") or {}
    if request.get("target"):
        if helping or pending is not None or request["target"] == agent or request["target"] not in available:
            raise ValueError("Ungueltige oder wiederholte Kollaboration.")
        return Command(update={"collaboration_request": {**request, "requester": agent, "task_id": task["task_id"]},
                               "pending_task": task}, goto=request["target"] + "_agent")
    actions = result["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError("Agent muss mindestens einen strukturierten Tool-Aufruf liefern.")
    if any(not isinstance(action, dict) for action in actions):
        raise ValueError("Jeder Tool-Aufruf muss ein Dictionary sein.")
    if helping and any(a.get("skill_id") not in {"get_inventory", "get_time", "list_events", "search"} for a in actions):
        raise ValueError("Kollaboration dient nur zur Informationsbeschaffung.")
    actions = [{**action, "task": task, "task_id": f"{task['task_id']}:{index}"}
               for index, action in enumerate(actions)]
    return Command(update={"pending_actions": actions}, goto="validate_action")

def finish(state):
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
                   "update_home_state": update_home_state, "finish": finish}.items():
    builder.add_node(name, node)
for agent in AGENT_REGISTRY:
    node = _node(agent)
    globals()[node.__name__] = node
    builder.add_node(node.__name__, node)
builder.add_edge(START, "human")
graph = builder.compile(checkpointer=MemorySaver())

def run_request(text, **setup):
    config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 200}
    for _ in graph.stream(create_initial_state(**setup), config):
        pass
    yield from graph.stream(Command(resume=text), config)

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
    interactive()
