"""All knowledge of MAS registries, settings and events lives in this wrapper."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from model_client import BackendUnavailable


def add_settings_arguments(parser):
    from smart_home_langgraph import add_settings_arguments as configure
    return configure(parser)


def settings_from_args(args):
    from smart_home_langgraph import settings_from_args as extract
    return extract(args)


def homebench_command(action, config):
    arguments = action.get("arguments", {})
    target = arguments.get("device_id", "unknown_device")
    device = config["devices"].get(target, {})
    name = device.get("homebench_name", str(target))
    skill = action.get("skill_id", "unknown_method")
    schema = device.get("parameters", {}).get(skill, {})
    names = [key for key in schema if key != "device_id"] or [key for key in arguments if key != "device_id"]
    def literal(value):
        return "(" + ",".join(literal(v) for v in value) + ")" if isinstance(value, list) else str(value)
    return f"{name}.{skill}({','.join(literal(arguments.get(k, 'missing_argument')) for k in names)})"


def homebench_prediction(report, config):
    attempts = {}
    for result in report.get("task_results", []):
        if result["status"] != "blocked":
            attempts.setdefault(result["task_id"], []).append(result)
    commands = []
    for proposal in report.get("proposed_actions", []):
        commands.extend(homebench_command(a, config) for a in attempts.get(proposal["task_id"], [proposal]))
    commands.extend("error_input" for _ in report.get("rejected_requests", []))
    return "{" + ",".join(commands) + "}"


class MAS:
    name = "mas"

    def __init__(self, settings=None, config_path=None, module=None):
        if module is None:
            import smart_home_langgraph as module
        self.module, self.config_path = module, config_path
        self.settings = module.resolve_settings(settings, config_path=config_path)

    def metadata(self):
        import llm_config
        llm_config.configure_models(self.config_path)
        root = Path(self.module.__file__).parent
        return {"name": self.name, "settings": self.settings,
                "models": {key: getattr(llm_config, key, None) for key in
                           ("DEFAULT_MODEL", "SAME_LLM_FOR_ALL_AGENTS", "ROLE_MODELS", "MODEL_SETTINGS")},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}}

    def run_episode(self, episode):
        session = episode.environment
        if episode.benchmark == "dexa":
            home, registry, bindings = dexa_home(episode)
            extra = {"environment": session}
        elif episode.benchmark == "homebench":
            from ..agent_registry_homebench import build_home, execute
            home, registry = build_home(session.initial_home)
            bindings = {(a, skill): execute for a, card in registry.items() for skill in card["operations"]}
            extra = {}
        elif episode.benchmark in {"simuhome", "homeassistant"}:
            from importlib import import_module
            build_home = import_module("external_benchmarks.agent_registry_" + episode.benchmark).build_home
            home, registry, bindings = build_home(session)
            extra = {"environment": session}
        else:
            raise NotImplementedError(episode.benchmark)
        simulation = {"benchmark": episode.benchmark, "search_results": {}, "search_available": False}
        if episode.benchmark == "dexa":
            simulation.update(now=episode.context["now"], search_available=True)
        if episode.benchmark != "homebench":
            simulation.update(domain_planning_only=True, answer_enabled=True,
                              user_location=episode.context.get("user_location"),
                              reference_time=episode.context.get("reference_time"))
        report, effective = {}, self.settings
        try:
            for event in self.module.run_request(episode.request, home_config=home, agent_registry=registry,
                    tool_bindings=bindings, simulation_data=simulation, config_path=self.config_path,
                    **extra, **self.settings):
                if "episode_result" in event:
                    report = deepcopy(event["episode_result"])
                if "experiment_config" in event:
                    effective = event["experiment_config"]["settings"]
        except Exception as exc:
            from langgraph.errors import GraphRecursionError
            model_failure = isinstance(exc, GraphRecursionError) or any(
                call.get("error") and "raw_response" in call for call in report.get("llm_calls", []))
            report.update(error=f"{type(exc).__name__}: {exc}", exception=type(exc).__name__,
                          technical_error=not model_failure)
        report["technical_error"] = report.get("technical_error", False) or any(
            call.get("technical_error") for call in report.get("llm_calls", []))
        report["status"] = "error" if report.get("error") else "finished"
        report["settings"] = effective
        report["backend_unavailable"] = isinstance(
            exc, BackendUnavailable
        )
        if episode.benchmark == "homebench":
            try:
                report["prediction"] = homebench_prediction(report, home)
            except (KeyError, TypeError, ValueError) as exc:
                report.update(prediction="{}", status="error", error=f"PredictionError: {exc}", technical_error=True)
        return report


def dexa_home(episode):
    """Translate only public tool schemas/state into MAS cards; never inspect gold labels."""
    context = episode.context
    home = {"rooms": deepcopy(context["rooms"]), "devices": {}, "services": {}}
    registry, bindings = {}, {}
    for target, entity in context["entities"].items():
        tools = [t for t in episode.tools if t["target_id"] == target]
        device = entity.get("room_id") is not None
        section, target_arg = ("devices", "device_id") if device else ("services", "service_id")
        prefix = ("device." if device else "service.") + target + "."
        pattern = ("device.{device_id}." if device else "service.{service_id}.")
        owner = tools[0]["domain"]
        card = registry.setdefault(owner, {"agent": owner + "_agent", "enabled": True,
            "description": f"DEXA {entity['type']}; use only the public tools and their actual observations.",
            "context": {"preferences": context["preferences"], "instructions": context["instructions"]},
            "capabilities": [], "operations": {}})
        config = {**deepcopy(entity), "room_id": entity.get("room_id"), "agent_id": owner, "skills": [], "parameters": {},
                  "initial": {k[len(prefix):]: (deepcopy(v), None) for k,v in context["state"].items() if k.startswith(prefix)}}
        for tool in tools:
            skill = tool["skill_id"]
            parameters = {target_arg: {"type": "string", "enum": [target], "default": target},
                          **deepcopy(tool["parameters"]["properties"])}
            config["skills"].append(skill)
            config["parameters"][skill] = parameters
            contract = deepcopy(tool["contract"])
            contract = json.loads(json.dumps(contract).replace(prefix, pattern))
            post = []
            field = {"set_temperature": "target_temperature_c", "set_brightness": "brightness_pct",
                     "set_color_temperature": "color_temperature_k", "set_color": "color",
                     "set_volume": "volume_pct", "set_mode": "mode"}.get(skill)
            if field:
                argument = "temperature_c" if skill == "set_temperature" else field
                post.append({"state": pattern + field, "equals_arg": argument})
            if skill in {"turn_on", "turn_off"}:
                post.append({"state": pattern + "power", "equals": skill == "turn_on"})
            if skill == "play":
                post = [{"state": pattern + "playback_status", "equals": "playing"},
                        {"state": pattern + "current_track", "equals_arg": "track"}]
            if skill == "show_content":
                post = [{"rule": "content_matches_request", "state": pattern + "current_content"}]
            card["operations"][skill] = {"description": tool["description"], "parameters": parameters,
                "output": {target_arg: {"type": "string"}, "data": {}},
                **contract, "postconditions": post,
                "max_retries": 0 if skill in {"set_timer", "set_alarm", "schedule", "create_event"} else 1}
            if contract["preconditions"] and any(t["skill_id"] == "turn_on" for t in tools):
                card["operations"][skill]["auto_prepare"] = {pattern + "power": "turn_on"}
            if skill not in card["capabilities"]: card["capabilities"].append(skill)
        names = {t["skill_id"]: t["name"] for t in tools}
        def invoke(skill, arguments, simulator, now, simulation, names=names, target_arg=target_arg):
            return {"data": episode.invoke(names[skill], {k:v for k,v in arguments.items() if k != target_arg})}
        for skill in names: bindings[(owner, skill, target)] = invoke
        home[section][target] = config
    # MAS dispatches by (domain, skill); the target argument selects the native tool.
    routed = {}
    for owner, card in registry.items():
        for skill in card["operations"]:
            def invoke(skill, args, simulator, now, simulation, owner=owner):
                target = args.get("device_id", args.get("service_id"))
                return bindings[(owner, skill, target)](skill, args, simulator, now, simulation)
            routed[(owner, skill)] = invoke
    return home, registry, routed

