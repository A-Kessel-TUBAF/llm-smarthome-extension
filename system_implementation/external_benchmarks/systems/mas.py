"""All knowledge of MAS registries, settings and events lives in this wrapper."""
from copy import deepcopy
import hashlib
from pathlib import Path


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
        if episode.benchmark == "homebench":
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
        if episode.benchmark == "homebench":
            try:
                report["prediction"] = homebench_prediction(report, home)
            except (KeyError, TypeError, ValueError) as exc:
                report.update(prediction="{}", status="error", error=f"PredictionError: {exc}", technical_error=True)
        return report
