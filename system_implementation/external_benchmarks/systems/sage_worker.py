"""Python 3.10 worker for original SAGE with native benchmark I/O."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

API_DEVICE = "00000000-0000-0000-0000-000000000001"
PROTOCOL = sys.stdout


def emit(message):
    PROTOCOL.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
    PROTOCOL.flush()


def rpc(kind, **payload):
    emit({"kind": kind, **payload})
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("Benchmark bridge closed.")
    reply = json.loads(line)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["result"]


def inject(name, **attrs):
    """Substitute an external service boundary before importing its consumers."""
    parent_name, child = name.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    module = ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    setattr(parent, child, module)


def load_original(root):
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Original SAGE requires a separate Python 3.10 environment (--sage-python).")
    sys.path.insert(0, str(root))
    os.environ["SMARTHOME_ROOT"] = str(root)
    # Upstream defines ClaudeConfig eagerly; this benchmark never constructs it.
    os.environ.setdefault("ANTHROPIC_API_KEY", "")
    from sage.base import BaseToolConfig

    class UnavailableConfig(BaseToolConfig):
        def __init__(self):
            super().__init__(_target=None)

        def instantiate(self, **kwargs):
            raise RuntimeError("This external service is not supplied by the benchmark.")

    class ColdMemory:
        """No benchmark provides personal interaction history or a user profile."""
        def read_from_json(self, path):
            self.history = json.loads(Path(path).read_text())

        def save_snapshot(self, path):
            Path(path).write_text(json.dumps(self.history))

    # These tools are excluded from tool_configs below, not simulated as successful.
    for module, names in {
        "sage.smartthings.device_disambiguation": ["DeviceDisambiguationToolConfig"],
        "sage.smartthings.persistent_command_tools": ["ConditionCheckerToolConfig", "NotifyOnConditionToolConfig"],
        "sage.smartthings.tv_schedules": ["QueryTvScheduleToolConfig"],
        "sage.misc_tools.weather_tool": ["WeatherToolConfig"],
    }.items():
        inject(module, **{name: UnavailableConfig for name in names})
    inject("sage.retrieval.memory_bank", MemoryBank=ColdMemory)
    # No original SmartThings/MongoDB transport is used. Device I/O is installed below.
    inject("sage.smartthings.db", DeviceCapabilityDb=UnavailableConfig)
    inject("sage.testing.fake_requests")
    from sage.coordinators.sage_coordinator import SAGECoordinator, SAGECoordinatorConfig
    from sage.smartthings import smartthings_tool
    return SAGECoordinator, SAGECoordinatorConfig, smartthings_tool


def run_episode(root, start):
    Coordinator, Config, native = load_original(root)
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.chat_models.base import BaseChatModel
    from langchain.schema import AIMessage, ChatGeneration, ChatResult
    from sage.base import BaseConfig, GlobalConfig
    from sage.coordinators.base import AgentConfig
    from sage.coordinators import base as coordinator_base
    from sage.smartthings.docmanager import DocManager

    context, tools = start["context"], {tool["name"]: tool for tool in start["tools"]}
    tools["reject_request"] = {
        "name": "reject_request", "description": "Explicitly report one unfulfillable user-request clause; complete valid clauses separately.",
        "parameters": {"type": "object", "properties": {"request": {"type": "string"}, "reason": {"type": "string"}},
                       "required": ["request", "reason"], "additionalProperties": False}}
    settings = start["settings"]

    class Model(BaseChatModel):
        @property
        def _llm_type(self):
            return "benchmark-http-bridge"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            roles = {"human": "user", "ai": "assistant", "system": "system"}
            payload = [{"role": roles[message.type], "content": message.content} for message in messages]
            text = rpc("model", messages=payload, stop=stop)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    class Trace(BaseCallbackHandler):
        def on_agent_action(self, action, **kwargs):
            emit({"kind": "event", "event": {"type": "agent_action", "tool": action.tool,
                "tool_input": action.tool_input, "log": action.log}})

        def on_agent_finish(self, finish, **kwargs):
            output = str(finish.return_values.get("output", ""))
            emit({"kind": "event", "event": {"type": "agent_finish", "output": output}})

    class Docs(DocManager):
        def __init__(self):
            self.default_devices = {API_DEVICE: {"name": start["benchmark"] + " native environment API"}}
            self.device_capabilities = {API_DEVICE: [{"capability_id": name} for name in (*tools, "benchmark_context")]}

        def capability_summary_for_devices(self):
            summaries = "\n".join(name + ": " + tool["description"] for name, tool in tools.items())
            devices = (f"Device ID: {API_DEVICE}. This is the benchmark environment API, not a physical device. "
                "Its capabilities are native benchmark tools. Preserve their exact names, targets and units. "
                "Use api_doc_retrieval before invoking a capability. The read-only capability benchmark_context "
                "has attribute context containing the initially supplied public information.\n"
                + json.dumps({"public_context": context, "capabilities": list(tools)}, ensure_ascii=False))
            return summaries, devices

        def device_capability_details(self, device_id, capability_id):
            if capability_id == "benchmark_context":
                return "Use get_attribute with capability benchmark_context and attribute context."
            tool = tools[capability_id]
            return json.dumps({"device_id": API_DEVICE, "component": "main", "capability": capability_id,
                "command": capability_id, "description": tool["description"], "native_parameters": tool["parameters"],
                "transport": "Use execute_command. args must be a list containing one object of native parameters, e.g. args=[{...}]. For parameterless tools args=[] also works. Never rename native targets or fields."})

        def has_refresh_capability(self, device_id):
            return False

    DocManager.from_json = classmethod(lambda cls, path: Docs())

    class Requests:
        def post(self, url, json, headers=None):
            try:
                if url != f"https://api.smartthings.com/v1/devices/{API_DEVICE}/commands":
                    raise ValueError("Unknown benchmark API device or route.")
                values = []
                for command in json["commands"]:
                    name, args = command["command"], command.get("arguments", [])
                    if command["component"] != "main" or command["capability"] != name:
                        raise ValueError("Use component main and the exact native tool name as both capability and command.")
                    if args == []:
                        params = {}
                    elif isinstance(args, list) and len(args) == 1 and isinstance(args[0], dict):
                        params = args[0]
                    else:
                        raise ValueError("args must contain one native-parameters object. See api_doc_retrieval.")
                    values.append(rpc("reject", **params) if name == "reject_request" else rpc("tool", name=name, arguments=params))
                return SimpleNamespace(status_code=200, json=lambda: {"results": values})
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                error = {"error": f"{type(exc).__name__}: {exc}"}
                return SimpleNamespace(status_code=400, json=lambda: error)

        def get(self, url, headers=None):
            expected = f"https://api.smartthings.com/v1/devices/{API_DEVICE}/components/main/capabilities/benchmark_context/status"
            if url == expected:
                return SimpleNamespace(status_code=200, json=lambda: {"context": {"value": context}})
            return SimpleNamespace(status_code=400, json=lambda: {"error": "Use execute_command to invoke a native benchmark tool with its parameters. See api_doc_retrieval."})

    # Logging is replaced; the original prompts, planner and agent execution are retained.
    coordinator_base.get_callback_handlers = native.get_callback_handlers = lambda *args, **kwargs: [Trace()]
    with tempfile.TemporaryDirectory(prefix="sage-benchmark-") as directory:
        directory = Path(directory)
        memory = directory / "memory.json"
        memory.write_text("{}")
        BaseConfig.global_config = GlobalConfig(logpath=str(directory), smartthings_token="",
                                               docmanager_cache_path=directory / "unused.json")
        config = Config(verbose=False, run_mode="test", memory_path=str(memory),
            enable_memory_updating=False, enable_human_interaction=False, enable_google=False,
            agent_config=AgentConfig(verbose=False), llm_config=SimpleNamespace(instantiate=Model),
            tool_configs=(native.SmartThingsToolConfig(tool_configs=(native.SmartThingsPlannerToolConfig(),
                native.ApiDocRetrievalToolConfig(), native.GetAttributeToolConfig(), native.ExecuteCommandToolConfig())),))
        coordinator = Coordinator(config)
        coordinator.agent.max_iterations = settings["max_steps"]
        coordinator.agent.max_execution_time = settings["episode_timeout"]
        coordinator.agent.verbose = False
        for tool in coordinator.tooldict["smartthings_tool"].tools:
            if hasattr(tool, "requests_module"):
                tool.requests_module = Requests()
        # Input contains only the same public information supplied to every participant.
        command = "benchmark_user: " + start["request"] + "\n\nPublic benchmark context: " + json.dumps(context, ensure_ascii=False)
        response = coordinator.execute(command)
        answer = response["output"]
        stopped = answer.startswith("Agent stopped due to")
        return {"answer": answer, "status": "max_steps" if stopped else "finished",
                "error": answer if stopped else None}


def main():
    sys.stdout = sys.stderr  # Upstream prints/logs cannot corrupt the JSON-lines protocol.
    root = Path(sys.argv[-1]).resolve()
    if "--probe" in sys.argv:
        Coordinator, _, _ = load_original(root)
        from importlib.metadata import version
        emit({"python": sys.version.split()[0], "langchain": version("langchain"),
              "coordinator": Coordinator.__module__ + "." + Coordinator.__name__})
        return
    start = json.loads(sys.stdin.readline())
    try:
        report = run_episode(root, start)
    except Exception as exc:
        report = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "exception": type(exc).__name__}
    emit({"kind": "result", "report": report})


if __name__ == "__main__":
    main()
