"""Native SimuHome episodes, simulator calls and evaluation."""
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from configuration import config_path as configured_path, load_config

from external_benchmarks.common import (Episode, cached_case, case_file, create_system,
    prepare_output, summarize_calls, system_report, write_json)

QUERY_TYPES = ("qt1", "qt2", "qt3", "qt4-1", "qt4-2", "qt4-3")
SUPPORTED_CATEGORIES = {f"{qt}/{case}" for qt in QUERY_TYPES for case in ("feasible", "infeasible")}


def needs_judge(episode):
    return episode["meta"]["query_type"] == "qt1" or episode["meta"]["case"] == "infeasible"


def evaluator(episode):
    qt, case = episode["meta"]["query_type"], episode["meta"]["case"]
    family, variant = ("qt4", qt[-1]) if qt.startswith("qt4-") else (qt, None)
    module = case + ("_" + variant if variant and case == "infeasible" else "")
    return importlib.import_module(f"src.pipelines.episode_evaluation.{family}.{module}").evaluate


class Session:
    """One live simulator per sequential episode; never checkpoint HTTP clients."""
    def __init__(self, client):
        self.client, self.calls, self.lock = client, [], RLock()
        self.cycle_predictions = {}
        self.technical_error = False

    def episode(self, request, context):
        from src.agents.tools import TOOL_REGISTRY
        tools = [{"name": name, "description": (fn.__doc__ or "").strip(),
                  "parameters": {"type": "object"}}
                 for name, fn in TOOL_REGISTRY.items()
                 if name not in {"finish", "add_device", "remove_device", "set_tick_interval"}]
        return Episode("simuhome", request, context, tools,
                       lambda name, args: self.call(name, args, origin="agent"), self)

    def native_call(self, tool, params):
        from src.agents.tools import get_tool_config, set_tool_config, ToolConfig, run_tool, TOOL_REGISTRY
        if tool not in TOOL_REGISTRY or tool in {"finish", "add_device", "remove_device", "set_tick_interval"}:
            raise ValueError("Unavailable SimuHome tool: " + tool)
        previous = get_tool_config()
        try:
            set_tool_config(ToolConfig(base_url=self.client.base_url, timeout=int(self.client.timeout)))
            return run_tool(tool, deepcopy(params))
        finally:
            set_tool_config(previous)

    def cycle_timing(self, home, device):
        with self.lock:
            attributes = device["attributes"]
            remaining = attributes["1.OperationalState.CountdownTime"]
            state = attributes["1.OperationalState.OperationalState"]
            end = (datetime.fromisoformat(home["current_time"]) + timedelta(seconds=remaining)
                   if state == 1 and remaining is not None else None)
            previous = self.cycle_predictions.get(device["device_id"])
            # Native timestamps/countdowns are whole seconds with independent rounding.
            if end and previous and previous["state"] == state and previous["ends_at"] and abs(
                    (datetime.fromisoformat(previous["ends_at"]) - end).total_seconds()) <= 1:
                return previous
            prediction = {"state": state, "ends_at": end.strftime("%Y-%m-%d %H:%M:%S") if end else None}
            self.cycle_predictions[device["device_id"]] = prediction
            return prediction

    def call(self, tool, params, *, origin="adapter"):
        params = deepcopy(params)
        with self.lock:
            call = {"tool": tool, "params": params, "origin": origin}
            try:
                if origin == "agent":
                    response = self.native_call(tool, params)
                elif tool == "schedule_workflow":
                    response = self.client.post("/schedule/workflow", json=params)
                elif tool in {"execute_command", "write_attribute"}:
                    args = dict(params)
                    device = args.pop("device_id")
                    route = "commands" if tool == "execute_command" else "attributes/write"
                    response = self.client.post(f"/devices/{device}/{route}", json=args)
                else:
                    routes = {"get_rooms": "/rooms", "get_home_state": "/home/state",
                              "get_room_devices": f"/rooms/{params.get('room_id')}/devices",
                              "get_device_structure": f"/devices/{params.get('device_id')}/structure",
                              "get_all_attributes": f"/devices/{params.get('device_id')}/attributes",
                              "get_room_states": f"/rooms/{params.get('room_id')}/states",
                              "get_current_time": "/time",
                              "get_workflow_status": f"/schedule/workflow/{params.get('workflow_id')}/status",
                              "get_environment_control_rules": f"/environment/control_rules/{params.get('state')}"}
                    response = self.client.get(routes[tool])
                # Repeated synchronization snapshots need no duplicate full-home payload in the log.
                observation = ({"current_time": response["data"]["current_time"],
                                "current_tick": response["data"]["current_tick"]}
                               if tool == "get_home_state" and origin == "adapter" and response["status"]["code"] == 200 else response)
                call.update(observation=observation, outcome={"ok": response["status"]["code"] == 200 and not response["error"],
                    "status_code": response["status"]["code"],
                    "error_type": (response["error"] or {}).get("type")})
                if not call["outcome"]["ok"]:
                    raise RuntimeError(str(response["error"]))
                return response
            except Exception as exc:
                call.setdefault("outcome", {"ok": False, "status_code": None, "error_type": type(exc).__name__})
                call["error"] = str(exc)
                response = getattr(exc, "response", None)
                code = response.status_code if response is not None else call["outcome"]["status_code"]
                if (code is not None and code >= 500) or (code is None and isinstance(exc, OSError)):
                    self.technical_error = True
                if response is not None:
                    call["outcome"]["status_code"] = response.status_code
                    call["observation"] = {"error": response.text}
                    call["error"] += ": " + response.text
                    raise RuntimeError(call["error"]) from exc
                raise
            finally:
                self.calls.append(call)

    def snapshot(self):
        state = self.call("get_home_state", {})["data"]
        values = {}
        for room_id, room in state["rooms"].items():
            for device in room.get("devices", []):
                values[f"device.{device['device_id']}.attributes"] = device["attributes"]
                if "1.OperationalState.CountdownTime" in device["attributes"]:
                    values[f"device.{device['device_id']}.cycle_timing"] = self.cycle_timing(state, device)
            for name, value in room.get("state", {}).items():
                values[f"room.{room_id}.{name}"] = value
        return state["current_time"], values


def load(path, category="all", case_ids=(), limit=0):
    if category != "all" and category not in SUPPORTED_CATEGORIES:
        raise ValueError("Unknown SimuHome category: " + category)
    path = Path(path).resolve()
    root = next((p for p in (path, *path.parents) if (p / "src/clients/smarthome_client.py").is_file()), None)
    if root is None:
        raise FileNotFoundError("--benchmark-path must point inside a SimuHome checkout (repository or data/benchmark).")
    directory = root / "data/benchmark" if path == root else path
    files = [directory] if directory.is_file() else sorted(directory.glob("*.json"))
    cases = []
    for file in files:
        episode = json.loads(file.read_text(encoding="utf-8"))
        case_category = episode["meta"]["query_type"] + "/" + episode["meta"]["case"]
        if (category == "all" or category == case_category) and (not case_ids or file.stem in case_ids):
            cases.append((file, episode))
    if not cases:
        raise ValueError("No supported SimuHome episodes match the selected category/IDs/path.")
    return root, cases[:limit] if limit else cases


def summarize(rows):
    evaluated = [r for r in rows if r["evaluation"]["score"] in (0, 1)]
    judge_usage = summarize_calls([call for row in rows for call in row.get("judge_calls", [])])
    return {"cases": len(rows), "incomplete_cases": sum(not r["completed"] for r in rows),
        "evaluated_cases": len(evaluated),
        "successes": sum(r["evaluation"]["score"] == 1 for r in evaluated),
        "success_rate": sum(r["evaluation"]["score"] == 1 for r in evaluated) / len(evaluated) if evaluated else None,
        "evaluation_errors": len(rows) - len(evaluated),
        "pending_evaluations": sum(r.get("evaluation_pending", False) for r in rows),
        "judge_calls": judge_usage["llm_calls"], "judge_call_errors": judge_usage["llm_call_errors"],
        "judge_call_latency_s": judge_usage["model_call_latency_s"],
        "errors": sum(bool(r["error"]) for r in rows),
        "runtime_errors": sum(bool(r["exception"]) for r in rows),
        **summarize_calls([c for r in rows for c in r["report"].get("llm_calls", [])]),
        "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows) if rows else 0}


class Judge:
    """Keep provider errors/raw votes visible before the native panel reduces them to A/B/Error."""
    def __init__(self, provider, seed):
        self.provider, self.seed, self.model = provider, seed, provider.model
        self.last_call = None

    def generate(self, messages):
        call = {"model": self.model, "seed": self.seed, "error": None,
                "input_tokens": None, "output_tokens": None}
        started = time.perf_counter()
        try:
            if self.model.split(":")[0] == "qwen3":
                messages = deepcopy(messages)
                messages[-1].content += "\n/no_think"
            answer = self.provider.generate(messages)
            call["raw_response"] = answer
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            if not answer:
                raise ValueError("Judge returned no verdict.")
            return answer
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            call["latency_s"] = time.perf_counter() - started
            self.last_call = call


def make_judges(config):
    """Create the configured native judge panel."""
    from src.agents.providers import OpenAIChatProvider
    key = os.getenv("SIMUHOME_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not config["model"] or not key:
        return []
    return [Judge(OpenAIChatProvider(model=config["model"], api_base=config["api_base"], api_key=key,
            temperature=config["temperature"], seed=seed, timeout=config["request_timeout"], max_retries=0), seed) for seed in config["seeds"]]


def run(path=None, *, category="all", limit=0, case_ids=(), output_dir=None, settings=None, system=None, config_path=None,
        judge_llms=None, resume=False):
    system = create_system(system, settings, config_path)
    native = load_config()["benchmarks"]["simuhome"]
    root, cases = load(path or os.getenv("SIMUHOME_PATH") or configured_path(native["path"]), category, case_ids, limit)
    sys.path.insert(0, str(root))
    from src.clients.smarthome_client import SmartHomeClient
    judge_config = dict(load_config()["judge"])
    for key in ("model", "api_base", "temperature"):
        value = os.getenv("SIMUHOME_JUDGE_" + key.upper())
        if value is not None:
            judge_config[key] = float(value) if key == "temperature" else value
    if judge_llms is None:
        judge_llms = make_judges(judge_config) if any(needs_judge(e) for _, e in cases) else []
    judge_config["models"] = [getattr(judge, "model", type(judge).__name__) for judge in judge_llms]

    client = SmartHomeClient(base_url=os.getenv("SIMULATOR_API_BASE_URL") or native["api_base"],
                             timeout=native["request_timeout"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    metadata = {"benchmark": "simuhome", "category": category, "system": system.metadata(), "judge": judge_config,
        "repository": str(root), "commit": commit.stdout.strip() or None, "api_base": client.base_url,
        "runtime_settings": native,
        "case_ids": [p.stem for p, _ in cases],
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in cases},
        "protocol": "Selected system owns discovery, planning and execution. ReAct uses upstream tool functions "
                    "and docstrings, excluding simulator administration. MAS registry conversion is confined to its wrapper. "
                    "Calls retain origin adapter/agent; MAS discovery/synchronization is counted as in the previous adapter. "
                    "No case labels, hidden eval goals or temporal_conflict metadata enter the public context.",
        "time_basis": "Simulator local virtual time; native format, no inferred timezone."}
    output = prepare_output(output_dir or f"logs/simuhome/{system.name}", metadata, resume)
    rows = []
    for file, episode in cases:
        cached = cached_case(output, file.stem, resume)
        if cached is not None:
            rows.append(cached)
            continue
        qt = episode["meta"]["query_type"]
        pending_judge = needs_judge(episode) and not judge_llms
        for judge in judge_llms:
            judge.last_call = None
        session, report, exception, error = Session(client), {}, None, None
        evaluation = {"score": -1}
        final_state = None
        started = time.perf_counter()
        try:
            reset = client.reset_simulation(episode["initial_home_config"])
            if reset["status"]["code"] != 200 or reset["error"]:
                raise RuntimeError(str(reset["error"]))
            public = {"user_location": episode.get("user_location"),
                      "reference_time": episode["initial_home_config"]["base_time"]}
            report = system_report(system, session.episode(episode["query"], public))
            exception, error = report.get("exception"), report.get("error")
            time.sleep(native["settle_time"])
            # Evaluator read-back is not an agent action and cannot fulfil required_actions.
            final_state = client.get("/home/state")["data"]
            steps = [{"action": call["tool"], "action_input": call["params"],
                      "observation": call.get("observation"), "origin": call["origin"]} for call in session.calls]
            steps.append({"action": "finish", "action_input": {"answer": report.get("answer", "")}})
            try:
                evaluation = ({"score": -1, "error_type": "Judge not configured", "judge": []} if pending_judge else
                    evaluator(episode)({"episode": episode, "final_home_state": final_state, "client": client,
                                       "tools_invoked": session.calls, "steps": steps}, judge_llms))
            except Exception as exc:
                evaluation = {"score": -1, "error_type": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:
            exception = error = f"{type(exc).__name__}: {exc}"
            evaluation = {"score": -1, "error_type": exception}
            session.technical_error = True
        elapsed = time.perf_counter() - started
        judge_calls = [judge.last_call for judge in judge_llms if getattr(judge, "last_call", None)]
        row = {"id": file.stem, "category": qt + "/" + episode["meta"]["case"], "input": episode["query"], "answer": report.get("answer", ""),
               "evaluation": evaluation, "error": error, "exception": exception, "latency_s": elapsed,
               "evaluation_pending": pending_judge,
               "evaluation_error": evaluation.get("error_type") if evaluation["score"] == -1 else None,
               "judge_calls": judge_calls, "judge_usage": summarize_calls(judge_calls),
               "report": report, "model_usage": report.get("model_usage", summarize_calls([])),
               "tools_invoked": session.calls, "final_home_state": final_state,
               "technical_error": bool(session.technical_error or report.get("technical_error") or report.get("backend_unavailable"))}
        row["completed"] = not (row["technical_error"] or pending_judge or evaluation["score"] == -1)
        rows.append(row)
        write_json(case_file(output, file.stem), row)
        write_json(output / "summary.json", {"overall": summarize(rows),
            "complete": len(rows) == len(cases) and all(r["completed"] for r in rows)})
        detail = error or row["evaluation_error"]
        print(f"[{len(rows)}/{len(cases)}] {file.stem}: score={evaluation['score']}, {elapsed:.2f}s"
              + (f" {'PENDING' if pending_judge else 'ERROR'}: {detail}" if detail else ""), flush=True)
        if report.get("backend_unavailable"):
            raise RuntimeError("Model backend unavailable; stopped after saving this case. Configure the endpoint, then --resume.")
    summary = {"overall": summarize(rows), "by_category": {
        name: summarize([row for row in rows if row["category"] == name]) for name in sorted({r["category"] for r in rows})}}
    summary["complete"] = all(r["completed"] for r in rows)
    write_json(output / "summary.json", summary)
    return summary
