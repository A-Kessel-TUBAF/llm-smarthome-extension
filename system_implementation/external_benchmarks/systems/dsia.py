"""DS-IA architecture adapted to native benchmark tools, not paper-score reproduction."""
import json
import time

from jsonschema import Draft202012Validator, ValidationError

from ..common import summarize_calls
from .react import BackendUnavailable, ChatClient
from configuration import system_settings

DEFAULTS = system_settings("dsia")

# Original intent-analysis prompt and early-rejection rule from xxscy/DS-IA-smart.
IA_PROMPT = """You are a smart home intent analyzer. Analyze if each operation in the user command is valid.
Check: 1) Does the room exist? 2) Does the device exist in that room? 3) Does the device support the action/attribute?
Output JSON: {"operations": [{"desc": "...", "valid": true/false, "reason": "..."}], "all_valid": true/false}"""

PLAN_PROMPT = """Generate a complete ordered batch of native smart-home tool calls for the request.
Use only the supplied environment, exact tool names, schemas, units and capabilities.
Return JSON: {"commands":[{"name":"<tool name>","arguments":{...}}],
"rejected_requests":[{"request":"<unfulfillable clause>","reason":"<reason>"}]}.
Complete valid parts of mixed requests; reject unavailable devices/abilities. For ambiguity, ask for
clarification in the rejection reason instead of choosing an arbitrary device. Preserve conditions,
ordering and timing using native tools (SimuHome schedule_workflow for future/conditional actions).
This batch will be executed once, without replanning or substituting later observation values.
Home Assistant: follow tool_instructions and the requesting device's room context. Use an explicit
native name/area/floor for implicit targets; domain alone can affect devices in other rooms.
HomeBench: do not repeat successful device commands; every rejected clause becomes error_input.
Do not invent tools, observations or physical success from accepted future workflows."""

ANSWER_PROMPT = """Answer the user's request using only the supplied public context and actual tool
observations. Report rejections and execution errors honestly. Acceptance of a command or workflow
does not prove future physical success. Do not generate more actions. Return JSON {"answer":"..."}."""


class DSIA:
    name = "dsia"

    def __init__(self, settings=None, config_path=None, client=None):
        backend = ChatClient(settings, env_prefix="DSIA", config_path=config_path)
        self.settings, self.client = backend.settings, client or backend

    def metadata(self):
        return {"name": self.name, "settings": self.settings, "backend": self.client.metadata(),
            "reference": "https://github.com/xxscy/DS-IA-smart/blob/main/code/model_test_ia.py",
            "implementation": "DS-IA adaptation: intent gate, independent batch generation, native tool grounding; no repair loop.",
            "protocol": "JSON native calls instead of HomeBench code/few-shot prompts; schema validation; "
                        "SimuHome public inventory/structure reads; native response synthesis after execution. "
                        "Not an exact reproduction of published DS-IA scores."}

    def run_episode(self, episode):
        report = {"answer": "", "rejected_requests": [], "trace": [], "llm_calls": [],
                  "status": "error", "error": None}
        deadline = time.monotonic() + self.settings["episode_timeout"]

        def remaining():
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise TimeoutError("DS-IA episode time budget exhausted.")
            return budget

        def ask(prompt, data):
            raw, call = self.client.generate([{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}], timeout=remaining())
            report["llm_calls"].append(call)
            return raw

        def invoke(name, arguments):
            remaining()
            entry = {"action": name, "action_input": arguments}
            report["trace"].append(entry)
            try:
                entry["observation"] = episode.invoke(name, arguments)
                return entry["observation"]
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                raise

        try:
            context = dict(episode.context)
            structures = {}
            if episode.benchmark == "simuhome":
                # DS-IA needs a complete catalogue before its two model stages, as in HomeBench.
                home = invoke("get_home_state", {})["data"]
                context["home_state"] = home
                for room in home["rooms"].values():
                    for device in room.get("devices", []):
                        key = device["device_id"]
                        structures[key] = invoke("get_device_structure", {"device_id": key})["data"]
                context["device_structures"] = structures
            public = {"request": episode.request, "context": context, "tools": episode.tools}
            raw = ask(IA_PROMPT, public)
            try:
                analysis = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
                operations = analysis.get("operations", [])
                invalid = [op for op in operations if not op.get("valid", True)]
                ia_status = "all_invalid" if invalid and len(invalid) == len(operations) else "mixed" if invalid else "all_valid"
            except (ValueError, TypeError, AttributeError):
                # The original implementation proceeds with generation when IA parsing fails.
                analysis, ia_status = {"raw": raw}, "all_valid"
            report.update(intent_analysis=analysis, ia_status=ia_status)
            if ia_status == "all_invalid":
                reason = "; ".join(op.get("reason") or op.get("desc", "Unavailable request") for op in invalid)
                report.update(status="finished", answer=reason,
                    rejected_requests=[{"request": episode.request, "reason": reason}])
            else:
                # Deliberately do not pass IA output to generation: this matches the source pipeline.
                raw = ask(PLAN_PROMPT, public)
                plan = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
                Draft202012Validator({"type": "object", "required": ["commands"], "properties": {
                    "commands": {"type": "array", "items": {"type": "object", "required": ["name", "arguments"],
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}}},
                    "rejected_requests": {"type": "array", "items": {"type": "object", "required": ["request", "reason"],
                        "properties": {"request": {"type": "string"}, "reason": {"type": "string"}}}}}}).validate(plan)
                report["rejected_requests"] = plan.get("rejected_requests", [])
                tools = {tool["name"]: tool for tool in episode.tools}
                for command in plan["commands"]:
                    name, args = command["name"], command["arguments"]
                    try:
                        if name not in tools:
                            raise ValueError("Unavailable tool: " + name)
                        Draft202012Validator(tools[name]["parameters"]).validate(args)
                        if episode.benchmark == "simuhome":
                            if "room_id" in args and args["room_id"] not in home["rooms"]:
                                raise ValueError("Unknown room: " + args["room_id"])
                            if "device_id" in args and args["device_id"] not in structures:
                                raise ValueError("Unknown device: " + args["device_id"])
                            if name in {"execute_command", "write_attribute", "get_attribute"}:
                                cluster = structures[args["device_id"]]["endpoints"][str(args["endpoint_id"])]["clusters"][args["cluster_id"]]
                                if name == "execute_command":
                                    if args["command_id"] not in cluster["commands"]:
                                        raise ValueError("Unknown command: " + args["command_id"])
                                else:
                                    attribute = cluster["attributes"][args["attribute_id"]]
                                    if name == "write_attribute" and attribute["readonly"]:
                                        raise ValueError("Read-only attribute: " + args["attribute_id"])
                        report["trace"].append({"grounding": "accepted", **command})
                    except (ValueError, KeyError, ValidationError) as exc:
                        report["rejected_requests"].append({"request": name, "reason": str(exc)})
                        report["trace"].append({"grounding": "rejected", **command, "reason": str(exc)})
                        continue
                    try:
                        invoke(name, args)
                    except TimeoutError:
                        raise
                    except Exception:
                        report["error"] = "One or more native tool calls failed; see trace."
                if episode.benchmark != "homebench":
                    raw = ask(ANSWER_PROMPT, {**public, "trace": report["trace"],
                        "rejected_requests": report["rejected_requests"]})
                    report["answer"] = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answer"]
                    if not isinstance(report["answer"], str):
                        raise ValueError("answer must be a string.")
                report["status"] = "error" if report["error"] else "finished"
        except BackendUnavailable as exc:
            report["llm_calls"].append(exc.call)
            report.update(status="backend_error", error=str(exc), backend_unavailable=True)
        except Exception as exc:
            report.update(status="timeout" if isinstance(exc, TimeoutError) else "error",
                          error=f"{type(exc).__name__}: {exc}", exception=type(exc).__name__)
        report["model_usage"] = summarize_calls(report["llm_calls"])
        return report
