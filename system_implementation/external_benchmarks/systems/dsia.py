"""DS-IA architecture adapted to native benchmark tools, not paper-score reproduction."""
import ast
import json
import re
import time

from jsonschema import Draft202012Validator, ValidationError

from model_client import BackendUnavailable, ChatClient, summarize_calls
from configuration import system_settings

DEFAULTS = system_settings("dsia")

# Original DS-IA templates; only the environment and command transport are adapted.
IA_PROMPT = """You are a smart home intent analyzer. Analyze if each operation in the user command is valid.
Check: 1) Does the room exist? 2) Does the device exist in that room? 3) Does the device support the action/attribute?
Output JSON: {"operations": [{"desc": "...", "valid": true/false, "reason": "..."}], "all_valid": true/false}"""

IA_USER = """<home_state>
{state}
</home_state>

<methods>
{methods}
</methods>

<command>
{command}
</command>

Analyze each operation and output JSON only."""

SYSTEM_PROMPT = """You are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed or answer the following question with the information provided only. The current status of the device and the methods it possesses are provided below, please only use the methods provided. Output "error_input" when operating non-existent attributes and devices. Only output machine instructions and enclose them in {}.
"""

EXAMPLES = """Here are a few examples, your output format should be consistent with the results provided in the example:
<example1>
    <User instructions:> Lower the air conditioner temperature in the guest bedroom by 1 degree.
    <Machine instructions:> {guest_bedroom.air_conditione.set_temperature(26)}
</example1>
<example2>
    <User instructions:> Turn off the lights in the living room.
    <Machine instructions:> {living_room.light.turn_off()}
</example2>
<example3>
    <User instructions:> Turn on the light in the living room.
    <Machine instructions:> {error_input}  (If there is no device or if there is no attribute or method that can be operated on the device, please output as "error_input".)
</example3>
<example4>
    <User instructions:> Increase the intensity of the aromatherapy in the corridor by 4%, set the brightness of the lights in the garage to 40, adjust the fan speed of the air conditioner in the living room to low, and set the volume of the media player in the living room to 0.
    <Machine instructions:> {corridor.aromatherapy.set_intensity(30),error_input,error_input,living_room.media_player.set_volume(0),} 
</example4>
"""

ANSWER_PROMPT = """Answer the user's request using only the supplied public context and actual tool
observations. Report rejections and execution errors honestly. Acceptance of a command or workflow
does not prove future physical success. Do not generate more actions. Return JSON {"answer":"..."}."""


def environment_text(episode, context):
    """Render HomeBench as upstream; expose other native tools through one API facade."""
    if episode.benchmark != "homebench":
        tools = {"environment." + tool["name"]: tool for tool in episode.tools}
        if episode.benchmark == "simuhome":
            tools = {name: {**tool, "unpack_arguments": True, "parameters": {
                "type": "object", "properties": {"args": tool["parameters"]},
                "required": ["args"], "additionalProperties": False}} for name, tool in tools.items()}
        methods = []
        for name, tool in tools.items():
            parameters = ",".join(key + ":" + json.dumps(schema, ensure_ascii=False)
                                  for key, schema in tool["parameters"].get("properties", {}).items())
            methods.append(f"{name}({parameters}); {tool.get('description', '')}\n")
        return json.dumps(context, ensure_ascii=False, indent=2), "".join(methods), tools

    home = episode.environment.initial_home
    lines = []
    for room, contents in home["home_status"].items():
        lines.append(room + ":\n")
        devices = {"": contents} if room == "VacuumRobot" else contents
        for device, state in devices.items():
            if device == "room_name":
                continue
            if device:
                lines.append("  " + device + "\n")
            indent = "    " if device else "  "
            lines.append(indent + "state: " + state["state"] + "\n")
            for name, attribute in state["attributes"].items():
                value = str(attribute["value"])
                if "options" in attribute:
                    value += " (options" + str(attribute["options"]) + ")"
                elif "lowest" in attribute:
                    value += " (range: " + str(attribute["lowest"]) + " - " + str(attribute["highest"]) + ")"
                lines.append(indent + name + ": " + value + "\n")
    tools = {tool["name"]: tool for tool in episode.tools if tool["name"] != "get_home_state"}
    methods = []
    for method in home["method"]:
        name = (method["device_name"] if method["room_name"] == "None" else
                method["room_name"] + "." + method["device_name"]) + "." + method["operation"]
        parameters = ",".join(p["name"] + ":" + p["type"] for p in method["parameters"])
        methods.append(f"{name}({parameters});\n")
    return "".join(lines), "".join(methods), tools


def machine_commands(raw, tools):
    """Parse literals only; generated code is never executed."""
    block = re.search(r"\{(.*)\}", raw, re.DOTALL)
    if block is None or not block[1].strip():
        raise ValueError("DS-IA returned no machine instructions.")
    source = "[" + block[1] + "]"
    nodes = ast.parse(source, mode="eval").body.elts
    def literal(node):
        try:
            return ast.literal_eval(node)
        except ValueError:
            return json.loads(ast.get_source_segment(source, node))
    commands, rejected = [], []
    for node in nodes:
        text = ast.get_source_segment(source, node)
        if isinstance(node, ast.Name) and node.id == "error_input":
            rejected.append({"request": "error_input", "reason": "Rejected by DS-IA."})
            continue
        if not isinstance(node, ast.Call):
            raise ValueError("Expected device method calls or error_input.")
        try:
            tool = tools[ast.unparse(node.func)]
            names = list(tool["parameters"].get("properties", {}))
            if len(node.args) > len(names):
                raise ValueError("Too many positional arguments.")
            arguments = dict(zip(names, map(literal, node.args)))
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg in arguments:
                    raise ValueError("Expanded or duplicate arguments are unavailable.")
                arguments[keyword.arg] = literal(keyword.value)
            arguments = json.loads(json.dumps(arguments, allow_nan=False))
            Draft202012Validator(tool["parameters"]).validate(arguments)
            commands.append({"name": tool["name"],
                             "arguments": arguments["args"] if tool.get("unpack_arguments") else arguments})
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            rejected.append({"request": text, "reason": str(exc)})
    return commands, rejected


class DSIA:
    name = "dsia"

    def __init__(self, settings=None, config_path=None, client=None):
        backend = ChatClient(settings, env_prefix="DSIA", config_path=config_path)
        backend.settings["json_mode"] = False  # Original command batches are not JSON.
        self.settings, self.client = backend.settings, client or backend

    def metadata(self):
        return {"name": self.name, "settings": self.settings, "backend": self.client.metadata(),
            "reference": "https://github.com/xxscy/DS-IA-smart/blob/main/code/model_test_ia.py",
            "implementation": "DS-IA adaptation: intent gate, independent batch generation, native tool grounding; no repair loop.",
            "protocol": "Original DS-IA prompts, few-shot examples and machine-instruction format; safe native call translation; "
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

        def ask(messages):
            raw, call = self.client.generate(messages, timeout=remaining())
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
            state_text, methods_text, tools = environment_text(episode, context)
            raw = ask([{"role": "system", "content": IA_PROMPT}, {"role": "user", "content":
                IA_USER.format(state=state_text, methods=methods_text, command=episode.request)}])
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
                reason = "; ".join(str(op.get("reason") or op.get("desc") or "Unavailable request") for op in invalid)
                report.update(status="finished", answer=reason,
                    rejected_requests=[{"request": episode.request, "reason": reason}])
            else:
                # Deliberately do not pass IA output to generation: this matches the source pipeline.
                prompt = (SYSTEM_PROMPT + "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"
                    + state_text + "\n</home_state>\n<device_method>\n     The following provides the methods to control each device in the current household:"
                    + methods_text + "\n</device_method>\n" + EXAMPLES
                    + "-------------------------------\nHere are the user instructions you need to reply to.\n<User instructions:> \n"
                    + episode.request + "\n<Machine instructions:>")
                raw = ask([{"role": "user", "content": prompt}])
                try:
                    commands, report["rejected_requests"] = machine_commands(raw, tools)
                except (ValueError, SyntaxError) as exc:
                    report.update(status="invalid_response", error=f"Invalid DS-IA response: {exc}")
                    commands = []
                for command in commands:
                    name, args = command["name"], command["arguments"]
                    try:
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
                    except (ValueError, KeyError) as exc:
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
                    raw = ask([{"role": "system", "content": ANSWER_PROMPT}, {"role": "user", "content":
                        json.dumps({**public, "trace": report["trace"], "rejected_requests": report["rejected_requests"]}, ensure_ascii=False)}])
                    try:
                        report["answer"] = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["answer"]
                        if not isinstance(report["answer"], str):
                            raise ValueError("answer must be a string.")
                    except (ValueError, KeyError, TypeError) as exc:
                        report["error"] = f"Invalid DS-IA answer: {exc}"
                report["status"] = "error" if report["error"] else "finished"
        except BackendUnavailable as exc:
            report["llm_calls"].append(exc.call)
            report.update(status="timeout" if exc.episode_timeout else "backend_error",
                          error=str(exc), backend_unavailable=not exc.episode_timeout)
        except Exception as exc:
            report.update(status="timeout" if isinstance(exc, TimeoutError) else "error",
                          error=f"{type(exc).__name__}: {exc}", exception=type(exc).__name__,
                          technical_error=not isinstance(exc, TimeoutError))
        report["model_usage"] = summarize_calls(report["llm_calls"])
        return report
