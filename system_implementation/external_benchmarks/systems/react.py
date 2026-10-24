"""One independent ReAct agent: model -> one tool -> observation -> model."""
import json
import time

from model_client import BackendUnavailable, ChatClient, summarize_calls
from configuration import system_settings


DEFAULTS = system_settings("react")

PROMPT = """You are a smart-home assistant. Fulfil the user's complete request using the available tools.
Return exactly one JSON object per turn:
{"action":"<exact tool name>","action_input":{...}}
You may add a short "thought" string. Call one tool, inspect its observation, then choose your next step.
Never invent devices, rooms, capabilities, arguments or observations. Use native names, units and schemas.
Discover or read missing information using tools. Preserve order, quantities, conditions and timing.
If a tool fails, reason from its error; do not claim success. Do not repeat a successful write unnecessarily.
Finish with {"action":"finish","action_input":{"answer":"<answer to the user>",
"rejected_requests":[{"request":"<unfulfillable clause>","reason":"<reason>"}]}}.
Use an empty rejected_requests list when none are rejected. Complete valid parts of a mixed request.
For ambiguity, ask a clarification question in the final answer; do not choose an arbitrary target.
Actions must be supplied tool names or finish. Report unfulfillable clauses only through
finish.action_input.rejected_requests; never invent an error-reporting tool.
HomeBench tools use a local state model, not a physics simulator.
For SimuHome, command or workflow acceptance alone does not establish future physical success.
For Home Assistant, use the native Assist tools and their targeting instructions.
"""



def parse_step(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    step = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(step, dict) or not isinstance(step.get("action"), str):
        raise ValueError("Expected an object with an action string.")
    args = step.get("action_input")
    if isinstance(args, str):  # Also accept SimuHome's JSON-string convention.
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("action_input must be a JSON object.")
    return step["action"], args


class ReAct:
    name = "react"

    def __init__(self, settings=None, config_path=None, client=None):
        backend = ChatClient(settings, config_path=config_path)
        self.settings, self.client = backend.settings, client or backend

    def metadata(self):
        return {"name": self.name, "settings": self.settings, "backend": self.client.metadata(),
                "implementation": "single model, explicit JSON ReAct, native benchmark tools; no MAS controller"}

    def run_episode(self, episode):
        # Only the public context and tool schemas enter the model prompt; no gold labels/goals.
        messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps({
            "benchmark": episode.benchmark, "request": episode.request,
            "context": episode.context, "tools": episode.tools}, ensure_ascii=False)}]
        report = {"answer": "", "rejected_requests": [], "trace": [], "llm_calls": [],
                  "status": "max_steps", "error": None}
        deadline = time.monotonic() + self.settings["episode_timeout"]
        for _ in range(self.settings["max_steps"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                report.update(status="timeout", error="Episode time budget exhausted.")
                break
            try:
                raw, call = self.client.generate(messages, timeout=remaining)
                report["llm_calls"].append(call)
            except BackendUnavailable as exc:
                report["llm_calls"].append(exc.call)
                report.update(status="timeout" if exc.episode_timeout else "backend_error",
                              error=str(exc), backend_unavailable=not exc.episode_timeout)
                break
            messages.append({"role": "assistant", "content": raw})
            try:
                action, args = parse_step(raw)
                if action == "finish":
                    answer, rejected = args.get("answer", ""), args.get("rejected_requests", [])
                    if not isinstance(answer, str) or not isinstance(rejected, list) or any(
                            not isinstance(r, dict) or not isinstance(r.get("request"), str) or
                            not isinstance(r.get("reason"), str) for r in rejected):
                        raise ValueError("finish requires an answer string and a list of request/reason objects.")
                    report.update(answer=answer, rejected_requests=rejected, status="finished")
                    report["trace"].append({"action": "finish", "action_input": args})
                    break
                event = {"action": action, "action_input": args}
                try:
                    event["observation"] = episode.invoke(action, args)
                except Exception as exc:
                    event["observation"] = {"error": f"{type(exc).__name__}: {exc}"}
                report["trace"].append(event)
                observation = event["observation"]
            except (ValueError, TypeError, KeyError) as exc:
                observation = {"error": "Invalid ReAct response: " + str(exc)}
                report["trace"].append({"raw_response": raw, "observation": observation})
            messages.append({"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)})
        if report["status"] == "max_steps":
            report["error"] = "Maximum ReAct steps reached without finish."
        report["model_usage"] = summarize_calls(report["llm_calls"])
        return report
