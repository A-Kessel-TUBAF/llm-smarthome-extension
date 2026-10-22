"""One independent ReAct agent: model -> one tool -> observation -> model."""
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..common import summarize_calls
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


class BackendUnavailable(RuntimeError):
    episode_timeout = False


class ChatClient:
    """Shared HTTP transport and model settings; no agent decisions."""
    def __init__(self, settings=None, env_prefix="REACT", config_path=None):
        defaults = system_settings(env_prefix.lower())
        cfg = system_settings(env_prefix.lower(), config_path)
        cfg["model"] = os.getenv(env_prefix + "_MODEL") or os.getenv("LLM_MODEL") or cfg["model"]
        cfg["api_base"] = os.getenv(env_prefix + "_API_BASE") or cfg["api_base"]
        cfg.update(settings or {})
        unknown = set(cfg) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {env_prefix} settings: " + ", ".join(sorted(unknown)))
        for name in ("max_steps", "max_llm_calls", "max_tokens", "context_window"):
            if name in cfg and (type(cfg[name]) is not int or cfg[name] <= 0):
                raise ValueError(name + " must be a positive integer.")
        for name in ("request_timeout", "episode_timeout", "temperature"):
            if not math.isfinite(cfg[name]) or cfg[name] < 0 or (name != "temperature" and not cfg[name]):
                raise ValueError(name + " must be finite and within its allowed range.")
        if type(cfg["seed"]) is not int or ("json_mode" in cfg and type(cfg["json_mode"]) is not bool):
            raise ValueError("seed must be an integer and json_mode a boolean.")
        self.settings = cfg
        self.model = cfg["model"]
        self.provider, separator, self.model_id = self.model.partition(":")
        if not separator or self.provider not in {"ollama", "openai"} or not self.model_id:
            raise ValueError("Model must be ollama:<model> or openai:<model> (also for compatible servers).")
        default = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") if self.provider == "ollama"
                   else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        self.base = (cfg["api_base"] or default).rstrip("/")
        parsed = urlsplit(self.base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("api_base must be an HTTP(S) URL without credentials or query parameters.")
        self.key = os.getenv(env_prefix + "_API_KEY") or (os.getenv("OPENAI_API_KEY") if self.provider == "openai" else None)

    def metadata(self):
        return {"model": self.model, "api_base": self.base, "credentials": "environment only"}

    def generate(self, messages, timeout=None, stop=None):
        cfg = self.settings
        if self.provider == "ollama":
            path = "/api/chat"
            body = {"model": self.model_id, "messages": messages, "stream": False,
                    "options": {"temperature": cfg["temperature"], "seed": cfg["seed"],
                                "num_predict": cfg["max_tokens"], "num_ctx": cfg["context_window"]}}
            if cfg.get("json_mode", False):
                body["format"] = "json"
            if stop:
                body["options"]["stop"] = stop
        else:
            path = "/chat/completions"
            body = {"model": self.model_id, "messages": messages, "temperature": cfg["temperature"],
                    "max_tokens": cfg["max_tokens"], "seed": cfg["seed"]}
            if stop:
                body["stop"] = stop
            if cfg.get("json_mode", False):
                body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        call = {"model": self.model, "input_tokens": None, "output_tokens": None, "error": None}
        started = time.perf_counter()
        try:
            request = Request(self.base + path, data=json.dumps(body, allow_nan=False).encode(), headers=headers)
            with urlopen(request, timeout=min(cfg["request_timeout"], timeout or cfg["request_timeout"])) as response:
                data = json.load(response)
            if self.provider == "ollama":
                content = data["message"]["content"]
                call.update(input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"))
            else:
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                call.update(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
            if not isinstance(content, str):
                raise ValueError("Model response has no text content.")
            call["raw_response"] = content
            return content, call
        except (HTTPError, URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            # Do not include server error bodies, request headers or API keys in logs.
            reason = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
            call["error"] = f"Model endpoint unavailable or invalid response ({reason})."
            error = BackendUnavailable(call["error"])
            error.call = call
            error.episode_timeout = (timeout is not None and timeout <= cfg["request_timeout"] and
                (isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)))
            raise error from None
        finally:
            call["latency_s"] = time.perf_counter() - started


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
